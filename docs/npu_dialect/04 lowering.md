# lowering 设计

`convert-top-to-npu` 负责把已经完成量化信息导入的 `top dialect` IR 转成 `npu dialect` IR。它的输入不是原始 `f32` Top IR，而是经过 `import-quant-info` 处理后的 Top IR：每个需要量化的 operand/result tensor type 已经携带 `quant::UniformQuantizedType` 或 `quant::UniformQuantizedPerAxisType`。

因此 lowering 的职责不是再解析 encodings JSON，而是：

- 读取 Top op 输入/输出 quantized tensor type
- 校验 dtype、per-tensor/per-channel、scale、zero point、axis 是否满足目标 chip 约束
- 将 `top.*` op 替换为对应的 `npu.*` op
- 生成硬件执行需要的整数参数、查找表、定点权重/偏置等属性
- 按 chip 分发到不同 lowering 实现，例如 `ada200`、`ada300`

参考实现：`/workspace/toolchains/compiler/backend/ada200/top_to_npu`。

## 1. 与 import-quant-info 的分工

`import-quant-info` 输出的是类型语义，例如：

```text
tensor<1x1x240xf32>
  -> tensor<1x1x240x!quant.uniform<i8:f32, 7.812500e-03>>
```

`convert-top-to-npu` 继续完成目标硬件语义：

```text
top.Add(input0, input1) -> npu.Add(input0, input1)
```

换句话说，quant type 描述“这个 tensor 如何解释为实数”，lowering 描述“这个 op 在 NPU dialect 中用什么 op 表达”。如果某个 NPU op 的 ODS 已经不需要硬件派生属性，lowering 不应再预计算并写入这些属性。

## 2. Pass 总体结构

建议 pass 入口保持现在的三层结构：

```text
convert-top-to-npu
  1. 读取 module::getChip()
  2. 根据 chip 注册 lowering patterns
  3. applyPatternsAndFoldGreedily()
  4. module::updateModuleTypes()
```

伪代码：

```text
function runConvertTopToNpu(module):
    patterns = RewritePatternSet(context)

    switch module::getChip():
      ada200:
        ada200::populateTopToNpuConversionPatterns(patterns)
      ada300:
        ada300::populateTopToNpuConversionPatterns(patterns)
      default:
        return failure("unsupported chip")

    applyPatternsAndFoldGreedily(module, patterns)
    module::updateModuleTypes()
```

参考 Python 版 `top_to_npu.py` 的做法：

- 遍历 op 列表
- 通过 op 类型查找 handler
- lowering 前处理常量张量定点化
- handler 生成 NPU op
- 用新 op 替换旧 op
- 最后删除无用 tensor

在 MLIR 中，`NPU_OP_TRANSFORMMAP` 对应 `RewritePatternSet`；每个 Python handler 对应一个 `OpRewritePattern<top::XxxOp>`。

为了对应 Python 版 `base.py` 中的两种注册方式，C++ 侧也提供两类 pattern 注册 API：

```text
registerSimpleTopToNpuPattern<top::AbsOp, npu::AbsOp>(patterns, ctx)
registerComplexTopToNpuPattern<ada200::AddLowering>(patterns, ctx)
```

- simple pattern：只做 `top op -> npu op` 替换，复制 operands、result types 和 attrs，不执行额外计算。
- complex pattern：注册手写 `OpRewritePattern`，用于 Add、Conv、MatMul、LUT 等需要特定逻辑的算子。

## 3. 基类与 chip 分层

建议保留通用 `TopLowering<OpTy>` 基类，负责按输出 quant type 分派：

```text
TopLowering<OpTy>::matchAndRewrite(op):
    quantType = inferTopToNpuQuantType(op)
    if quantType == Int8:
        return LoweringINT8(rewriter, op)
    if quantType == Int16:
        return LoweringINT16(rewriter, op)
    return failure()
```

chip 侧通过宏或显式结构声明每个 op 的 lowering：

```text
ada200::AddLowering : TopLowering<top::AddOp>
  LoweringINT8(...)
  LoweringINT16(...)

ada300::AddLowering : TopLowering<top::AddOp>
  LoweringINT8(...)
  LoweringINT16(...)
```

目录建议：

```text
include/tpu_mlir/Conversion/TopToNpu/
  TopLowering.h
  LoweringAda200.h
  LoweringAda300.h

lib/Conversion/TopToNpu/
  TopToNpuPass.cpp
  TopLowering.cpp
  LoweringAda200.cpp
  LoweringAda300.cpp
  Ada200/Add.cpp
  Ada200/Conv.cpp
  Ada200/MatMul.cpp
  Ada200/Requantize.cpp
  Ada200/Lut.cpp
  Ada300/...
```

## 4. 量化信息读取

lowering 必须优先从 type 读取量化参数，不再读取 JSON 或依赖 `input_quant_*` 调试属性。

通用 helper：

```text
readQuantInfo(value):
    rankedType = value.type as RankedTensorType
    elementType = rankedType.elementType

    if elementType is UniformQuantizedType:
        return PerTensorQuantInfo {
          storageWidth,
          signedness,
          scale,
          zeroPoint,
          storageMin,
          storageMax
        }

    if elementType is UniformQuantizedPerAxisType:
        return PerAxisQuantInfo {
          storageWidth,
          signedness,
          scales[],
          zeroPoints[],
          quantizedDimension,
          storageMin,
          storageMax
        }
```

首版可以先支持 per-tensor INT8/INT16；遇到 per-channel 激活或当前硬件不支持的组合时应明确报错。Conv/MatMul 权重 per-channel 可作为下一阶段重点支持。

## 5. 常量张量定点化

Python 参考实现中，`top_to_npu.py` 在大多数算子 lowering 前调用：

```text
convert_constant_tensor_to_fixed_point(tensor)
```

MLIR 版也需要同等能力，但不建议混在每个 pattern 内重复实现。建议抽象成通用 helper：

```text
quantizeWeightIfNeeded(weightOp/resultValue, quantType):
    if value is not top.Weight:
        return value
    if weight payload is already integer:
        return value
    read scale / zeroPoint / storage range from value type
    quantized = round(floatData / scale) + zeroPoint
    clamp to storageMin/storageMax
    write back WeightOp payload or emit npu op attrs
```

实现策略有两种：

1. 修改 `top.Weight` 的 weight data，使后续 npu op 继续引用 weight value
2. 在 lowering 时把硬件需要的定点数组直接写入 `npu.*` 属性，例如 `weight_value`、`bias_value`

当前 `npu.Conv2d` / `npu.MatMul` 的 TableGen 已经倾向第二种属性化形式，可以先使用属性化方案，后续再根据 weight-map/codegen 设计调整。

注意 `BatchNorm2d` 等算子在 Python 版属于 `SKIP_CONST_QUANT_OPS`，首版也应保留跳过策略，避免误量化仍需浮点语义的常量。

## 6. 算子分类

参考 `lowering/base.py`，算子可以分为三类。

### 6.1 简单改名类

这类算子主要复制 operands/results/attrs，不需要生成复杂硬件参数：

- `Gather -> npu.Gather`
- `Permute/Transpose -> npu.Permute`
- `Abs -> npu.Abs`
- `Relu -> npu.Relu`
- `Pad -> npu.Pad`
- `ReduceMean/ReduceMax/ReduceSum -> npu.Reduce*`

设计：

```text
SimpleTopToNpuLowering<top::XxxOp, npu::XxxOp>
  - 检查输入/输出 tensor type 已量化
  - 复制 operands
  - 复制 result types
  - 复制 attrs
  - create npu::XxxOp
  - replace top::XxxOp
```

简单类也不能完全无校验：需要确认 dtype、rank、axis、shape 等满足 NPU op 约束。

### 6.2 参数派生类

这类算子需要从量化参数和权重内容生成硬件属性：

- `Add/Sub/Mul`
- `Concat`
- `Conv2d/ConvTranspose`
- `MatMul/Gemm`
- `Requantize`

这些 pattern 应独立实现，不使用简单改名。

### 6.3 LUT / PWL 类

非线性函数需要生成查找表或分段线性参数：

- `Sigmoid`
- `Tanh`
- `Exp`
- `Sin`
- `Gelu`
- `Pow`
- `Reciprocal`
- `Rsqrt`
- `Sqrt`

Python 版通过 `generate_lut(net, name, inputTensor)` 生成 `lookup_table`。MLIR 版建议统一成：

```text
generateLut(functionKind, inputQuant, outputQuant, targetChip)
  -> DenseI64ArrayAttr / DenseElementsAttr / external weight table ref
```

首版如果还没有 LUT 生成器，可以先使用 `npu.Pwl` 表达，并保留：

- `origin_op_name`
- `num_segments`
- `segment_data`

后续由 `graph-opt` 或 codegen 阶段补齐真实 LUT/PWL 数据。

## 7. Ada200 关键 lowering 规则

### 7.1 Add

当前 `npu.Add` 的 ODS 只包含两个输入和一个输出，不再保存 `multipliers`、`rshifts`、`zps`、`target_dtype` 或 `n_work/delta_n*` 等预计算属性。因此 MLIR 版 Add lowering 只做：

```text
top.Add(input0, input1) -> npu.Add(input0, input1)
```

处理流程：

- 校验 operand/result 数量
- 校验输入和输出都已经是 quantized tensor type
- 按输出 quant type 分派到 INT8 / INT16 lowering 分支
- 创建 `npu.Add`，保留原 operand 和 result type
- 不写入 Add 专用量化属性

Add 的量化参数仍然保存在 operand/result 的 quant type 中，后续 `codegen` 或更靠近硬件编码的阶段如果需要 `zp`、`shift`、`multiplier`，应从 type 中读取并计算，而不是在 lowering 阶段提前固化到 `npu.Add` 属性上。

标量常量 Add 是否 lowering 成 `npu.AddConst` 或专用 tensor-consistency op，应由 `npu.AddConst` 的 ODS 和后续 codegen 需求决定；在 `npu.Add` 不带预计算属性的前提下，不应复用 Python 版 `add.py` 中的 `n_work/delta_n*` 设计。

### 7.2 Conv2d / ConvTranspose

Python 版 `conv.py` 的核心派生：

```text
q_fused_bias = -Zx * sum(qw) + qb * (Sb / (Sw * Sx))
scale_work = Sw * Sx / Sy
shift_work = GetQuantShift(scale_work)
zp_y = output.zero_point
```

MLIR 版输入：

- input quant type：`Sx`, `Zx`
- weight quant type：`Sw`，可 per-tensor 或 per-channel
- bias quant type：`Sb`
- output quant type：`Sy`, `Zy`
- weight/bias data：来自 `top.Weight`

输出：

```text
npu.Conv2d {
  weight_value = [...]
  bias_value = [...]
  shift_work = [...]
  zp_y = ...
  kernel_shape / pads / strides / dilations / group / shape attrs
}
```

约束：

- 首版支持 `INT8` / `INT16`
- 权重 per-channel 时 `scale_work` 长度应等于 output channel
- 输出 zero point 若为 per-channel，必须所有通道相同，否则报错
- `ConvTranspose` 需要额外设置 `conv_type` 或使用独立 `npu.ConvTranspose`

### 7.3 MatMul / Gemm

Python 版 `matmul.py` 与 Conv 类似：

```text
sum_qw = sum(weight, axis=0)
q_fused_bias = -Zx * sum_qw + qb * (Sb / (Sw * Sx))
scale_work = Sw * Sx / Sy
shift_work = GetQuantShift(scale_work)
zp_y = output.zero_point
```

MLIR lowering 需要处理：

- weight shape 规范化，例如 `[1, K, C] -> [K, C]`
- bias 可选
- `right_transpose` / `left_transpose` / `output_transpose`
- per-channel weight scale 与输出通道对齐

输出 `npu.MatMul` 或 `npu.Gemm`，并写入：

- `weight_value`
- `bias_value`
- `shift_work`
- `zp_y`

### 7.4 Requantize

Python 版语义：

```text
out = clip(round((in - in_zp) * scale_in / scale_out) + out_zp)
```

若使用 power-of-2 量化：

```text
shift = n_in - n_out
```

MLIR 版推荐统一使用 scale：

```text
realScale = input.scale / output.scale
(multiplier, rshift) = toMultiplierShift(realScale)
```

若硬件 requantize op 只接受 shift，则需要约束 `scale_in / scale_out` 必须为 2 的整数幂，否则 lowering 报错或走更通用的 multiplier/rshift 表达。

### 7.5 Concat

Python 版保存每个输入和输出的 `n/zp`：

```text
input_n_list
input_zp_list
output_n
output_zp
```

MLIR 版应读取每个输入/output quant type：

- 若所有输入 type 与输出 type 完全一致，可直接 `top.Concat -> npu.Concat`
- 若 scale/zp 不一致，硬件若支持 concat 内部 requant，则写入输入/输出量化数组
- 若硬件不支持，要求上游插入 `top.Requantize`，lowering 阶段报错

## 8. 广播、shape 与 tensor consistency

Python 版 `broadcast_elemwise_inputs()` 会为 elementwise 常量输入做形状广播。MLIR 版建议把广播处理分成两层：

1. Top IR 规范化阶段尽量显式插入 `top.Reshape` / `top.Tile` / `top.Broadcast`
2. lowering 阶段只处理硬件可直接支持的简单广播或 scalar pattern

对于 `Add/Sub/Mul`：

- 两个动态 tensor：优先要求 shape 已一致，或硬件明确支持广播
- tensor + scalar constant：可 lowering 成专用 npu op 或写入 immediate 属性
- tensor + broadcast constant：可在 lowering 中把 constant payload broadcast 到目标 shape，但要避免对大 tensor 造成 IR 膨胀

## 9. 错误处理原则

lowering 不应静默猜测硬件行为。以下情况应直接报错：

- operand/result 缺少 quantized tensor type
- dtype 不在当前 chip 支持范围内
- per-channel axis 与硬件期望不一致
- per-channel zero point 非常量但硬件只支持 per-tensor zero point
- weight/bias payload 缺失或 shape 不匹配
- 同一 op 的输入输出量化类型组合需要 requantize，但 IR 中没有显式边界
- 当前 chip 未实现该 op

错误信息应包含：

- op name / loc
- op 类型
- 失败的 operand/result index
- 期望 dtype / shape / quant type

## 10. 首版实现范围

建议按风险递进：

1. 完成通用 quant type reader
2. 完成 `TopLowering` 基类和 chip pattern 注册
3. 完成 `top.Add -> npu.Add`
   - INT8
   - INT16
   - per-tensor only
4. 完成简单改名类算子
5. 完成 `Requantize`
6. 完成 `Concat`
7. 完成 `Conv2d` / `MatMul`
8. 完成 LUT/PWL 类算子
9. 补齐 per-channel 权重路径

当前 Add 样例的最小链路：

```text
top.Input / top.Add
  --import-quant-info-->
top.Input / top.Add with quantized tensor type
  --convert-top-to-npu-->
npu.Add with quantized tensor type
```

## 11. 与后续 pass 的契约

`convert-top-to-npu` 输出的 IR 应满足：

- module 中业务算子已从 `top.*` 替换为 `npu.*`
- npu op operands/results 保留 quantized tensor type
- npu op attrs 已包含 codegen 所需的硬件参数
- 权重/偏置要么已经定点化，要么以明确属性形式写入 npu op
- 不再依赖 encodings JSON

后续 pass 分工：

- `graph-opt`：做 NPU dialect 内部图优化、融合、PWL/Hypot 等模式
- `weight-map`：建立权重和常量数据映射
- `layer-group`：做 layer group 和 tiling 策略
- `address-assign`：分配 GMEM/LMEM 地址
- `codegen`：把 npu op 和 attrs 编码成 rxmodel / backend 指令数据
