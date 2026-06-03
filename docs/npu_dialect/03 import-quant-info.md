# import-quant-info 设计

`import-quant-info` 是 `Top -> Npu` lowering 前的量化信息导入 pass。它的核心职责不是给 op 附加一组私有量化属性，而是把 encodings JSON 中的量化信息写入 MLIR value type：每个节点的输入、输出 tensor type 都应被转换成带 MLIR 标准量化 element type 的 ranked tensor type。

类型约定：

- per-tensor 使用 `quant::UniformQuantizedType`
- per-channel 使用 `quant::UniformQuantizedPerAxisType`

这样后续 `convert-top-to-npu` 只需要读取 operand/result type，就能得到 scale、zero point、axis 等量化信息，不再重复解析 JSON 或依赖命名规则。

## 1. 输入数据来源

`import-quant-info` 读取 encodings JSON，主要使用三类字段：

- `activation_encodings`：节点输入/输出量化的主数据源。它按 `op_name` 索引，每个节点包含 `input[]` 和 `output[]`，顺序分别对应 op operands 和 op results。
- `param_encodings`：权重、bias 等参数张量的主数据源。导入时按张量名匹配 op 的输入 value。
- `tensor_encodings`：按张量名补充、兼容和一致性校验的数据源，主要用于 graph input、standalone weight/result 等不能完全依赖节点顺序的 value。

内部建议统一成下面的数据结构：

```text
TensorEncoding:
  name
  dtype
  scales[]
  zeroPoints[]
  n[]
  realMin
  realMax
  symmetric
  encType
  quantizedDimension

NodeEncoding:
  nodeName
  inputs[]
  outputs[]
  internalOps[]
```

encoding 解析需要参考 `/workspace/toolchains/compiler/frontend/quantization.py` 中 `_normalize_encoding_entry()` 的行为：

- 单个 dict 直接使用
- list 中只有一个 dict 时直接使用该项
- list 中有多个 dict 时，标量字段取第一个非空值，`scale`、`zero_point`、`n`、`real_min`、`real_max` 等数组字段拼接为 per-channel 数组

## 2. 节点与 value 绑定

节点级绑定依赖 Top IR 中的 `op_name` 属性：

```text
TopOpNameIndex:
  opNameToOp : Map<string, Operation*>
```

构建规则：

1. 遍历 Top 模块中的所有 op
2. 读取 `op_name` 字符串属性
3. 若属性存在，则登记到 `opNameToOp`
4. 若重复出现同名 `op_name`，导入阶段直接报错

找到 op 后，节点输入输出按顺序绑定：

- `activation_encodings[op_name].input[i]` 绑定到 `op->getOperand(i)`
- `activation_encodings[op_name].output[i]` 绑定到 `op->getResult(i)`
- 数量不匹配时报错，避免静默错配

张量名解析只作为补充能力，主要用于：

- `top.Input`：按输入 op 的 `loc` 名或函数输入约定匹配 JSON 张量名
- `top.Weight`：按 weight 的 loc 名匹配 `param_encodings` 或 `tensor_encodings`
- 普通算子 result：缺失节点级信息时，才退回到 loc/result 来源关系等兼容规则

## 3. 类型构造规则

`enc_type` 判定与前端保持一致：

- `enc_type == "PER_TENSOR"`：构造 `quant::UniformQuantizedType`
- `enc_type == "PER_CHANNEL"`：构造 `quant::UniformQuantizedPerAxisType`
- 若 `enc_type` 缺失，则根据 `scale` 或 `zero_point` 的元素个数推断；任一字段长度大于 1 时按 per-channel，否则按 per-tensor

`zero_point` 语义也与前端保持一致：普通 tensor 导入时使用 `abs(zero_point)` 作为有效 zero point。

伪代码：

```text
function buildQuantizedElementType(q):
    q = normalizeEncoding(q)
    encType = q.encType
    if encType is null:
        encType = inferFromScaleAndZeroPointCount(q)

    storageType = integerTypeFromDtype(q.dtype)
    expressedType = f32
    storageMin = integerLowerBound(q.bitWidth, q.symmetric)
    storageMax = integerUpperBound(q.bitWidth, q.symmetric)
    zeroPoints = abs(q.zeroPoints)

    if encType == "PER_TENSOR":
        return UniformQuantizedType(
            storageType,
            expressedType,
            q.scales[0],
            zeroPoints[0],
            storageMin,
            storageMax)

    if encType == "PER_CHANNEL":
        return UniformQuantizedPerAxisType(
            storageType,
            expressedType,
            q.scales,
            zeroPoints,
            q.quantizedDimension,
            storageMin,
            storageMax)

    return failure("unsupported quant encoding type")
```

`dtype` 与量化范围：

- `INT2`、`INT4`、`INT8`、`INT16`、`INT32` 分别对应 2/4/8/16/32 bit signed 存储
- `UINT2`、`UINT4` 若后端没有独立 storage type，可用 unsigned 8-bit storage 承载，同时保留真实 bit width
- `symmetric=false` 时按 unsigned range：`[0, 2^bitWidth - 1]`
- `symmetric=true` 时按 signed range：`[-2^(bitWidth-1), 2^(bitWidth-1)-1]`

## 4. Pass 处理流程

```text
function importQuantInfoForAllOps(topMlirModule, encodingsJson):
    enc = parseEncodingsJson(encodingsJson)
    nodeMap = enc.activationEncodings
    paramMap = enc.paramEncodings
    tensorMap = enc.tensorEncodings

    opNameIndex = buildTopOpNameIndex(topMlirModule)

    for each (nodeName, nodeEncoding) in nodeMap:
        op = opNameIndex.lookup(nodeName)
        if op is null:
            emitWarningOrError(nodeName, "cannot find top op by op_name")
            continue

        inputQuants = normalizeEncodingList(nodeEncoding.input)
        outputQuants = normalizeEncodingList(nodeEncoding.output)

        if inputQuants.size != op.numOperands:
            return failure("activation input count mismatch")
        if outputQuants.size != op.numResults:
            return failure("activation output count mismatch")

        updateOperandValueTypes(op, inputQuants)
        updateResultTypes(op, outputQuants)
        attachDebugQuantAttrsIfNeeded(op, inputQuants, outputQuants, nodeEncoding)

    for each op in topMlirModule:
        for each operand in op.operands:
            tensorName = resolveTensorName(operand)
            q = paramMap.lookup(tensorName)
            if q is not null:
                updateValueType(operand, normalizeEncoding(q))

    applyMatMulAddBiasQuantOverride(topMlirModule, paramMap)
    fillMissingQuantTypesInDimensionChains(topMlirModule)

    for each top.Input/top.Weight not covered above:
        tensorName = resolveTensorName(op.result)
        q = tensorMap.lookup(tensorName)
        if q is not null:
            updateStandaloneResultType(op, normalizeEncoding(q))

    updateFunctionTypesFromBodies(topMlirModule)
```

更新类型时需要注意：

- `OpResult` 可以通过重建 op 或更新 result type 的方式落地，具体取决于 op 是否允许 in-place type mutation
- `BlockArgument` 需要同步更新 function type 和 block argument type
- operand use 本身不能单独拥有独立类型；如果同一个 value 被多个消费者以不同量化参数使用，需要插入显式 cast/requantize 或在前序图规范化中拆分 value
- `none`、控制类 operand 或无 tensor element type 的 value 不参与量化 type 更新

## 5. 与前端一致的补充规则

`import-quant-info` 应覆盖 `quantization.py` 中对图级量化参数的关键修正逻辑：

1. 参数张量量化：遍历每个 op 的输入 value，若张量名命中 `param_encodings`，则使用该 encoding 更新权重或 bias 的 tensor type。
2. MatMul+Add bias 模式：若 `Add` 的一个输入来自 `MatMul` 输出，另一个输入是常量 bias，且 bias 名称存在于 `param_encodings`，则将 `Add` 输出 type 替换为 `MatMul` 输出 type，避免 bias 的 per-channel 量化污染 Add 输出。
3. 维度算子链继承：`Slice`、`Transpose`、`Reshape`、`Unsqueeze`、`Squeeze`、`Gather`、`Concat`、`Pad` 等仅改变形状或布局的算子，如果输出缺少量化 type，应在该连通域内继承最近已量化 value 的 element type，并保留当前 tensor shape。
4. 混合精度边界：如果同一个 producer value 被多个消费者要求不同 bit width 或不同 scale，`import-quant-info` 不应静默覆盖 type；应依赖前序 QDQ/requantize 插入，或在本 pass 中显式报错并要求插入 requantize 边界。

## 6. Add 样例

以 `test/npu/ops/add/add_top.mlir` 和 `test/npu/ops/add/add.json` 为例：

- `activation_encodings["module_add"].input[0]` 对应 `top.Add` 的第 0 个 operand
- `activation_encodings["module_add"].input[1]` 对应 `top.Add` 的第 1 个 operand
- `activation_encodings["module_add"].output[0]` 对应 `top.Add` 的第 0 个 result

这些 encoding 会直接驱动 type 改写：

- 第 0 个输入 value：`UniformQuantizedType(scale=0.0078125, zeroPoint=0)`
- 第 1 个输入 value：`UniformQuantizedType(scale=0.0078125, zeroPoint=0)`
- 第 0 个输出 value：`UniformQuantizedType(scale=0.015625, zeroPoint=0)`

同时可以用 `tensor_encodings` 做输入/输出张量名的一致性校验和 graph input 补齐：

- `top.Input loc("t.1")` 对应 `tensor_encodings["t.1"]`
- `top.Input loc("t.3")` 对应 `tensor_encodings["t.3"]`
- `top.Add loc("5_Add")` 的输出可按结果名兼容匹配到 `tensor_encodings["5"]`

因此首版实现推荐以 `activation_encodings` 的节点输入/输出顺序作为主绑定关系，`tensor_encodings` 作为张量名补齐和一致性校验来源。

## 7. 输出与后续 lowering 契约

`import-quant-info` 的核心输出是类型变化，例如：

```text
tensor<1x1x240xf32>
  -> tensor<1x1x240x!quant.uniform<i8:f32, 7.812500e-03>>
```

对于 per-channel 权重或激活，输出 element type 应是 `UniformQuantizedPerAxisType`，并保留每个 channel 的 scales、zero points 和 quantized dimension。

必要时可以额外保留轻量属性用于调试或过渡期兼容，例如：

- `op_name`
- `input_quant_scales`
- `input_quant_zps`
- `output_quant_scales`
- `output_quant_zps`
- `quant_dtype`
- `quant_axis`
- `quant_symmetric`

但 lowering 的主输入应优先是 operand/result type，而不是这些附加属性。对于 `Add` 这样的首批落地算子，也不建议让 lowering 依赖 Add 专用量化属性；如果确实需要过渡期兼容，应把属性读取封装在 helper 中，并逐步切换到 type 读取。
