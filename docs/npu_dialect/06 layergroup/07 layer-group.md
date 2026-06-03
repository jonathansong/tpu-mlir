# NPU Layer Group 设计文档

## 1. 摘要

`npu-layer-group` pass 负责把连续可调度的 `npu.*` compute op 组织成 layer group，并在 MLIR IR 中使用 `npu.Group` region 直接表达分组边界。

`npu.Group` 是结构化 wrapper：它接收 group 外部输入，在 body 中保存 grouped ops，并通过 `npu.Yield` 返回 group 对外可见输出。额外文本 dump 仅用于调试，不作为 layer group 的语义来源。

## 2. 背景

Layer group 是 NPU 后端调度和局部内存规划的基础。分组结果需要被后续 address assignment 和调试 dump 等模块消费。

当前设计把 layer group materialize 为 `npu.Group` region。设计分层遵循 `tpu-mlir` 的风格：IR 结构负责语义，dump/debug 文件负责展示，helper 负责局部查询。

相关模块：

- `include/tpu_mlir/Dialect/Npu/Backend/`
- `include/tpu_mlir/Dialect/Npu/IR/NpuOps.td`
- `lib/Dialect/Npu/Backend/`
- `lib/Dialect/Npu/Transforms/LayerGroup/`
- `lib/Dialect/Npu/Transforms/AddressAssign/`

## 3. 目标

- 在 MLIR IR 中直接表达 layer group 边界。
- 保留原始 grouped ops，不把 compute op 折叠成外部文件或 opaque blob。
- 让 `npu.Group` 成为后续 pass 可识别的调度边界。
- 保持 address assignment 面向真实 compute op 工作，不让 wrapper op 生成硬件 task。
- 提供稳定的结构化 IR、文本 dump/debug 输出和 helper 查询接口，便于调试、FileCheck 和后续 pass 消费。

## 4. 非目标

- 不在 `npu-layer-group` 中完成具体硬件调度或 descriptor 生成。
- 不让 `npu.Group` 本身对应硬件指令。
- 不在 `npu-layer-group` 中执行 `top.*` 到 `npu.*` 的 lowering。
- 不在当前阶段实现跨非 NPU dialect op 的分组。
- 不以文本 dump 作为编译管线中的语义输入。
- 不在本设计中定义完整 vertical tiling 执行策略；当前仅保留显式失败逻辑。后续如需展示 tiling / schedule 信息，应通过结构化属性或外部 debug 输出设计。

## 5. 需求和约束

Layer group 需要满足以下约束：

- 只分组真实 NPU compute op，`npu.Group` / `npu.Yield` 不参与再次分组。
- 非 NPU dialect op、`func.return` 等会切断 base group。
- group 的工作集需要满足目标设备 SRAM / tensor memory 约束。
- region 内部 SSA value 不能直接逃逸到 region 外部，必须通过 terminator 显式返回。
- 后续 address assignment 仍需要看到 body 内部真实 compute op。
- group 后 IR 需要能被 lit/FileCheck 稳定验证。

当前设备约束由 NPU backend target model 提供，例如 Ada200 / Ada300 的 tensor memory、SRAM block 大小和 block 数量。LayerGroup 只消费这些能力参数，不拥有芯片相关实现。

## 6. 设计概览

整体 pass 流程：

```text
平铺的 NPU function
    |
    v
收集连续的 NPU compute op
    |
    v
按 grouping method 和内存约束切分 group
    |
    v
materialize npu.Group region
    |
    v
address assignment 消费 group 后 IR
```

`npu.Group` 的职责是表达结构边界，不负责生成硬件 task。后续 pass walk IR 时会进入 group body，处理内部真实 compute op，并跳过 `npu.Group` / `npu.Yield` wrapper。

## 7. 详细设计

### 7.1 IR 操作

`npu.Group` 定义为 variadic operands、variadic results、单 region op：

```mlir
%group_result = "npu.Group"(%external_input0, %external_input1) ({
^bb0(%group_input0: tensor<...>, %group_input1: tensor<...>):
  // grouped npu compute ops
  "npu.Yield"(%internal_output) : (tensor<...>) -> ()
}) : (...) -> (...)
```

`npu.Yield` 是 `npu.Group` body 的 terminator。它类似 `func.return` / `scf.yield`，用于把 region 内部 value 映射为 `npu.Group` results。

### 7.2 IR 示例

分组前：

```mlir
%0 = "npu.Add"(%arg0, %arg1) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
%1 = "npu.Sub"(%0, %arg1) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
return %1 : tensor<1x4xf32>
```

分组后：

```mlir
%0 = "npu.Group"(%arg0, %arg1) ({
^bb0(%in0: tensor<1x4xf32>, %in1: tensor<1x4xf32>):
  %1 = "npu.Add"(%in0, %in1) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %2 = "npu.Sub"(%1, %in1) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  "npu.Yield"(%2) : (tensor<1x4xf32>) -> ()
}) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
return %0 : tensor<1x4xf32>
```

对应关系：

- `npu.Group` operands 是 group 外部输入。
- group body block arguments alias 到对应 operands。
- `npu.Yield` operands 是 group 内部输出。
- `npu.Group` results alias 到对应 `npu.Yield` operands。
- group 外部只能使用 `npu.Group` results，不能直接使用 body 内部 value。

### 7.3 Group 收集

`collectBaseGroups` 在每个 `func.func` 中按 block 顺序收集连续 NPU compute op：

- `npu.*` compute op 加入当前 base group。
- 非 NPU dialect op 会 flush 当前 base group。
- `npu.Group` / `npu.Yield` 不是 compute op，不会被再次收集。

当前不跨 block 做分组。

### 7.4 Group 切分

当前支持的 grouping method：

- `simple`：默认策略，按连续 NPU op 和内存约束切分。
- `dp` / `opt2`：动态规划策略，用于搜索更优切分。

每个候选 group 会计算：

- `working_set_bytes`
- `max_tensor_bytes`
- `requires_vertical_tiling`

当 group 超过设备内存约束且当前还没有可用的 vertical tiling 执行策略时，pass 显式失败，而不是静默生成非法 group。

### 7.5 Materialization 流程

每个 `LayerGroupInfo` 会被 materialize 成一个 `npu.Group`：

1. 收集 group 外部输入作为 `npu.Group` operands。
2. 收集 group 外部可见输出作为 `npu.Group` results。
3. 在 group body 中创建 block arguments。
4. 将 grouped ops 的外部 operand 替换为对应 block argument。
5. 将 group 外部 uses 替换为对应 `npu.Group` result。
6. 将 grouped ops 移入 group body。
7. 在 body 末尾创建 `npu.Yield`。

这个过程保证 SSA 合法性：region 内部 value 不直接被 region 外部使用。

### 7.6 摘要、调试和查询

结构语义只以 `npu.Group` / `npu.Yield` region 为准。

需要展示分组摘要时，从 IR 中实时遍历 `npu.Group` 生成 dump/debug 输出。例如 dump 可以打印 group 数量、group 内 op 列表、工作集估算、内存约束检查结果等，但这些信息只用于人工查看和调试，不作为后续 pass 的输入语义。

后续 pass 需要局部查询 group 关系时，应使用 helper API 封装结构查询，而不是读取属性。典型查询包括：

- 判断某个 op 是否位于 `npu.Group` body 内。
- 获取 op 所属的 parent `npu.Group`。
- 遍历 `npu.Group` body 中的真实 compute op。
- 通过 `npu.Group` operands / body block arguments / `npu.Yield` operands / results 解析 group 边界 value。
- 从 `npu.Group` 结构生成稳定的 FileCheck 或文本 dump 摘要。

这样可以保持三层职责清晰：IR 结构负责语义，dump/debug 文件负责展示，helper 负责查询。

### 7.7 Alias 解析

引入 group alias 解析，避免 region 边界影响 address 和 size 查询：

- group body block argument alias 到对应的 `npu.Group` operand。
- `npu.Group` result alias 到对应的 `npu.Yield` operand。

读取 address、size、dtype 或 shape 相关信息时，应先解析 alias，再读取真实 value 上的属性或类型。

### 7.8 后续 Pass 约定

`npu.Group` 是结构 wrapper，不是硬件 compute task。

后续 pass 约定：

- Address assignment 只统计真实 NPU compute op 的生命周期，跳过 `npu.Group` / `npu.Yield`。
- IR dump 统计 NPU op 数量时跳过 `npu.Group` / `npu.Yield`。
- Weight map pass 若只面向 compute op，也应跳过 `npu.Group` / `npu.Yield`。

这样可以同时满足两个需求：IR 中有结构化 group 边界，后端 task 仍由真实 compute op 生成。

### 7.9 与 TopToNpu Lowering 的边界

`npu-layer-group` 的输入应当已经是 NPU dialect IR。它只消费 `npu.*` compute op，不负责把 `top.*` op 临时转换成 `npu.*` op。

如果为了打通 pipeline 需要临时把简单 `top` 算子替换成同名或近似同构的 `npu` 算子，应在 `TopToNpu` conversion 注册点完成，例如：

```cpp
registerSimpleTopToNpuPattern<top::AbsOp, npu::AbsOp>(patterns, ctx);
```

只有当 `npu.*` op 已经存在之后，LayerGroup 才参与分组；如果仍然看到 `top.*`，它会切断 base group，而不是在 LayerGroup 内做 lowering。

## 8. 替代方案

### 8.1 输出外部 Dump 文件

通过文本文件记录 group 列表，便于人工查看和调试。

缺点是外部文件不属于 IR，不适合作为 pass 间语义传递方式。当前保留 dump/debug 文件作为展示功能，但不作为语义来源。

### 8.2 使用 Function-level Kernel Graph

也可以把 group 变成单独 `func.func` 或 kernel graph function，再通过 call 表达边界。

这种方式适合更强的 kernel 抽象，但会引入 symbol、call graph、函数参数和返回值维护成本。当前需求只是表达同一 function 内的调度分组，因此 region wrapper 更直接。

## 9. 限制和后续工作

当前限制：

- 只处理同一 block 内连续 NPU compute op。
- 不跨非 NPU dialect op 分组。
- `npu.Group` 目前只表达结构边界，不携带完整硬件 schedule。
- vertical tiling 尚未结构化表达，完整执行策略需要后续设计。
- group alias 解析需要在读取 address / size 的公共路径中持续保持一致。

后续可以考虑：

- 为 `npu.Group` 增加 verifier，检查 `npu.Yield` operand 数量和 result 数量一致。
- 如后续需要保存 schedule / tiling 信息，优先设计为 `npu.Group` 上的结构化 attr，或外部 debug dump。
- 在调试 dump 中显式打印 group 层级、op 列表和内存估算。
- 增加 helper API，统一封装 group parent 查询、body op 遍历和 group value alias 解析。
- 引入更完整的 tiling / schedule 表达。

## 10. 验证

构建目标：

```bash
cd /workspace/rx-mlir
source envsetup.sh
cmake --build build --target npu-opt tpuc-opt -j$(nproc)
```

重点回归：

```bash
PATH="/workspace/rx-mlir/build/bin:/workspace/rx-mlir/install/bin:$PATH" \
  /workspace/rx-mlir/build/bin/llvm-lit -sv \
  test/Transforms/Npu/LayerGroup.mlir \
  test/Transforms/Npu/PipelineSkeleton.mlir
```

验证点：

- `--npu-layer-group` 输出包含 `npu.Group`。
- grouped ops 位于 `npu.Group` body 内。
- group body 以 `npu.Yield` 结束。
- 文本 dump/debug 输出可从 `npu.Group` 结构生成稳定摘要。
- 后续 pass 通过 helper 查询 group 关系。
- `--npu-weight-map --npu-layer-group --npu-address-assign` 能处理 group 后 IR。
