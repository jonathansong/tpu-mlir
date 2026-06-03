# NPU LayerGroup 软件架构文档

## 1. 定位

本文描述 `npu-layer-group` 的软件架构、模块边界和团队协作约定。设计语义、IR
形式和替代方案见 `07 layer-group.md`；本文重点说明代码如何组织、数据如何流动、
如何扩展和如何验证。

LayerGroup 的架构原则是：

- IR 结构负责语义：分组边界由 `npu.Group` / `npu.Yield` region 表达。
- dump/debug 文件负责展示：文本 dump 只用于人工查看，不作为 pass 间语义输入。
- helper 负责查询：后续 pass 通过封装 API 查询 group 关系。

## 2. 代码布局

LayerGroup 相关代码分布如下：

- `include/tpu_mlir/Dialect/Npu/Transforms/LayerGroup/LayerGroupDriver.h`
  和 `lib/Dialect/Npu/Transforms/LayerGroup/LayerGroupDriver.cpp`：
  pass 主驱动，负责选择分组方法、materialize `npu.Group`、清理旧属性并触发 dump。
- `LayerGroupPlan.h` / `LayerGroupPlan.cpp`：
  基础数据结构、连续 NPU op 收集、内存统计和 simple 分组策略。
- `LayerGroupMethod.h` / `LayerGroupMethod.cpp`：
  grouping method 注册和查询，目前支持 `simple`、`dp`、`opt2`。
- `method/DynamicProgramming.h` / `method/DynamicProgramming.cpp`：
  DP/opt2 分组策略，使用 `cyclesim` cost model 评估候选 group。
- `LayerGroupDump.h` / `LayerGroupDump.cpp`：
  从 `npu.Group` IR 结构重新收集分组信息并输出文本 dump。
- `LayerGroupUtils.h` / `LayerGroupUtils.cpp`：
  面向后续 pass 的 group 查询 helper。
- `include/tpu_mlir/Dialect/Npu/IR/NpuOps.td`：
  定义 `npu.Group` 和 `npu.Yield`。
- `include/tpu_mlir/Dialect/Npu/Transforms/Passes.td`：
  定义 `--npu-layer-group` pass 及其 options。

相关外部依赖：

- `include/tpu_mlir/Dialect/Npu/Backend/` 和
  `lib/Dialect/Npu/Backend/` 提供 target device 能力，例如 tensor memory、
  SRAM block 大小和 target name。
- `cyclesim` submodule 提供 DP LayerGroup cost model。

## 3. Pass 入口

`LayerGroup.cpp` 是 MLIR pass facade。它从 TableGen 生成的 option 中读取：

- `enable_vertical_tiling`
- `dump-output`
- `grouping-method`

然后调用：

```cpp
layer_group::runLayerGroup(getOperation(), enableVerticalTiling, dumpOutput,
                           groupingMethod)
```

实际逻辑集中在 `LayerGroupDriver.cpp`，避免 pass wrapper 和算法实现耦合。

命令行入口示例：

```bash
npu-opt input.mlir \
  --npu-layer-group="grouping-method=dp dump-output=layer_group_dump.txt" \
  -o output.mlir
```

默认 `grouping-method` 是 `simple`。`opt2` 是 `dp` 的别名，用于兼容既有命名。

## 4. 主流程

`runLayerGroup` 的执行顺序如下：

```text
ModuleOp
  -> module::init
  -> backend::getDevice
  -> lookupGroupingMethod
  -> walk func.func
  -> collectBaseGroups
  -> method->buildGroups
  -> materializeGroupOp
  -> writeLayerGroupDump
```

关键约定：

- 输入 IR 应该已经完成 `top.*` 到 `npu.*` 的 lowering。
- LayerGroup 只消费 `npu.*` compute op。
- 非 NPU dialect op、`func.return`、`npu.Group`、`npu.Yield` 都会切断或跳过分组。
- 当前不跨 block 分组。
- 如果分组需要 vertical tiling 但 pass 未启用该路径，会显式失败。

## 5. 核心数据结构

`GroupStats` 描述候选 group 的资源状态：

- `workingSetBytes`：候选 group 内 tensor working set 估算。
- `maxTensorBytes`：候选 group 中最大 tensor 大小。
- `requiresVerticalTiling`：target device 判断该 group 是否需要 vertical tiling。

`LayerGroupInfo` 是 planner 和 driver 之间的边界对象：

- `ops`：连续 NPU compute op 列表。
- `stats`：该 group 的资源统计。

`BaseGroup` 是同一 block 内连续 NPU compute op 片段。`BaseGroupList` 是一个
function 内所有可独立切分的 base group 集合。

这些结构只在 planning/materialization 阶段使用。分组完成后，语义以 IR 中的
`npu.Group` region 为准，不依赖这些 C++ 临时结构。

## 6. Base Group 收集

`collectBaseGroups` 在每个 `func.func` 中顺序扫描 block：

- 遇到真实 `npu.*` compute op 时加入当前 base group。
- 遇到非 NPU dialect op 时 flush 当前 base group。
- 遇到 `npu.Group` 或 `npu.Yield` 时不把它们当作 compute op。

`isNpuOp` 的判断逻辑是：

- dialect namespace 是 NPU。
- op 不是 `npu.Group`。
- op 不是 `npu.Yield`。

这个规则保证 LayerGroup 不会重复分组已经 materialize 的 group，也不会把 region
terminator 当作硬件 compute task。

## 7. 分组策略层

分组策略通过 `GroupingMethod` 注册。新增策略时只需要提供同一签名的 builder：

```cpp
void buildXxxGroups(const backend::Device &device,
                    llvm::ArrayRef<mlir::Operation *> baseGroup,
                    llvm::SmallVector<LayerGroupInfo> &groups);
```

### 7.1 simple

`buildSimpleGroups` 使用设备内存约束切分连续 NPU op：

- 对候选区间调用 `isLayerGroupValid`。
- 如果整个区间满足 target memory 约束，则尽量保留为一个 group。
- 如果不满足，则移动切分点，直到找到合法片段。
- 单 op 情况直接生成 group，并由后续 vertical tiling 检查决定是否失败。

### 7.2 dp / opt2

`buildDynamicProgrammingGroups` 枚举 base group 的所有连续区间：

- 先为每个候选区间计算 `GroupStats`。
- 对合法区间调用 `cyclesim::estimateLayerGroupCost`。
- 用 DP 表搜索总代价更低的切分。
- `opt2` 与 `dp` 走同一实现。

DP 策略依赖 `cyclesim` submodule。干净 clone 后需要初始化：

```bash
git submodule update --init --recursive cyclesim
```

如果没有初始化 submodule，包含 DP cost model 的构建和可移植 case 流程都不应被认为
是完整环境。

## 8. IR Materialization

`materializeGroupOp` 把一个 `LayerGroupInfo` 改写为 `npu.Group` region。流程是：

1. 建立 `groupOps` 集合。
2. 收集 group 外部输入，作为 `npu.Group` operands。
3. 收集 group 外部可见输出，作为 `npu.Group` results。
4. 在 first op 之前创建 `npu.Group`。
5. 为 region body 创建 block arguments。
6. 将 group 内 op 对外部 value 的使用替换为 block arguments。
7. 将 group 外部对内部 result 的使用替换为 `npu.Group` results。
8. 创建 `npu.Yield`，并将 grouped ops 移入 group body。

这个过程维护 MLIR region 的 SSA 合法性：

- group 内部可以使用 block argument 访问外部输入。
- group 外部只能使用 `npu.Group` result。
- region 内部 value 不直接逃逸到 region 外部。

`npu.Group` 是结构 wrapper，不对应硬件 task。硬件 task 仍来自 group body 内部的
真实 `npu.*` compute op。

## 9. Dump 和 Debug

`writeLayerGroupDump` 不读取 planner 阶段的临时数据。它会：

1. walk 当前 `ModuleOp` 中的 `npu.Group`。
2. 用 `collectGroupComputeOps` 收集 group body 内真实 NPU compute op。
3. 重新计算 `GroupStats`。
4. 输出 summary 和 group 明细。

`dump-output` 支持两种形式：

- 空字符串：不输出 dump。
- `-`：输出到 stdout。
- 文件路径：写入指定文本文件。

dump 是展示层，只服务人工查看、case 说明和 lit/FileCheck。后续 pass 不应把 dump
作为输入。

## 10. Helper API

`LayerGroupUtils` 提供当前最小查询接口：

- `getParentGroup(op)`：获取 op 所属的 parent `npu.Group`。
- `isInsideGroupBody(op)`：判断 op 是否在 group body 中。
- `walkGroupComputeOps(groupOp, callback)`：遍历 group body 内真实 compute op。
- `collectGroupComputeOps(groupOp, ops)`：收集 group body 内真实 compute op。

后续如果 address、size、dtype、shape 查询需要跨 group boundary，优先在 helper 或
公共 value 查询路径中扩展 alias 解析，而不是让各 pass 直接手写 region 结构遍历。

## 11. 后续 Pass 契约

LayerGroup 后的 IR 对后续 pass 有以下契约：

- `npu.Group` 只表示结构边界，不生成硬件 descriptor。
- `npu.Yield` 只表示 region 输出，不生成硬件 descriptor。
- Address assignment、weight map、IR dump 等 pass 应处理 body 内真实 compute op。
- 如果 pass 只统计硬件 task，应跳过 `npu.Group` / `npu.Yield`。
- 如果 pass 需要知道 group 关系，应使用 helper 查询。

这个契约保证了两件事：

- IR 中有明确的调度边界。
- 后端 task 粒度仍保持在真实 NPU compute op 上。

## 12. 可移植验证流程

团队成员在新机器上复现 transformer LayerGroup case 时，应使用仓库内可追踪资源：

```bash
git submodule update --init --recursive cyclesim
source envsetup.sh
PATH="$PWD/build/bin:$PATH" bash test/case/transformer_block/run_layergroup_flows.sh
```

该脚本覆盖：

- ONNX 生成。
- TOP pass-by-pass IR dump。
- NPU graph/weight/layergroup/address pass。
- simple LayerGroup dump。
- DP LayerGroup dump。
- cyclesim topology/layout/block timeline 提取。
- 使用 `test/case/transformer_block/cyclesim_ada200.cfg` 生成 ADA200 report。

不要在通用文档或脚本中依赖本机私有的 `cyclesim-main/` 目录。

## 13. 测试覆盖

当前 LayerGroup 相关测试和 case 包括：

- `test/Transforms/Npu/LayerGroup.mlir`：
  验证 `npu.Group` 结构和 dump 输出。
- `test/case/transformer_block/run_pass_ir.sh`：
  生成 pass-by-pass IR，并运行 simple LayerGroup。
- `test/case/transformer_block/run_layergroup_flows.sh`：
  运行 simple、DP 和 cyclesim report 的完整可移植流程。

推荐的基础验证命令：

```bash
llvm-lit test/Transforms/Npu/LayerGroup.mlir
PATH="$PWD/build/bin:$PATH" bash test/case/transformer_block/run_layergroup_flows.sh
```

## 14. 扩展指南

新增 grouping method：

1. 在 `LayerGroupMethod.cpp` 注册新名字和 builder。
2. 复用 `LayerGroupPlan` 的 `GroupStats`、`isLayerGroupValid` 和
   `computeGroupStats`。
3. 输出 `LayerGroupInfo`，不要直接创建 IR。
4. 在 `test/Transforms/Npu/LayerGroup.mlir` 或新增 lit case 中验证。

新增 dump 字段：

1. 优先从 `npu.Group` IR 结构重新收集信息。
2. 从 `npu.Group` / `npu.Yield` 结构获取分组信息。
3. 确保字段稳定，适合 FileCheck。

新增后续 pass 查询：

1. 优先扩展 `LayerGroupUtils` 或公共 value helper。
2. 避免在多个 pass 中复制 region traversal 或 alias 解析逻辑。
3. 明确区分 wrapper op 和真实 compute op。

## 15. 当前限制

当前实现仍有以下限制：

- 只在同一 block 内收集连续 NPU compute op。
- 不跨非 NPU dialect op 分组。
- `npu.Group` 不携带完整 schedule 或 tile plan。
- `enable_vertical_tiling` 目前只作为显式路径开关，完整 vertical tiling 执行策略仍需后续设计。
- DP cost model 是工程估算模型，适合分组排序和调试，不应被解释为最终硬件 cycle 精确模型。

这些限制不影响当前 LayerGroup 的核心职责：用 IR 结构表达分组语义，并为后续
address assignment、debug dump 和 case 验证提供稳定边界。
