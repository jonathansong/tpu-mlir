# Ada300 LLM Top Dialect 到 TPU Dialect Lowering Pipeline 设计

## 1. 文档目标

本文设计面向 Ada300 的 LLM inference 编译链路：

```text
LLM Top dialect
  -> Ada300 TPU dialect
  -> TPU graph optimization
  -> weight map
  -> layer group / tiling
  -> memory allocation
  -> Ada300 schedule
  -> Ada300 codegen
```

首个可交付版本聚焦纯文本 Qwen2、Qwen3、Llama 类模型。优先跑通单层
`block_0` 的 prefill 和 decode，再扩展到多层模型、embedding、lm-head 和量化优化。

本文是设计计划，不表示所列能力已经实现。

## 2. 范围与非目标

### 2.1 首版范围

- 静态 shape。
- `batch = 1`。
- 文本 LLM transformer block。
- 独立编译 prefill 与 decode 子网。
- FP16 activation 和 KV cache。
- FP16 weight baseline。
- W8A16、W4A16 作为 baseline 跑通后的优化项。
- flat TPU pipeline 先行，layer group 和 tiling 后续加入。

### 2.2 首版非目标

- VLM / VIT。
- LoRA。
- 动态 batch。
- `greedy_head` / `sample_head`。
- 跨层融合。
- 跨 block layer group。
- 在硬件 ABI 未确认前固化 descriptor 字段。

## 3. 输入 Top IR

LLM Python converter 不经过 ONNX，而是直接使用 MLIR Python bindings 构造
`top.*` IR。模型被拆成多个子网：

| 子网 | 用途 |
|---|---|
| `embedding` | prompt token embedding |
| `embedding_cache` | decode 阶段单 token embedding |
| `block_i` | 第 `i` 层 prefill |
| `block_cache_i` | 第 `i` 层 decode，显式接收 KV cache |
| `lm_head` | hidden state 转 logits 或 token id |
| `greedy_head` / `sample_head` | 可选采样逻辑 |
| `vit` | VLM 可选视觉编码器 |

首版只要求 `block_0` 和 `block_cache_0` 可编译、可运行、可验证。

### 3.1 Prefill Block

典型 `block_i` 数据流：

```text
input_states
  -> top.RMSNorm
  -> top.MatMul / top.A16MatMul for Q, K, V
  -> top.Reshape
  -> top.Rope
  -> top.FAttention
  -> top.MatMul for O projection
  -> top.Add residual
  -> RMSNorm + MLP
  -> top.Add residual
  -> return output_states, k_cache, v_cache
```

### 3.2 Decode Block

典型 `block_cache_i` 输入：

```text
input_states, position_ids, attention_mask, history_k, history_v
```

单 token Q、K、V 计算完成后，新 KV 使用 `top.Concat` 或 `top.Insert`
更新 cache，再进入 `top.FAttention`。

## 4. 当前仓库能力与缺口

### 4.1 已有能力

- `convert-top-to-tpu` 已按 `module::getChip()` 分发 Ada200 / Ada300 lowering。
- Ada300 已有设备模型：
  - 最小地址粒度：64 B。
  - SRAM block：256 KiB。
  - SRAM block 数量：5。
  - 本地共享内存总量：1.25 MiB。
- `tpu-layer-group` 已能通过 `tpu.target` 获取设备约束。
- TPU dialect 已有基础算子，例如 `tpu.Add`、`tpu.MatMul`、`tpu.Concat`、
  `tpu.Permute`、`tpu.Slice`、`tpu.Softmax`、`tpu.LoadInput`、
  `tpu.LoadWeight` 和 `tpu.Store`。
- Ada200 codegen 已有 registry、task、engine dependency 和 memory pool 骨架。

### 4.2 必须补齐的缺口

- Ada300 lowering 目前只有 `top.Add` 占位实现，并明确返回失败。
- Ada300 codegen 尚未实现。
- TPU dialect 缺少 LLM 核心算子：
  - `tpu.A16MatMul`
  - `tpu.RMSNorm`
  - `tpu.Rope`
  - `tpu.FAttention`
  - `tpu.KVCacheUpdate`
- 地址分配仍包含 Ada200 假设：
  - GMEM 容量硬编码为 4 MiB。
  - SMEM 使用双页模型。
  - API 名称仍为 `assignAda200GmemAddresses()` 和
    `assignAda200SmemAddresses()`。
- `getCurrentDevice()` 固定返回 Ada200，目标设备信息没有完全贯穿 pipeline。

## 5. 关键设计原则

### 5.1 保留 LLM 高层语义

首版不应把 `RMSNorm`、`RoPE` 和 `FAttention` 过早拆成大量低层 elementwise
op。TPU dialect 应保留硬件友好的高层语义边界，由 Ada300 codegen 决定：

- 是否调用单 kernel。
- 是否拆分为多个 kernel。
- 如何 tiling。
- 如何组织 DMA 与 compute overlap。

### 5.2 正确性基线先于量化优化

先使用 FP16 weight 和 FP16 activation 跑通闭环，再加入 W8A16 和 W4A16。
不要一开始同时调试 lowering、量化格式、SRAM tiling 和硬件 kernel。

### 5.3 Prefill 与 Decode 分开优化

两类子网形态不同：

```text
Prefill:
  mq > 1
  序列维分块
  tiled QK^T -> streaming softmax -> PV

Decode:
  mq = 1
  流式读取 history K/V
  KV cache 原地更新
  避免整体搬运 cache
```

### 5.4 结构语义保留在 IR 中

- `tpu.Group` 表达 layer group。
- `tpu.KVCacheUpdate` 表达 cache 更新。
- tile / schedule 使用结构化属性表达。
- 外部文本 dump 只用于调试，不作为 pass 间语义来源。

## 6. 目标 Pipeline

### 6.1 Flat Baseline Pipeline

首版使用平铺 TPU IR：

```bash
tpuc-opt block_0.mlir \
  --tpu-init="chip=ada300" \
  --tpu-llm-normalize \
  --convert-top-to-tpu \
  --graph-opt \
  --tpu-weight-map \
  --tpu-gmem-alloc \
  --tpu-smem-alloc \
  --tpu-codegen \
  -o block_0_ada300.mlir
```

### 6.2 优化版 Pipeline

```text
Top LLM IR
  -> tpu-init
  -> tpu-llm-normalize
  -> convert-top-to-tpu
  -> graph-opt
  -> tpu-weight-map
  -> tpu-layer-group
  -> tpu-tile-plan
  -> tpu-gmem-alloc
  -> group-local SMEM allocation
  -> tpu-schedule
  -> tpu-codegen
  -> Ada300 command stream / executable artifact
```

## 7. Phase 0：设备模型收口

这一阶段必须先完成。否则后续 layer group、tiling 和 address assignment
可能建立在错误设备约束上。

### 7.1 Target Propagation

执行：

```bash
--tpu-init="chip=ada300"
```

应同时写入：

```mlir
module.chip = "ada300"
tpu.target = "ada300"
```

### 7.2 Device API

扩展 `tpu::backend::Device`：

```cpp
virtual uint64_t getGmemBytes() const = 0;
virtual uint64_t getSramBankCount() const = 0;
virtual uint64_t getSramBankBytes() const = 0;
virtual uint64_t getWeightMemoryBytes() const = 0;
```

可根据硬件文档继续增加 dtype、alignment 和 engine 能力查询。

### 7.3 移除 Ada200 固定假设

将：

```cpp
assignAda200GmemAddresses(moduleOp)
assignAda200SmemAddresses(moduleOp)
```

重命名为：

```cpp
assignGmemAddresses(moduleOp)
assignSmemAddresses(moduleOp)
```

内部使用：

```cpp
const Device &device = backend::getDevice(moduleOp);
```

不要继续依赖固定返回 Ada200 的 `getCurrentDevice()`。

### 7.4 验收

- Ada200 回归不变。
- Ada300 `tpu.mem.summary` 使用 Ada300 约束。
- Ada300 SRAM 规划显示 5 个 256 KiB block，而不是 Ada200 双页模型。
- 未知 target 明确报错，不静默回退到 Ada200。

## 8. Phase 1：定义 LLM TPU Dialect

### 8.1 首版新增 Ops

```text
tpu.A16MatMul
tpu.RMSNorm
tpu.Rope
tpu.FAttention
tpu.KVCacheUpdate
```

继续复用已有 ops：

```text
tpu.LoadInput
tpu.LoadWeight
tpu.MatMul
tpu.Add
tpu.Mul
tpu.Concat
tpu.Permute
tpu.Slice
tpu.Store
```

`Reshape`、部分 `Permute` 可以优先作为 alias / view，由 layout 分析决定是否生成
硬件 task。

### 8.2 建议 ODS

```text
tpu.A16MatMul
  input, weight, scale, zero_point, bias
  weight_bits, q_group_size, right_transpose

tpu.RMSNorm
  input, gamma
  eps, accumulate_dtype

tpu.Rope
  input, sin, cos
  rope_mode

tpu.FAttention
  q, k, v, mask, scratch
  batch, q_head, kv_head, head_dim, mq, mk, scale

tpu.KVCacheUpdate
  history, current, position
  axis, mode = concat | insert
```

### 8.3 Verifier

每个新增 op 应增加 verifier：

- shape 与 rank 校验。
- dtype 校验。
- `head_dim`、`q_head`、`kv_head` 一致性校验。
- `mq`、`mk` 与输入 shape 一致性校验。
- W4A16 / W8A16 scale 和 zero point shape 校验。
- KV cache history/current shape 校验。

## 9. Phase 2：LLM Normalize

新增：

```bash
--tpu-llm-normalize
```

该 pass 位于 `convert-top-to-tpu` 前，只负责将不同来源的 Top IR 收敛成稳定形式：

```text
top.MatMul(weight quant metadata)
  -> canonical top.A16MatMul

top.Insert / top.Concat used for KV cache
  -> canonical KV update form

top.Reshape
  -> view-like reshape where possible

RMSNorm 展开图
  -> top.RMSNorm

RoPE 展开图
  -> top.Rope

attention 展开图
  -> top.FAttention
```

当前 LLM Python converter 已经直接生成多数高层 op。Normalize pass 仍然有价值：

- 保证不同 converter 输出一致。
- 允许未来 ONNX 前端复用 Ada300 LLM 后端。
- 为 codegen 提供稳定契约。

## 10. Phase 3：Ada300 TopToTpu Lowering

目录建议：

```text
lib/Conversion/TopToTpu/Ada300/
  Input.cpp
  Weight.cpp
  Return.cpp
  MatMul.cpp
  A16MatMul.cpp
  RMSNorm.cpp
  Rope.cpp
  FAttention.cpp
  KVCacheUpdate.cpp
  Add.cpp
  Mul.cpp
  SiLU.cpp
  View.cpp
```

### 10.1 P0：跑通 Transformer Block

| Top op | TPU op | 说明 |
|---|---|---|
| `top.Input` | `tpu.LoadInput` | GMEM 到 SRAM |
| `top.Weight` | `tpu.LoadWeight` | 权重映射 |
| `top.A16MatMul` | `tpu.A16MatMul` | LLM linear 核心 |
| `top.MatMul` | `tpu.MatMul` | FP16 fallback |
| `top.RMSNorm` | `tpu.RMSNorm` | 建议 FP32 accumulate |
| `top.Rope` | `tpu.Rope` | 保留 `rope_mode` |
| `top.FAttention` | `tpu.FAttention` | 区分 prefill / decode |
| `top.Add` | `tpu.Add` | residual |
| `top.Mul` | `tpu.Mul` | gated MLP |
| `top.SiLU` | `tpu.SiLU` 或 LUT / PWL | 激活 |
| `top.Insert` | `tpu.KVCacheUpdate` | decode cache |
| `top.Concat` | `tpu.KVCacheUpdate` 或 `tpu.Concat` | decode fallback |
| `func.return` | `tpu.Store` + return | 输出回写 |

### 10.2 P1：外围子网

```text
Gather
TopK
Slice
Permute
Reshape
Tile
Unsqueeze
Softmax
```

这些算子用于 embedding、lm-head 和部分模型变体。

### 10.3 P2：后续能力

```text
Mlp 融合
采样 head
LoRA
VLM / VIT
```

## 11. Phase 4：量化策略

### 11.1 正确性 Baseline

```text
Activation: FP16
Weight:     FP16
Norm:       FP32 accumulate
KV cache:   FP16
```

### 11.2 优化顺序

```text
FP16 baseline
  -> W8A16
  -> W4A16
  -> 可选 KV cache 量化
```

Top dialect 已有 `top.A16MatMul`，用于表达 W8A16 / W4A16 LLM linear：

```text
y_f16 = x_f16 x (quantized_w.to(f16) * scale_f16)
```

### 11.3 Dtype 建议

现有 `TopToTpuQuantType` 只覆盖 INT8 / INT16 activation，不足以描述 LLM。
建议增加：

```cpp
enum class TpuDataType {
  F32,
  F16,
  BF16,
  I8,
  I4,
  W8A16,
  W4A16,
};
```

Lowering 需要显式校验：

- activation dtype。
- weight dtype。
- scale dtype。
- per-channel / per-group shape。
- `q_group_size`。
- zero point 是否存在。

## 12. Phase 5：内存规划

### 12.1 分层

```text
External memory:
  model weights
  KV cache
  graph input / output
  command buffers

SRAM:
  当前 tile activation
  Q / K / V tile
  attention partial state
  RMSNorm reduction scratch
  DMA 双缓冲
```

KV cache 不应整体进入 SRAM。decode 阶段按 head 或 token block 流式搬运。

### 12.2 Ada300 SRAM

Ada300 当前设备模型：

```text
5 banks * 256 KiB = 1.25 MiB
```

初始建议，仅作为启发式策略：

```text
bank0: input / Q tile
bank1: weight tile A
bank2: weight tile B 或 K tile
bank3: output / V tile
bank4: scratch / reduction / ping-pong
```

具体 bank 映射必须由 kernel 资源需求驱动，不应写死为全局规则。

### 12.3 Group 属性

为 `tpu.Group` 增加结构化计划属性：

```mlir
tpu.Group {
  schedule = "streaming",
  tile_m = ...,
  tile_n = ...,
  tile_k = ...,
  sram_banks = [...],
  double_buffer = true
}
```

### 12.4 Address Assignment

职责拆分：

```text
tpu-gmem-alloc:
  graph input / output
  external KV cache
  weight source address

tpu-smem-alloc:
  flat baseline 的 activation

group-local smem allocation:
  tile buffer
  scratch
  double buffer
  bank conflict avoidance

tpu-codegen:
  descriptor
  config
  param
  instruction resources
```

## 13. Phase 6：Prefill 与 Decode 调度

### 13.1 Prefill

输入特征：

```text
mq > 1
mk = prompt length
```

Attention 建议：

```text
for each Q tile:
  for each K/V tile:
    DMA K/V tile
    compute QK^T
    update streaming softmax state
    accumulate PV
  write output tile
```

目标：

- 避免完整 attention matrix 落入 external memory。
- Q/K/V tile 尽量驻留 SRAM。
- DMA 与 TE / VE 执行重叠。

### 13.2 Decode

输入特征：

```text
mq = 1
mk = history length + 1
```

Attention 建议：

```text
compute current Q/K/V
update external KV cache in-place
for each history K/V tile:
  DMA K/V tile
  compute QK^T
  update softmax state
  accumulate PV
write output token
```

目标：

- 避免复制完整 KV cache。
- 将 external memory bandwidth 作为主要优化指标。
- KV cache 更新与 attention 流式读取明确建模。

## 14. Phase 7：Ada300 Codegen

新增目录：

```text
lib/Dialect/Tpu/Transforms/CodeGen/Ada300/
  Driver.cpp
  OpRegister.cpp
  LoadInput.cpp
  LoadWeight.cpp
  Store.cpp
  MatMul.cpp
  A16MatMul.cpp
  RMSNorm.cpp
  Rope.cpp
  FAttention.cpp
  KVCacheUpdate.cpp
  Add.cpp
  Mul.cpp
  SiLU.cpp
```

### 14.1 Registry

沿用 Ada200 registry 结构：

```cpp
OpRegistry createAda300OpCodeGenRegistry();
```

### 14.2 Adapter 职责

每个 op codegen adapter 负责：

```text
1. 校验 dtype、shape、layout 和 assigned memory
2. 读取 tile plan
3. 申请 descriptor / scratch
4. 生成 DMA、TE、VE task
5. 建立跨 engine dependency
6. 输出结构化调试摘要
```

### 14.3 Engine 调度

建议显式区分：

```text
DMA / TMA:
  input、weight、KV cache、output 搬运

TE:
  MatMul、A16MatMul、QK^T、PV

VE:
  RMSNorm、RoPE、Add、Mul、SiLU、softmax reduction
```

具体 engine 命名与能力以 Ada300 ISA 和 runtime ABI 为准。

## 15. Graph Optimization

### 15.1 首版

只做不改变数值语义的规范化：

- 移除冗余 view。
- 合并连续 reshape。
- 识别 KV cache 更新。
- 标记 alias。
- 保留高层 LLM op。

### 15.2 后续融合

```text
RMSNorm + QKV projection
RMSNorm + MLP gate/up
SiLU + Mul
RoPE + attention input preparation
decode KVCacheUpdate + FAttention
```

融合必须由 kernel 能力驱动。不要只为减少 IR op 数量而融合。

## 16. 验证策略

### 16.1 Level 1：Top 到 TPU FileCheck

单 op：

```text
A16MatMul
RMSNorm
Rope
FAttention
KVCacheUpdate
Add
Mul
SiLU
```

子图：

```text
prefill block
decode block
```

### 16.2 Level 2：TPU Pass FileCheck

```text
target propagation
weight map
layer group
5-bank SRAM plan
alias propagation
tiling failure diagnostics
```

### 16.3 Level 3：Codegen Golden

```text
descriptor 字段
task 顺序
DMA / TE / VE dependency
memory summary
command stream snapshot
```

### 16.4 Level 4：数值验证

```text
单 op 对比 NumPy / PyTorch
单层 prefill
单层 decode
多 token decode
2 层 tiny Llama 端到端
完整小模型
```

FP16 baseline 建议阈值：

```text
cosine similarity > 0.999
```

W8A16 和 W4A16 应分别建立精度基线。

### 16.5 Level 5：性能验证

```text
prefill tokens/s
decode tokens/s
time-to-first-token
external memory bandwidth
KV cache bandwidth
SRAM utilization
kernel 时间占比
DMA / compute overlap
```

## 17. 建议测试文件

```text
test/Transforms/lowering/Ada300LlmA16MatMul.mlir
test/Transforms/lowering/Ada300LlmRMSNorm.mlir
test/Transforms/lowering/Ada300LlmRope.mlir
test/Transforms/lowering/Ada300LlmFAttention.mlir
test/Transforms/lowering/Ada300LlmKVCacheUpdate.mlir
test/Transforms/lowering/Ada300LlmPrefillBlock.mlir
test/Transforms/lowering/Ada300LlmDecodeBlock.mlir

test/Transforms/Tpu/Ada300Target.mlir
test/Transforms/Tpu/Ada300MemoryAlloc.mlir
test/Transforms/Tpu/Ada300LayerGroup.mlir
test/Transforms/Tpu/Ada300CodeGen.mlir

test/tpu/llm/tiny_llama/
  export_top.sh
  compile_ada300.sh
  compare.py
```

## 18. 交付里程碑

### M0：设备模型收口

- 修复 target propagation。
- 移除 Ada200 固定设备依赖。
- 参数化 GMEM / SRAM。
- Ada200 回归不变。

### M1：FP16 MatMul + Add

- Ada300 `Input`、`Weight`、`MatMul`、`Add`、`Store` lowering。
- Ada300 flat memory allocation。
- Ada300 codegen 最小闭环。

### M2：单层 Prefill

- `RMSNorm`。
- `RoPE`。
- `FAttention` prefill。
- MLP。
- 单层数值对比通过。

### M3：单层 Decode

- `KVCacheUpdate`。
- external KV cache。
- decode attention 流式读取。
- 多 token decode 数值对比通过。

### M4：Tiny Llama 端到端

- 多层 block。
- embedding。
- lm-head。
- 完整生成流程。

### M5：量化与 Tiling

- W8A16。
- W4A16。
- layer group。
- 5-bank SRAM tile plan。
- DMA / compute overlap。

### M6：外围能力

- sampling head。
- LoRA。
- 动态 batch。

### M7：VLM

- VIT。
- 多模态 position embedding。
- VLM 专用算子。

## 19. Steps 1-5 详细实施清单

本节将前五个交付步骤展开为可执行任务。每一步都应形成独立、可验证的
patch。不要在前一步尚未通过回归时提前叠加后续能力。

### 19.1 Step 1：M0 设备模型收口

#### 目标

让 Ada300 target 信息完整贯穿 lowering、layer group、memory allocation 和
codegen。此阶段不实现 LLM op。

当前部分公共 helper 固定返回 Ada200，地址分配也仍使用 Ada200 的 4 MiB
GMEM 和双页 SMEM 模型。如果不先修复，后续 `tpu.FAttention` 和
`tpu.A16MatMul` 的内存规划会建立在错误假设上。

#### 代码改动

1. 修改 `--tpu-init="chip=ada300"`，同时写入：

```mlir
module.chip = "ada300"
tpu.target = "ada300"
```

2. 扩展 `tpu::backend::Device`：

```cpp
virtual uint64_t getGmemBytes() const = 0;
virtual uint64_t getSramBankCount() const = 0;
virtual uint64_t getSramBankBytes() const = 0;
virtual uint64_t getWeightMemoryBytes() const = 0;
```

3. 将地址分配 API 改为 target-neutral：

```cpp
assignGmemAddresses(ModuleOp)
assignSmemAddresses(ModuleOp)
```

4. 地址分配通过：

```cpp
const Device &device = backend::getDevice(moduleOp);
```

获取容量、对齐和 bank 配置，不再硬编码 Ada200。

5. 删除或停止使用固定返回 Ada200 的 `getCurrentDevice()`。

6. 为 Ada300 先实现保守 flat allocator：

- 正确分配。
- 正确计算 peak。
- 内存不足时明确报错。
- 暂不实现复杂 tiling。

#### 测试

新增：

```text
test/Transforms/Tpu/Ada300Target.mlir
test/Transforms/Tpu/Ada300MemoryAlloc.mlir
```

验证：

- `module.chip = "ada300"`。
- `tpu.target = "ada300"`。
- Ada300 memory summary 使用 `5 * 256 KiB` SRAM。
- unknown target 明确失败，不静默回退 Ada200。
- Ada200 原有 lit 回归保持通过。

#### 完成标准

Ada200 与 Ada300 均通过同一套 target-neutral memory API 工作。Ada300 的
memory summary、layer group 约束和 allocator 使用同一个设备模型。

### 19.2 Step 2：M1 FP16 MatMul + Add 最小闭环

#### 目标

跑通 Ada300 上最短但真实的 Top -> TPU -> codegen 链路：

```text
top.Input
  -> top.MatMul(weight)
  -> top.Add(bias / residual)
  -> func.return
```

lower 为：

```text
tpu.LoadInput
  -> tpu.LoadWeight
  -> tpu.MatMul
  -> tpu.Add
  -> tpu.Store
  -> func.return
```

首版只支持 FP16。遇到 INT8、W4A16 或动态 shape 时明确报错。

#### Lowering 改动

在 `lib/Conversion/TopToTpu/Ada300/` 增加：

```text
Input.cpp
Weight.cpp
MatMul.cpp
Add.cpp
Return.cpp
```

在 `LoweringAda300.cpp` 注册对应 pattern。

#### Codegen 改动

新增：

```text
lib/Dialect/Tpu/Transforms/CodeGen/Ada300/
  Driver.cpp
  OpRegister.cpp
  LoadInput.cpp
  LoadWeight.cpp
  MatMul.cpp
  Add.cpp
  Store.cpp
```

在通用 `tpu-codegen` pass 中调用：

```cpp
runAda300CodeGen(moduleOp)
```

替换当前 Ada300 直接失败分支。

#### 测试

新增：

```text
test/Transforms/lowering/Ada300MatMulAdd.mlir
test/Transforms/Tpu/Ada300CodeGen.mlir
test/tpu/llm/matmul_add/
```

验证：

```text
Top -> TPU FileCheck
TPU -> Ada300 task / descriptor golden
小规模 FP16 数值对比
```

#### 完成标准

以下命令成功执行：

```bash
tpuc-opt matmul_add.mlir \
  --tpu-init="chip=ada300" \
  --convert-top-to-tpu \
  --tpu-gmem-alloc \
  --tpu-smem-alloc \
  --tpu-codegen \
  -o matmul_add_codegen.mlir
```

输出与 NumPy FP16 baseline 一致。

### 19.3 Step 3：M2 单层 Transformer Prefill

#### 目标

让 `block_0.mlir` 在 Ada300 上完成 prompt 阶段推理，暂不处理历史 KV
cache：

```text
input_states
  -> RMSNorm
  -> Q / K / V projection
  -> RoPE
  -> Flash Attention
  -> O projection + residual
  -> RMSNorm
  -> gate / up projection
  -> SiLU * up
  -> down projection + residual
  -> output_states, k_cache, v_cache
```

#### TPU Dialect 改动

在 `TpuOps.td` 中定义：

```text
tpu.RMSNorm
tpu.Rope
tpu.FAttention
tpu.A16MatMul
tpu.SiLU
```

`tpu.A16MatMul` 首版可以支持 FP16 fallback，但 IR 契约应保留：

```text
weight_bits
q_group_size
scale
zero_point
```

#### Lowering 改动

在 `lib/Conversion/TopToTpu/Ada300/` 增加：

```text
RMSNorm.cpp
Rope.cpp
FAttention.cpp
A16MatMul.cpp
SiLU.cpp
Mul.cpp
View.cpp
```

规则：

- `top.Reshape` 优先 lower 为 alias / view，不生成硬件 task。
- `top.FAttention` 保持高层语义，不在 lowering 中生成完整 attention matrix。
- `top.RMSNorm` 建议使用 FP32 accumulate。
- Prefill attention 使用 tile / streaming 方案，避免 `QK^T` 完整落外存。

#### Codegen 改动

新增 Ada300 adapter：

```text
RMSNorm.cpp
Rope.cpp
FAttention.cpp
A16MatMul.cpp
SiLU.cpp
Mul.cpp
```

如果复合 kernel 暂时不存在，可在 codegen 内 fallback 展开，但不要破坏
TPU IR 的高层契约。

#### 测试

新增：

```text
test/Transforms/lowering/Ada300LlmRMSNorm.mlir
test/Transforms/lowering/Ada300LlmRope.mlir
test/Transforms/lowering/Ada300LlmFAttention.mlir
test/Transforms/lowering/Ada300LlmPrefillBlock.mlir
test/Transforms/Tpu/Ada300LlmPrefillCodeGen.mlir
```

建议数值测试规格：

```text
hidden_size = 64
num_heads = 4
head_dim = 16
seq_length = 8
layers = 1
```

#### 完成标准

- `block_0.mlir` 可完成 Top -> TPU -> Ada300 codegen。
- 输出 `output_states`、`k_cache` 和 `v_cache`。
- 与 PyTorch FP16 baseline 对比：

```text
cosine similarity > 0.999
```

- Attention 使用 tile / streaming 方案，没有生成完整外存 `QK^T` buffer。

### 19.4 Step 4：M3 单层 Decode

#### 目标

跑通 `block_cache_0.mlir`：每次输入一个 token，读取历史 KV cache，追加
当前 K/V，再完成 attention。

```text
input token
  -> Q / K / V
  -> RoPE
  -> KV cache update
  -> streaming attention over history K/V
  -> O projection + MLP
  -> output token
```

#### TPU Dialect 改动

新增：

```text
tpu.KVCacheUpdate
```

建议语义：

```text
history, current, position
axis = 1
mode = insert | concat
```

#### Lowering 与 Codegen 改动

- 将 `top.Insert` lower 为 `tpu.KVCacheUpdate`。
- 将用于 KV cache 更新的 `top.Concat` lower 为 `tpu.KVCacheUpdate`。
- 保留普通 `top.Concat -> tpu.Concat`。
- KV cache 常驻 external memory，不整体搬入 SRAM。
- Decode attention 使用 `mq = 1` 专用 codegen。
- 按 token block 或 head 分块 DMA 读取历史 K/V。
- 新 K/V 原地写入 cache。
- 不生成完整 attention score 外存 buffer。

#### 测试

新增：

```text
test/Transforms/lowering/Ada300LlmKVCacheUpdate.mlir
test/Transforms/lowering/Ada300LlmDecodeBlock.mlir
test/Transforms/Tpu/Ada300LlmDecodeCodeGen.mlir
```

#### 完成标准

- 单层 decode 连续运行多个 token。
- KV cache 每轮正确更新。
- 与 PyTorch FP16 baseline 对比：

```text
cosine similarity > 0.999
```

- task graph 可观察到 KV cache 分块读取以及 DMA / compute 顺序。

### 19.5 Step 5：M4 Tiny Llama 端到端

#### 目标

将单层能力扩展为可执行的小型文本生成模型：

```text
embedding
  -> block_0 ... block_n
  -> lm_head
  -> token id
```

同时编译：

```text
block_i
block_cache_i
```

#### 建议模型规格

```text
layers       = 2
hidden_size  = 64 或 128
num_heads    = 4
head_dim     = 16 或 32
seq_length   = 16 或 32
batch        = 1
dtype        = FP16
```

#### Lowering 与 Runtime 改动

- 增加 `top.Gather -> tpu.Gather`，支持 embedding。
- 增加 lm-head 的 `MatMul + Reshape + TopK` lowering。
- 如果 Ada300 暂无 `TopK` kernel，首版允许在 host 执行 token selection，
  但必须明确记录 host / device 边界。
- 定义模型打包格式：
  - 子网列表。
  - 权重地址。
  - KV cache 布局。
  - runtime 调用顺序。
- 确保多层 KV cache 按 layer 独立分配。
- 增加端到端 runtime harness。

#### 测试

新增：

```text
test/tpu/llm/tiny_llama/
  export_top.sh
  compile_ada300.sh
  compare.py
```

验证流程：

```text
prompt
  -> prefill
  -> first token
  -> repeated decode
  -> generated tokens
```

#### 完成标准

- 固定 prompt 下，与 PyTorch baseline 对比每步 logits 或 hidden state。
- 生成 token 序列一致，或满足 FP16 误差约定。
- 输出：

```text
time-to-first-token
decode tokens/s
```

- 达到 `seq_length` 时不越界。
- 多层 KV cache 地址互不覆盖。

完成 Step 5 后，再进入 W8A16、W4A16、layer group、5-bank SRAM tiling 和
DMA / compute overlap 优化。

## 20. 风险与决策点

开始编码前需要确认：

1. Ada300 ISA、runtime ABI 和 descriptor 格式。
2. FP16 MatMul、W8A16、W4A16 kernel 是否存在。
3. RMSNorm、RoPE、Flash Attention 是否有硬件 kernel 或算子库实现。
4. 5 个 SRAM block 的并发访问规则和 bank conflict 约束。
5. external memory、weight memory 和 KV cache 地址空间。
6. DMA、TE、VE 的真实 engine 模型和同步机制。
7. command stream 或 rxmodel 的最终产物格式。

如果硬件暂时缺少复合 kernel，仍应保留高层 `tpu.*` op，由 codegen 内部选择
fallback 展开。这样可以保持 TPU IR 契约稳定，并减少后续重构。

## 21. 推荐实施顺序

```text
Phase 0 设备模型收口
  -> M1 FP16 MatMul + Add
  -> 新增 LLM TPU ops 与 verifier
  -> M2 单层 prefill
  -> M3 单层 decode
  -> M4 tiny Llama
  -> M5 W8A16 / W4A16 与 tiling
  -> M6 外围能力
  -> M7 VLM
```

最重要的取舍是：先用高层 `tpu.FAttention`、`tpu.RMSNorm` 和 `tpu.Rope`
建立稳定后端契约，再在 Ada300 codegen 内逐步实现 tiling、融合和硬件映射。
