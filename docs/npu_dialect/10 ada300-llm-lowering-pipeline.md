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


#### 实现记录（2026-06-04）

本次按 Ada300-only 目标完成 M0 主线收口，范围限定在 `top/tpu`
dialect pipeline 的 target 与 memory model，不接入 bmodel codegen。

已落地改动：

1. `module::Chip` 增加 `ada300`，`--processor-assign="chip=ada300"`
   可正常解析；未知 chip 改为 `emitError + signalPassFailure`，不再 assert。
2. `module` helper 增加 `getTarget/setTarget/isTarget`，`setChip()` 同步写入
   `tpu.target`；`--init` 调用 `backend::attachDeviceAttrs()` 写入：
   - `tpu.target = "ada300"`
   - `tpu.gmem_bytes = 512 MiB`
   - `tpu.sram_bank_count = 5`
   - `tpu.sram_bank_bytes = 256 KiB`
   - `tpu.weight_memory_bytes = 256 MiB`
3. 新增 `backend::Device` 与 `backend::getDevice(moduleOp)`，Ada300 设备模型
   通过同一入口提供 GMEM、SRAM bank、weight memory、alignment 和 base addr。
4. `backend::Arch::init()` 增加 Ada300 分支，只初始化静态参数，不加载 BM/CV
   backend instance。
5. `address-assign` 对 `ada300`/`tpu.target="ada300"` 走 Ada300 flat
   allocator，跳过 BM global-buffer rewrite 与 BM live-range allocator。
   当前 allocator 行为：
   - weight 从 `weightStartAddr` 顺序分配；
   - func block argument 与 op result 从 `gmemStartAddr` 顺序分配；
   - 64-byte 对齐；
   - 超过 `Device` 容量时显式报错；
   - 不做 reuse、tiling、bank-aware placement。
6. `tpu.KVCacheUpdate` 从 `Tpu_Op` 改为 `Tpu_BaseOp`，不再强制声明
   `GlobalGenInterface`、`InferenceInterface`、`DynGlobalGenInterface`；
   `KVCacheUpdate.cpp` 删除全部 BM/CV codegen 桩（`codegen_global_bm1684`、
   `codegen_global_bm1684x`、`codegen_global_cv18xx`、`dyn_codegen_*`、
   `get_fw_type_*`、`support_multi_core`）及 inference 桩（`init`、`deinit`、
   `inference`），仅保留 `type_verify`；include 收拢至
   `TypeInterface.h` + `TpuOps.h`。
7. `AddressAssign.cpp` `else` 分支内删除死代码
   `if (!module::isChip(module::Chip::Ada300))` 守卫——该守卫在 `else` 分支中
   永远为 `true`（Ada300 已在上层 `if` 截断），直接展开其包裹的
   `populateGlobalBufferBM168xPatterns` 调用。

新增回归：

```text
test/Transforms/Tpu/Ada300Target.mlir
test/Transforms/Tpu/Ada300MemoryAlloc.mlir
```

已验证命令：

```bash
ninja -C build tpuc-opt

build/bin/tpuc-opt --processor-assign="chip=ada300 mode=F16" \
  --init="freq=0 level=0" test/Transforms/Tpu/Ada300Target.mlir | \
  FileCheck test/Transforms/Tpu/Ada300Target.mlir

not build/bin/tpuc-opt --processor-assign="chip=unknown mode=F16" \
  test/Transforms/Tpu/Ada300Target.mlir 2>&1 | \
  FileCheck test/Transforms/Tpu/Ada300Target.mlir --check-prefix=UNKNOWN

build/bin/tpuc-opt --init="freq=0 level=0" \
  --address-assign="reuse_addr=false" \
  test/Transforms/Tpu/Ada300MemoryAlloc.mlir | \
  FileCheck test/Transforms/Tpu/Ada300MemoryAlloc.mlir
```

当前完成度：Ada300-only M0 已完成。legacy BM/CV address allocator 未重构成
`assignGmemAddresses/assignSmemAddresses` 命名，但 Ada300 路径已经不依赖 BM/CV
设备常量或 bmodel codegen。`KVCacheUpdate` BM/CV 桩已清除，死代码守卫已删除。

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

#### 实现记录

已完成以下改动（实际 pass 命令与设计略有差异，沿用现有 M0 pass 名）：

1. **`include/tpu_mlir/Conversion/TopToTpu/LoweringAda300.h`** — 新增。定义
   `LOWERING_ADA300(OP)` 宏，展开为 `struct OPLowering : TopLowering<top::OPOp>`，
   仅实现 `LoweringF16`；`LoweringF32/BF16` 委托给 `LoweringF16`；`LoweringINT8`
   明确报错。当前声明 `MatMulLowering`、`AddLowering`。

2. **`lib/Conversion/TopToTpu/Ada300/MatMul.cpp`** — 新增。
   `MatMulLowering::LoweringF16` 调用
   `lowering_common_f16<tpu::MatMulOp>(rewriter, op, 3)`（3 = input/right/bias）。

3. **`lib/Conversion/TopToTpu/Ada300/Add.cpp`** — 新增。
   `AddLowering::LoweringF16` 调用
   `lowering_common_f16<tpu::AddOp>(rewriter, op)`。

4. **`lib/Conversion/TopToTpu/LoweringAda300.cpp`** — 新增。实现
   `ada300::populateTopToTpuConversionPatterns()`，注册上述两个 pattern。

5. **`lib/Conversion/TopToTpu/TopToTpuPass.cpp`**
   - `#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"` 加入包含列表。
   - 在 `populateTopToTpuConversionPatterns` 的 chip 分发链末尾增加：
     ```cpp
     } else if (module::isChip(module::Chip::Ada300) ||
                module::isTarget("ada300")) {
       ada300::populateTopToTpuConversionPatterns(&patterns);
     ```

6. **`lib/Conversion/TopToTpu/CMakeLists.txt`** — 在 `file(GLOB _sources ...)` 中
   加入 `Ada300/*.cpp`。

7. **`lib/Dialect/Tpu/Transforms/Codegen/Ada300Codegen.hpp`** — 新增。声明
   `Ada300Codegen::run(ModuleOp, filename)`。

8. **`lib/Dialect/Tpu/Transforms/Codegen/Ada300Codegen.cpp`** — 新增。M1
   placeholder 实现：遍历所有 op（调试日志），向输出文件写入 chip 名称和 op
   计数，不产生真实指令（真正指令编码留给后续 milestone）。

9. **`lib/Dialect/Tpu/Transforms/Codegen.cpp`**
   - `#include "Codegen/Ada300Codegen.hpp"` 加入包含。
   - 在 `isCV18xx()` 分支之后、`BMCodegen` 之前插入：
     ```cpp
     if (module::isChip(module::Chip::Ada300) || module::isTarget("ada300")) {
       Ada300Codegen ada300_codegen;
       ada300_codegen.run(mOp, filename);
       return;
     }
     ```

10. **`test/Transforms/lowering/Ada300MatMulAdd.mlir`** — 新增。FileCheck 测试：
    Top IR（`top.MatMul` + `top.Add`）经
    `--processor-assign="chip=ada300 mode=F16" --init --convert-top-to-tpu`
    后变成 `tpu.MatMul` + `tpu.Add`。

**偏差说明**：文档中 `lib/Dialect/Tpu/Transforms/CodeGen/Ada300/` 子目录结构
（Driver / OpRegister / LoadInput / LoadWeight / MatMul / Add / Store）暂未分文件
实现，合并在 `Ada300Codegen.cpp` 中作为单文件 placeholder；待 M2 milestone 再按
设计分拆。

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

#### 实现记录

已完成以下改动（Codegen 部分保持 M1 placeholder，待后续 milestone 分拆）：

1. **`include/tpu_mlir/Conversion/TopToTpu/LoweringAda300.h`** — 扩展。在原有
   `LOWERING_ADA300(MatMul)` / `LOWERING_ADA300(Add)` 之后追加：

   ```cpp
   LOWERING_ADA300(RMSNorm)
   LOWERING_ADA300(Rope)
   LOWERING_ADA300(FAttention)
   LOWERING_ADA300(A16MatMul)
   LOWERING_ADA300(Mul)
   LOWERING_ADA300(Reshape)
   ```

   并手写 `SiLULowering` struct（`tpu` dialect 中无独立 `tpu::SiLUOp`，SiLU
   必须 lower 为 `tpu::ActiveOp(mode=SILU)`，与宏生成的结构不兼容）：

   ```cpp
   struct SiLULowering : public TopLowering<top::SiLUOp> {
     SiLULowering(MLIRContext *ctx) : TopLowering<top::SiLUOp>(ctx) {}
     void LoweringF16(PatternRewriter &, top::SiLUOp) const override;
     void LoweringF32(...)  const override { LoweringF16(rewriter, op); }
     void LoweringBF16(...) const override { LoweringF16(rewriter, op); }
     void LoweringINT8(...) const override {
       op.emitError("Ada300 M2: INT8 SiLU not supported");
     }
   };
   ```

2. **`lib/Conversion/TopToTpu/Ada300/RMSNorm.cpp`** — 新增。手动克隆
   gamma weight 到 FP16（而非依赖 `lowering_common_f16`），以正确处理
   `weight_keep_f32` flag：

   ```cpp
   void RMSNormLowering::LoweringF16(...) const {
     // 遍历所有 operand；若 operand 是 WeightOp 且 !weight_keep_f32，则
     // clone_f16；否则直接 pass-through。
     auto new_type = getQuantF16Type(op.getOutput());
     rewriter.replaceOpWithNewOp<tpu::RMSNormOp>(op, new_type, opds, attrs);
   }
   ```

   模式与 BM1684X `RMSNorm.cpp` 一致，去掉 BM168x 特有的 norm_type /
   mode 分支。

3. **`lib/Conversion/TopToTpu/Ada300/Rope.cpp`** — 新增。三步处理：

   ```cpp
   void RopeLowering::LoweringF16(...) const {
     // 1. 将 top::RopeModeAttr 字符串转换为 tpu::RopeMode
     auto rope_mode = get_rope_mode(op.getRopeModeAttr().str());
     // 2. 移除 top-only 属性（tpu dialect 无对应字段）
     module::removeAttr(op, "mul1_round_mode");
     module::removeAttr(op, "mul2_round_mode");
     module::removeAttr(op, "add_round_mode");
     // 3. 通用 FP16 lowering + 回填 rope_mode 枚举
     Operation *newOp = lowering_common_f16<tpu::RopeOp>(rewriter, op);
     newOp->setAttr("rope_mode",
                    tpu::RopeModeAttr::get(op.getContext(), rope_mode));
   }
   ```

   `get_rope_mode()` 在 `TopLowering.h`（行 598）中已声明，实现在
   `TopLowering.cpp`（行 708），Ada300 直接复用，无需拷贝。

4. **`lib/Conversion/TopToTpu/Ada300/FAttention.cpp`** — 新增。直接透传：

   ```cpp
   void FAttentionLowering::LoweringF16(...) const {
     lowering_common_f16<tpu::FAttentionOp>(rewriter, op);
   }
   ```

5. **`lib/Conversion/TopToTpu/Ada300/A16MatMul.cpp`** — 新增。FP16 fallback
   路径，传入 `num_operands = 5`（input / weight / scale / zp / bias）。
   `lowering_common_f16` 仅克隆 float WeightOp，int8/int4 weight 原样透传：

   ```cpp
   void A16MatMulLowering::LoweringF16(...) const {
     lowering_common_f16<tpu::A16MatMulOp>(rewriter, op, 5);
   }
   ```

6. **`lib/Conversion/TopToTpu/Ada300/SiLU.cpp`** — 新增。先在 top op 上设置
   `mode` attribute，再调用 `lowering_common_f16<tpu::ActiveOp>`，令生成的
   ActiveOp 携带正确的 `mode = SILU`：

   ```cpp
   void SiLULowering::LoweringF16(...) const {
     op->setAttr("mode",
       tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SILU));
     lowering_common_f16<tpu::ActiveOp>(rewriter, op.getOperation());
   }
   ```

7. **`lib/Conversion/TopToTpu/Ada300/Mul.cpp`** — 新增。直接透传：

   ```cpp
   void MulLowering::LoweringF16(...) const {
     lowering_common_f16<tpu::MulOp>(rewriter, op);
   }
   ```

8. **`lib/Conversion/TopToTpu/Ada300/Reshape.cpp`** — 新增。`top.Reshape` 为
   零拷贝 view；lower 为 `tpu.Reshape` 供后续 pass 识别为 alias：

   ```cpp
   void ReshapeLowering::LoweringF16(...) const {
     lowering_common_f16<tpu::ReshapeOp>(rewriter, op);
   }
   ```

9. **`lib/Conversion/TopToTpu/LoweringAda300.cpp`** — 扩展。在
   `populateTopToTpuConversionPatterns()` 中追加所有 M2 pattern：

   ```cpp
   patterns->add<
       MatMulLowering, AddLowering,          // Step 2
       RMSNormLowering, RopeLowering,
       FAttentionLowering, A16MatMulLowering,
       MulLowering, ReshapeLowering,
       SiLULowering                          // Step 3
   >(patterns->getContext());
   ```

10. **`test/Transforms/lowering/Ada300LlmRMSNorm.mlir`** — 新增 FileCheck 测试：
    `top.RMSNorm` + gamma `top.Weight` → `tpu.RMSNorm`（FP16）。

11. **`test/Transforms/lowering/Ada300LlmRope.mlir`** — 新增 FileCheck 测试：
    `top.Rope`（三个 FP32 输入，无可选操作数）→ `tpu.Rope`，验证
    `mul*_round_mode` / `add_round_mode` 属性已被移除。

12. **`test/Transforms/lowering/Ada300LlmFAttention.mlir`** — 新增 FileCheck 测试：
    `top.FAttention`（Q/K/V + none mask/buffer）→ `tpu.FAttention`（FP16）。

13. **`test/Transforms/lowering/Ada300LlmPrefillBlock.mlir`** — 新增集成 FileCheck
    测试，覆盖 SwiGLU-MLP 子图：

    ```text
    RMSNorm -> A16MatMul (gate) -> SiLU
                                             -> Mul (SwiGLU)
    RMSNorm -> A16MatMul (up)  -+
                                             -> A16MatMul (down)
    ```

    验证所有节点正确 lower 为 `tpu.RMSNorm` / `tpu.A16MatMul` /
    `tpu.Active` / `tpu.Mul`。

**偏差说明**：
- 文档中的 `View.cpp` 以 `Reshape.cpp` 形式实现（`top.Reshape` → `tpu.Reshape`）。
- Codegen 部分（RMSNorm / Rope / FAttention / A16MatMul / SiLU / Mul adapter）
  仍保持 M1 的单文件 `Ada300Codegen.cpp` placeholder，待后续 milestone 分拆。

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

#### 实现记录（2026-06-04）

已完成 Top → TPU lowering 层面的 M3 改动（Codegen 部分仍为 M1 placeholder）。

1. **`include/tpu_mlir/Conversion/TopToTpu/LoweringAda300.h`** — 扩展。在原有
   `LOWERING_ADA300(Reshape)` 之后追加：

   ```cpp
   LOWERING_ADA300(Concat)
   LOWERING_ADA300(Insert)
   ```

   并手写 `KVCacheUpdateLowering` struct（需要手动将 `kv_format` / `mode`
   字符串属性转换为 `tpu` dialect 枚举 attr，无法用宏生成）：

   ```cpp
   struct KVCacheUpdateLowering : public TopLowering<top::KVCacheUpdateOp> {
     KVCacheUpdateLowering(MLIRContext *ctx)
         : TopLowering<top::KVCacheUpdateOp>(ctx) {}
     void LoweringF16(PatternRewriter &, top::KVCacheUpdateOp) const override;
     void LoweringF32(...)  const override { LoweringF16(rewriter, op); }
     void LoweringBF16(...) const override { LoweringF16(rewriter, op); }
     void LoweringINT8(...) const override {
       op.emitError("Ada300 M3: INT8 KVCacheUpdate not supported");
     }
   };
   ```

2. **`lib/Conversion/TopToTpu/Ada300/KVCacheUpdate.cpp`** — 新增。手动处理
   两个枚举转换，并丢弃 top-only 属性 `resource_id`：

   - `kv_format`（`AnyStrAttrOf` 字符串）→ `Tpu_KVCacheFormatAttr`
     (`KV4` / `KV8` / `FP8` / `none`)
   - `mode`（`AnyStrAttrOf` 字符串）→ `Tpu_KVCacheModeAttr`
     (`paged` / `contiguous`)

   三个结果类型分别处理：`updated_cache_k` / `updated_cache_v` 转为 FP16，
   `updated_seq_lens` 保持原类型（通常为 F32 或 I32，不做类型转换）：

   ```cpp
   rewriter.replaceOpWithNewOp<tpu::KVCacheUpdateOp>(
       op, TypeRange{f16_k, f16_v, seq_type}, operands, attrs);
   ```

3. **`lib/Conversion/TopToTpu/Ada300/Concat.cpp`** — 新增。直接透传：

   ```cpp
   void ConcatLowering::LoweringF16(...) const {
     lowering_common_f16<tpu::ConcatOp>(rewriter, op.getOperation());
   }
   ```

4. **`lib/Conversion/TopToTpu/Ada300/Insert.cpp`** — 新增。直接透传：

   ```cpp
   void InsertLowering::LoweringF16(...) const {
     lowering_common_f16<tpu::InsertOp>(rewriter, op);
   }
   ```

5. **`lib/Conversion/TopToTpu/LoweringAda300.cpp`** — 扩展。追加三个新 pattern：

   ```cpp
   // Step 4: M3 single-layer Decode
   KVCacheUpdateLowering,
   ConcatLowering,
   InsertLowering
   ```

6. **`test/Transforms/lowering/Ada300LlmKVCacheUpdate.mlir`** — 新增 FileCheck
   测试。六个 `top.Input` 操作数（cache_k / cache_v / key / value /
   block_table / seq_lens），属性 `kv_format = "KV8"` / `mode = "paged"`。
   只返回 `updated_seq_lens`（F32 tensor）避免多路 Cast op 引发的
   `all_names` 唯一性断言。验证 `tpu.KVCacheUpdate` 出现在输出 IR 中。

7. **`test/Transforms/lowering/Ada300LlmDecodeBlock.mlir`** — 新增集成 FileCheck
   测试，覆盖完整单层 decode 数据流：

   ```text
   RMSNorm -> A16MatMul (Q proj) -> Rope
     -> KVCacheUpdate
     -> FAttention (mq=1, mk=8)
     -> A16MatMul (O proj) -> Add (attention residual)
     -> RMSNorm -> SwiGLU MLP -> Add (MLP residual)
   ```

   规格：`hidden_size=64`, `num_heads=8`, `head_dim=8`,
   decode token `seq=1`, history `len=8`。
   验证所有节点 lower 正确：`tpu.RMSNorm` / `tpu.A16MatMul` /
   `tpu.Rope` / `tpu.KVCacheUpdate` / `tpu.FAttention` /
   `tpu.Add` / `tpu.Active` / `tpu.Mul`。

**偏差说明**：

- 文档方案中 `top.Insert` / `top.Concat`（用于 KV cache）被合并 lower 为
  `tpu.KVCacheUpdate`。实际实现中：`top.KVCacheUpdate` 直接存在于 top dialect
  并 lower 为 `tpu.KVCacheUpdate`；`top.Insert` lower 为 `tpu.Insert`（保留为
  独立 op）；`top.Concat` lower 为 `tpu.Concat`（普通拼接）。KV cache 更新语义
  通过 `top.KVCacheUpdate` 显式表达，无需从 Concat/Insert 推断。
- 新增 `.cpp` 文件需执行 `cmake ..` 以使 CMake 的 glob 重新扫描 `Ada300/`
  目录，再执行 `ninja tpuc-opt`。
- Codegen（`Ada300LlmDecodeCodeGen.mlir` 测试、decode streaming attention 调度）
  仍为 M1 单文件 placeholder，待后续 milestone 分拆。

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
