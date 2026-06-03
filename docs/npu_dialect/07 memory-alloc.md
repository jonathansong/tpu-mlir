# NPU 内存分配设计文档

## 1. 背景

当前 `rx-mlir` 的 NPU flat pipeline 使用独立的 GMEM/SMEM 分配 pass：

```bash
model_transform.py
  -> tpuc-opt
     --npu-init="chip=ada200"
     --convert-top-to-npu
     --npu-gmem-alloc
     --npu-smem-alloc
     --npu-codegen
```

旧的组合入口 `--npu-address-assign` 不再保留。拆分的原因是后续版本中 SMEM 分配时机可能下沉到 layergroup 内部，而 GMEM 图输入/输出分配仍然可以在图级完成。

本文档描述首版 flat pipeline 的内存分配方式；不覆盖 layergroup 内分块、tiling 或跨 group 的局部内存规划。

## 2. 目标

首版方案按职责拆成三个阶段：

- `--npu-gmem-alloc`：只分配图输入、图输出的 GMEM 地址。
- `--npu-smem-alloc`：只分配中间 tensor 的 SMEM 地址，并尽量让 compute op 输入/输出分到不同 SMEM page。
- `--npu-codegen`：按 op 生成 desc/config/param/quant/weight 等资源时，再从 GMEM/SMEM/CMEM pool 申请地址。

tensor value 的地址写入 `RankedTensorType` encoding。codegen 阶段生成的非 SSA tensor 资源不提前写入 type，而是由 `CodeGenContext` 的 memory pool 返回地址后写入 desc/config/rxmodel。

## 3. 地址空间

### 3.1 GMEM

Ada200 约束：

- 相对地址空间：0 ~ 4MB
- 对齐：当前按 64B 分配

`--npu-gmem-alloc` 只处理：

- graph input
- graph output

单图内输入、输出不复用；多图/多 function 之间可以在图边界后复用。

codegen 阶段继续使用同一个 GMEM pool，在 `address_assign_gmem_peak` 之后申请：

- weight / const 原始数据
- bias / quant / lut 原始数据
- TMA desc
- VE/TE addr、dim、param、config
- 指令或 rxmodel 静态数据

### 3.2 SMEM

Ada200 约束：

- 相对地址空间：0 ~ 256KB
- 逻辑上分为两个 128KB page：
  - page0: `[0, 128KB)`
  - page1: `[128KB, 256KB)`
- 对齐：当前按 64B 分配

`--npu-smem-alloc` 只处理中间 tensor：

- `npu.LoadInput` 的 result
- compute op 的 result
- compute op operand 中已经位于 SMEM 的 value

`npu.Store` 的 result 是 graph output，属于 GMEM，不分配 SMEM。

SMEM page 规则是性能偏好，不是正确性约束：

- 如果一个 compute op 的 SMEM 输入都在同一 page，则输出优先放到另一个 page，实现读写 page ping-pong。
- 如果输入已经分散在不同 page，则不报错、不 copy，输出退化为选择其中一个 page 分配。

codegen 阶段如果需要 bias、quant、lut 等临时片上数据，可继续从 SMEM pool 申请。

### 3.3 CMEM

Ada200 约束：

- 相对地址空间：0 ~ 512KB
- 地址按 8KB 对齐

CMEM 只由 `LoadWeight` codegen 申请，用于 weight 的片上/存算目的地址。GMEM/SMEM 分配 pass 不使用 CMEM。

## 4. 编译流程

```text
--npu-init="chip=ada200"
  -> --convert-top-to-npu
  -> --npu-gmem-alloc
       1. 收集 NPU op/value
       2. 标记 graph input / graph output
       3. 分配 GMEM 地址
       4. 写回 RankedTensorType encoding
       5. 写入 address_assign_gmem_peak
  -> --npu-smem-alloc
       1. 收集 NPU op/value
       2. 标记中间 tensor
       3. 规划 SMEM page
       4. 分配 SMEM 地址
       5. 写回 RankedTensorType encoding
       6. 写入 address_assign_smem_peak
  -> --npu-codegen
       1. 从 npu.mem.summary 恢复 GMEM/SMEM pool
       2. 创建 CMEM pool
       3. 每个 op codegen 时按需申请 desc/config/param/quant/weight 等资源
       4. 写入 codegen_gmem_peak / codegen_smem_peak / codegen_cmem_peak
```

后续 layergroup 版本可以只移动 `--npu-smem-alloc` 的执行位置；`--npu-gmem-alloc` 仍可保持图级执行。

## 5. IR 表达

tensor value 地址使用 `RankedTensorType` encoding。当前实现用 dictionary encoding，保存：

```mlir
{space = "SMEM", addr = 0 : i64, size = 5760 : i64, align = 64 : i64}
```

示例：

```mlir
%0 = "npu.LoadInput"(%arg0)
  : (tensor<1x6x240xi8,
       {space = "GMEM", addr = 0 : i64, size = 5760 : i64, align = 64 : i64}>)
 -> tensor<1x6x240xi8,
       {space = "SMEM", addr = 0 : i64, size = 5760 : i64, align = 64 : i64}>
```

module 级别使用 `npu.mem.summary` 保存各阶段 pool 的摘要信息：

```mlir
npu.mem.summary = {
  target = "ada200",
  gmem_bytes = 4194304 : i64,
  smem_bytes = 262144 : i64,
  cmem_bytes = 524288 : i64,
  address_assign_gmem_peak = ...,
  address_assign_smem_peak = ...,
  codegen_gmem_peak = ...,
  codegen_smem_peak = ...,
  codegen_cmem_peak = ...
}
```

codegen 阶段才生成的资源，例如 weight、desc、config、quant，不提前写入 SSA value type encoding。

## 6. 分配策略

### 6.1 GMEM 分配

`--npu-gmem-alloc` 的输入是 flat NPU IR。

分配对象：

- `func.func` 的 block argument：graph input
- `npu.Store` 的 result / `func.return` operand：graph output

生命周期：

- 每个 function 内，graph input 和 graph output 都从该 function 第一个 NPU op 活到最后一个 NPU op。
- 因此单图内输入、输出不复用。
- 多 function 之间，如果前一个 function 的 GMEM value 已结束，则地址可以回收到 GMEM free list，供后续 function 复用。

### 6.2 SMEM 分配

`--npu-smem-alloc` 的输入可以是已经执行过 `--npu-gmem-alloc` 的 IR。

分配对象：

- 非 `npu.Store`、非 `npu.LoadWeight` 的 NPU op result
- 典型包括 `npu.LoadInput` result、`npu.Add` result 等中间 activation

分配流程：

1. 收集所有 SMEM value 的 live range。
2. 规划每个 SMEM value 的 page。
3. 遍历 op 序列，在每个 op 前释放已结束生命周期的 SMEM value。
4. 从对应 page 的 pool 中分配地址。

page 规划：

```text
if compute op 的 SMEM 输入都在同一 page:
  output page = 另一个 page
else:
  output page = 选定输入 page
```

该规则只用于提高同时读写性能；如果无法满足输入输出跨 page，不影响正确性。

### 6.3 Codegen Pool

`--npu-codegen` 的 `CodeGenContext` 持有 `MemoryPoolManager`：

```text
allocGmem(size, align, purpose) -> addr
allocSmem(size, align, purpose) -> addr
allocCmem(size, align, purpose) -> addr
```

初始化方式：

- GMEM pool 从 `address_assign_gmem_peak` 后继续分配。
- SMEM pool 从 `address_assign_smem_peak` 后继续分配。
- CMEM pool 从 0 开始分配。

典型使用：

- `LoadInput`：读取输入 GMEM / 输出 SMEM encoding，申请 TMA load desc 的 GMEM 地址。
- `Add`：读取输入/输出 SMEM encoding，申请 VE addr/dim/param config 的 GMEM 地址。
- `Store`：读取输入 SMEM / 输出 GMEM encoding，申请 TMA store desc 的 GMEM 地址。
- `LoadWeight`：申请 weight GMEM 源地址、CMEM 目的地址和 TMA desc GMEM 地址。

## 7. Add 示例

当前 add case 的关键地址：

```text
GMEM tensor:
%arg0 : [0, 5760)
%arg1 : [5760, 11520)
%3    : [11520, 17280)

SMEM page0:
%0    : [0, 5760)
%1    : [5760, 11520)

SMEM page1:
%2    : [131072, 136832)

GMEM codegen:
LoadInput0 desc : [17280, 17344)
LoadInput1 desc : [17344, 17408)
Add addr_cfg    : [17408, 17472)
Add dim_cfg     : [17472, 17536)
Add param_cfg   : [17536, 17600)
Store desc      : [17600, 17664)
```

数据流：

```text
GMEM.%arg0 -> TMA Load -> SMEM page0.%0
GMEM.%arg1 -> TMA Load -> SMEM page0.%1
SMEM page0.%0 + SMEM page0.%1 -> Add -> SMEM page1.%2
SMEM page1.%2 -> TMA Store -> GMEM.%3
```

## 8. 验证

核心测试：

- `test/Transforms/Npu/PipelineSkeleton.mlir`
- `test/Transforms/Npu/CodeGen.mlir`
- `test/npu/ops/add/compile.sh`

常用验证命令：

```bash
source envsetup.sh
cmake --build $PROJECT_ROOT/build --target npu-opt tpuc-opt -j2
PATH=$PROJECT_ROOT/build/bin:$PATH \
  $PROJECT_ROOT/build/bin/llvm-lit -sv \
  $PROJECT_ROOT/test/Transforms/Npu/PipelineSkeleton.mlir \
  $PROJECT_ROOT/test/Transforms/Npu/CodeGen.mlir
```

add case：

```bash
cd $PROJECT_ROOT/test/npu/ops/add
PATH=$PROJECT_ROOT/build/bin:$PATH bash ./compile.sh
```

## 9. 取舍

- GMEM 和 SMEM 分配拆成两个独立 pass，不保留旧组合 pass。
- GMEM 只负责 graph input/output。
- SMEM 只负责中间 tensor，并做双页性能优化。
- SMEM 输入输出跨 page 是性能偏好，不是强制约束。
- weight、desc、config、param、dim、quant 等资源延后到 op codegen 分配。
- 首版不处理 layergroup 内 tiling 和复杂跨 group 复用。
