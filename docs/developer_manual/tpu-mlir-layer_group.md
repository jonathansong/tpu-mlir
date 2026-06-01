# Layer Group Pass

## The Core Problem

Sophgo TPUs have two memory tiers:
- **DDR (Global Memory)** — large, slow, shared by all operations
- **Local SRAM (LMEM)** — small, fast, private to each NPU core (typically tens of KiB per lane)

Every `tpu.Load` and `tpu.Store` is a DMA transfer between them. The goal of Layer Group is to **keep intermediate tensors in LMEM across multiple consecutive ops**, eliminating round-trips to DDR. This is the single biggest contributor to inference speedup.

### Source Location

| File | Purpose |
|---|---|
| `lib/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.cpp` | Top-level DP algorithm and group partitioning |
| `lib/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.cpp` | Timestep table construction |
| `lib/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.cpp` | Greedy nearest-layer timestep scheduling |
| `lib/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.cpp` | LMEM interval-based buffer packing |
| `lib/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.cpp` | Per-op DDR traffic / cycle cost estimation |
| `lib/Dialect/Tpu/Transforms/LayerGroup/SwPipeline.cpp` | 3-stage software pipeline scheduling |
| `lib/Dialect/Tpu/Transforms/LayerGroup/GroupOps.cpp` | Final MLIR emission (tpu.GroupOp construction) |
| `lib/Dialect/Tpu/Transforms/LayerGroup/GroupOverlap.cpp` | Cross-group LMEM overlap optimization |
| `lib/Dialect/Tpu/Transforms/LayerGroup/opt3/IlpTimeStep.cpp` | ILP-based timestep scheduling for MatMul groups |
| `include/tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h` | GroupMethod class declaration |

---

## Overall Flow

```
subnet_ops (topologically sorted TPU ops)
    ↓
get_base_groups()               partition into independent contiguous segments
    ↓
get_group_clusters()            cluster ops within each base group (≤50 clusters)
    ↓
dynamic_programming_kernel()    find optimal cut points minimizing DDR traffic
    ↓
is_layer_group_valid()          verify each candidate group fits in LMEM
    ↓
GroupOps::buildGroupOps()       emit tpu.GroupOp with tpu.Load/tpu.Store + lg attrs
```

---

## Step 1: Base Groups

`get_base_groups()` breaks the subnet into **base groups** — maximal sequences of ops that have no data-parallel obstacles (no cross-subnet dependencies, no CPU ops, no dynamic-shape ops). Each base group is a contiguous run of fusable candidates.

---

## Step 2: Op Clustering

Within each base group, ops are **clustered** by structural affinity. The reason is that DP on 1000 individual ops would be O(N²) or worse — clustering reduces the problem size to at most `MAX_GROUP_CLUSTER = 50` clusters.

---

## Step 3: Group Type Detection

Each candidate group is classified into one of several **group types** that determine the tiling strategy:

| Group Type | Tiling Dimensions | Use Case |
|---|---|---|
| `GROUP_NORMAL` | N × H slices | Default: conv + elementwise chains |
| `GROUP_3D` | N × D × H slices | 3D convolution, 3D pooling |
| `GROUP_SMALL_C` | N × (C or H) slices | Thin channels: LLM attention/norm layers |
| `GROUP_MM_OPT` (opt3) | M / K / N tiles | MatMul-dominant groups (LLM linear layers) |
| `GROUP_OVERLAP` | with lookahead | Adjacent groups sharing a prefetch window |

Detection functions:
- `can_be_group_3d()` — activates when `Conv3DOp` or `Pool3DOp` is present, or when `LayerNormOp` input has rank 5.
- `can_be_group_small_c()` — activates for elementwise + norm + matmul + softmax chains where C is too small to fill all NPU lanes on the H dimension.

---

## Step 4: Dynamic Programming (opt=2)

The core algorithm fills a 2D **cost table** where `cost_table[i][j]` is the minimum DDR traffic to execute ops `[i..j]` fused as a single group, or the optimal split cost.

```cpp
dynamic_programming_kernel(base_group, clusters, cost_table, cut_points, ...)
```

For each candidate group `[i..j]`:
1. Call `is_layer_group_valid(lg_info, calc_cost=true, &group_cost)` — verify LMEM fit and compute cost.
2. Compare against splitting at every interior point `k`: `cost[i][k] + cost[k+1][j]`.
3. Record the optimal cut point in `cut_points`.

`sweep_for_min_cost()` scans the DP table to extract the globally optimal grouping.

The **cost metric** is total GDMA bytes (DDR traffic), estimated by `CycleCalculator` using each chip's memory bandwidth and op latency models.

> **opt=1 (simple_layer_group):** Greedy — tries to extend the current group by one op at a time; accepts if LMEM still fits.

---

## Step 5: Validity Check — `is_layer_group_valid()`

This is the innermost loop of the DP. For a candidate group, it:

1. **Determines tiling** (`shape_secs`): how many slices along N / C / H / D / W to partition tensors.
2. **Calls `BasicTimeStep::assignTimeStep()`**: assigns compute and DMA transfers to a timestep schedule.
3. **Calls `LmemAllocator::assignLmemAddrWithSecs()`**: tries to fit all live tensors in LMEM simultaneously.
4. Returns success + cost if LMEM fits; failure if LMEM overflows.

Tiling is progressively coarsened until the group fits:
```
try 1 slice → fits? → done
too large   → try 2 slices → fits? → done
...
```

---

## Step 6: Timestep Scheduling — `BasicTimeStep`

For a given group and tiling, the timestep table interleaves TIU (compute) and GDMA (DMA) steps:

```
Timestep 0:  GDMA load  layer0.inputs
Timestep 1:  TIU        layer0 compute   +  GDMA load  layer1.inputs
Timestep 2:  TIU        layer1 compute   +  GDMA store layer0.outputs  +  GDMA load layer2.inputs
...
Timestep N:  GDMA store lastLayer.outputs
```

Two scheduling strategies are available:

### Greedy: `layer_nearest_timestep_assignment`
- Used for normal/3D/small-C groups.
- Each op is assigned to its own timestep.
- The next layer's inputs are pre-loaded in the same timestep as the current layer's compute.
- Previous layer's outputs are stored in the same timestep as the current layer's compute.
- Simple, fast, and effective for conv-dominated networks.

### ILP: `IlpTimeStep` (opt=3, `opt3/IlpTimeStep.cpp`)
- Used for MatMul-heavy groups (LLM transformer blocks).
- Integer Linear Programming formulation finds the schedule that maximally overlaps GDMA and TIU.
- Also manages L2 SRAM as an intermediate cache between DDR and LMEM.
- Supports the `opt3/MatmulGroup.cpp` specialization for tiling transformer layers.

---

## Step 7: LMEM Allocation — `LmemAllocator`

Given the timestep table, each tensor has a **live interval** `[start_ts, end_ts]`. The allocator packs them into LMEM using a modified **interval graph coloring / bin-packing** approach.

### Sorting Priority

Tensors are sorted before placement by:
1. **Conflict count** — tensors alive during many timesteps (high contention) go first.
2. **Area** — `time_span × byte_size` (larger tensors placed first to reduce fragmentation).
3. **Start timestep** — earlier tensors get priority among ties.

### Buffer Types

| Type | Description |
|---|---|
| `LMEM_ACTIVATION` | Input/output activation tensors |
| `LMEM_WEIGHT` | Weight tensors (may be `hold_in_lmem` = loaded once, kept forever) |
| `LMEM_OPERATION` | Per-op scratch buffers (e.g., layer norm temporaries) |

### Bank Conflict Awareness

`find_used_banks()` tracks which LMEM banks each allocation occupies. The allocator avoids placing simultaneously-live tensors in the same bank to prevent TIU/GDMA bank conflicts that would stall the pipeline.

---

## Step 8: Software Pipeline — `SoftwarePipeline`

When a group processes multiple slices (e.g. N=3 height slices), the pipeline hides DMA latency via **3-stage software pipelining**:

```
Stage 0 (prologue):     Preload slice[n+1] inputs   (GDMA only)
Stage 1 (steady state): Compute slice[n]             (TIU)   +  prefetch next + writeback prev
Stage 2 (epilogue):     Store  slice[n-1] outputs    (GDMA only)
```

`tensor_swpipl_stage_` maps each tensor value to its pipeline stage (0, 1, or 2). The first and last timestep rows (pure GDMA) become the prologue and epilogue respectively.

Pipeline is enabled only when `nsecs * hsecs > 2` (enough slices to fill all 3 stages).

---

## Step 9: Group Overlap

`GroupOverlap` is a post-DP optimization: if adjacent groups have headroom in their LMEM watermark, ops from the **next** group are migrated into the tail timesteps of the previous group.

This creates an **overlap window** that hides the startup DMA latency between groups:

```
Group A [last timestep]:   TIU A_last  +  GDMA load B_first_inputs   ← overlap
Group B [first timestep]:  TIU B_first                                ← no idle cycles
```

`up_overlap_depth()` computes how many timesteps from the next group can be absorbed without exceeding the upstream group's LMEM budget.

---

## Step 10: MLIR Emission — `GroupOps::buildGroupOps()`

Final valid groups are materialized into `tpu.GroupOp` regions:

```mlir
tpu.GroupOp {
    tpu.Load  %ddr_input   → %lmem0  {tpu.lg = {out_addr=0,    n_slice=1, h_slice=32, ...}}
    tpu.Conv2D %lmem0      → %lmem1  {tpu.lg = {out_addr=4096, n_slice=1, h_slice=28, ...}}
    tpu.Relu   %lmem1      → %lmem2  {tpu.lg = {out_addr=8192, can_merge=true, ...}}
    tpu.Store  %lmem2      → %ddr_out {tpu.lg = {...}}
}
```

Ops that cannot be grouped become **global ops** with plain `tpu.Load` / `tpu.Store` / `tpu.D2D`.

### `tpu.lg` Attribute Fields (`LayerGroupAttr`)

| Field | Type | Description |
|---|---|---|
| `out_addr` | `int64` | LMEM base address of this op's output slice |
| `out_size` | `int64` | Byte size of the output slice in LMEM |
| `buffer_addr` | `int64` | LMEM address of op's scratch buffer |
| `buffer_size` | `int64` | Scratch buffer size in bytes |
| `eu_align` | `bool` | Output must be EU (execution-unit) aligned |
| `can_merge` | `bool` | GDMA transfer can be merged with adjacent transfers |
| `n_idx` / `n_slice` | `int64[]` | N-dimension tile start index / tile size |
| `c_idx` / `c_slice` | `int64[]` | C-dimension tile start index / tile size |
| `h_idx` / `h_slice` | `int64[]` | H-dimension tile start index / tile size |
| `d_idx` / `d_slice` | `int64[]` | D-dimension tile start index / tile size |
| `w_idx` / `w_slice` | `int64[]` | W-dimension tile start index / tile size |
| `in_hslice_offset` | `int64[]` | Halo rows needed from previous slice (conv padding overlap) |

---

## Cache and Debugger

The pass is expensive for large networks, so its results are cached and can be debugged interactively.

### Cache File

```
{model_name}_{chip}_{mode}.layer_group_cache.json
```

If the IR hash matches a cached run, the pass result is loaded directly — skipping the DP re-computation entirely.

### Debugger File

```
{model_name}_{chip}_{mode}.layer_group_debugger.json
```

### `--debugger` Modes

| Mode | Behavior |
|---|---|
| `0` | Normal run, no debug output (default) |
| `1` | Run layer group and write debugger file |
| `2` | Only write debugger file, do not recompute groups |
| `3` | Run layer group using a previously written debugger file |
| `4` | Partial re-run: only recompute groups specified in the debugger file |
| `5` | Check a single group interval given by the debugger file |

### Pass Options

```
--layer-group
  opt=<1|2>               # 1: greedy, 2: dynamic programming (default: 2)
  group_by_cores=<auto>   # force or auto-detect core-parallel grouping
  compress_mode=<none>    # weight compress mode
  lgcache=<true>          # enable/disable cache file
  enable_lghash=<false>   # embed hash into cache filename
  lghash_dir=<"">         # directory for lghash file
  debugger=<0>            # debugger mode (0–5)
  debugger_filename=<"">  # path to debugger JSON
  disable_group_overlap   # disable cross-group LMEM overlap optimization
  config_filename=<"">    # manual group configuration JSON
```

---

## Worked Example

Given three consecutive ops in DDR-only mode:

```
DDR traffic (no grouping):
  input  → [Load] → Conv → [Store] → DDR
  DDR    → [Load] → BN   → [Store] → DDR
  DDR    → [Load] → Relu → [Store] → DDR
  Total: 6 DDR accesses
```

After layer group fuses all three into one group with H-slicing:

```
DDR traffic (grouped, 2 H-slices):
  Slice 0: [Load input slice0] → Conv → BN → Relu → [Store output slice0]
  Slice 1: [Load input slice1] → Conv → BN → Relu → [Store output slice1]
  Total: 4 DDR accesses  (input load + output store per slice)
  BN and Relu intermediates never touch DDR.
```

With software pipelining across slices, the loads for slice 1 overlap with the compute for slice 0, hiding memory latency entirely.
