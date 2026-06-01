# Layer Group Optimization Pass

## Overview

**Layer grouping** is the most critical optimization pass in tpu-mlir. Its goal is to minimize expensive DRAM↔LMEM data transfers by running chains of TPU operations on small tensor *slices* that all fit in local memory (LMEM), instead of executing each op on the full tensor with round-trips to global memory between every op. It also enables hardware pipelining (overlapping GDMA data movement with BDC computation).

---

## The Core Problem

A TPU has two memory tiers:
- **Global memory (DRAM)**: Large, slow (~100s of GB/s bandwidth)
- **Local memory (LMEM/SRAM)**: Small (~128KB per core), very fast

Without grouping, naively compiled code does:

```
Load tensor A from DRAM → LMEM
Execute Conv
Store result to DRAM
Load result from DRAM → LMEM
Execute BatchNorm
Store result to DRAM
...
```

Every operation incurs a full DRAM round-trip — the execution is **memory-bandwidth bound**, not compute-bound.

With grouping, tensors are sliced into small chunks, and for each chunk the **entire chain** runs through LMEM without touching DRAM for intermediates:

```
Load slice[0] of A → LMEM
  Conv(slice[0])       → LMEM
  BatchNorm(slice[0])  → LMEM
  ReLU(slice[0])       → LMEM
Store slice[0] result → DRAM
Load slice[1] of A → ...
```

This is analogous to **software-managed cache blocking**: the compiler explicitly decides what data to bring in, when, and in what order — the hardware won't do it automatically.

---

## Key Concepts

### Layer Group (`LgInfo`)

A set of consecutive, compatible TPU operations processed together. Defined by:

| Field | Description |
|---|---|
| `group_ops` | The operations in this group |
| `group_ins` / `group_outs` | Input/output tensors crossing the group boundary |
| `type` | `GROUP_NORMAL`, `GROUP_3D`, `GROUP_SMALL_C` |
| `shape_secs` | How to slice tensors (`nsecs`, `csecs`, `hsecs`, `dsecs`, `wsecs`) |
| `group_cost` | Estimated cost used during group search |

### Tiling / Shape Sections

You must choose *how* to slice tensors along N, C, H, W, D dimensions. The slice must be small enough that all live intermediate tensors fit in LMEM simultaneously. This is essentially a **bin-packing / constraint satisfaction** problem.

### Tensor Lifetime Analysis

Within a group, every intermediate tensor has a liveness range: `[produced_at_timestep, last_used_at_timestep]`. Two tensors whose lifetimes overlap must both fit in LMEM at the same time. This is equivalent to **register allocation** but for scratchpad memory.

### Time Steps (`BasicTimeStep`)

The group's execution is divided into discrete `TimestepRow`s, each containing:
- **`TpuTsField`**: compute ops (BDC — the tensor processor)
- **`GdmaTsField`**: data transfers (GDMA loads/stores from/to DRAM)

The scheduler maximizes **overlap** between GDMA and BDC, since they are independent hardware units.

### Software Pipeline (`SwPipeline`)

Classic 3-stage software pipeline:

| Stage | Action |
|---|---|
| Stage 0 | Load first slice |
| Stage 1 | Compute slice N while loading slice N+1 |
| Stage 2 | Store results of slice N-1 while computing |

This hides DMA latency behind computation — equivalent to Lam-style software pipelining for loop iterations.

### Grouping Problem = Graph Partitioning

Given a DAG of operations, partition it into contiguous chains where each chain:
- Has all tensors fitting in LMEM (feasibility constraint)
- Minimizes total DRAM traffic (optimization objective)

In tpu-mlir this is solved with **dynamic programming**: `cost[i][j]` = cost of making ops `i..j` one group, then find the minimum-cost partition.

---

## The Pass Pipeline

For normal mode (opt ≠ 3), `InternalLgOptimizer` runs these passes in order:

| Order | Pass | Source File |
|---|---|---|
| 1 | `LayerGroupSearchPass` | `GroupMethod.cpp` |
| 2 | `GroupPostTransformPass` | `GroupPostTransform.cpp` |
| 3 | `TimeStepAssignmentPass` | `TimeStepMethod.cpp` |
| 4 | `LocalMemoryAllocationPass` | `LmemAllocator.cpp` |
| 5 | `TimeStepCombinePass` | `TimeStepCombine.cpp` |
| 6 | `GroupDataMoveOverlapPass` | `GroupOverlap.cpp` |
| 7 | `LayerGroupProfilePass` | `LayerGroupProfile.cpp` |

For **opt=3**, an ILP-based solver (`opt3/IlpTimeStep.cpp`, `opt3/MatmulGroup.cpp`) replaces the scheduling and allocation passes with a globally optimal formulation using Google OR-Tools.

---

## Data Flow

```
TPU ops in MLIR function
       │
       ▼  GroupMethod — DP-based grouping
   LgInfo list (groups with shape_secs)
       │
       ▼  TimeStepMethod — scheduling
   BasicTimeStep tables (one per group)
       │
       ▼  LmemAllocator — address assignment
   time_steps with concrete LMEM addresses
       │
       ▼  TimeStepCombine + GroupOverlap — refinement
   Optimized schedule
       │
       ▼  GroupOps — MLIR emission
   GroupOp + LoadOp + StoreOp in IR
```

---

## Component Breakdown

### `GroupMethod.cpp` — Group Search

Finds optimal groupings using **dynamic programming**.

1. **Base Groups**: splits the network into candidate groups based on operation compatibility (`can_be_group_3d`, `can_be_group_small_c`, etc.)
2. **DP Optimization**: builds a `cost[i][j]` table and solves for minimum-cost cut points
3. **Validation**: `is_layer_group_valid()` checks LMEM feasibility by attempting actual allocation
4. **Caching**: uses `LgCache` with hash-based lookup to avoid recomputing known group costs

### `BasicTimeStep.cpp` — Timestep Data Structure

Holds the per-group schedule as a `timestep_table_`: a vector of `TimestepRow`, each with a `TpuTsField` (compute) and `GdmaTsField` (DMA) list.

### `TimeStepMethod.cpp` — Scheduler

Two assignment strategies:

| Strategy | Description |
|---|---|
| `layer_nearest_timestep_assignment` | Simple: each op gets its own timestep; loads immediately before, stores immediately after |
| `memory_aware_timestep_assignment` | Advanced: computes `slack = layer_cycles − gdma_cycles` and moves GDMA loads earlier to maximize GDMA-BDC overlap |

### `LmemAllocator.cpp` — Memory Allocation

Assigns concrete LMEM addresses to tensors. Key responsibilities:
- Avoids **bank conflicts** (tensors accessed simultaneously must be in different memory banks)
- Tries multiple shape-section combinations (`sc_method_brute_force`, `sc_method_quick_search`, `sc_method_multi_core`, etc.)
- Picks the allocation with the lowest cost (LMEM usage vs. GDMA traffic)

Key data structures:
```
lmem_buffer_: map<mem_buffer_key_t, mem_buffer_value_t>
  key:   { type (WEIGHT/ACTIVATION/OPERATION), value, op, conflict }
  value: { addr, size, start_ts, end_ts, bank_id }
```

### `CycleCalculator.cpp` — Cycle Estimation

Chip-specific cycle counts used by the scheduler and allocator for cost-driven decisions:
- `getLocalLayerCycle()` — BDC compute cycles for a sliced tensor
- `getGdmaCycle()` — GDMA transfer cycles based on tensor shape and size
- `getGroupCycle()` — total group cycles on the critical path

Subclasses: `Bm168xCycleCalculator`, `Cv18xxCycleCalculator`.

### `SwPipeline.cpp` — Software Pipelining

Restructures the timestep table into a 3-stage circular pipeline to maximize GDMA/BDC concurrency. Assigns each tensor to a pipeline stage (0=load, 1=compute, 2=store), then merges first/last timesteps to close the pipeline loop.

### `GroupOps.cpp` — MLIR Emission

Replaces original TPU ops with `GroupOp` in the IR, surrounded by `LoadOp`/`StoreOp`. Encodes the entire schedule, memory layout, and slicing parameters as MLIR attributes on the `GroupOp`.

### `TimeStepCombine.cpp` — Timestep Merging

Reduces dispatch overhead by merging timesteps that can be safely combined without violating tensor lifetime or LMEM capacity constraints.

### `GroupOverlap.cpp` — Inter-Group Overlap

Moves GDMA loads for group N+1 into group N's execution window, hiding memory latency at group boundaries:

```
Without overlap:   [Group N exec] → [Load Group N+1] → [Group N+1 exec]
With overlap:      [Group N exec + Load Group N+1]    → [Group N+1 exec]
```

### `CoeffReloadOpt.cpp` — Weight Reload Strategy

Decides which weight tensors to hold resident in LMEM across slices vs. reload each time, based on computed GDMA slack. Holds weights when `slack ≥ gdma_reload_cost`, otherwise marks them for reload to free LMEM for other tensors.

### `opt3/IlpTimeStep.cpp` — ILP-Based Scheduling

Uses **Google OR-Tools** MIP solver to formulate timestep assignment and memory allocation as an integer linear program:
- **Variables**: binary variables for tensor placement at each (timestep, slice)
- **Objective**: minimize total cycles or memory usage
- **Constraints**: data dependencies, LMEM capacity, bank conflicts, tensor lifetimes

More compute-intensive than the heuristic approaches but finds globally optimal solutions. Used for complex patterns like multi-head attention in LLMs.

### `opt3/MatmulGroup.cpp` — MatMul Chain Optimization

Detects and optimizes chains of connected MatMul operations (common in transformer/LLM layers). Identifies shared intermediate results and computes optimal grouping to minimize both computation and data movement across the chain.

---

## Classical Compiler Analogies

| Layer Group Concept | Classical Compiler Equivalent |
|---|---|
| LMEM address allocation | Register allocation (linear scan / graph coloring) |
| Timestep scheduling | Instruction scheduling for two parallel execution units |
| Shape section search | Loop tiling / cache blocking |
| Software pipeline | Lam-style software pipelining (loop prologue/epilogue) |
| Tensor liveness analysis | Live range analysis |
| Group search (DP partition) | Graph partitioning / optimal basic block layout |
| ILP scheduling (opt3) | ILP-based instruction scheduling |

---

## Key Data Structures Summary

| Structure | Purpose | Key Fields |
|---|---|---|
| `LgInfo` | One layer group | `group_ops`, `group_ins/outs`, `type`, `shape_secs`, `group_cost` |
| `BasicTimeStep` | Group schedule | `timestep_table_` (vector of `TimestepRow`) |
| `TimestepRow` | One timestep | `tpu0_ts_field` (compute), `gdma0_ts_field` (DMA) |
| `shape_secs_t` | Tensor slicing params | `nsecs`, `csecs`, `hsecs`, `dsecs`, `wsecs` |
| `tensor_info_t` | Per-tensor info | `mode` (load/store), `slice_info`, `hold_in_lmem`, `stage` |
| `mem_buffer_key_t` | Buffer identity key | `type`, `value`, `op`, `conflict` |
| `mem_buffer_value_t` | Allocated buffer | `addr`, `size`, `start_ts`, `end_ts` |
| `group_cycle_info_t` | Cycle estimation input | slice indices, memory addresses, sizes, pipeline stage |

---

## FAQ

**Why not put everything in one group?**
Intermediate tensors from diverging branches can't all fit in LMEM simultaneously.

**What if LMEM is too small for even a single op's slice?**
That op becomes a single-op group and falls back to global execution (full DRAM round-trip).

**How does multi-core change things?**
Each core has its own LMEM, so tiling across cores adds another dimension to the shape-section search.

**Why use ILP for opt=3?**
Heuristic schedulers can miss optimal solutions for complex patterns with many interdependencies (e.g., attention layers). ILP guarantees optimality at the cost of compile time.

**How is grouping validated?**
By attempting actual LMEM allocation with real tensor sizes. If every tensor's live range fits within the LMEM budget, the group is valid.
