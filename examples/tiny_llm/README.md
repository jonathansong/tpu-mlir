# Tiny LLM Ada300 Demo

A minimal but complete **LLM inference loop** running on the Ada300 RISC-V NPU
under QEMU.  Uses five compiled subgraphs from real **Qwen3.5-2B-int4** MLIR,
mirroring the structure of the BM1684X `chat.cpp` reference implementation:

```text
PREFILL (once per prompt)
  embedding  →  block_7  →  lm_head  →  first token

DECODE (one iteration per generated token)
  embedding_cache  →  block_cache_7  →  lm_head  →  next token
                            ↑                   ↑
                       KV cache append      argmax (TopK-1)
```

All five subgraphs are real `TOP_F32` MLIR files extracted from Qwen3.5-2B-int4
via `llm_convert.py`.  All ops run in **f16** on Ada300 after the TopToTpu pass.

## What This Demo Validates

```text
TOP_F32 MLIR  (real Qwen subgraphs, module.platform = "LLM_QUANTIZED")
  → processor-assign{chip=ada300 mode=F16}      step 2
  → init{freq=0 level=0}                         step 2
  → convert-top-to-tpu                           step 3  (Ada300 f16 lowerings)
  → address-assign{reuse_addr=false               step 4
                   skip_weight_check=true}
  → convert-tpu-to-rxops                         step 5  (RMSNorm, A16MatMul,
                                                          RoPE, FAttention, …)
  → one-shot-bufferize                           step 6  (tensor → memref)
  → finalize-memref-to-llvm + convert-func-to-llvm  step 7
  → mlir-translate --mlir-to-llvmir              step 8  → .ll
  → rx-llvm clang  → RISC-V object files
  → ld.lld         → Ada300 baremetal ELF
  → rx-qemu ada300s_evk  → PASS
```

A successful run prints:

```text
[TinyLLM] Ada300 Qwen3.5-2B-int4 minimal inference demo
[TinyLLM] SEQ=1024 DIM=2048 KV_HIST=2048 MAX_NEW=4
[TinyLLM] --- PREFILL (SEQ=1024) ---
[TinyLLM] embedding ok
[TinyLLM] block_7 (prefill) ok
[TinyLLM] prefill done — first token=XXXXX (history=1024)
[TinyLLM] --- DECODE (max 4 tokens) ---
[TinyLLM] decode step 0: token=XXXXX (history=1025)
[TinyLLM] decode step 1: token=XXXXX (history=1026)
[TinyLLM] decode step 2: token=XXXXX (history=1027)
[TinyLLM] decode step 3: token=XXXXX (history=1028)
[TinyLLM] done (total history=1028)
```

Token values reflect zero-initialised weight tensors; numerical correctness is
not the goal.  QEMU exits automatically via a RISC-V semihosting `SYS_EXIT` call.

> **Note**: With SEQ=1024, block_7 prefill is compute-heavy in software QEMU
> simulation (~5–15 minutes).  The decode steps are fast by comparison.

## Directory Layout

```text
examples/tiny_llm/
  CMakeLists.txt
  README.md
  tiny_llm_baremetal_main.c          prefill + decode inference loop + QEMU exit
  rx_ops_bridge_ada300_compat.c      flat C ABI bridge (MLIR ciface → rx-ops)
  rxops_c_init_stub.c                rxops_init() stub for baremetal
  rxops_tiny_ada300_setup.c          f16 callback registration for Ada300
  rxops_ada300_tensor_utils_stub.c   prepare_tensor_as / release_tensor_view stub
  mlir/                              real Qwen3.5-2B-int4 TOP_F32 subgraphs
    embedding.mlir                   (1,1024) si32 → (1,1024,2048) f32
    embedding_cache.mlir             (1,1)    si32 → (1,1,2048)    f32  [decode]
    block_7.mlir
    block_cache_7.mlir
    lm_head.mlir
  output/                            intermediate MLIR/LLVM IR (generated, see below)
  build/                             compiled objects and ELF (generated)
```

  embedding_top_weights.npz          Qwen weight file (shared by embedding + embedding_cache)
  block_7_top_weights.npz
  block_cache_7_top_weights.npz
  lm_head_top_weights.npz

### Generated output files

Each subgraph produces 8 intermediate files in `output/`:

| Step | File | Description |
|------|------|-------------|
| 1 | `{name}_top.mlir` | Source copy |
| 2 | `{name}_top_assigned.mlir` | After `processor-assign` + `init` |
| 3 | `{name}_tpu.mlir` | After `convert-top-to-tpu` |
| 4 | `{name}_tpu_addr.mlir` | After `address-assign` |
| 5 | `{name}_rxops.mlir` | After `convert-tpu-to-rxops` |
| 6 | `{name}_rxops_bufferize.mlir` | After `one-shot-bufferize` |
| 7 | `{name}_rxops_finished.mlir` | After LLVM dialect lowering |
| 8 | `{name}_rxops.ll` | Final LLVM IR (compiled to `.o`) |

## Weight Files

Weight files are required.  Generate them with:

```bash
python3 python/tools/llm_convert.py \
  -m /workspace/Qwen3.5-2B-int4-AutoRound \
  -s 2048 -q w4bf16 -c bm1684x \
  --only_mlir --debug \
  -o /tmp/qwen_weights

cp /tmp/qwen_weights/*/embedding_top_weights.npz        examples/tiny_llm/
cp /tmp/qwen_weights/*/block_7_top_weights.npz          examples/tiny_llm/
cp /tmp/qwen_weights/*/block_cache_7_top_weights.npz    examples/tiny_llm/
cp /tmp/qwen_weights/*/lm_head_top_weights.npz          examples/tiny_llm/
# embedding_cache shares the embedding weight table:
cp examples/tiny_llm/embedding_top_weights.npz \
   examples/tiny_llm/embedding_cache_top_weights.npz
```

## Toolchain Prerequisites

```text
/workspace/tpu-mlir/build/bin/tpuc-opt
/workspace/rx-llvm/build/bin/clang
/workspace/rx-llvm/build/bin/llvm-objdump
/workspace/rx-qemu/build/qemu-system-riscv64
/workspace/embed-system/ada300_snpu_sdk
/usr/lib/picolibc/riscv64-unknown-elf      (picolibc-riscv64-unknown-elf pkg)
```

## Configure

```bash
cmake -S examples/tiny_llm \
  -B examples/tiny_llm/build \
  -DMLIR_DIR=/workspace/rx-llvm/build/lib/cmake/mlir \
  -DTPUC_OPT=/workspace/tpu-mlir/build/bin/tpuc-opt \
  -DADA300_LLVM_ROOT=/workspace/rx-llvm/build \
  -DADA300_SDK_ROOT=/workspace/embed-system/ada300_snpu_sdk \
  -DADA300_QEMU=/workspace/rx-qemu/build/qemu-system-riscv64 \
  -DADA300_NEWLIB_SYSROOT=/usr/lib/picolibc/riscv64-unknown-elf
```

## Build and Run

```bash
cd examples/tiny_llm/build

# MLIR → LLVM IR (host step, runs all 8 pipeline steps for each subgraph)
make tiny-llm-rxops-ll

# Cross-compile → Ada300 RISC-V ELF
make tiny-llm-baremetal

# Boot under rx-qemu (exits automatically via semihosting)
make run-tiny-llm-baremetal
```

Expected build artifacts:

```text
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_top.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_top_assigned.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_tpu.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_tpu_addr.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_rxops.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_rxops_bufferize.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_rxops_finished.mlir
output/{embedding,embedding_cache,block_7,block_cache_7,lm_head}_rxops.ll
build/tiny_llm.elf
build/tiny_llm.asm
build/tiny_llm.map
```

## Compiler Pass Pipeline (per subgraph)

| Step | Pass | Note |
|------|------|------|
| 2 | `processor-assign{chip=ada300 mode=F16}` + `init` | sets chip target, F16 mode |
| 3 | `convert-top-to-tpu` | Top → TPU dialect (f16 quantisation) |
| 4 | `address-assign{reuse_addr=false skip_weight_check=true}` | flat LPDDR layout; `skip_weight_check` bypasses the 256 MiB hardware limit for QEMU |
| 5 | `convert-tpu-to-rxops` | TPU → `func.call @rxops_bridge_*` |
| 6 | `one-shot-bufferize` | tensor SSA → memref allocations |
| 7 | `finalize-memref-to-llvm` + `convert-func-to-llvm` + `reconcile-unrealized-casts` | LLVM dialect |
| 8 | `mlir-translate --mlir-to-llvmir` + `perl` symbol rename | `.ll` LLVM IR |

## MLIR C-Interface ABI

The generated `_mlir_ciface_*` functions use the MLIR ranked-memref ABI.
Each tensor argument becomes a pointer to a descriptor struct:

```c
// Rank-2 (e.g. position_ids [3×1024])
typedef struct { void *allocated, *aligned; int64_t offset;
                 int64_t sizes[2]; int64_t strides[2]; } MemRef2D;

// Rank-3 (e.g. hidden_states [1×1024×2048])
typedef struct { void *allocated, *aligned; int64_t offset;
                 int64_t sizes[3]; int64_t strides[3]; } MemRef3D;

// Rank-4 (e.g. attention_mask [1×1×1024×1024], kv_cache [1×1024×2×256])
typedef struct { void *allocated, *aligned; int64_t offset;
                 int64_t sizes[4]; int64_t strides[4]; } MemRef4D;
```

Result tensors are returned by writing a packed struct of descriptors to the
first argument pointer:

| Subgraph | Signature |
|----------|-----------|
| `embedding_main` | `(MemRef3D *out, MemRef2D *token_ids)` |
| `embedding_cache_main` | `(MemRef3D *out, MemRef2D *token_id)` |
| `block7_main` | `(Block7Result *out, MemRef3D *states, MemRef2D *pos_ids, MemRef4D *mask)` |
| `block_cache7_main` | `(BlockCache7Result *out, MemRef3D *states, MemRef2D *pos_ids, MemRef4D *mask, MemRef4D *hist_k, MemRef4D *hist_v)` |
| `lm_head_main` | `(MemRef2D *out, MemRef2D *hidden)` |

where:
```c
typedef struct { MemRef3D out_states; MemRef4D k_cache; MemRef4D v_cache; } Block7Result;
typedef struct { MemRef3D out_states; MemRef4D new_k;   MemRef4D new_v;   } BlockCache7Result;
```

`lm_head` includes a `top.TopK` op that returns the token index directly as `si32`;
no separate argmax step is needed on the host.

## Subgraph Shapes

| Subgraph | Phase | Inputs | Outputs |
|----------|-------|--------|----------|
| `embedding` | prefill | token_ids `(1,1024)` si32 | hidden `(1,1024,2048)` f16 |
| `embedding_cache` | decode | token_id `(1,1)` si32 | hidden `(1,1,2048)` f16 |
| `block_7` | prefill | states `(1,1024,2048)` f16 · pos_ids `(3,1024)` si32 · mask `(1,1,1024,1024)` f16 | out_states `(1,1024,2048)` f16 · k_cache `(1,1024,2,256)` f16 · v_cache `(1,1024,2,256)` f16 |
| `block_cache_7` | decode | states `(1,1,2048)` f16 · pos_ids `(3,1)` si32 · mask `(1,1,1,2049)` f16 · hist_k `(1,2048,2,256)` f16 · hist_v `(1,2048,2,256)` f16 | out_states `(1,1,2048)` f16 · new_k `(1,1,2,256)` f16 · new_v `(1,1,2,256)` f16 |
| `lm_head` | both | hidden `(1,2048)` f16 | token_id `(1,1)` si32 |

**KV cache constants** (from compiled shapes):
```
SEQ     = 1024   prefill sequence length
DIM     = 2048   hidden size
KV_HIST = 2048   max KV history slots
NHEADS  = 2      KV heads per layer
HDIM    = 256    KV head dimension
KV_STEP = 512    = NHEADS × HDIM  (elements per token slot)
```

**Attention masks** (f16, `MASK_NEG_INF = 0xF0E2 ≈ −10004`):

| Phase | Shape | Rule |
|-------|-------|------|
| Prefill (causal) | `(1,1,SEQ,SEQ)` | `mask[i][j] = 0 if j≤i, else MASK_NEG_INF` |
| Decode (history) | `(1,1,1,KV_HIST+1)` | `mask[i] = 0 if i<history, else MASK_NEG_INF` |

## Ada300 Op Coverage

All ops emitted by the real Qwen3.5-2B block subgraphs are supported:

| TPU op | Bridge function | Ada300 kernel |
|--------|-----------------|---------------|
| `tpu.RMSNorm` | `rxops_bridge_rms_norm_f16` | `rxnn_npu_rms_norm_fp16` |
| `tpu.A16MatMul` | `rxops_bridge_a16matmul_f16` | `rxnn_npu_fullyconnected_fp16` |
| `tpu.Rope` | `rxops_bridge_rope_contiguous_f16` | `rxnn_npu_rope_fp16` |
| `tpu.FAttention` | `rxops_bridge_fattention_f16` | `rxnn_npu_full_attention_fp16` |
| `tpu.Active(SILU)` | `rxops_bridge_silu_f16` | `rxnn_npu_silu_fp16` |
| `tpu.Active(SIGMOID)` | `rxops_bridge_sigmoid_f16` | `rxnn_npu_sigmoid_fp16` |
| `tpu.MatMul` | `rxops_bridge_matmul_f16` | `rxnn_npu_matmul_fp16` |
| `tpu.Reshape` | `rxops_bridge_reshape_bytes` | `rxnn_npu_reshape_copy` |
| `tpu.Permute` | `rxops_bridge_transpose_nd` | `rxnn_npu_transpose_fp16` |
| `tpu.Gather` | `rxops_bridge_gather_nd` | `rxnn_npu_gather_fp16` |
| `tpu.Concat` | `rxops_bridge_concat{2,3,4}_nd` | `rxnn_npu_concat_fp16` |
| `tpu.TopK` | `rxops_bridge_topk_f16` | `rxnn_npu_topk_fp16` |
| `tpu.Arg` | `rxops_bridge_topk_f16` (argmax mode) | `rxnn_npu_argmax_fp16` |
| `tpu.Add` | `rxops_bridge_add_f16` | `rxnn_npu_add_fp16` |
| `tpu.Mul` | `rxops_bridge_mul_f16` | `rxnn_npu_mul_fp16` |
| `tpu.Slice` | `rxops_bridge_slice_nd` | (copy-based slice) |

## Inference Loop Design

`tiny_llm_baremetal_main.c` mirrors `Qwen3_5::forward_first` + `forward_next`
from `LLM-TPU/models/Qwen3_5/cpp_demo/chat.cpp`, adapted for bare-metal MLIR
C-interface ABI instead of bmruntime:

| Concept | BM1684X (`chat.cpp`) | Ada300 (`tiny_llm_baremetal_main.c`) |
|---------|----------------------|--------------------------------------|
| Sub-network dispatch | `bmrt_launch_tensor_ex` | `_mlir_ciface_*` direct call |
| Device memory | `bm_device_mem_t` (TPU DRAM) | `uint16_t[]` in LPDDR |
| KV cache storage | aliased into `.bmodel` static input buffers | `calloc`'d flat array |
| KV append (decode) | `bm_mem_from_device(addr + offset)` → output redirect | `memcpy` into `kv_k/v[history * KV_STEP]` |
| Token sampling | `greedy_head` sub-network or host argmax | embedded in `lm_head` via `top.TopK` |
| Position IDs | built by `get_rope_index()` in pipeline.cpp | `pos[d][i] = i` (prefill), `{h,h,h}` (decode) |

## Linking Model

```text
embed-system BSP
  startup.S, bsp_main.c, uart.c, mini_printf.c, mr_heap.c
tiny_llm runtime
  tiny_llm_baremetal_main.c       — prefill + decode loop, KV cache, QEMU exit
  rx_ops_bridge_ada300_compat.c   — _mlir_ciface_rxops_bridge_* wrappers
  rxops_c_init_stub.c             — rxops_init() no-op stub
  rxops_tiny_ada300_setup.c       — f16 op callback registration
  rxops_ada300_tensor_utils_stub.c — prepare_tensor_as / release_tensor_view stub
compiled MLIR objects
  {embedding,embedding_cache,block_7,block_cache_7,lm_head}_rxops_rv64.o
rx-ops Ada300 f16 backend (selected ops only)
picolibc libm.a
```

`rxops_tiny_ada300_setup.c` registers **only f16 callbacks** for ops the Qwen
model uses.  All fp32 kernel variants (which live in `quant.c` and pull
unrelated asm dependencies) are excluded.

## Current Limitations

- Numerical correctness is not validated; all weight tensors are
  zero-initialised at runtime.
- Only layer 7 (`block_7` / `block_cache_7`) is compiled; a full 28-layer run
  would require weight files for all blocks.
- All active layers in this demo are Full Attention (`is_FA(7)` = true).
  No Mamba/linear-attention blocks are present in the single-layer demo.

