# Tiny LLM Ada300 Demo

This directory compiles a real Qwen3.5-2B-int4 LLM subgraph pipeline for the
Ada300 RISC-V NPU and boots it under Ada300 QEMU:

```text
embedding -> block_7 -> lm_head -> token id
```

The three subgraphs are real `TOP_F32` MLIR files extracted from Qwen3.5-2B-int4
via `llm_convert.py`. All ops run in **f16** on Ada300 after the TopToTpu pass.

## What This Demo Validates

```text
TOP_F32 MLIR  (real Qwen subgraphs, module.platform = "LLM_QUANTIZED")
  -> processor-assign{chip=ada300 mode=F16}
  -> init
  -> convert-top-to-tpu             (Ada300 f16 lowerings)
  -> address-assign{reuse_addr=false}
  -> convert-tpu-to-rxops           (all LLM ops: RMSNorm, A16MatMul, RoPE,
                                      FAttention, SiLU, Gather, Concat, ...)
  -> one-shot-bufferize
  -> finalize-memref-to-llvm
  -> convert-func-to-llvm
  -> mlir-translate --mlir-to-llvmir
  -> rx-llvm clang  -> RISC-V object files
  -> ld.lld  -> Ada300 baremetal ELF  (embed-system BSP + rx-ops f16 backend)
  -> rx-qemu ada300s_evk  -> PASS
```

A successful run prints:

```text
[TinyLLM] Ada300 real Qwen pipeline
[TinyLLM] graph: embedding -> block_7 -> lm_head -> token id
[TinyLLM] embedding ok
[TinyLLM] block_7 ok
[TinyLLM] lm_head ok
[TinyLLM] token=0
[TinyLLM] done
```

`token=0` reflects argmax on zero-initialized weight tensors; numerical
correctness is not the goal of this demo.  QEMU exits automatically after
`done` via a RISC-V semihosting `SYS_EXIT` call.

## Directory Layout

```text
examples/tiny_llm/
  CMakeLists.txt
  README.md
  tiny_llm_baremetal_main.c          baremetal runner + QEMU semihosting exit
  rx_ops_bridge_ada300_compat.c      flat C ABI bridge (MLIR -> rx-ops public API)
  rxops_c_init_stub.c                rxops_init() stub for baremetal
  rxops_tiny_ada300_setup.c          f16 callback registration for Ada300
  rxops_ada300_tensor_utils_stub.c   prepare_tensor_as / release_tensor_view stub
  mlir/                              real Qwen3.5-2B-int4 TOP_F32 subgraphs
    embedding.mlir
    block_7.mlir
    lm_head.mlir
  smoke_mlir/                        synthetic fallback graphs (no weight files needed)
    embedding.mlir
    block_7.mlir
    lm_head.mlir
  embedding_top_weights.npz          Qwen weight file for embedding subgraph
  block_7_top_weights.npz            Qwen weight file for block_7 subgraph
  lm_head_top_weights.npz            Qwen weight file for lm_head subgraph
  output/                            intermediate MLIR/LLVM IR (generated)
  build/                             compiled objects and ELF (generated)
```

### MLIR source selection

`-DTINY_LLM_USE_REAL_MLIR=ON` (default OFF) switches the input from
`smoke_mlir/` to `mlir/`.  Real MLIR requires the three `*_top_weights.npz`
files to be present in this directory.

Generate the weight files with:

```bash
python3 python/tools/llm_convert.py \
  -m /workspace/Qwen3.5-2B-int4-AutoRound \
  -s 2048 -q w4bf16 -c bm1684x \
  --only_mlir --debug \
  -o /tmp/qwen_weights

cp /tmp/qwen_weights/*/embedding_top_weights.npz examples/tiny_llm/
cp /tmp/qwen_weights/*/block_7_top_weights.npz   examples/tiny_llm/
cp /tmp/qwen_weights/*/lm_head_top_weights.npz   examples/tiny_llm/
```

## Toolchain Prerequisites

```text
/workspace/tpu-mlir/build/bin/tpuc-opt
/workspace/rx-llvm/build/bin/clang
/workspace/rx-llvm/build/bin/llvm-objdump
/workspace/rx-qemu/build/qemu-system-riscv64
/workspace/embed-system/ada300_snpu_sdk
/usr/lib/picolibc/riscv64-unknown-elf        (picolibc-riscv64-unknown-elf pkg)
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
  -DADA300_NEWLIB_SYSROOT=/usr/lib/picolibc/riscv64-unknown-elf \
  -DTINY_LLM_USE_REAL_MLIR=ON
```

Omit `-DTINY_LLM_USE_REAL_MLIR=ON` to use the synthetic `smoke_mlir/` inputs
(no weight files required).

## Build and Run

```bash
# MLIR -> LLVM IR (host step, requires tpuc-opt)
make -C examples/tiny_llm/build tiny-llm-rxops-ll

# Cross-compile -> Ada300 RISC-V ELF
make -C examples/tiny_llm/build tiny-llm-baremetal

# Boot under rx-qemu (exits automatically via semihosting)
make -C examples/tiny_llm/build run-tiny-llm-baremetal
```

Expected build artifacts:

```text
examples/tiny_llm/build/tiny_llm.elf
examples/tiny_llm/build/tiny_llm.asm
examples/tiny_llm/build/tiny_llm.map
examples/tiny_llm/output/*_rxops.ll
```

## Compiler Pass Pipeline

Each subgraph runs through these `tpuc-opt` passes (the `convert-tpu-to-rxops`
pass auto-runs the first four when it sees `TOP_F32` state):

| Pass | Output state |
|---|---|
| `processor-assign{chip=ada300 mode=F16}` | — |
| `init{freq=0 level=0}` | — |
| `convert-top-to-tpu` | `TPU_LOWERED` |
| `address-assign{reuse_addr=false}` | `TPU_ADDRESSED` |
| `convert-tpu-to-rxops` | LLVM + `func.call @rxops_bridge_*` |
| `one-shot-bufferize` | bufferized memref |
| `finalize-memref-to-llvm` + `convert-func-to-llvm` | LLVM dialect |
| `mlir-translate --mlir-to-llvmir` | `.ll` LLVM IR |

## Ada300 Op Coverage

All ops emitted by the real Qwen3.5-2B block_7 subgraph are supported:

| TPU op | Bridge function | Ada300 kernel |
|---|---|---|
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

## Linking Model

```text
embed-system BSP (startup.S, bsp_main.c, uart.c, mini_printf.c, mr_heap.c)
tiny_llm runtime (tiny_llm_baremetal_main.c, rx_ops_bridge_ada300_compat.c,
                  rxops_c_init_stub.c, rxops_tiny_ada300_setup.c,
                  rxops_ada300_tensor_utils_stub.c)
compiled MLIR objects (embedding_rxops_rv64.o, block_7_rxops_rv64.o,
                       lm_head_rxops_rv64.o)
rx-ops Ada300 f16 backend (selected ops only - avoids full upstream registry)
picolibc (libm.a)
```

`rxops_tiny_ada300_setup.c` registers **only f16 callbacks** for the ops the
real Qwen model uses.  All fp32 kernel variants (which live in `quant.c` and
pull unrelated asm deps) are excluded.  `rxops_ada300_tensor_utils_stub.c`
provides the `rxnn_ada300_prepare_tensor_as` / `rxnn_ada300_release_tensor_view`
helpers required by `full_attention.c`, without pulling in the rest of
`quant.c`.

## Current Limitations

- Numerical correctness is not validated; weight tensors are zero-initialized
  at runtime by the baremetal runner.
- KV-cache decode path (`tpu.Store` into KV buffer, `cache_matmul`) is not
  yet implemented in `convert-tpu-to-rxops`.
- Only `block_7` is compiled; all 28 Qwen blocks would require a full
  `llm_convert.py --debug` run (without `--only_mlir`) to generate weight
  files for each block.
