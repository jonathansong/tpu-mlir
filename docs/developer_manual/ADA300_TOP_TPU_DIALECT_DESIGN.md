# ADA300 TOP/TPU Dialect Extension Design

This document proposes an ADA300 extension of the existing two-level MLIR
dialect design for LLM inference compilation. It intentionally reuses the
existing `top` and `tpu` dialects. It does not introduce a third
dialect.

The existing compiler structure is kept:

```text
top  ->  tpu  ->  LLVM / assembly / binary
```

The main improvement is to make the boundary sharper:

| Level | Responsibility | Should Not Contain |
|---|---|---|
| `top` | Model and operator semantics | SRAM bank choices, exact ISA encodings, tensor CSR fields |
| `tpu` | Existing target-independent TPU ops plus ADA300 executable semantics | Import-framework artifacts, vague unfused model patterns |
| backend | Register allocation, instruction encoding, binary emission | High-level graph rewrites |

## Design Goals

1. Preserve LLM intent long enough for model-level optimization.
2. Expose ADA300 execution details early enough for tiling, memory planning, and
   synchronization.
3. Keep exact opcode/funct bit encoding out of `top` and mostly out of generic
   optimization passes.
4. Make quantization and KV-cache behavior visible in the IR.
5. Allow `tpu` ops to lower naturally to the ADA300 tensor and vector custom ISA.
6. Reuse the existing `TopToTpu`, layer-group, memory-planning, and codegen
   infrastructure where its contracts fit ADA300.

## TOP Dialect

`top` should represent what the model computes. It should stay portable across
targets and should not mention SRAM, ping-pong buffers, tensor CSRs, or raw
instruction fields.

### Existing Useful TOP Ops

The current `top` dialect already has many LLM building blocks:

| Existing Op | LLM Use |
|---|---|
| `top.MatMul`, `top.A16MatMul` | dense projections, attention matmuls, MLP |
| `top.Attention`, `top.FAttention` | fused attention forms |
| `top.ScaleDotProductAttention` | PyTorch SDPA import |
| `top.RMSNorm`, `top.LayerNorm` | normalization |
| `top.Rope` | rotary position embedding |
| `top.Softmax` | attention probability |
| `top.SiLU`, `top.GELU`, `top.Sigmoid` | MLP activation |
| `top.DequantInt`, `top.QuantizeLinear`, `top.DequantizeLinear` | quantized graph import |

### Recommended New TOP Ops

Add TOP ops only when they preserve semantic information that is hard to recover
after decomposition.

| Proposed Op | Purpose |
|---|---|
| `top.QKVProj` | Fused Q/K/V projection with head metadata |
| `top.PrefillAttention` | Attention in prefill mode |
| `top.DecodeAttention` | Attention in decode mode with current-token K/V and KV-cache inputs |
| `top.KVCacheLoad` | Logical KV-cache read |
| `top.KVCacheStore` | Logical KV-cache update |
| `top.GatedMLP` | `up_proj`, `gate_proj`, activation, `down_proj` pattern |
| `top.MoERouter` | router logits and top-k expert selection |
| `top.MoEDispatch` | token-to-expert dispatch |
| `top.MoECombine` | expert output combine |

### TOP Side Effects

Most TOP ops should remain pure tensor transforms. KV-cache ops are the major
exception because they model state that persists across decode invocations.

| Op | Memory Effects | Notes |
|---|---|---|
| `top.KVCacheLoad` | `MemRead` on `KVCacheResource` | Reads external cache state and must not be hoisted across stores to the same cache slot |
| `top.KVCacheStore` | `MemWrite` on `KVCacheResource` | Updates external cache state and must not be dead-code eliminated when results are unused |

The cache handle operand should participate in alias analysis. At minimum, each
cache object should have a stable symbol or resource ID. Optimizations may
reorder loads/stores only when they can prove the cache object and token range do
not alias.

### TOP Attributes

Use attributes to retain LLM information through canonicalization and fusion.

| Attribute | Example Values | Applies To |
|---|---|---|
| `phase` | `prefill`, `decode` | attention, blocks |
| `num_heads` | `32` | attention, QKV |
| `num_kv_heads` | `8` | GQA/MQA attention |
| `head_dim` | `128` | attention, RoPE |
| `rope_mode` | `interleaved_pairs`, `contiguous_halves` | RoPE, attention |
| `causal` | `true`, `false` | attention |
| `quant_scheme` | `W4A8KV4`, `W8A8`, `A16W8` | matmul, attention, MLP |
| `weight_format` | `Q4_0`, `Q4_1`, `Q8_0`, `Q8_1`, `Q4_K` | weight users |
| `kv_format` | `KV4`, `KV8`, `FP8` | KV-cache ops |
| `group_size` | `64`, `128`, `256` | quantized tensors |
| `expert_topk` | `1`, `2`, `4` | MoE router |

`phase` can be attached to individual attention ops for local rewrites, but the
preferred representation is a function or region attribute when the whole entry
point has a fixed execution mode:

```mlir
func.func @decode_step(...) attributes {top.phase = "decode"} {
  ...
}
```

For mixed graphs, such as a single module containing both prefill and decode
entry points, use function-level attributes and allow op-level attributes only as
an override.

### TOP Example

```mlir
%q, %k, %v = top.QKVProj %x, %wqkv {
  num_heads = 32,
  num_kv_heads = 8,
  head_dim = 128,
  quant_scheme = "W4A8"
} : tensor<1x4096xf16>, tensor<4096x6144xi4>
  -> tensor<1x32x128xf16>, tensor<1x8x128xf16>, tensor<1x8x128xf16>

%k_old, %v_old = top.KVCacheLoad %cache, %pos {
  kv_format = "KV4",
  group_size = 64
} : !top.kv_cache<"KV4", group_size = 64>, index
  -> tensor<?x8x128xi4>, tensor<?x8x128xi4>

%y = top.DecodeAttention %q, %k, %v, %k_old, %v_old {
  causal = true,
  rope_mode = "interleaved_pairs",
  num_heads = 32,
  num_kv_heads = 8,
  kv_format = "KV4"
} : tensor<1x32x128xf16>, tensor<1x8x128xf16>, tensor<1x8x128xf16>,
    tensor<?x8x128xi4>, tensor<?x8x128xi4>
  -> tensor<1x4096xf16>

top.KVCacheStore %cache, %pos, %k, %v {
  kv_format = "KV4",
  group_size = 64
} : !top.kv_cache<"KV4", group_size = 64>, index,
    tensor<1x8x128xf16>, tensor<1x8x128xf16>
```

## TPU Dialect

The existing `tpu` dialect should continue to represent lowered TPU operations
such as `tpu.MatMul`, `tpu.Softmax`, and `tpu.RMSNorm`. ADA300-specific lowering
adds executable TPU ops for tensor core, vector, DMA, SRAM, quantization, and
synchronization semantics. These ops should still avoid raw final instruction
encodings where possible.

The examples below use concise names such as `tpu.gmm`. During implementation,
the TableGen op classes should carry an explicit ADA300 prefix, for example
`Tpu_ADA300GmmOp`, so target-specific ops remain easy to identify next to
generic TPU ops.

### TPU Type And Memory Model

Recommended memory-space attributes or types should use enum attributes instead
of stringly typed names:

```mlir
!tpu.sram<#tpu.sram_kind<asram>, side = #tpu.pp<ping>>
!tpu.sram<#tpu.sram_kind<asram>, side = #tpu.pp<pong>>
!tpu.sram<#tpu.sram_kind<wsram>, side = #tpu.pp<ping>>
!tpu.sram<#tpu.sram_kind<wsram>, side = #tpu.pp<pong>>
!tpu.sram<#tpu.sram_kind<rsram>>
!tpu.vreg
!tpu.global
```

The recommended implementation is to reuse `memref` where possible:

```mlir
memref<16x32xi8, #tpu.memory_space<asram, ping>>
memref<16x64xi4, #tpu.memory_space<wsram, pong>>
memref<16x32xi16, #tpu.memory_space<rsram>>
```

ADA300 has ASRAM ping/pong, WSRAM ping/pong, and RSRAM. Do not model the
ping/pong side twice as both a bank number and a side attribute. Add a separate
bank attribute only if a later controller specification exposes an independent
bank-selection dimension.

Custom `!tpu.sram` types are useful only if the backend needs stronger tile
semantics than `memref` can express. If custom types are used, provide explicit
conversion to and from `memref` before bufferization and backend emission.

Recommended tensor tile type metadata:

```mlir
!tpu.tile<16x16xbf16>
!tpu.tile<16x32xi8>
!tpu.tile<16x64xi4>
```

### TPU Tensor Core Ops

These ops map directly to the ADA300 tensor instruction family.

| TPU Op | ISA Form | Meaning |
|---|---|---|
| `tpu.gmm` | `gmm.mm` | `C = A * W`, output to SRAM |
| `tpu.gmma` | `gmma.mm` | `C += A * W`, output to SRAM |
| `tpu.gmma_trans` | `gmma.mt` | `C += A * W^T`, output to SRAM |
| `tpu.gmv` | `gmv.mm` | `C = A * W`, output to vector register |
| `tpu.gmva` | `gmva.mm` | `C += A * W`, output to vector register |
| `tpu.gmva_trans` | `gmva.mt` | `C += A * W^T`, output to vector register |

ADA300 also supports a tensor dot-product mode. Model it as a closed `mode`
enum on `tpu.tc_config`; do not invent a separate ISA mnemonic unless the final
controller specification defines one.

Prefer an explicit use-def link for tensor core configuration. This avoids
ambiguous dominance rules when two tensor ops with different configs appear in
the same block.

```mlir
%cfg = tpu.tc_config {
  target = "gmm",
  output = "sram",
  mode = "inner",
  row_size = 16,
  col_size = 32,
  acc_size = 4096,
  blk_cnt_a = 4,
  blk_cnt_w = 4,
  act_type = "int8",
  weight_type = "int8",
  out_type = "int16"
}

%c = tpu.gmma %cfg, %a, %w, %c_init {
  transpose_w = false
} : !tpu.tc_config,
    memref<16x32xi8, #tpu.memory_space<asram, ping>>,
    memref<16x32xi8, #tpu.memory_space<wsram, ping>>,
    memref<16x32xi16, #tpu.memory_space<rsram>>
  -> memref<16x32xi16, #tpu.memory_space<rsram>>
```

An attached config attribute is acceptable for final instruction-selection IR,
but the scheduling IR should use the SSA config value because it gives
verifiers, CSE, and rewrites a precise dependency.

### TPU Vector Ops

Standard vector arithmetic:

| TPU Op | ISA Form | Cycles |
|---|---|---|
| `tpu.vfadd` | `vfadd.vv` | 3 |
| `tpu.vfsub` | `vfsub.vv` | 3 |
| `tpu.vfmul` | `vfmul.vv` | 3 |
| `tpu.vfmin` | `vfmin.vv` | 3 |
| `tpu.vfmax` | `vfmax.vv` | 3 |
| `tpu.vslideup` | `vslideup.vs` | 3 |
| `tpu.vslidedown` | `vslidedown.vs` | 3 |

Custom vector ops:

| TPU Op | ISA Form | Cycles |
|---|---|---|
| `tpu.vfpwnl` | `vfpwnl.*` | 8 |
| `tpu.vfmul_cvt` | `vfmul.{low/high}.*.*.*.vv` | 3 |
| `tpu.vfcvt` | `vfcvt.*` | 3 |
| `tpu.vrepeat` | `vrepeat.vs` | 3 |
| `tpu.vpick` | `vpick.vs` | 1 |
| `tpu.vset` | `vset.ss` | 3 |
| `tpu.vfbredsum` | `vfbredsum.v` | 6 |
| `tpu.vfbredmin` | `vfbredmin.v` | 3 |
| `tpu.vfbredmax` | `vfbredmax.v` | 3 |

PWNL should use semantic attributes:

```mlir
%rsqrt = tpu.vfpwnl %x {
  kind = "rsqrt",
  segments = 16,
  input_type = "f32",
  output_type = "f32"
} : !tpu.vreg -> !tpu.vreg
```

The native `tpu.vfpwnl` kinds are closed and should match the ADA300 ISA
3-bit type field:

| Kind | Meaning |
|---|---|
| `exp` | exponential |
| `sqrt` | square root |
| `rsqrt` | reciprocal square root |
| `sigmoid` | sigmoid |
| `log` | natural logarithm |
| `sin` | sine |
| `cos` | cosine |
| `reciprocal` | `1 / x` |

Higher-level functions such as `silu`, `softplus`, `softmax`, RMS, and RMSNorm
must lower into native PWNL and vector ops. Centralize this mapping in enums
instead of letting ad-hoc strings leak into passes. The hardware architecture
document describes 16- and 31-segment approximations; the final controller
specification must define how the segment choice is encoded.

Precision conversion should keep width-splitting explicit:

```mlir
%lo = tpu.vfcvt %src {
  half = "low",
  src_type = "f16",
  dst_type = "f32"
} : !tpu.vreg -> !tpu.vreg

%hi = tpu.vfcvt %src {
  half = "high",
  src_type = "f16",
  dst_type = "f32"
} : !tpu.vreg -> !tpu.vreg
```

### TPU Memory And Sync Ops

| Op | Purpose |
|---|---|
| `tpu.sram_alloc` | Allocate logical SRAM region |
| `tpu.sram_view` | Tile/view into SRAM allocation |
| `tpu.dma_load` | Move global/LPDDR/HBF/NAND data into SRAM; carry an `engine` attribute |
| `tpu.dma_store` | Store SRAM result to global memory; carry an `engine` attribute |
| `tpu.vload` | Load SRAM data into vector register |
| `tpu.vstore` | Store vector register to SRAM |
| `tpu.sync_tc` | Wait for tensor core completion |
| `tpu.sync_vector` | Wait for vector pipeline completion |
| `tpu.sync_result`, `tpu.sync_act_ping`, `tpu.sync_act_pong` | Wait for result or activation SRAM visibility |
| `tpu.sync_wht_ping`, `tpu.sync_wht_pong`, `tpu.sync_sram_all` | Wait for weight SRAM side or all SRAM resources |
| `tpu.sync_odma0`, `tpu.sync_odma1` | Wait for DMA engines |
| `tpu.sync_all` | Full barrier |

ADA300 DMA engine access rules:

| Engine | Legal SRAM resources |
|---|---|
| `odma0` | WSRAM ping/pong and RSRAM |
| `odma1` | ASRAM ping/pong and RSRAM |
| `vlsu` | WSRAM ping/pong, ASRAM ping/pong, and RSRAM |

Ping-pong verification rule:

```text
When tensor reads one side of ASRAM/WSRAM, ODMA/VLSU must access the other side.
RESULT SRAM does not have the same ping-pong restriction.
```

Scheduling IR should use SSA completion tokens where practical. Sync ops remain
explicit because they map to scalar ISA barriers, but token dependencies make
DMA, tensor-core, and vector ordering verifiable before instruction selection.

### TPU Quantization Ops

Quantization should be visible at TPU level because it affects memory traffic,
tile shape, and instruction choice.

| Op | Purpose |
|---|---|
| `tpu.quant_act` | FP16/BF16 activation to INT8/INT16 |
| `tpu.dequant_weight` | INT4 weight to INT8 or FP16/BF16 |
| `tpu.dequant_kv` | KV4/KV8 cache expansion |
| `tpu.requant` | accumulator/result requantization |
| `tpu.pack` | pack low-bit tensor representation |
| `tpu.unpack` | unpack low-bit tensor representation |

Represent ordinary block quantization and K-quant separately. Ordinary formats
such as `Q4_0`, `Q4_1`, `Q8_0`, and `Q8_1` commonly use 64-element blocks.
K-quant formats such as `Q2_K` through `Q6_K` use a 256-element super-block
with lower-precision sub-block scale and min metadata. A single `group_size`
attribute is not sufficient to describe both families.

Example:

```mlir
%a_i8 = tpu.quant_act %a_f16 {
  src_type = "f16",
  dst_type = "int8",
  group_size = 64
}

%w_i8 = tpu.dequant_weight %w_q4 {
  src_format = "Q4_1",
  dst_type = "int8",
  group_size = 64
}
```

## Lowering Pipeline

### Full Pipeline

```text
Import
  -> top graph

TOP canonicalization
  -> shape inference
  -> constant folding
  -> QDQ cleanup

TOP LLM recognition
  -> recognize QKV projection
  -> recognize RMSNorm
  -> recognize RoPE
  -> recognize prefill/decode attention
  -> recognize GatedMLP
  -> recognize MoE router/dispatch/combine

TOP LLM optimization
  -> fuse QKV
  -> fuse RMSNorm + MatMul when profitable
  -> fuse activation patterns
  -> attach quant/cache attributes
  -> select prefill/decode strategy

TOP to TPU
  -> lower matmul/attention skeletons
  -> lower nonlinear ops to PWNL or vector arithmetic
  -> lower quant ops to vfcvt/dequant/requant
  -> choose tensor vs vector implementation
  -> choose gmm/gmv/gmma/gmva variants

TPU verification
  -> verify coarse TPU structural invariants

TPU scheduling
  -> tile tensors
  -> assign SRAM regions
  -> plan ping-pong buffers
  -> assign DMA engines and insert load/store ops
  -> insert sync ops and completion-token dependencies
  -> verify SRAM alignment and access conflicts

TPU lowering
  -> allocate vector/scalar registers
  -> materialize tensor CSR writes
  -> select exact ADA300 instructions
  -> encode opcode/funct fields

Backend
  -> assembly
  -> object/binary
```

### Pass Sketch

| Pass | Input | Output |
|---|---|---|
| `top-shape-infer` | imported `top` | shaped `top` |
| `top-llm-patterns` | primitive `top` | semantic LLM `top` |
| `top-quant-annotate` | `top` | `top` with quant/cache attrs |
| `top-to-tpu-matmul` | `top` matmul/QKV/MLP | coarse tensor `tpu` |
| `top-to-tpu-attention` | `top` attention/cache ops | coarse attention `tpu` |
| `top-to-tpu-nonlinear` | `top` nonlinear ops | vector/PWNL `tpu` |
| `top-to-tpu-quant` | `top` quant attrs/ops | explicit TPU quant/dequant |
| `tpu-verify-coarse` | coarse `tpu` | verified coarse `tpu` |
| `tpu-ada300-tile` | coarse `tpu` | tiled `tpu` |
| `tpu-verify-tiled` | tiled `tpu` | verified tiled `tpu` |
| `tpu-ada300-memory-plan` | tiled `tpu` | SRAM-assigned `tpu` |
| `tpu-verify-memory` | SRAM-assigned `tpu` | verified memory placement |
| `tpu-ada300-insert-dma` | SRAM-assigned `tpu` | DMA-explicit `tpu` |
| `tpu-ada300-insert-sync` | DMA-explicit `tpu` | synchronized `tpu` |
| `tpu-verify-scheduled` | synchronized `tpu` | verified scheduled `tpu` |
| `tpu-ada300-isel` | synchronized `tpu` | instruction-level `tpu` or LLVM |

The pass manager may still expose a convenience `convert-top-to-tpu` pipeline,
but internally it should be composed from the smaller passes above so each class
of lowering has focused tests and failure modes.

## Control Flow And Dynamic Shapes

LLM inference needs explicit loop structure for variable sequence length, batch
size, and continuous batching. Recommended representation:

1. Keep high-level decode and prefill entry points as separate `func.func`s when
   possible. Attach `top.phase = "prefill"` or `top.phase = "decode"`.
2. Represent token loops with `scf.for` or `scf.while` before TPU scheduling.
3. Keep dynamic sequence length and active batch metadata as SSA values, not
   hidden global state.
4. Lower loop-carried KV-cache state through explicit cache handles plus
   `top.KVCacheLoad/Store` memory effects.
5. During TPU scheduling, tile loops and DMA loops should be made explicit before
   SRAM allocation, so memory planning can see live ranges.

Prefill and decode should diverge before final TPU scheduling:

| Phase | Preferred Codegen Strategy |
|---|---|
| prefill | larger GEMM-oriented tiles, SRAM result reuse, batched Q/K/V materialization |
| decode | GEMV-oriented tiles, KV-cache streaming, vector-register postprocessing |

## Migration Strategy

The current `tpu` dialect already contains generic lowered ops such as
`tpu.MatMul`, `tpu.Softmax`, and `tpu.ReduceSum`. The migration should be staged:

1. Keep existing generic `tpu` ops as compatibility ops.
2. Add ADA300 executable ops next to them.
3. Introduce canonical lowering patterns from generic `tpu` ops to executable
   `tpu` ops.
4. Move ADA300-specific backend code to consume executable ops first, falling
   back to generic ops during transition.
5. Once coverage is complete, reject generic ops late in the ADA300 pipeline.

Example migration path:

```text
top.MatMul
  -> tpu.MatMul                 # compatibility path
  -> tpu.tc_config + tpu.gmm    # executable path
```

## Testing Strategy

Use tests at every IR contract boundary:

| Test Type | What To Check |
|---|---|
| `lit` parser tests | New attrs/types print and parse, including SRAM enums and PWNL kinds |
| verifier tests | Invalid config scope, invalid SRAM ping-pong conflict, unsupported PWNL kind |
| FileCheck lowering tests | TOP LLM patterns lower to expected TPU ops |
| scheduling tests | DMA and sync insertion around tensor/vector dependencies |
| encoding tests | Final instruction selection emits expected opcode/funct fields |
| integration tests | Small RMSNorm, softmax, decode attention, and quantized matmul kernels |

## Example Lowerings

### RMSNorm

TOP:

```mlir
%y = top.RMSNorm %x, %weight {
  epsilon = 1.0e-6
}
```

TPU:

```text
tpu.vfmul        # x * x
tpu.vfbredsum    # sum(x * x)
tpu.vfmul        # multiply by 1 / hidden_size
tpu.vfadd        # add epsilon
tpu.vfpwnl       # rsqrt
tpu.vfmul        # x * rsqrt(...)
tpu.vfmul        # apply weight
```

### Decode Attention

TOP:

```mlir
%y = top.DecodeAttention %q, %k, %v, %k_cache, %v_cache {
  causal = true,
  num_heads = 32,
  num_kv_heads = 8,
  kv_format = "KV4"
}
```

TPU:

```text
tpu.dma_load / tpu.dequant_kv
tpu.tc_config target=gmv mode=inner
tpu.gmva_trans       # Q * K^T -> VR
tpu.vfmul            # scale by 1/sqrt(head_dim)
tpu.vfbredmax
tpu.vfsub
tpu.vfpwnl kind=exp
tpu.vfbredsum
tpu.vfpwnl kind=reciprocal
tpu.vfmul            # softmax probabilities
tpu.tc_config target=gmv mode=inner
tpu.gmva             # P * V
tpu.sync_tc
tpu.sync_vector
```

### Quantized MatMul

TOP:

```mlir
%y = top.MatMul %x, %w {
  quant_scheme = "W4A8",
  weight_format = "Q4_1",
  group_size = 64
}
```

TPU:

```text
tpu.dma_load weight_q4 -> WSRAM pong
tpu.dequant_weight Q4_1 -> INT8
tpu.quant_act F16/BF16 -> INT8
tpu.tc_config act_type=int8 weight_type=int8 out_type=int16
tpu.gmm/gmma
tpu.sync_tc
tpu.requant or tpu.vfcvt
tpu.dma_store
```

## Verification Rules

TPU verifier should check:

1. Tensor ops consume an explicit compatible `tpu.tc_config` SSA value.
2. `gmm/gmma` output to SRAM; `gmv/gmva` output to vector register.
3. `.mt` variants are used only when the logical operation is `A * W^T`.
4. `row_size`, `col_size`, and data type agree with ADA300 limits.
5. `blk_cnt_a == blk_cnt_w` for inner-product tensor mode.
6. SRAM alignment constraints are checked by SRAM kind, engine, and access
   direction.
7. `odma0`, `odma1`, and `vlsu` access only their legal SRAM resources.
8. Ping-pong SRAM conflicts are rejected.
9. Vector low/high conversion variants are paired when full-width conversion is
   required.
10. `vfpwnl` kind is one of the eight native ISA functions. Segment count is 16
    or 31 when selected explicitly, pending final controller encoding details.
11. Sync ops or their completion tokens dominate consumers that depend on tensor,
    DMA, or vector results.

## Summary

The recommended direction is not to replace the existing `top` and `tpu`
dialects. Instead:

```text
top: keep and extend as the LLM semantic dialect
tpu: make more ADA300-executable
backend: handle exact instruction encoding
```

This keeps model-level optimization clean while giving the ADA300 backend enough
structure to schedule tensor cores, vector custom instructions, SRAM ping-pong,
DMA, quantization, and synchronization correctly.
