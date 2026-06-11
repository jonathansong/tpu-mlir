/* Tiny LLM Ada300 minimal inference demo for rx-qemu.
 *
 * Implements a complete prefill + decode loop using 5 compiled subgraphs,
 * mirroring the structure of Qwen3_5::forward_first + forward_next in
 * the LLM-TPU/models/Qwen3_5/cpp_demo/chat.cpp reference implementation.
 *
 * Subgraph signatures (after tpuc-opt 8-step pipeline, mode=F16):
 *
 *   embedding         : (1, SEQ)        si32  → (1, SEQ, DIM)          f16
 *   embedding_cache   : (1,   1)        si32  → (1,   1, DIM)          f16
 *   block_7           : (1,SEQ,DIM) f16,          [prefill FA layer]
 *                       (3,SEQ)     si32,
 *                       (1,1,SEQ,SEQ) f16
 *                       → (1,SEQ,DIM) f16,
 *                         (1,SEQ,NHEADS,HDIM) f16 × 2  (k_cache, v_cache)
 *   block_cache_7     : (1,1,DIM)          f16,  [decode FA layer]
 *                       (3,1)              si32,
 *                       (1,1,1,KV_HIST+1)  f16,
 *                       (1,KV_HIST,NHEADS,HDIM) f16 × 2  (hist_k, hist_v)
 *                       → (1,1,DIM)              f16,
 *                         (1,1,NHEADS,HDIM)      f16 × 2  (new_k, new_v)
 *   lm_head           : (1, DIM)        f16   → (1, 1) si32  [TopK index]
 *
 * Fixed model dimensions (from compiled .mlir files):
 *   SEQ     = 1024        prefill sequence length  (block_7.mlir input dim)
 *   DIM     = 2048        hidden size
 *   KV_HIST = 2048        max KV history slots     (block_cache_7.mlir)
 *   NHEADS  = 2           KV heads
 *   HDIM    = 256         KV head dimension
 *   KV_STEP = 512         NHEADS * HDIM  (elements per history slot)
 *   VOCAB   = 248320      vocabulary size
 *
 * Attention masks (f16, MASK_NEG_INF = 0xF0E2 ≈ -10004.0):
 *   Prefill: causal lower-triangular  (1, 1, SEQ, SEQ)
 *   Decode:  history-aware            (1, 1, 1, KV_HIST+1)
 *             positions [0..history-1]   → 0          (visible past tokens)
 *             positions [history..KV_HIST-1] → neg-inf  (padding)
 *             position  [KV_HIST]        → 0          (new token's own KV)
 *
 * KV cache layout (per-layer):
 *   kv_k / kv_v : uint16_t[KV_HIST * KV_STEP]  (flat, row-major)
 *   Prefill writes slots  [0 .. SEQ-1]  from block_7 output.
 *   Each decode step appends one slot at [history] and increments history.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mr_heap.h"

void *malloc(size_t n) { return mr_alloc(n, MR_REGION_LPDDR); }
void free(void *p) { mr_free(p); }
void *calloc(size_t nmemb, size_t size) { return mr_calloc(nmemb, size, MR_REGION_LPDDR); }
void *rxnn_mem_alloc(int64_t size) { return mr_alloc((size_t)size, MR_REGION_LPDDR); }
void rxnn_mem_free(void *ptr) { mr_free(ptr); }

void mr_smp_secondary_entry(unsigned long hartid) {
  (void)hartid;
  for (;;) { __asm__ volatile("wfi"); }
}

/* ---------- memrefCopy (called from generated LLVM IR) ---------------- */
typedef struct { int64_t rank; void *descriptor; } UnrankedMemRef;

void memrefCopy(int64_t elem_size, UnrankedMemRef *src_u, UnrankedMemRef *dst_u) {
  if (!src_u || !dst_u || !src_u->descriptor || !dst_u->descriptor) return;
  char *src_desc = (char *)src_u->descriptor;
  char *dst_desc = (char *)dst_u->descriptor;
  void *src_data = *(void **)(src_desc + sizeof(void *));
  void *dst_data = *(void **)(dst_desc + sizeof(void *));
  int64_t *sizes = (int64_t *)(src_desc + sizeof(void *) * 2 + sizeof(int64_t));
  int64_t elems = 1;
  for (int64_t i = 0; i < src_u->rank; ++i) elems *= sizes[i];
  memcpy(dst_data, src_data, (size_t)(elems * elem_size));
}

/* ---------- Model dimensions ------------------------------------------ */
#define SEQ       1024   /* prefill sequence length (fixed by block_7.mlir) */
#define DIM       2048   /* hidden size                                      */
#define KV_HIST   2048   /* max KV history (block_cache_7.mlir past_k dim1) */
#define NHEADS       2   /* KV heads                                         */
#define HDIM       256   /* KV head dimension                                */
#define KV_STEP    512   /* NHEADS * HDIM; elements per history slot         */
#define VOCAB   248320   /* vocabulary (lm_head weight rows)                 */

/* float16 large-negative for attention masking (0xF0E2 ≈ -10004.0 in f16) */
#define MASK_NEG_INF ((uint16_t)0xF0E2)

#define MAX_NEW_TOKENS  4        /* decode steps to run (keep small for QEMU) */
#define EOS_TOKEN  151645        /* Qwen3 <|im_end|> token id                 */

/* ---------- Ranked memref descriptor types ---------------------------- */
typedef struct {
  void    *allocated;
  void    *aligned;
  int64_t  offset;
  int64_t  sizes[2];
  int64_t  strides[2];
} MemRef2D;

typedef struct {
  void    *allocated;
  void    *aligned;
  int64_t  offset;
  int64_t  sizes[3];
  int64_t  strides[3];
} MemRef3D;

typedef struct {
  void    *allocated;
  void    *aligned;
  int64_t  offset;
  int64_t  sizes[4];
  int64_t  strides[4];
} MemRef4D;

/* ---------- Result structs -------------------------------------------- */
typedef struct { MemRef3D out_states; MemRef4D k_cache; MemRef4D v_cache; } Block7Result;
typedef struct { MemRef3D out_states; MemRef4D new_k;   MemRef4D new_v;   } BlockCache7Result;

/* ---------- Descriptor helpers ---------------------------------------- */
static MemRef2D make2d(void *data, int64_t d0, int64_t d1) {
  MemRef2D m;
  m.allocated = data; m.aligned = data; m.offset = 0;
  m.sizes[0] = d0;   m.sizes[1] = d1;
  m.strides[0] = d1; m.strides[1] = 1;
  return m;
}
static MemRef3D make3d(void *data, int64_t d0, int64_t d1, int64_t d2) {
  MemRef3D m;
  m.allocated = data; m.aligned = data; m.offset = 0;
  m.sizes[0] = d0;      m.sizes[1] = d1;      m.sizes[2] = d2;
  m.strides[0] = d1*d2; m.strides[1] = d2;    m.strides[2] = 1;
  return m;
}
static MemRef4D make4d(void *data, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  MemRef4D m;
  m.allocated = data; m.aligned = data; m.offset = 0;
  m.sizes[0] = d0;          m.sizes[1] = d1;      m.sizes[2] = d2;      m.sizes[3] = d3;
  m.strides[0] = d1*d2*d3;  m.strides[1] = d2*d3; m.strides[2] = d3;    m.strides[3] = 1;
  return m;
}

/* ---------- Subgraph extern declarations ------------------------------ *
 * Signatures derived from compiled LLVM IR (8-step tpuc-opt pipeline,   *
 * mode=F16).  All floating-point tensors become f16 (uint16_t data);     *
 * integer tensors stay si32.  Result structs are passed as first arg.    *
 * ----------------------------------------------------------------------- */

/* (1,SEQ) si32 → (1,SEQ,DIM) f16 */
extern void _mlir_ciface_embedding_main(MemRef3D *result,
                                        MemRef2D *token_ids);

/* (1,1) si32 → (1,1,DIM) f16 */
extern void _mlir_ciface_embedding_cache_main(MemRef3D *result,
                                              MemRef2D *token_id);

/* (1,SEQ,DIM) f16, (3,SEQ) si32, (1,1,SEQ,SEQ) f16
 *   → { (1,SEQ,DIM) f16,  (1,SEQ,NHEADS,HDIM) f16 × 2 } */
extern void _mlir_ciface_block7_main(Block7Result *result,
                                     MemRef3D     *in_states,
                                     MemRef2D     *pos_ids,
                                     MemRef4D     *attn_mask);

/* (1,1,DIM) f16, (3,1) si32, (1,1,1,KV_HIST+1) f16,
 *  (1,KV_HIST,NHEADS,HDIM) f16 × 2
 *   → { (1,1,DIM) f16,  (1,1,NHEADS,HDIM) f16 × 2 } */
extern void _mlir_ciface_block_cache7_main(BlockCache7Result *result,
                                           MemRef3D          *in_states,
                                           MemRef2D          *pos_ids,
                                           MemRef4D          *attn_mask,
                                           MemRef4D          *hist_k,
                                           MemRef4D          *hist_v);

/* (1,DIM) f16 → (1,1) si32   (TopK-1 returns token index as si32) */
extern void _mlir_ciface_lm_head_main(MemRef2D *result,
                                      MemRef2D *hidden);

/* ---------- main ------------------------------------------------------- */
int main(void) {
  printf("[TinyLLM] Ada300 Qwen3.5-2B-int4 minimal inference demo\n");
  printf("[TinyLLM] SEQ=%d DIM=%d KV_HIST=%d MAX_NEW=%d\n",
         SEQ, DIM, KV_HIST, MAX_NEW_TOKENS);

  /* ==================================================================== *
   * Persistent KV cache — (1, KV_HIST, NHEADS, HDIM) f16, flat array.   *
   * Zero-init via calloc; prefill fills [0..SEQ-1], decode appends.      *
   * Analogous to past_key[] / past_value[] in Qwen3_5::init.             *
   * ==================================================================== */
  uint16_t *kv_k = (uint16_t *)calloc((size_t)(KV_HIST * KV_STEP), sizeof(uint16_t));
  uint16_t *kv_v = (uint16_t *)calloc((size_t)(KV_HIST * KV_STEP), sizeof(uint16_t));
  if (!kv_k || !kv_v) {
    printf("[TinyLLM] ERROR: KV cache alloc failed\n");
    return 1;
  }
  int history = 0;   /* next free KV slot; updated after prefill and each decode step */

  /* ==================================================================== *
   * PREFILL — analogous to Qwen3_5::forward_first                        *
   * ==================================================================== */
  printf("[TinyLLM] --- PREFILL (SEQ=%d) ---\n", SEQ);

  /* 1. Token IDs: (1, SEQ) si32.
   *    Use a simple 1-based pattern (non-zero) as a smoke-test prompt.
   *    Replace with real tokenized input for genuine inference. */
  int32_t *token_ids = (int32_t *)calloc((size_t)(SEQ), sizeof(int32_t));
  if (!token_ids) { printf("[TinyLLM] ERROR: token_ids alloc failed\n"); return 1; }
  for (int i = 0; i < SEQ; i++)
    token_ids[i] = (int32_t)(i % 100 + 1);

  MemRef2D emb_in  = make2d(token_ids, 1, SEQ);
  MemRef3D emb_out;                               /* (1, SEQ, DIM) f16 — malloc'd by callee */
  _mlir_ciface_embedding_main(&emb_out, &emb_in);
  free(token_ids);
  printf("[TinyLLM] embedding ok\n");

  /* 2. Causal attention mask: (1, 1, SEQ, SEQ) f16.
   *    mask[i][j] = 0 if j <= i  (token i can attend to j),
   *               = MASK_NEG_INF otherwise.
   *    Analogous to the attention_mask in Qwen3_5::forward_first. */
  uint16_t *causal_mask = (uint16_t *)calloc((size_t)(SEQ * SEQ), sizeof(uint16_t));
  if (!causal_mask) { printf("[TinyLLM] ERROR: causal_mask alloc failed\n"); return 1; }
  for (int i = 0; i < SEQ; i++)
    for (int j = 0; j < SEQ; j++)
      causal_mask[i * SEQ + j] = (j <= i) ? (uint16_t)0 : MASK_NEG_INF;
  MemRef4D mask4 = make4d(causal_mask, 1, 1, SEQ, SEQ);

  /* 3. Position IDs: (3, SEQ) si32.
   *    3D RoPE — all three dimensions use the same 1-D position sequence
   *    for plain text.  Analogous to get_position_ids() in pipeline.cpp. */
  int32_t *pos_ids = (int32_t *)calloc((size_t)(3 * SEQ), sizeof(int32_t));
  if (!pos_ids) { printf("[TinyLLM] ERROR: pos_ids alloc failed\n"); return 1; }
  for (int d = 0; d < 3; d++)
    for (int i = 0; i < SEQ; i++)
      pos_ids[d * SEQ + i] = (int32_t)i;
  MemRef2D pos2d = make2d(pos_ids, 3, SEQ);

  /* 4. block_7 — the single Full Attention prefill layer. */
  Block7Result b7_res;
  _mlir_ciface_block7_main(&b7_res, &emb_out, &pos2d, &mask4);
  free(emb_out.allocated);
  free(causal_mask);
  free(pos_ids);
  printf("[TinyLLM] block_7 (prefill) ok\n");

  /* 5. Copy prefill KV output into persistent cache, slots [0..SEQ-1].
   *    Analogous to bm_memcpy_d2d_byte(past_key[i], 0, output_mem, 0, ...)
   *    in Qwen3_5::forward_first. */
  memcpy(kv_k, b7_res.k_cache.aligned, (size_t)(SEQ * KV_STEP) * sizeof(uint16_t));
  memcpy(kv_v, b7_res.v_cache.aligned, (size_t)(SEQ * KV_STEP) * sizeof(uint16_t));
  history = SEQ;
  free(b7_res.k_cache.allocated);
  free(b7_res.v_cache.allocated);

  /* 6. lm_head on the last prefill token to get the first generated token.
   *    Slice the last row of out_states: offset = (SEQ-1)*DIM elements.
   *    Analogous to the in_tensors[0].device_mem = bm_mem_from_device(
   *    out_mem.addr + (token_length-1)*bytes, bytes) in forward_first. */
  uint16_t *last_hidden = (uint16_t *)b7_res.out_states.aligned
                          + (int64_t)(SEQ - 1) * DIM;
  MemRef2D lmh_in  = make2d(last_hidden, 1, DIM);
  MemRef2D lmh_out;
  _mlir_ciface_lm_head_main(&lmh_out, &lmh_in);
  free(b7_res.out_states.allocated);

  /* TopK returns the index as si32 (matches Qwen3_5 lmhead_with_topk path). */
  int32_t token = *(int32_t *)lmh_out.aligned;
  free(lmh_out.allocated);
  printf("[TinyLLM] prefill done — first token=%d (history=%d)\n",
         (int)token, history);

  /* ==================================================================== *
   * DECODE LOOP — analogous to Qwen3_5::forward_next                     *
   * Each iteration:                                                        *
   *   embed one token → block_cache_7 (reads full KV, writes new slot)   *
   *   → lm_head → next token                                              *
   * ==================================================================== */
  printf("[TinyLLM] --- DECODE (max %d tokens) ---\n", MAX_NEW_TOKENS);

  for (int step = 0; step < MAX_NEW_TOKENS; step++) {
    if (token == EOS_TOKEN) {
      printf("[TinyLLM] EOS at step %d\n", step);
      break;
    }
    if (history >= KV_HIST) {
      printf("[TinyLLM] KV cache full at step %d\n", step);
      break;
    }

    /* 7. Single-token embedding: (1,1) si32 → (1,1,DIM) f16.
     *    Analogous to embedding_cache in Qwen3_5::forward_next. */
    int32_t single = token;
    MemRef2D ec_in  = make2d(&single, 1, 1);
    MemRef3D ec_out;
    _mlir_ciface_embedding_cache_main(&ec_out, &ec_in);

    /* 8. Decode attention mask: (1, 1, 1, KV_HIST+1) f16.
     *    positions [0 .. history-1]   → 0           (visible past KV slots)
     *    positions [history .. KV_HIST-1] → neg-inf  (padding / unused slots)
     *    position  [KV_HIST]          → 0           (new token's own KV in concat)
     *    Analogous to the attention_mask in Qwen3_5::forward_next. */
    uint16_t *dec_mask = (uint16_t *)calloc((size_t)(KV_HIST + 1), sizeof(uint16_t));
    if (!dec_mask) { printf("[TinyLLM] ERROR: dec_mask alloc failed\n"); break; }
    for (int i = history; i < KV_HIST; i++)
      dec_mask[i] = MASK_NEG_INF;
    /* dec_mask[KV_HIST] stays 0: new token attends to its own KV in concat */
    MemRef4D dec_mask4 = make4d(dec_mask, 1, 1, 1, KV_HIST + 1);

    /* 9. Decode position IDs: (3, 1) si32 — all dims = history.
     *    Analogous to following_position_ids = {max_posid, max_posid, max_posid}
     *    in ChatPipe::run_once. */
    int32_t dec_pos[3] = { (int32_t)history, (int32_t)history, (int32_t)history };
    MemRef2D dec_pos2  = make2d(dec_pos, 3, 1);

    /* 10. Full KV cache descriptors for block_cache_7.
     *     Analogous to in_tensors[3/4] = past_key/value buffers in
     *     Qwen3_5::net_launch_decode. */
    MemRef4D full_kv_k = make4d(kv_k, 1, KV_HIST, NHEADS, HDIM);
    MemRef4D full_kv_v = make4d(kv_v, 1, KV_HIST, NHEADS, HDIM);

    /* 11. block_cache_7 — single FA decode step. */
    BlockCache7Result bc_res;
    _mlir_ciface_block_cache7_main(&bc_res, &ec_out, &dec_pos2, &dec_mask4,
                                   &full_kv_k, &full_kv_v);
    free(ec_out.allocated);
    free(dec_mask);

    /* 12. Append new KV to cache at position `history`.
     *     Analogous to bm_mem_from_device(past_key[i].addr + kv_offset, KV_BYTES)
     *     / writing the output directly into the offset slot. */
    memcpy(kv_k + history * KV_STEP, bc_res.new_k.aligned,
           (size_t)KV_STEP * sizeof(uint16_t));
    memcpy(kv_v + history * KV_STEP, bc_res.new_v.aligned,
           (size_t)KV_STEP * sizeof(uint16_t));
    history++;
    free(bc_res.new_k.allocated);
    free(bc_res.new_v.allocated);

    /* 13. lm_head on the single decode hidden state → next token.
     *     bc_res.out_states is (1,1,DIM); treat as (1,DIM) by passing
     *     the same data pointer with a 2D descriptor. */
    uint16_t *dec_hidden = (uint16_t *)bc_res.out_states.aligned;
    MemRef2D dec_lmh_in  = make2d(dec_hidden, 1, DIM);
    MemRef2D dec_lmh_out;
    _mlir_ciface_lm_head_main(&dec_lmh_out, &dec_lmh_in);
    free(bc_res.out_states.allocated);

    token = *(int32_t *)dec_lmh_out.aligned;
    free(dec_lmh_out.allocated);

    printf("[TinyLLM] decode step %d: token=%d (history=%d)\n",
           step, (int)token, history);
  }

  /* ==================================================================== *
   * Cleanup                                                               *
   * ==================================================================== */
  free(kv_k);
  free(kv_v);
  printf("[TinyLLM] done (total history=%d)\n", history);

  /* Exit QEMU via RISC-V semihosting SYS_EXIT (0x18). */
  register long a0 asm("a0") = 0x18;     /* SYS_EXIT */
  register long a1 asm("a1") = 0x20026;  /* ADP_Stopped_ApplicationExit */
  __asm__ volatile(".option push\n"
                   ".option norvc\n"
                   "slli zero, zero, 0x1f\n"
                   "ebreak\n"
                   "srai zero, zero, 7\n"
                   ".option pop\n"
                   : : "r"(a0), "r"(a1));
  return 0;
}

