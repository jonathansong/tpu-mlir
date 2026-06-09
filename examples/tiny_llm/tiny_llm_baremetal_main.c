/* Tiny LLM Ada300 runner for rx-qemu. */
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
  for (;;) {
    __asm__ volatile("wfi");
  }
}

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

typedef struct {
  float *base_ptr;
  float *data_ptr;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} F32MemRef2D;

static F32MemRef2D make_f32_2d(float *data, int64_t d0, int64_t d1) {
  F32MemRef2D m;
  m.base_ptr = data;
  m.data_ptr = data;
  m.offset = 0;
  m.sizes[0] = d0;
  m.sizes[1] = d1;
  m.strides[0] = d1;
  m.strides[1] = 1;
  return m;
}

extern void _mlir_ciface_embedding_main(F32MemRef2D *result, F32MemRef2D *input);
extern void _mlir_ciface_block7_main(F32MemRef2D *result, F32MemRef2D *hidden, F32MemRef2D *weight);
extern void _mlir_ciface_lm_head_main(F32MemRef2D *result, F32MemRef2D *hidden, F32MemRef2D *weight);

static float g_input[8];
static float g_block_w[64];
static float g_lm_w[8];

static void init_inputs(void) {
  for (int i = 0; i < 8; ++i) {
    g_input[i] = (float)(i + 1);
    g_lm_w[i] = 1.0f;
  }
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c)
      g_block_w[r * 8 + c] = (r == c) ? 1.0f : 0.0f;
}

int main(void) {
  printf("[TinyLLM] Ada300 real Qwen pipeline\n");
  printf("[TinyLLM] graph: embedding -> block_7 -> lm_head -> token id\n");
  init_inputs();

  F32MemRef2D input = make_f32_2d(g_input, 1, 8);
  F32MemRef2D block_w = make_f32_2d(g_block_w, 8, 8);
  F32MemRef2D lm_w = make_f32_2d(g_lm_w, 8, 1);
  F32MemRef2D hidden0;
  F32MemRef2D hidden1;
  F32MemRef2D token;

  _mlir_ciface_embedding_main(&hidden0, &input);
  printf("[TinyLLM] embedding ok\n");
  _mlir_ciface_block7_main(&hidden1, &hidden0, &block_w);
  printf("[TinyLLM] block_7 ok\n");
  _mlir_ciface_lm_head_main(&token, &hidden1, &lm_w);
  printf("[TinyLLM] lm_head ok\n");
  int token_id = token.data_ptr ? (int)token.data_ptr[0] : -1;
  printf("[TinyLLM] token=%d\n", token_id);
  printf("[TinyLLM] done\n");

  free(hidden0.data_ptr);
  free(hidden1.data_ptr);
  free(token.data_ptr);

  /* Exit QEMU via RISC-V semihosting SYS_EXIT (0x18).
   * a0 = operation number, a1 = ADP_Stopped_ApplicationExit reason. */
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
