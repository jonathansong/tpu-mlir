/*
 * Baremetal tiny_llm intentionally uses the rx-ops Ada300 backend.  Keep the
 * reference-C backend init as a no-op so linking a small demo does not pull in
 * the full libc-heavy reference operator suite.
 */
void rxnn_c_init(void) {}

#include <stddef.h>
#include "rx_memory.h"
#include "rx_ops_data_type.h"

void *rxmem_malloc(enum rxmem_type type, size_t size) {
  (void)type;
  (void)size;
  return 0;
}

void rxmem_free(enum rxmem_type type, void *ptr) {
  (void)type;
  (void)ptr;
}

void ada300_matmul_tensor_f16(const __fp16_t *a, const __fp16_t *b,
                              __fp16_t *out, int *params) {
  (void)a;
  (void)b;
  (void)out;
  (void)params;
}
