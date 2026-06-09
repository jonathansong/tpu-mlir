/* Ada300 callback registration for tiny_llm — all LLM-relevant ops.
 *
 * Each entry wires a (op_enum, dtype_enum) pair to the corresponding Ada300
 * NPU kernel.  The bridge C file (rx_ops_bridge_ada300_compat.c) builds
 * rxops_tensor descriptors and calls the public rxops_<op>() API which then
 * dispatches through this table.
 */
#include <string.h>

#include "backend/ada300/rxnn_npu.h"
#include "internal/rx_ops_register.h"
#include "interface/rx_ops_data_structure.h"

static void *tiny_ada300_cb_map(int op, int dtype) {
  static struct rxops_callback cb_map[RXOPS_OP_SIZE][RXOPS_DTYPE_SIZE];
  static int initialized;
  if (!initialized) {
    memset(cb_map, 0, sizeof(cb_map));

    /* ── The real Qwen model runs entirely in f16 after TopToTpu lowering.
     *    Only register f16 callbacks to avoid pulling fp32 kernel sources
     *    (quant.c, fp32/*.c) and their transitive asm dependencies.
     *    The reshape/concat shape-copy primitives are dtype-agnostic and
     *    used by both the f32 and f16 kernel paths. ───────────────────── */

    /* Element-wise binary */
    cb_map[RXOPS_OP_ADD][RXOPS_DTYPE_FLOAT16].exec  = rxnn_npu_add_fp16;
    cb_map[RXOPS_OP_MUL][RXOPS_DTYPE_FLOAT16].exec  = rxnn_npu_mul_fp16;

    /* Matrix multiply / fully-connected */
    cb_map[RXOPS_OP_MATMUL][RXOPS_DTYPE_FLOAT16].exec        = rxnn_npu_matmul_fp16;
    cb_map[RXOPS_OP_FULLYCONNECTED][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_fullyconnected_fp16;

    /* Normalization */
    cb_map[RXOPS_OP_RMS_NORM][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_rms_norm_fp16;

    /* Activations */
    cb_map[RXOPS_OP_SILU][RXOPS_DTYPE_FLOAT16].exec    = rxnn_npu_silu_fp16;
    cb_map[RXOPS_OP_SIGMOID][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_sigmoid_fp16;

    /* Data movement / shape ops */
    cb_map[RXOPS_OP_RESHAPE][RXOPS_DTYPE_FLOAT16].exec   = rxnn_npu_reshape_copy;
    cb_map[RXOPS_OP_RESHAPE][RXOPS_DTYPE_FLOAT32].exec   = rxnn_npu_reshape_copy;
    cb_map[RXOPS_OP_CONCAT][RXOPS_DTYPE_FLOAT16].exec    = rxnn_npu_concat_fp16;
    cb_map[RXOPS_OP_CONCAT][RXOPS_DTYPE_FLOAT32].exec    = rxnn_npu_concat_copy;
    cb_map[RXOPS_OP_TRANSPOSE][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_transpose_fp16;
    cb_map[RXOPS_OP_GATHER][RXOPS_DTYPE_FLOAT16].exec    = rxnn_npu_gather_fp16;

    /* Positional encoding */
    cb_map[RXOPS_OP_ROPE][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_rope_fp16;

    /* Attention */
    cb_map[RXOPS_OP_FULL_ATTENTION][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_full_attention_fp16;

    /* Top-K / ArgMax */
    cb_map[RXOPS_OP_TOPK][RXOPS_DTYPE_FLOAT16].exec   = rxnn_npu_topk_fp16;
    cb_map[RXOPS_OP_ARGMAX][RXOPS_DTYPE_FLOAT16].exec = rxnn_npu_argmax_fp16;

    initialized = 1;
  }
  if (op < 0 || op >= RXOPS_OP_SIZE || dtype < 0 || dtype >= RXOPS_DTYPE_SIZE)
    return 0;
  return &cb_map[op][dtype];
}

void rxnn_ada300_init(void) {
  rxnn_register_op_callback(RXOPS_ADA300, tiny_ada300_cb_map);
}
