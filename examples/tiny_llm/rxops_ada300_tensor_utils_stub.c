/* Minimal stub for Ada300 tensor view helpers used by full_attention.c.
 *
 * The real implementations live in quant.c which pulls in matmul_fp32,
 * fullyconnected_fp32 and other asm symbols not needed for the f16-only
 * Qwen model.  These stubs replicate only the logic paths exercised at
 * runtime: tensors are always already in the target dtype (f16 on Ada300),
 * so prepare_tensor_as always takes the no-copy path and owns_data is
 * never set — making release_tensor_view a safe no-op.
 */
#include "ada300/rxnn_npu_quant.h"
#include "ada300/rxnn_npu.h"

int rxnn_ada300_prepare_tensor_as(struct rxnn_ada300_tensor_view *view,
                                  struct rxops_tensor *src,
                                  enum rxops_dtype_enum compute_dtype,
                                  enum rxops_quant_enum quant_hint)
{
    (void)quant_hint;
    if (view == NULL || src == NULL)
        return RXOPS_FALSE;
    view->tensor   = *src;
    view->owns_data = 0;
    if (src->dtype == compute_dtype && src->mtype == RXOPS_MEM_TYPE_CPU)
        return RXOPS_TRUE;
    return RXOPS_UNSUPPORT_DTYPE;
}

void rxnn_ada300_release_tensor_view(struct rxnn_ada300_tensor_view *view)
{
    /* owns_data is never set in the no-copy path above; nothing to free. */
    if (view != NULL) {
        view->tensor.data = NULL;
        view->owns_data   = 0;
    }
}
