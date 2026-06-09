// top.Permute -> tpu.Permute (FP16, Ada300)

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void PermuteLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::PermuteOp op) const {
  lowering_common_f16<tpu::PermuteOp>(rewriter, op, 2);
}

} // namespace ada300
} // namespace tpu_mlir
