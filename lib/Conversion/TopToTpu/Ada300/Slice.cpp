// top.Slice -> tpu.Slice (FP16, Ada300)

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void SliceLowering::LoweringF16(PatternRewriter &rewriter,
                                top::SliceOp op) const {
  lowering_common_f16<tpu::SliceOp>(rewriter, op, 5);
}

} // namespace ada300
} // namespace tpu_mlir
