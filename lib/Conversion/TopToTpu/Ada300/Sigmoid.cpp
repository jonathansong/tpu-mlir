// top.Sigmoid -> tpu.Active[SIGMOID] (FP16, Ada300)

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void SigmoidLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SigmoidOp op) const {
  bool log = op.getLog();
  auto active_mode =
      log ? tpu::ActiveMode::LOG_SIGMOID : tpu::ActiveMode::SIGMOID;
  op->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(), active_mode));
  lowering_common_f16<tpu::ActiveOp>(rewriter, op.getOperation());
}

} // namespace ada300
} // namespace tpu_mlir
