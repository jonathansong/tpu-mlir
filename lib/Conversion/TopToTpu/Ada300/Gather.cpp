// top.Gather -> tpu.Gather (FP16, Ada300)

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

static void LoweringGather(PatternRewriter &rewriter, top::GatherOp op,
                           mlir::Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  if (module::isWeight(op.getInput())) {
    auto wOp = op.getInput().getDefiningOp<top::WeightOp>();
    auto stype = module::getStorageType(type);
    if (stype.isF16())
      operands.push_back(wOp.clone_f16(op));
    else
      operands.push_back(op.getInput());
  } else {
    operands.push_back(op.getInput());
  }
  if (module::isWeight(op.getIndices())) {
    auto wOp = op.getIndices().getDefiningOp<top::WeightOp>();
    operands.push_back(wOp.clone_int(op));
  } else {
    operands.push_back(op.getIndices());
  }
  operands.push_back(module::getNoneOp(op));
  rewriter.replaceOpWithNewOp<tpu::GatherOp>(op, type, operands,
                                             op->getAttrs());
}

void GatherLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
  LoweringGather(rewriter, op, new_type);
}

} // namespace ada300
} // namespace tpu_mlir
