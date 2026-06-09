// top.TopK -> tpu.TopK (FP16, Ada300)

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void TopKLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  rewriter.setInsertionPointAfter(op);

  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (op.getKT())
    operands.push_back(op.getKT());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs())
    attrs.push_back(attr);

  std::vector<mlir::Type> new_types;
  // Values output: preserve type from top op
  new_types.push_back(op.getValues().getType());
  // Indices output: i32 tensor
  if (!module::isNone(op.getIndices())) {
    auto shape = module::getShape(op.getIndices());
    new_types.push_back(RankedTensorType::get(shape, rewriter.getI32Type()));
  } else {
    new_types.push_back(op.getIndices().getType());
  }

  // buffer_val and buffer_idx operands (None placeholders)
  auto noneOp0 = module::getNoneOp(op);
  auto noneOp1 = module::getNoneOp(op);
  operands.push_back(noneOp0);
  operands.push_back(noneOp1);

  rewriter.replaceOpWithNewOp<tpu::TopKOp>(op, new_types, operands, attrs);
}

} // namespace ada300
} // namespace tpu_mlir
