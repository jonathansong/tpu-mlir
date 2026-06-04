//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.RMSNorm -> tpu.RMSNorm (FP16, Ada300)
//
// Mirrors the BM1684X implementation: clone gamma weight to FP16 explicitly
// rather than relying on the generic lowering_common_f16, so that
// weight_keep_f32 is respected.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void RMSNormLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::RMSNormOp op) const {
  rewriter.setInsertionPointAfter(op);
  const int nInputs = op->getNumOperands();
  std::vector<Value> opds;
  bool weight_keep_f32 = op.getWeightKeepF32();
  for (int i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    auto def = opd.getDefiningOp();
    if (def && isa<top::WeightOp>(def)) {
      auto weightOp = cast<top::WeightOp>(def);
      if (weight_keep_f32)
        opds.push_back(opd);
      else
        opds.push_back(weightOp.clone_f16(op));
    } else {
      opds.push_back(opd);
    }
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs())
    attrs.push_back(attr);

  auto new_type = getQuantF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::RMSNormOp>(op, new_type, opds, attrs);
}

} // namespace ada300
} // namespace tpu_mlir
