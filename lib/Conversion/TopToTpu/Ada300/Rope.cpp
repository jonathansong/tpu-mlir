//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.Rope -> tpu.Rope (FP16, Ada300)
//
// Mirrors the BM1684X F16 path:
//   1. Remove top-only round-mode attrs (not present in tpu dialect).
//   2. Use lowering_common_f16 to clone inputs and create tpu.RopeOp.
//   3. Convert the top::RopeMode string to tpu::RopeMode and set it on the
//      newly created op (lowering_common copies attrs as-is, but the enum
//      types differ between dialects).
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void RopeLowering::LoweringF16(PatternRewriter &rewriter,
                               top::RopeOp op) const {
  auto rope_mode = get_rope_mode(op.getRopeModeAttr().str());
  module::removeAttr(op, "mul1_round_mode");
  module::removeAttr(op, "mul2_round_mode");
  module::removeAttr(op, "add_round_mode");
  Operation *newOp = lowering_common_f16<tpu::RopeOp>(rewriter, op);
  newOp->setAttr("rope_mode",
                 tpu::RopeModeAttr::get(op.getContext(), rope_mode));
}

} // namespace ada300
} // namespace tpu_mlir
