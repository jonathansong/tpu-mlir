//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.SiLU -> tpu.Active(mode=SILU) (FP16, Ada300)
//
// The tpu dialect has no dedicated SiLUOp; all activation functions are
// represented by tpu.ActiveOp with an ActiveMode attribute.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void SiLULowering::LoweringF16(PatternRewriter &rewriter,
                               top::SiLUOp op) const {
  op->setAttr("mode",
              tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::SILU));
  lowering_common_f16<tpu::ActiveOp>(rewriter, op.getOperation());
}

} // namespace ada300
} // namespace tpu_mlir
