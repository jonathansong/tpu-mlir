//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.Mul -> tpu.Mul (FP16, Ada300)
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void MulLowering::LoweringF16(PatternRewriter &rewriter, top::MulOp op) const {
  lowering_common_f16<tpu::MulOp>(rewriter, op);
}

} // namespace ada300
} // namespace tpu_mlir
