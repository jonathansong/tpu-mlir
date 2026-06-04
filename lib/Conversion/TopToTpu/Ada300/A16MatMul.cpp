//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.A16MatMul -> tpu.A16MatMul (FP16 baseline, Ada300)
//
// M2 FP16 fallback: activations are lowered to FP16; the int8/int4 weight
// WeightOp is not float, so lowering_common_f16 leaves it untouched.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void A16MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::A16MatMulOp op) const {
  // Pass all 5 operands: input, weight, scale, zp, bias.
  // lowering_common_f16 only clones WeightOps that hold float data, so the
  // int8/int4 weight operand is passed through unchanged.
  lowering_common_f16<tpu::A16MatMulOp>(rewriter, op, 5);
}

} // namespace ada300
} // namespace tpu_mlir
