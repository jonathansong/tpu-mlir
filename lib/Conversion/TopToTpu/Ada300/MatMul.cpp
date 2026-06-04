//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.MatMul -> tpu.MatMul (FP16, Ada300)
//
// Ada300 M1 scope: FP16 activation, FP16 weight.
// Unsupported modes redirect to an explicit error rather than silently
// producing incorrect IR.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

// Lower top.MatMul to tpu.MatMul with FP16 output.
// tpu.MatMulOp has 5 operands: input, right, bias, multi, buffer.
// top.MatMulOp has 3: input, right, bias.
// Passing num_operands=5 makes lowering_common_f16 copy the 3 top operands
// (cloning weight tensors to FP16) and pad the remaining 2 with noneOp.
void MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  lowering_common_f16<tpu::MatMulOp>(rewriter, op.getOperation(), 5);
}

} // namespace ada300
} // namespace tpu_mlir
