//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.Add -> tpu.Add (FP16, Ada300)
//
// top.Add is variadic (Variadic<AnyTensor>:$inputs).  lowering_common_f16
// copies all operands through the weight-clone path and replaces the op with
// tpu.Add keeping all attrs.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void AddLowering::LoweringF16(PatternRewriter &rewriter, top::AddOp op) const {
  lowering_common_f16<tpu::AddOp>(rewriter, op.getOperation());
}

} // namespace ada300
} // namespace tpu_mlir
