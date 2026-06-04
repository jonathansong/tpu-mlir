//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.Insert -> tpu.Insert (FP16, Ada300)
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void InsertLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::InsertOp op) const {
  lowering_common_f16<tpu::InsertOp>(rewriter, op);
}

} // namespace ada300
} // namespace tpu_mlir
