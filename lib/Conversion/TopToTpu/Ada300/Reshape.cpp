//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.Reshape -> tpu.Reshape (FP16 / alias, Ada300)
//
// Reshape is a layout-only op (zero-copy view). We lower it to tpu.Reshape
// so that later passes can treat it as an alias.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void ReshapeLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::ReshapeOp op) const {
  lowering_common_f16<tpu::ReshapeOp>(rewriter, op);
}

} // namespace ada300
} // namespace tpu_mlir
