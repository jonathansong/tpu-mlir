//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      // Step 2: M1 MatMul + Add
      MatMulLowering,
      AddLowering,
      // Step 3: M2 single-layer Transformer Prefill
      RMSNormLowering,
      RopeLowering,
      FAttentionLowering,
      A16MatMulLowering,
      MulLowering,
      ReshapeLowering,
      SiLULowering,
      // Step 4: M3 single-layer Decode
      KVCacheUpdateLowering,
      ConcatLowering,
      InsertLowering,
      // Shape / indexing ops for real Qwen subgraphs
      SliceLowering,
      PermuteLowering,
      TopKLowering,
      SigmoidLowering,
      GatherLowering
  >(patterns->getContext());
  // clang-format on
}

} // namespace ada300
} // namespace tpu_mlir
