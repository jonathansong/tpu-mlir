//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Interfaces/TypeInterface.h"

mlir::Type tpu::KVCacheUpdateOp::type_verify(uint64_t opd_idx,
                                             TypeCastMode &mode) {
  // block_table and seq_lens are indexing/state tensors. Even in FP16-only
  // Ada300 lowering they must retain their original integer type.
  if (opd_idx == 4 || opd_idx == 5) {
    return do_nothing(mode);
  }
  return type_verify_case_same(getOperation(), opd_idx, mode);
}
