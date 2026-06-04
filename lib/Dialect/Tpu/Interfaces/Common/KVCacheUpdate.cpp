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
  return type_verify_case_same(getOperation(), opd_idx, mode);
}
