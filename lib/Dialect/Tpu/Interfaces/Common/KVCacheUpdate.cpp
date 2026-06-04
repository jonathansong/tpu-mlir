//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::KVCacheUpdateOp::init(InferenceParameter &p) {
  return success();
}

void tpu::KVCacheUpdateOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::KVCacheUpdateOp::inference(InferenceParameter &p) {
  return failure();
}

mlir::Type tpu::KVCacheUpdateOp::type_verify(uint64_t opd_idx,
                                             TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

void tpu::KVCacheUpdateOp::codegen_global_bm1684() {}

void tpu::KVCacheUpdateOp::codegen_global_bm1684x() {}

void tpu::KVCacheUpdateOp::codegen_global_cv18xx(int64_t layer_id) {}

bool tpu::KVCacheUpdateOp::support_multi_core() { return false; }

uint32_t tpu::KVCacheUpdateOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  return 0;
}

int64_t tpu::KVCacheUpdateOp::dyn_codegen_global_bm1684x(void *buffer) {
  return 0;
}

int64_t tpu::KVCacheUpdateOp::get_fw_type_bm1684x() { return -1; }

int64_t tpu::KVCacheUpdateOp::get_fw_type_bm1684() { return -1; }
