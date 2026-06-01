//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Top/IR/TopOpsDialect.cpp.inc"

void TopDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
      >();
  addOperations<
#define GET_OP_LIST
#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Top Operator Definitions.
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/Top/IR/TopOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ADA300 LLM Op Stubs (InferenceInterface / FlopsInterface / ShapeInterface).
// These ops lower to tpu dialect before CPU inference; bodies are intentionally
// empty.
//===----------------------------------------------------------------------===//

int64_t top::QKVProjOp::getFLOPs() { return 0; }
LogicalResult top::QKVProjOp::init(InferenceParameter &p) { return success(); }
void top::QKVProjOp::deinit(InferenceParameter &p) {}
LogicalResult top::QKVProjOp::inference(InferenceParameter &p) {
  return success();
}
void top::QKVProjOp::shape_inference() {
  common_shape_inference(getOperation());
}

int64_t top::PrefillAttentionOp::getFLOPs() { return 0; }
LogicalResult top::PrefillAttentionOp::init(InferenceParameter &p) {
  return success();
}
void top::PrefillAttentionOp::deinit(InferenceParameter &p) {}
LogicalResult top::PrefillAttentionOp::inference(InferenceParameter &p) {
  return success();
}
void top::PrefillAttentionOp::shape_inference() {
  common_shape_inference(getOperation());
}

int64_t top::DecodeAttentionOp::getFLOPs() { return 0; }
LogicalResult top::DecodeAttentionOp::init(InferenceParameter &p) {
  return success();
}
void top::DecodeAttentionOp::deinit(InferenceParameter &p) {}
LogicalResult top::DecodeAttentionOp::inference(InferenceParameter &p) {
  return success();
}
void top::DecodeAttentionOp::shape_inference() {
  common_shape_inference(getOperation());
}

int64_t top::GatedMLPOp::getFLOPs() { return 0; }
LogicalResult top::GatedMLPOp::init(InferenceParameter &p) { return success(); }
void top::GatedMLPOp::deinit(InferenceParameter &p) {}
LogicalResult top::GatedMLPOp::inference(InferenceParameter &p) {
  return success();
}
void top::GatedMLPOp::shape_inference() {
  common_shape_inference(getOperation());
}

int64_t top::MoERouterOp::getFLOPs() { return 0; }
LogicalResult top::MoERouterOp::init(InferenceParameter &p) {
  return success();
}
void top::MoERouterOp::deinit(InferenceParameter &p) {}
LogicalResult top::MoERouterOp::inference(InferenceParameter &p) {
  return success();
}
void top::MoERouterOp::shape_inference() {
  common_shape_inference(getOperation());
}

int64_t top::MoEDispatchOp::getFLOPs() { return 0; }
LogicalResult top::MoEDispatchOp::init(InferenceParameter &p) {
  return success();
}
void top::MoEDispatchOp::deinit(InferenceParameter &p) {}
LogicalResult top::MoEDispatchOp::inference(InferenceParameter &p) {
  return success();
}
void top::MoEDispatchOp::shape_inference() {
  common_shape_inference(getOperation());
}

int64_t top::MoECombineOp::getFLOPs() { return 0; }
LogicalResult top::MoECombineOp::init(InferenceParameter &p) {
  return success();
}
void top::MoECombineOp::deinit(InferenceParameter &p) {}
LogicalResult top::MoECombineOp::inference(InferenceParameter &p) {
  return success();
}
void top::MoECombineOp::shape_inference() {
  common_shape_inference(getOperation());
}
