//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.FAttention -> tpu.FAttention (FP16, Ada300)
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void FAttentionLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::FAttentionOp op) const {
  // lowering_common copies top attrs as-is; tpu.FAttentionOp expects
  // kv_cache_mode as a Tpu_KVCacheModeAttr enum, but top.FAttentionOp stores
  // it as a string. Build the tpu op manually so we can convert the attr.
  auto newType = getQuantF16Type(op.getOutput());
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  // Collect operands via the generic lowering helper (handles weight cloning).
  int nOp = op->getNumOperands();
  for (int i = 0; i < nOp; ++i)
    operands.push_back(op->getOperand(i));
  // Build attrs, converting kv_cache_mode string -> enum.
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "kv_cache_mode") {
      auto strVal = attr.getValue().cast<StringAttr>().getValue();
      auto mode = llvm::StringSwitch<tpu::KVCacheMode>(strVal)
                      .Case("paged", tpu::KVCacheMode::paged)
                      .Case("contiguous", tpu::KVCacheMode::contiguous)
                      .Default(tpu::KVCacheMode::paged);
      attrs.push_back({attr.getName(),
                       tpu::KVCacheModeAttr::get(rewriter.getContext(), mode)});
    } else {
      attrs.push_back(attr);
    }
  }
  rewriter.replaceOpWithNewOp<tpu::FAttentionOp>(op, newType, operands, attrs);
}

} // namespace ada300
} // namespace tpu_mlir
