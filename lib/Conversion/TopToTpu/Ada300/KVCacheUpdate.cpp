//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// top.KVCacheUpdate -> tpu.KVCacheUpdate (FP16, Ada300)
//
// The top dialect stores kv_format and mode as AnyStrAttrOf string attrs.
// tpu.KVCacheUpdate expects Tpu_KVCacheFormatAttr and Tpu_KVCacheModeAttr
// typed enums.  This lowering converts those two attrs manually and drops the
// top-only resource_id optional attr.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringAda300.h"

namespace tpu_mlir {
namespace ada300 {

void KVCacheUpdateLowering::LoweringF16(PatternRewriter &rewriter,
                                        top::KVCacheUpdateOp op) const {
  // Collect all 6 tensor operands in order:
  // (cache_k, cache_v, key, value, block_table, seq_lens)
  std::vector<Value> operands;
  for (auto opd : op->getOperands())
    operands.push_back(opd);

  // Build output types: cache tensors → F16; seq_lens keeps original type.
  auto f16_k = getQuantF16Type(op.getUpdatedCacheK());
  auto f16_v = getQuantF16Type(op.getUpdatedCacheV());
  auto seq_type = op.getUpdatedSeqLens().getType();

  // Convert string attrs to typed enum attrs, dropping top-only resource_id.
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "kv_format") {
      auto strVal = attr.getValue().cast<StringAttr>().getValue();
      auto fmt = llvm::StringSwitch<tpu::KVCacheFormat>(strVal)
                     .Case("KV4", tpu::KVCacheFormat::KV4)
                     .Case("KV8", tpu::KVCacheFormat::KV8)
                     .Case("FP8", tpu::KVCacheFormat::FP8)
                     .Default(tpu::KVCacheFormat::none);
      attrs.push_back({attr.getName(), tpu::KVCacheFormatAttr::get(
                                           rewriter.getContext(), fmt)});
    } else if (attr.getName() == "mode") {
      auto strVal = attr.getValue().cast<StringAttr>().getValue();
      auto mode = llvm::StringSwitch<tpu::KVCacheMode>(strVal)
                      .Case("paged", tpu::KVCacheMode::paged)
                      .Case("contiguous", tpu::KVCacheMode::contiguous)
                      .Default(tpu::KVCacheMode::paged);
      attrs.push_back({attr.getName(),
                       tpu::KVCacheModeAttr::get(rewriter.getContext(), mode)});
    } else if (attr.getName() == "resource_id") {
      // resource_id is top-dialect-only; tpu.KVCacheUpdate does not have it.
    } else {
      attrs.push_back(attr);
    }
  }

  rewriter.replaceOpWithNewOp<tpu::KVCacheUpdateOp>(
      op, TypeRange{f16_k, f16_v, seq_type}, operands, attrs);
}

} // namespace ada300
} // namespace tpu_mlir
