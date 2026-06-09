//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Conversion/TopToTpu/TopLowering.h"

namespace tpu_mlir {
namespace ada300 {

void populateTopToTpuConversionPatterns(RewritePatternSet *patterns);

// Each struct owns exactly one lowering method: LoweringF16.
// INT8/BF16/F32 entry points are declared so the TopLowering<> base does not
// fire UNREACHABLE; they all delegate to LoweringF16 or emit a clear error.

#define LOWERING_ADA300(OP)                                                    \
  struct OP##Lowering : public TopLowering<top::OP##Op> {                      \
    OP##Lowering(MLIRContext *ctx) : TopLowering<top::OP##Op>(ctx) {}          \
    void LoweringF16(PatternRewriter &rewriter,                                \
                     top::OP##Op op) const override;                           \
    void LoweringF32(PatternRewriter &rewriter,                                \
                     top::OP##Op op) const override {                          \
      LoweringF16(rewriter, op);                                               \
    }                                                                          \
    void LoweringBF16(PatternRewriter &rewriter,                               \
                      top::OP##Op op) const override {                         \
      LoweringF16(rewriter, op);                                               \
    }                                                                          \
    void LoweringINT8(PatternRewriter &rewriter, top::OP##Op op,               \
                      bool asymmetric) const override {                        \
      op.emitError("Ada300 M1 lowering does not support INT8 for " #OP);       \
    }                                                                          \
  };

LOWERING_ADA300(MatMul)
LOWERING_ADA300(Add)
LOWERING_ADA300(RMSNorm)
LOWERING_ADA300(Rope)
LOWERING_ADA300(FAttention)
LOWERING_ADA300(A16MatMul)
LOWERING_ADA300(Mul)
LOWERING_ADA300(Reshape)
// Step 4: M3 single-layer Decode
LOWERING_ADA300(Concat)
LOWERING_ADA300(Insert)
// Shape / indexing ops needed by the real Qwen MLIR subgraphs
LOWERING_ADA300(Slice)
LOWERING_ADA300(Permute)
LOWERING_ADA300(TopK)

// SiLU lowers to tpu::ActiveOp (not tpu::SiLUOp), so it cannot use the macro.
struct SiLULowering : public TopLowering<top::SiLUOp> {
  SiLULowering(MLIRContext *ctx) : TopLowering<top::SiLUOp>(ctx) {}
  void LoweringF16(PatternRewriter &rewriter, top::SiLUOp op) const override;
  void LoweringF32(PatternRewriter &rewriter, top::SiLUOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringBF16(PatternRewriter &rewriter, top::SiLUOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op,
                    bool asymmetric) const override {
    op.emitError("Ada300 M2: INT8 SiLU not supported");
  }
};

// Sigmoid lowers to tpu::ActiveOp[SIGMOID], similar to SiLU.
struct SigmoidLowering : public TopLowering<top::SigmoidOp> {
  SigmoidLowering(MLIRContext *ctx) : TopLowering<top::SigmoidOp>(ctx) {}
  void LoweringF16(PatternRewriter &rewriter, top::SigmoidOp op) const override;
  void LoweringF32(PatternRewriter &rewriter,
                   top::SigmoidOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringBF16(PatternRewriter &rewriter,
                    top::SigmoidOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringINT8(PatternRewriter &rewriter, top::SigmoidOp op,
                    bool asymmetric) const override {
    op.emitError("Ada300: INT8 Sigmoid not supported");
  }
};

// Gather: custom struct because weight cloning logic differs from the macro.
struct GatherLowering : public TopLowering<top::GatherOp> {
  GatherLowering(MLIRContext *ctx) : TopLowering<top::GatherOp>(ctx) {}
  void LoweringF16(PatternRewriter &rewriter, top::GatherOp op) const override;
  void LoweringF32(PatternRewriter &rewriter, top::GatherOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringBF16(PatternRewriter &rewriter,
                    top::GatherOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringINT8(PatternRewriter &rewriter, top::GatherOp op,
                    bool asymmetric) const override {
    LoweringF16(rewriter, op);
  }
};

// KVCacheUpdate: needs manual kv_format (string→Tpu_KVCacheFormatAttr) and
// mode (string→Tpu_KVCacheModeAttr) conversion, so it cannot use the macro.
struct KVCacheUpdateLowering : public TopLowering<top::KVCacheUpdateOp> {
  KVCacheUpdateLowering(MLIRContext *ctx)
      : TopLowering<top::KVCacheUpdateOp>(ctx) {}
  void LoweringF16(PatternRewriter &rewriter,
                   top::KVCacheUpdateOp op) const override;
  void LoweringF32(PatternRewriter &rewriter,
                   top::KVCacheUpdateOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringBF16(PatternRewriter &rewriter,
                    top::KVCacheUpdateOp op) const override {
    LoweringF16(rewriter, op);
  }
  void LoweringINT8(PatternRewriter &rewriter, top::KVCacheUpdateOp op,
                    bool asymmetric) const override {
    op.emitError("Ada300 M3: INT8 KVCacheUpdate not supported");
  }
};

} // namespace ada300
} // namespace tpu_mlir
