//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
// Avoid including Conversion.h here: it transitively pulls in
// TopToLinalg/TopLowering.h and TopToTosa/TopLowering.h, both of which have
// `using namespace llvm;` at global scope.  That leaks llvm::Value / llvm::Type
// into the global namespace and conflicts with mlir::Value / mlir::Type when
// mlir/Dialect/LLVMIR/LLVMDialect.h is also in scope.  Include only what this
// translation unit actually needs.
namespace mlir {
#define GEN_PASS_DECL
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstring>

namespace tpu_mlir {

#define GEN_PASS_DEF_CONVERTTPUTORXOPS
#include "tpu_mlir/Conversion/Passes.h.inc"

namespace {

// Maximum rank supported for N-dimensional bridge calls.
static constexpr int kMaxRxOpsBridgeRank = 6;

static int64_t getNumElements(llvm::ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (auto d : shape)
    n *= d;
  return n;
}

static RankedTensorType getF32TensorLike(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(),
                               Float32Type::get(type.getContext()));
}

static MemRefType getMemRefForTensor(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static bool getBoolAttr(Operation *op, llvm::StringRef name) {
  if (auto attr = op->getAttrOfType<BoolAttr>(name))
    return attr.getValue();
  return false;
}

static LogicalResult requireStaticShape(Operation *op, RankedTensorType type,
                                        llvm::StringRef label) {
  if (!type.hasStaticShape())
    return op->emitError("convert-tpu-to-rxops requires static shape for ")
           << label;
  return success();
}

// Return element size in bytes for a type, or 0 if unknown.
static int64_t elemBytes(mlir::Type eltTy) {
  if (eltTy.isF32() || eltTy.isInteger(32))
    return 4;
  if (eltTy.isF16() || eltTy.isBF16() || eltTy.isInteger(16))
    return 2;
  if (eltTy.isF64() || eltTy.isInteger(64))
    return 8;
  if (eltTy.isInteger(8))
    return 1;
  if (eltTy.isInteger(4))
    return 1; // packed, caller handles nibbles
  return 4;   // default f32
}

// Return "_f16" or "_f32" suffix for bridge function names.
static llvm::StringRef dtypeSuffix(mlir::Type eltTy) {
  if (eltTy.isF16() || eltTy.isBF16())
    return "_f16";
  return "_f32";
}

// Bit-cast a double value to int64 for passing as a bridge argument.
static int64_t f64ToI64Bits(double v) {
  int64_t bits;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

struct RxOpsBuilder {
  ModuleOp module;
  MLIRContext *ctx;
  OpBuilder &builder;
  mlir::Type i32Ty;
  mlir::Type i64Ty;
  mlir::Type ptrTy;

  RxOpsBuilder(ModuleOp module, OpBuilder &builder)
      : module(module), ctx(module.getContext()), builder(builder),
        i32Ty(mlir::IntegerType::get(ctx, 32)),
        i64Ty(mlir::IntegerType::get(ctx, 64)),
        ptrTy(LLVM::LLVMPointerType::get(ctx)) {}

  void ensureBridge(llvm::StringRef name, llvm::ArrayRef<mlir::Type> inputs) {
    if (module.lookupSymbol<func::FuncOp>(name))
      return;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto fnTy = builder.getFunctionType(inputs, i32Ty);
    auto fn = builder.create<func::FuncOp>(module.getLoc(), name, fnTy);
    fn.setPrivate();
  }

  mlir::Value i64(mlir::Location loc, int64_t v) {
    return builder.create<LLVM::ConstantOp>(loc, i64Ty,
                                            builder.getI64IntegerAttr(v));
  }

  mlir::Value ptr(mlir::Location loc, mlir::Value memref) {
    auto idx =
        builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
    auto asI64 = builder.create<arith::IndexCastOp>(loc, i64Ty, idx);
    return builder.create<LLVM::IntToPtrOp>(loc, ptrTy, asI64);
  }

  mlir::Value toMemRef(mlir::Location loc, mlir::Value tensor) {
    auto ranked = dyn_cast<RankedTensorType>(tensor.getType());
    auto memTy = getMemRefForTensor(ranked);
    return builder.create<bufferization::ToMemrefOp>(loc, memTy, tensor, false);
  }

  mlir::Value alloc(mlir::Location loc, RankedTensorType tensorTy) {
    auto memTy = getMemRefForTensor(tensorTy);
    return builder.create<memref::AllocOp>(loc, memTy);
  }

  mlir::Value toTensor(mlir::Location loc, mlir::Value memref,
                       RankedTensorType tensorTy) {
    // Strip any TPU-specific encoding (e.g. address attributes like `192 :
    // i64`) so the result type matches the plain memref produced by alloc().
    if (tensorTy.getEncoding()) {
      tensorTy =
          RankedTensorType::get(tensorTy.getShape(), tensorTy.getElementType());
    }
    return builder.create<bufferization::ToTensorOp>(loc, tensorTy, memref,
                                                     true, false);
  }

  void keepReturnCode(mlir::Location loc, mlir::Value rc) {
    auto scratchTy = MemRefType::get({1}, i32Ty);
    auto scratch = builder.create<memref::AllocOp>(loc, scratchTy);
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    builder.create<memref::StoreOp>(loc, rc, scratch, mlir::ValueRange{zero});
  }

  // Emit a bridge call and discard the i32 return code.
  void callBridge(mlir::Location loc, llvm::StringRef name,
                  llvm::ArrayRef<mlir::Type> argTys,
                  llvm::ArrayRef<mlir::Value> args) {
    ensureBridge(name, argTys);
    auto call =
        builder.create<func::CallOp>(loc, name, mlir::TypeRange{i32Ty}, args);
    call->setAttr("ada300.rxops.keep", builder.getBoolAttr(true));
    keepReturnCode(loc, call.getResult(0));
  }

  // Append up to kMaxRxOpsBridgeRank i64 dimension constants from `shape`,
  // padding with 1 if shape.size() < kMaxRxOpsBridgeRank.
  void appendDims(mlir::Location loc, SmallVector<mlir::Value> &args,
                  llvm::ArrayRef<int64_t> shape) {
    for (int i = 0; i < kMaxRxOpsBridgeRank; ++i)
      args.push_back(i64(loc, i < (int)shape.size() ? shape[i] : 1));
  }

  // ── Existing bridge calls ──────────────────────────────────────────────

  void callAdd(mlir::Location loc, mlir::Value out, mlir::Value lhs,
               mlir::Value rhs, int64_t n, mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, i64Ty};
    SmallVector<mlir::Value> args = {ptr(loc, out), ptr(loc, lhs),
                                     ptr(loc, rhs), i64(loc, n)};
    callBridge(loc, ("rxops_bridge_add" + dtypeSuffix(eltTy)).str(), tys, args);
  }

  void callMatMul(mlir::Location loc, mlir::Value out, mlir::Value lhs,
                  mlir::Value rhs, int64_t m, int64_t n, int64_t k,
                  mlir::Type eltTy = {}) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty};
    SmallVector<mlir::Value> args = {ptr(loc, out), ptr(loc, lhs),
                                     ptr(loc, rhs), i64(loc, m),
                                     i64(loc, n),   i64(loc, k)};
    // matmul bridge always converts to f16 internally for Ada300;
    // use the dtype-specific variant when element type is known.
    llvm::StringRef suffix = (eltTy && eltTy.isF16()) ? "_f16" : "_f32";
    callBridge(loc, ("rxops_bridge_matmul" + suffix).str(), tys, args);
  }

  // ── New bridge calls ───────────────────────────────────────────────────

  // Elementwise mul (same shape)
  void callMul(mlir::Location loc, mlir::Value out, mlir::Value lhs,
               mlir::Value rhs, int64_t n, mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, i64Ty};
    SmallVector<mlir::Value> args = {ptr(loc, out), ptr(loc, lhs),
                                     ptr(loc, rhs), i64(loc, n)};
    callBridge(loc, ("rxops_bridge_mul" + dtypeSuffix(eltTy)).str(), tys, args);
  }

  // Unary activation (SiLU / Sigmoid / etc.) by function name, f32
  // fnBase is e.g. "rxops_bridge_silu"; suffix "_f16" or "_f32" is appended.
  void callUnary(mlir::Location loc, llvm::StringRef fnBase, mlir::Type eltTy,
                 mlir::Value out, mlir::Value in, int64_t n) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, i64Ty};
    SmallVector<mlir::Value> args = {ptr(loc, out), ptr(loc, in), i64(loc, n)};
    callBridge(loc, (fnBase + dtypeSuffix(eltTy)).str(), tys, args);
  }

  // RMSNorm: batch = product(shape[:-1]), hidden = shape[-1]
  void callRMSNorm(mlir::Location loc, mlir::Value out, mlir::Value in,
                   mlir::Value gamma, int64_t batch, int64_t hidden, double eps,
                   mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty};
    SmallVector<mlir::Value> args = {
        ptr(loc, out),   ptr(loc, in),     ptr(loc, gamma),
        i64(loc, batch), i64(loc, hidden), i64(loc, f64ToI64Bits(eps))};
    callBridge(loc, ("rxops_bridge_rms_norm" + dtypeSuffix(eltTy)).str(), tys,
               args);
  }

  // A16MatMul: input [m,k], weight [n,k/2] (ui8 packed int4),
  //            scales [n,ngroups] (f32), zp [n,ngroups] (ui8)
  //            output [m,n]; eltTy is the input/output dtype (f16 or f32)
  void callA16MatMul(mlir::Location loc, mlir::Value out, mlir::Value in,
                     mlir::Value qweight, mlir::Value scales, mlir::Value zp,
                     int64_t m, int64_t k, int64_t n, int64_t q_group_size,
                     int64_t weight_bits, mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
                                   i64Ty, i64Ty, i64Ty, i64Ty, i64Ty};
    SmallVector<mlir::Value> args = {
        ptr(loc, out),        ptr(loc, in), ptr(loc, qweight),
        ptr(loc, scales),     ptr(loc, zp), i64(loc, m),
        i64(loc, k),          i64(loc, n),  i64(loc, q_group_size),
        i64(loc, weight_bits)};
    callBridge(loc, ("rxops_bridge_a16matmul" + dtypeSuffix(eltTy)).str(), tys,
               args);
  }

  // Reshape: just a memcpy of total bytes
  void callReshape(mlir::Location loc, mlir::Value out, mlir::Value in,
                   int64_t nbytes) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, i64Ty};
    SmallVector<mlir::Value> args = {ptr(loc, out), ptr(loc, in),
                                     i64(loc, nbytes)};
    callBridge(loc, "rxops_bridge_reshape_bytes", tys, args);
  }

  // Slice ND: supports up to kMaxRxOpsBridgeRank dims.
  // Passes: out*, in*, elem_bytes, rank, out_dims[6], in_dims[6],
  //         offsets[6], steps[6]
  void callSliceNd(mlir::Location loc, mlir::Value out, mlir::Value in,
                   int64_t eb, int64_t rank, llvm::ArrayRef<int64_t> outShape,
                   llvm::ArrayRef<int64_t> inShape,
                   llvm::ArrayRef<int64_t> offsets,
                   llvm::ArrayRef<int64_t> steps) {
    // Fixed arg types: ptr, ptr, i64, i64 + 4 * kMaxRxOpsBridgeRank * i64
    int nDimArgs = 4 * kMaxRxOpsBridgeRank;
    SmallVector<mlir::Type> tys(4 + nDimArgs, i64Ty);
    tys[0] = ptrTy;
    tys[1] = ptrTy;

    SmallVector<mlir::Value> args;
    args.push_back(ptr(loc, out));
    args.push_back(ptr(loc, in));
    args.push_back(i64(loc, eb));
    args.push_back(i64(loc, rank));
    appendDims(loc, args, outShape);
    appendDims(loc, args, inShape);
    appendDims(loc, args, offsets);
    appendDims(loc, args, steps);
    callBridge(loc, "rxops_bridge_slice_nd", tys, args);
  }

  // Gather ND: passes out*, data*, indices*, axis, elem_bytes,
  //            data_rank, out_rank, data_dims[6], idx_dims[6], out_dims[6]
  void callGatherNd(mlir::Location loc, mlir::Value out, mlir::Value data,
                    mlir::Value indices, int64_t axis, int64_t eb,
                    llvm::ArrayRef<int64_t> dataShape,
                    llvm::ArrayRef<int64_t> idxShape,
                    llvm::ArrayRef<int64_t> outShape) {
    int nDimArgs = 3 * kMaxRxOpsBridgeRank;
    SmallVector<mlir::Type> tys(5 + nDimArgs, i64Ty);
    tys[0] = ptrTy;
    tys[1] = ptrTy;
    tys[2] = ptrTy;

    SmallVector<mlir::Value> args;
    args.push_back(ptr(loc, out));
    args.push_back(ptr(loc, data));
    args.push_back(ptr(loc, indices));
    args.push_back(i64(loc, axis));
    args.push_back(i64(loc, eb));
    appendDims(loc, args, dataShape);
    appendDims(loc, args, idxShape);
    appendDims(loc, args, outShape);
    callBridge(loc, "rxops_bridge_gather_nd", tys, args);
  }

  // Transpose ND: passes out*, in*, elem_bytes, rank,
  //               perm[6], in_dims[6], out_dims[6]
  void callTransposeNd(mlir::Location loc, mlir::Value out, mlir::Value in,
                       int64_t eb, int64_t rank, llvm::ArrayRef<int64_t> perm,
                       llvm::ArrayRef<int64_t> inShape,
                       llvm::ArrayRef<int64_t> outShape) {
    int nDimArgs = 3 * kMaxRxOpsBridgeRank;
    SmallVector<mlir::Type> tys(4 + nDimArgs, i64Ty);
    tys[0] = ptrTy;
    tys[1] = ptrTy;

    SmallVector<mlir::Value> args;
    args.push_back(ptr(loc, out));
    args.push_back(ptr(loc, in));
    args.push_back(i64(loc, eb));
    args.push_back(i64(loc, rank));
    appendDims(loc, args, perm);
    appendDims(loc, args, inShape);
    appendDims(loc, args, outShape);
    callBridge(loc, "rxops_bridge_transpose_nd", tys, args);
  }

  // Concat 2 inputs ND: out*, in0*, in1*, elem_bytes, rank, axis,
  //                     out_dims[6], in0_dims[6], in1_dims[6]
  void callConcat2Nd(mlir::Location loc, mlir::Value out, mlir::Value in0,
                     mlir::Value in1, int64_t eb, int64_t rank, int64_t axis,
                     llvm::ArrayRef<int64_t> outShape,
                     llvm::ArrayRef<int64_t> in0Shape,
                     llvm::ArrayRef<int64_t> in1Shape) {
    int nDimArgs = 3 * kMaxRxOpsBridgeRank;
    SmallVector<mlir::Type> tys(6 + nDimArgs, i64Ty);
    tys[0] = ptrTy;
    tys[1] = ptrTy;
    tys[2] = ptrTy;

    SmallVector<mlir::Value> args;
    args.push_back(ptr(loc, out));
    args.push_back(ptr(loc, in0));
    args.push_back(ptr(loc, in1));
    args.push_back(i64(loc, eb));
    args.push_back(i64(loc, rank));
    args.push_back(i64(loc, axis));
    appendDims(loc, args, outShape);
    appendDims(loc, args, in0Shape);
    appendDims(loc, args, in1Shape);
    callBridge(loc, "rxops_bridge_concat2_nd", tys, args);
  }

  // Concat 3 inputs ND
  void callConcat3Nd(mlir::Location loc, mlir::Value out, mlir::Value in0,
                     mlir::Value in1, mlir::Value in2, int64_t eb, int64_t rank,
                     int64_t axis, llvm::ArrayRef<int64_t> outShape,
                     llvm::ArrayRef<int64_t> in0Shape,
                     llvm::ArrayRef<int64_t> in1Shape,
                     llvm::ArrayRef<int64_t> in2Shape) {
    int nDimArgs = 4 * kMaxRxOpsBridgeRank;
    SmallVector<mlir::Type> tys(7 + nDimArgs, i64Ty);
    tys[0] = ptrTy;
    tys[1] = ptrTy;
    tys[2] = ptrTy;
    tys[3] = ptrTy;

    SmallVector<mlir::Value> args;
    args.push_back(ptr(loc, out));
    args.push_back(ptr(loc, in0));
    args.push_back(ptr(loc, in1));
    args.push_back(ptr(loc, in2));
    args.push_back(i64(loc, eb));
    args.push_back(i64(loc, rank));
    args.push_back(i64(loc, axis));
    appendDims(loc, args, outShape);
    appendDims(loc, args, in0Shape);
    appendDims(loc, args, in1Shape);
    appendDims(loc, args, in2Shape);
    callBridge(loc, "rxops_bridge_concat3_nd", tys, args);
  }

  // Concat 4 inputs ND
  void callConcat4Nd(mlir::Location loc, mlir::Value out, mlir::Value in0,
                     mlir::Value in1, mlir::Value in2, mlir::Value in3,
                     int64_t eb, int64_t rank, int64_t axis,
                     llvm::ArrayRef<int64_t> outShape,
                     llvm::ArrayRef<int64_t> in0Shape,
                     llvm::ArrayRef<int64_t> in1Shape,
                     llvm::ArrayRef<int64_t> in2Shape,
                     llvm::ArrayRef<int64_t> in3Shape) {
    int nDimArgs = 5 * kMaxRxOpsBridgeRank;
    SmallVector<mlir::Type> tys(8 + nDimArgs, i64Ty);
    tys[0] = ptrTy;
    tys[1] = ptrTy;
    tys[2] = ptrTy;
    tys[3] = ptrTy;
    tys[4] = ptrTy;

    SmallVector<mlir::Value> args;
    args.push_back(ptr(loc, out));
    args.push_back(ptr(loc, in0));
    args.push_back(ptr(loc, in1));
    args.push_back(ptr(loc, in2));
    args.push_back(ptr(loc, in3));
    args.push_back(i64(loc, eb));
    args.push_back(i64(loc, rank));
    args.push_back(i64(loc, axis));
    appendDims(loc, args, outShape);
    appendDims(loc, args, in0Shape);
    appendDims(loc, args, in1Shape);
    appendDims(loc, args, in2Shape);
    appendDims(loc, args, in3Shape);
    callBridge(loc, "rxops_bridge_concat4_nd", tys, args);
  }

  // Rope (contiguous_halves mode): q [batch, seq, n_heads, head_dim],
  //   cos/sin [batch, seq, cos_heads, head_dim/2_or_full]
  void callRopeContiguous(mlir::Location loc, mlir::Value out, mlir::Value q,
                          mlir::Value cos, mlir::Value sin, int64_t seq,
                          int64_t n_heads, int64_t head_dim, int64_t cos_heads,
                          mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, ptrTy,
                                   i64Ty, i64Ty, i64Ty, i64Ty};
    SmallVector<mlir::Value> args = {ptr(loc, out),      ptr(loc, q),
                                     ptr(loc, cos),      ptr(loc, sin),
                                     i64(loc, seq),      i64(loc, n_heads),
                                     i64(loc, head_dim), i64(loc, cos_heads)};
    callBridge(loc, ("rxops_bridge_rope_contiguous" + dtypeSuffix(eltTy)).str(),
               tys, args);
  }

  // Flash/Full attention (static prefill, no paged cache):
  //   q  [1, mq, np_q, d_head], k [1, mk, np_kv, d_head],
  //   v  [1, mk, np_kv, d_head], mask [1,1,mq,mk] or nullptr
  //   out [1, mq, np_q*d_head]
  void callFAttention(mlir::Location loc, mlir::Value out, mlir::Value q,
                      mlir::Value k, mlir::Value v, mlir::Value mask,
                      int64_t batch, int64_t mq, int64_t mk, int64_t np_q,
                      int64_t np_kv, int64_t d_head, double scale,
                      int64_t has_mask, mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy,
                                   i64Ty, i64Ty, i64Ty, i64Ty, i64Ty,
                                   i64Ty, i64Ty, i64Ty};
    SmallVector<mlir::Value> args = {
        ptr(loc, out),     ptr(loc, q),      ptr(loc, k),
        ptr(loc, v),       ptr(loc, mask),   i64(loc, batch),
        i64(loc, mq),      i64(loc, mk),     i64(loc, np_q),
        i64(loc, np_kv),   i64(loc, d_head), i64(loc, f64ToI64Bits(scale)),
        i64(loc, has_mask)};
    callBridge(loc, ("rxops_bridge_fattention" + dtypeSuffix(eltTy)).str(), tys,
               args);
  }

  // TopK: input [n_before, k_size, n_after], k, axis_size
  //       values_out and indices_out same shape
  void callTopK(mlir::Location loc, mlir::Value vals_out, mlir::Value idx_out,
                mlir::Value in, int64_t k, int64_t n_before, int64_t axis_size,
                int64_t n_after, mlir::Type eltTy) {
    SmallVector<mlir::Type> tys = {ptrTy, ptrTy, ptrTy, i64Ty,
                                   i64Ty, i64Ty, i64Ty};
    SmallVector<mlir::Value> args = {
        ptr(loc, vals_out), ptr(loc, idx_out),   ptr(loc, in),     i64(loc, k),
        i64(loc, n_before), i64(loc, axis_size), i64(loc, n_after)};
    callBridge(loc, ("rxops_bridge_topk" + dtypeSuffix(eltTy)).str(), tys,
               args);
  }
};

// ── Helper: get static shape from a mapped mlir::Value ──────────────────────

static SmallVector<int64_t> getMappedShape(mlir::Value v) {
  if (auto ty = dyn_cast<RankedTensorType>(v.getType()))
    return SmallVector<int64_t>(ty.getShape().begin(), ty.getShape().end());
  return {};
}

// Get a null-pointer memref value for a "none" tensor operand.
static mlir::Value getNullPtr(mlir::Location loc, OpBuilder &builder,
                              LLVM::LLVMPointerType ptrTy) {
  return builder.create<LLVM::NullOp>(loc, ptrTy);
}

struct ConvertTpuToRxOps
    : public impl::ConvertTpuToRxOpsBase<ConvertTpuToRxOps> {
  LogicalResult lowerFunc(ModuleOp module, func::FuncOp func) {
    if (func.isExternal())
      return success();

    auto loc = func.getLoc();
    auto oldFuncType = func.getFunctionType();
    SmallVector<mlir::Type> newInputs;
    SmallVector<mlir::Type> newResults;
    auto stripTensorEncoding = [](mlir::Type type) -> mlir::Type {
      if (auto ranked = dyn_cast<RankedTensorType>(type))
        return RankedTensorType::get(ranked.getShape(),
                                     ranked.getElementType());
      return type;
    };
    for (auto type : oldFuncType.getInputs())
      newInputs.push_back(stripTensorEncoding(type));
    for (auto type : oldFuncType.getResults())
      newResults.push_back(stripTensorEncoding(type));
    auto funcType =
        mlir::FunctionType::get(module.getContext(), newInputs, newResults);
    func.setType(funcType);

    auto &oldBlock = func.getBody().front();
    SmallVector<Operation *> oldOps;
    for (auto &op : oldBlock)
      oldOps.push_back(&op);

    auto *entry = new Block();
    func.getBody().push_back(entry);
    for (auto inputType : funcType.getInputs())
      entry->addArgument(inputType, loc);

    OpBuilder builder(entry, entry->end());
    RxOpsBuilder rx(module, builder);
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;

    for (auto it : llvm::enumerate(funcType.getInputs()))
      valueMap[oldBlock.getArgument(it.index())] =
          entry->getArgument(it.index());

    auto *mctx = module.getContext();
    auto genLoc = UnknownLoc::get(mctx);
    auto f32Ty = Float32Type::get(mctx);

    for (auto *op : oldOps) {
      if (isa<func::FuncOp>(op))
        continue;

      // ── Return ──────────────────────────────────────────────────────────
      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        SmallVector<mlir::Value> returns;
        SmallVector<mlir::Type> actualResultTypes;
        for (auto indexed : llvm::enumerate(ret.getOperands())) {
          auto mapped = valueMap.lookup(indexed.value());
          if (!mapped)
            return ret.emitError(
                "convert-tpu-to-rxops cannot map return operand");
          auto expected = funcType.getResult(indexed.index());
          if (mapped.getType() != expected) {
            auto mappedRanked = dyn_cast<RankedTensorType>(mapped.getType());
            auto expectedRanked = dyn_cast<RankedTensorType>(expected);
            // If only the element type changed (e.g. f32->f16 after chip
            // lowering), tensor.cast is not valid.  Update the function
            // return type to the actual lowered type instead.
            if (mappedRanked && expectedRanked &&
                mappedRanked.getShape() == expectedRanked.getShape() &&
                mappedRanked.getElementType() !=
                    expectedRanked.getElementType()) {
              // Use the actual lowered type; update the function signature
              // below.
            } else {
              mapped = builder.create<tensor::CastOp>(ret.getLoc(), expected,
                                                      mapped);
            }
          }
          actualResultTypes.push_back(mapped.getType());
          returns.push_back(mapped);
        }
        // If any result type changed due to chip-mode lowering, update the
        // FuncOp signature to match so no incompatible tensor.cast is emitted.
        if (actualResultTypes != llvm::to_vector(funcType.getResults())) {
          auto newFuncType = mlir::FunctionType::get(
              module.getContext(), funcType.getInputs(), actualResultTypes);
          func.setType(newFuncType);
        }
        builder.create<func::ReturnOp>(ret.getLoc(), returns);
        continue;
      }

      // ── Pass-through ops ─────────────────────────────────────────────────
      if (auto input = dyn_cast<top::InputOp>(op)) {
        valueMap[input.getResult()] = valueMap.lookup(input.getOperand());
        continue;
      }
      if (isa<top::NoneOp>(op))
        continue;
      // Weight ops: lower to tensor.empty (zero-initialized placeholder).
      // The rx-ops baremetal demo zero-initializes weight memory at runtime;
      // mlir-opt (next step) does not have the top dialect registered, so
      // top.Weight cannot survive past this pass.
      if (auto weightOp = dyn_cast<top::WeightOp>(op)) {
        auto origTy =
            dyn_cast<RankedTensorType>(weightOp.getResult().getType());
        if (!origTy)
          return weightOp.emitError(
              "convert-tpu-to-rxops: Weight result is not ranked tensor");
        // Strip any TPU address encoding so the type is plain.
        auto plainElt = origTy.getElementType();
        auto shape = origTy.getShape();
        auto emptyTensor =
            builder.create<tensor::EmptyOp>(genLoc, shape, plainElt);
        valueMap[weightOp.getResult()] = emptyTensor.getResult();
        continue;
      }
      if (auto castOp = dyn_cast<tpu::CastOp>(op)) {
        valueMap[castOp.getResult()] = valueMap.lookup(castOp.getOperand());
        continue;
      }
      // Load/Store ops from address-assign are structural; skip them.
      if (isa<tpu::LoadOp>(op) || isa<tpu::StoreOp>(op))
        continue;

      // ── Add ─────────────────────────────────────────────────────────────
      if (auto add = dyn_cast<tpu::AddOp>(op)) {
        auto lhs = valueMap.lookup(add.getOperand(0));
        auto rhs = valueMap.lookup(add.getOperand(1));
        if (!lhs || !rhs)
          return add.emitError("convert-tpu-to-rxops cannot map Add operands");
        auto outOrigTy = dyn_cast<RankedTensorType>(add.getResult().getType());
        if (!outOrigTy ||
            failed(requireStaticShape(add, outOrigTy, "Add output")))
          return failure();
        auto eltTy = outOrigTy.getElementType();
        auto outTy = outOrigTy;
        auto lhsMem = rx.toMemRef(genLoc, lhs);
        auto rhsMem = rx.toMemRef(genLoc, rhs);
        auto outMem = rx.alloc(genLoc, outTy);
        rx.callAdd(genLoc, outMem, lhsMem, rhsMem,
                   getNumElements(outTy.getShape()), eltTy);
        valueMap[add.getResult()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Mul ─────────────────────────────────────────────────────────────
      if (auto mul = dyn_cast<tpu::MulOp>(op)) {
        auto lhs = valueMap.lookup(mul.getOperand(0));
        auto rhs = valueMap.lookup(mul.getOperand(1));
        if (!lhs || !rhs)
          return mul.emitError("convert-tpu-to-rxops cannot map Mul operands");
        auto outOrigTy = dyn_cast<RankedTensorType>(mul.getResult().getType());
        if (!outOrigTy ||
            failed(requireStaticShape(mul, outOrigTy, "Mul output")))
          return failure();
        auto eltTy = outOrigTy.getElementType();
        auto outTy = outOrigTy;
        auto lhsMem = rx.toMemRef(genLoc, lhs);
        auto rhsMem = rx.toMemRef(genLoc, rhs);
        auto outMem = rx.alloc(genLoc, outTy);
        rx.callMul(genLoc, outMem, lhsMem, rhsMem,
                   getNumElements(outTy.getShape()), eltTy);
        valueMap[mul.getResult()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Active (SiLU / Sigmoid / etc.) ──────────────────────────────────
      if (auto active = dyn_cast<tpu::ActiveOp>(op)) {
        auto in = valueMap.lookup(active.getInput());
        if (!in)
          return active.emitError(
              "convert-tpu-to-rxops cannot map Active input");
        auto outOrigTy =
            dyn_cast<RankedTensorType>(active.getOutput().getType());
        if (!outOrigTy ||
            failed(requireStaticShape(active, outOrigTy, "Active output")))
          return failure();
        auto eltTy = outOrigTy.getElementType();
        auto outTy = outOrigTy;
        auto inMem = rx.toMemRef(genLoc, in);
        auto outMem = rx.alloc(genLoc, outTy);
        int64_t n = getNumElements(outTy.getShape());

        auto mode = active.getMode();
        llvm::StringRef fnBase;
        switch (mode) {
        case tpu::ActiveMode::SILU:
          fnBase = "rxops_bridge_silu";
          break;
        case tpu::ActiveMode::SIGMOID:
          fnBase = "rxops_bridge_sigmoid";
          break;
        case tpu::ActiveMode::RELU:
          fnBase = "rxops_bridge_relu";
          break;
        case tpu::ActiveMode::TANH:
          fnBase = "rxops_bridge_tanh";
          break;
        case tpu::ActiveMode::GELU:
          fnBase = "rxops_bridge_gelu";
          break;
        case tpu::ActiveMode::TGELU:
          fnBase = "rxops_bridge_gelu";
          break;
        default:
          return active.emitError(
                     "convert-tpu-to-rxops unsupported ActiveMode: ")
                 << (int)mode;
        }
        rx.callUnary(genLoc, fnBase, eltTy, outMem, inMem, n);
        valueMap[active.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── MatMul ──────────────────────────────────────────────────────────
      if (auto matmul = dyn_cast<tpu::MatMulOp>(op)) {
        auto lhs = valueMap.lookup(matmul.getOperand(0));
        auto rhs = valueMap.lookup(matmul.getOperand(1));
        if (!lhs || !rhs)
          return matmul.emitError(
              "convert-tpu-to-rxops cannot map MatMul operands");
        auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
        auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
        auto outOrigTy =
            dyn_cast<RankedTensorType>(matmul.getResult().getType());
        if (!lhsTy || !rhsTy || !outOrigTy)
          return matmul.emitError(
              "convert-tpu-to-rxops MatMul needs ranked tensors");
        if (lhsTy.getRank() < 2 || rhsTy.getRank() < 2)
          return matmul.emitError(
              "convert-tpu-to-rxops MatMul requires rank >= 2");
        // Flatten batch dims: m = product(shape[:-1]), k = shape[-1]
        auto lhsShape = lhsTy.getShape();
        auto rhsShape = rhsTy.getShape();
        int64_t m = 1;
        for (int i = 0; i < (int)lhsShape.size() - 1; ++i)
          m *= lhsShape[i];
        int64_t k = lhsShape.back();
        int64_t n = rhsShape.back();
        auto outTy = RankedTensorType::get(outOrigTy.getShape(),
                                           outOrigTy.getElementType());
        auto lhsMem = rx.toMemRef(genLoc, lhs);
        auto rhsMem = rx.toMemRef(genLoc, rhs);
        auto outMem = rx.alloc(genLoc, outTy);
        rx.callMatMul(genLoc, outMem, lhsMem, rhsMem, m, n, k,
                      outOrigTy.getElementType());
        valueMap[matmul.getResult()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── RMSNorm ─────────────────────────────────────────────────────────
      if (auto rmsNorm = dyn_cast<tpu::RMSNormOp>(op)) {
        auto in = valueMap.lookup(rmsNorm.getInput());
        if (!in)
          return rmsNorm.emitError(
              "convert-tpu-to-rxops cannot map RMSNorm input");
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        if (!inTy || failed(requireStaticShape(rmsNorm, inTy, "RMSNorm input")))
          return failure();
        auto outOrigTy =
            dyn_cast<RankedTensorType>(rmsNorm.getOutput().getType());
        if (!outOrigTy)
          return rmsNorm.emitError(
              "convert-tpu-to-rxops RMSNorm needs ranked output");
        auto outTy = outOrigTy; /* preserve element type (f16 or f32) */
        auto inShape = inTy.getShape();
        int64_t hidden = inShape.back();
        int64_t batch = getNumElements(inShape) / hidden;
        double eps = rmsNorm.getEps().convertToDouble();
        auto eltTy = inTy.getElementType();

        auto inMem = rx.toMemRef(genLoc, in);
        auto outMem = rx.alloc(genLoc, outTy);

        // Gamma may be an optional none-typed value
        mlir::Value gammaMem;
        auto gammaVal = rmsNorm.getGamma();
        if (gammaVal && !gammaVal.getType().isa<NoneType>()) {
          auto mappedGamma = valueMap.lookup(gammaVal);
          if (mappedGamma)
            gammaMem = rx.toMemRef(genLoc, mappedGamma);
        }
        if (!gammaMem) {
          // Allocate a placeholder gamma
          auto gTy = RankedTensorType::get({hidden}, eltTy);
          gammaMem = rx.alloc(genLoc, gTy);
        }
        rx.callRMSNorm(genLoc, outMem, inMem, gammaMem, batch, hidden, eps,
                       eltTy);
        valueMap[rmsNorm.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── A16MatMul (int4 AWQ dequant + matmul) ───────────────────────────
      if (auto a16 = dyn_cast<tpu::A16MatMulOp>(op)) {
        auto in = valueMap.lookup(a16.getInput());
        auto weight = valueMap.lookup(a16.getWeight());
        // scale and zp are optional AnyTensorOrNone
        mlir::Value scaleMem, zpMem;
        if (a16.getScale() && !a16.getScale().getType().isa<NoneType>()) {
          auto s = valueMap.lookup(a16.getScale());
          if (s)
            scaleMem = rx.toMemRef(genLoc, s);
        }
        if (a16.getZp() && !a16.getZp().getType().isa<NoneType>()) {
          auto z = valueMap.lookup(a16.getZp());
          if (z)
            zpMem = rx.toMemRef(genLoc, z);
        }
        if (!in || !weight)
          return a16.emitError(
              "convert-tpu-to-rxops cannot map A16MatMul operands");

        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        auto wTy = dyn_cast<RankedTensorType>(weight.getType());
        auto outOrigTy = dyn_cast<RankedTensorType>(a16.getOutput().getType());
        if (!inTy || !wTy || !outOrigTy)
          return a16.emitError(
              "convert-tpu-to-rxops A16MatMul needs ranked tensors");

        int64_t weight_bits = a16.getWeightBits();
        int64_t q_group_size = a16.getQGroupSize();
        // weight shape: [N, K_packed] where K_packed = K/(8/weight_bits)
        int64_t N = wTy.getShape()[0];
        int64_t K_packed = wTy.getShape()[1];
        int64_t K = K_packed * (8 / weight_bits); // unpack logical K
        // input is [batch..., K], flatten batch dims
        auto inShape = inTy.getShape();
        int64_t M = 1;
        for (int i = 0; i < (int)inShape.size() - 1; ++i)
          M *= inShape[i];
        // output shape: [batch..., N]
        SmallVector<int64_t> outShape(inShape.begin(), inShape.end());
        outShape.back() = N;
        auto eltTy = inTy.getElementType();
        auto outTy = RankedTensorType::get(outShape, eltTy);

        auto inMem = rx.toMemRef(genLoc, in);
        auto wMem = rx.toMemRef(genLoc, weight);
        auto outMem = rx.alloc(genLoc, outTy);

        // Create dummy scale/zp memrefs if not provided
        if (!scaleMem) {
          auto sTy = RankedTensorType::get({1}, f32Ty);
          scaleMem = rx.alloc(genLoc, sTy);
        }
        if (!zpMem) {
          auto zTy = RankedTensorType::get(
              {1},
              mlir::IntegerType::get(mctx, 8, mlir::IntegerType::Unsigned));
          zpMem = rx.alloc(genLoc, zTy);
        }
        rx.callA16MatMul(genLoc, outMem, inMem, wMem, scaleMem, zpMem, M, K, N,
                         q_group_size, weight_bits, eltTy);
        valueMap[a16.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Reshape ─────────────────────────────────────────────────────────
      if (auto reshape = dyn_cast<tpu::ReshapeOp>(op)) {
        auto in = valueMap.lookup(reshape.getInput());
        if (!in)
          return reshape.emitError(
              "convert-tpu-to-rxops cannot map Reshape input");
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        auto outOrigTy =
            dyn_cast<RankedTensorType>(reshape.getOutput().getType());
        if (!inTy || !outOrigTy ||
            failed(requireStaticShape(reshape, outOrigTy, "Reshape output")))
          return failure();
        auto outTy =
            RankedTensorType::get(outOrigTy.getShape(), inTy.getElementType());
        auto inMem = rx.toMemRef(genLoc, in);
        auto outMem = rx.alloc(genLoc, outTy);
        int64_t nb =
            getNumElements(outTy.getShape()) * elemBytes(inTy.getElementType());
        rx.callReshape(genLoc, outMem, inMem, nb);
        valueMap[reshape.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Slice ────────────────────────────────────────────────────────────
      if (auto slice = dyn_cast<tpu::SliceOp>(op)) {
        auto in = valueMap.lookup(slice.getInput());
        if (!in)
          return slice.emitError("convert-tpu-to-rxops cannot map Slice input");
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        auto outOrigTy =
            dyn_cast<RankedTensorType>(slice.getOutput().getType());
        if (!inTy || !outOrigTy ||
            failed(requireStaticShape(slice, inTy, "Slice input")) ||
            failed(requireStaticShape(slice, outOrigTy, "Slice output")))
          return failure();
        int64_t rank = inTy.getRank();
        if (rank > kMaxRxOpsBridgeRank)
          return slice.emitError("convert-tpu-to-rxops Slice rank exceeds max");

        auto offsetAttr = slice.getOffset();
        auto stepsAttr = slice.getSteps();
        SmallVector<int64_t> offsets, steps;
        for (auto a : offsetAttr.getAsRange<IntegerAttr>())
          offsets.push_back(a.getInt());
        for (auto a : stepsAttr.getAsRange<IntegerAttr>())
          steps.push_back(a.getInt());
        // Pad to rank if needed
        while ((int)offsets.size() < rank)
          offsets.push_back(0);
        while ((int)steps.size() < rank)
          steps.push_back(1);

        auto outTy =
            RankedTensorType::get(outOrigTy.getShape(), inTy.getElementType());
        auto inMem = rx.toMemRef(genLoc, in);
        auto outMem = rx.alloc(genLoc, outTy);
        SmallVector<int64_t> inShape(inTy.getShape());
        SmallVector<int64_t> outShape(outTy.getShape());
        int64_t eb = elemBytes(inTy.getElementType());
        rx.callSliceNd(genLoc, outMem, inMem, eb, rank, outShape, inShape,
                       offsets, steps);
        valueMap[slice.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Gather ───────────────────────────────────────────────────────────
      if (auto gather = dyn_cast<tpu::GatherOp>(op)) {
        auto data = valueMap.lookup(gather.getInput());
        auto indices = valueMap.lookup(gather.getIndices());
        if (!data || !indices)
          return gather.emitError(
              "convert-tpu-to-rxops cannot map Gather operands");
        auto dataTy = dyn_cast<RankedTensorType>(data.getType());
        auto idxTy = dyn_cast<RankedTensorType>(indices.getType());
        auto outOrigTy =
            dyn_cast<RankedTensorType>(gather.getOutput().getType());
        if (!dataTy || !idxTy || !outOrigTy)
          return gather.emitError(
              "convert-tpu-to-rxops Gather needs ranked tensors");
        if (!dataTy.hasStaticShape() || !idxTy.hasStaticShape() ||
            !outOrigTy.hasStaticShape())
          return gather.emitError(
              "convert-tpu-to-rxops Gather needs static shapes");

        int64_t axis = gather.getAxis();
        int64_t eb = elemBytes(dataTy.getElementType());
        auto outTy = RankedTensorType::get(outOrigTy.getShape(),
                                           dataTy.getElementType());
        auto dataMem = rx.toMemRef(genLoc, data);
        auto idxMem = rx.toMemRef(genLoc, indices);
        auto outMem = rx.alloc(genLoc, outTy);
        SmallVector<int64_t> dataShape(dataTy.getShape());
        SmallVector<int64_t> idxShape(idxTy.getShape());
        SmallVector<int64_t> outShape(outTy.getShape());
        rx.callGatherNd(genLoc, outMem, dataMem, idxMem, axis, eb, dataShape,
                        idxShape, outShape);
        valueMap[gather.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Permute (transpose) ──────────────────────────────────────────────
      if (auto permute = dyn_cast<tpu::PermuteOp>(op)) {
        auto in = valueMap.lookup(permute.getInput());
        if (!in)
          return permute.emitError(
              "convert-tpu-to-rxops cannot map Permute input");
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        auto outOrigTy =
            dyn_cast<RankedTensorType>(permute.getOutput().getType());
        if (!inTy || !outOrigTy ||
            failed(requireStaticShape(permute, inTy, "Permute input")) ||
            failed(requireStaticShape(permute, outOrigTy, "Permute output")))
          return failure();
        int64_t rank = inTy.getRank();
        if (rank > kMaxRxOpsBridgeRank)
          return permute.emitError(
              "convert-tpu-to-rxops Permute rank exceeds max");

        SmallVector<int64_t> perm;
        for (auto a : permute.getOrder().getAsRange<IntegerAttr>())
          perm.push_back(a.getInt());
        int64_t eb = elemBytes(inTy.getElementType());
        auto outTy =
            RankedTensorType::get(outOrigTy.getShape(), inTy.getElementType());
        auto inMem = rx.toMemRef(genLoc, in);
        auto outMem = rx.alloc(genLoc, outTy);
        SmallVector<int64_t> inShape(inTy.getShape());
        SmallVector<int64_t> outShape(outTy.getShape());
        rx.callTransposeNd(genLoc, outMem, inMem, eb, rank, perm, inShape,
                           outShape);
        valueMap[permute.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Concat ───────────────────────────────────────────────────────────
      if (auto concat = dyn_cast<tpu::ConcatOp>(op)) {
        auto inputs = concat.getInputs();
        int64_t numInputs = inputs.size();
        if (numInputs < 2 || numInputs > 4)
          return concat.emitError(
              "convert-tpu-to-rxops Concat supports 2-4 inputs");
        SmallVector<mlir::Value> mappedInputs;
        for (auto inp : inputs) {
          auto mv = valueMap.lookup(inp);
          if (!mv)
            return concat.emitError(
                "convert-tpu-to-rxops cannot map Concat input");
          mappedInputs.push_back(mv);
        }
        auto outOrigTy =
            dyn_cast<RankedTensorType>(concat.getOutput().getType());
        if (!outOrigTy ||
            failed(requireStaticShape(concat, outOrigTy, "Concat output")))
          return failure();
        auto inTy0 = dyn_cast<RankedTensorType>(mappedInputs[0].getType());
        int64_t rank = outOrigTy.getRank();
        if (rank > kMaxRxOpsBridgeRank)
          return concat.emitError(
              "convert-tpu-to-rxops Concat rank exceeds max");
        int64_t axis = concat.getAxis();
        int64_t eb = elemBytes(inTy0.getElementType());
        auto outTy =
            RankedTensorType::get(outOrigTy.getShape(), inTy0.getElementType());
        SmallVector<mlir::Value> memrefs;
        SmallVector<SmallVector<int64_t>> shapes;
        for (auto mv : mappedInputs) {
          memrefs.push_back(rx.toMemRef(genLoc, mv));
          auto ty = dyn_cast<RankedTensorType>(mv.getType());
          shapes.push_back(SmallVector<int64_t>(ty.getShape()));
        }
        auto outMem = rx.alloc(genLoc, outTy);
        SmallVector<int64_t> outShape(outTy.getShape());
        if (numInputs == 2) {
          rx.callConcat2Nd(genLoc, outMem, memrefs[0], memrefs[1], eb, rank,
                           axis, outShape, shapes[0], shapes[1]);
        } else if (numInputs == 3) {
          rx.callConcat3Nd(genLoc, outMem, memrefs[0], memrefs[1], memrefs[2],
                           eb, rank, axis, outShape, shapes[0], shapes[1],
                           shapes[2]);
        } else {
          rx.callConcat4Nd(genLoc, outMem, memrefs[0], memrefs[1], memrefs[2],
                           memrefs[3], eb, rank, axis, outShape, shapes[0],
                           shapes[1], shapes[2], shapes[3]);
        }
        valueMap[concat.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── Rope ─────────────────────────────────────────────────────────────
      if (auto rope = dyn_cast<tpu::RopeOp>(op)) {
        // input1 = Q [batch, seq, n_heads, head_dim]
        // input2 = sin [batch, seq, cos_heads, head_dim]  ← confirmed from
        // inference input3 = cos [batch, seq, cos_heads, head_dim]  ← weight1 *
        // q = q * cos Bridge formula: out[d] = q[d]*cos[d] - q[d+h]*sin[d]
        // (standard RoPE)
        auto q = valueMap.lookup(rope.getInput1());
        auto sin = valueMap.lookup(rope.getInput2());
        auto cos = valueMap.lookup(rope.getInput3());
        if (!q || !cos || !sin)
          return rope.emitError(
              "convert-tpu-to-rxops cannot map Rope operands");
        auto qTy = dyn_cast<RankedTensorType>(q.getType());
        auto cosTy = dyn_cast<RankedTensorType>(cos.getType());
        auto outOrigTy = dyn_cast<RankedTensorType>(rope.getOutput().getType());
        if (!qTy || !cosTy || !outOrigTy ||
            failed(requireStaticShape(rope, qTy, "Rope Q")))
          return failure();
        auto qShape = qTy.getShape();
        // Expect rank >= 4: [batch, seq, n_heads, head_dim]
        int64_t seq = (qShape.size() >= 2) ? qShape[qShape.size() - 3] : 1;
        int64_t n_heads = (qShape.size() >= 2) ? qShape[qShape.size() - 2] : 1;
        int64_t head_dim = qShape.back();
        auto cosShape = cosTy.getShape();
        int64_t cos_heads =
            (cosShape.size() >= 2) ? cosShape[cosShape.size() - 2] : 1;

        auto eltTy = qTy.getElementType();
        auto outTy = RankedTensorType::get(outOrigTy.getShape(), eltTy);
        auto qMem = rx.toMemRef(genLoc, q);
        auto cosMem = rx.toMemRef(genLoc, cos);
        auto sinMem = rx.toMemRef(genLoc, sin);
        auto outMem = rx.alloc(genLoc, outTy);
        rx.callRopeContiguous(genLoc, outMem, qMem, cosMem, sinMem, seq,
                              n_heads, head_dim, cos_heads, eltTy);
        valueMap[rope.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── FAttention ───────────────────────────────────────────────────────
      if (auto fattn = dyn_cast<tpu::FAttentionOp>(op)) {
        auto q = valueMap.lookup(fattn.getQueries());
        auto k = valueMap.lookup(fattn.getKeys());
        auto v = valueMap.lookup(fattn.getValues());
        if (!q || !k || !v)
          return fattn.emitError(
              "convert-tpu-to-rxops cannot map FAttention QKV");

        // Mask is optional
        mlir::Value maskMem;
        auto maskVal = fattn.getMask();
        bool hasMask = false;
        if (maskVal && !maskVal.getType().isa<NoneType>()) {
          auto mappedMask = valueMap.lookup(maskVal);
          if (mappedMask) {
            maskMem = rx.toMemRef(genLoc, mappedMask);
            hasMask = true; // infer from operand presence, not attribute
          }
        }
        if (!maskMem) {
          // Emit null pointer for mask
          auto nullVal = builder.create<LLVM::NullOp>(
              genLoc, LLVM::LLVMPointerType::get(mctx));
          (void)nullVal; // will use ptr helper instead
          // Allocate a 1-element scratch so ptr() can run
          auto scratchTy = RankedTensorType::get({1}, f32Ty);
          maskMem = rx.alloc(genLoc, scratchTy);
        }

        auto outOrigTy =
            dyn_cast<RankedTensorType>(fattn.getOutput().getType());
        if (!outOrigTy)
          return fattn.emitError(
              "convert-tpu-to-rxops FAttention needs ranked output");
        auto outTy = RankedTensorType::get(outOrigTy.getShape(),
                                           outOrigTy.getElementType());

        auto qMem = rx.toMemRef(genLoc, q);
        auto kMem = rx.toMemRef(genLoc, k);
        auto vMem = rx.toMemRef(genLoc, v);
        auto outMem = rx.alloc(genLoc, outTy);

        int64_t batch = fattn.getBatch();
        int64_t mq = fattn.getMq();
        int64_t mk = fattn.getMk();
        int64_t np_q = fattn.getQHead();
        int64_t np_kv = fattn.getKvHead();
        int64_t d_head = fattn.getDim();
        double scale = fattn.getScale().convertToDouble();

        rx.callFAttention(genLoc, outMem, qMem, kMem, vMem, maskMem, batch, mq,
                          mk, np_q, np_kv, d_head, scale, hasMask ? 1 : 0,
                          outOrigTy.getElementType());
        valueMap[fattn.getOutput()] = rx.toTensor(genLoc, outMem, outTy);
        continue;
      }

      // ── TopK (values + indices, configurable k) ──────────────────────────
      if (auto topkOp = dyn_cast<tpu::TopKOp>(op)) {
        auto in = valueMap.lookup(topkOp.getInput());
        if (!in)
          return topkOp.emitError("convert-tpu-to-rxops cannot map TopK input");
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        if (!inTy || failed(requireStaticShape(topkOp, inTy, "TopK input")))
          return failure();
        int64_t axis = topkOp.getAxis();
        int64_t k = topkOp.getK();
        auto inShape = inTy.getShape();
        int64_t axis_size = inShape[axis];
        int64_t n_before = 1, n_after = 1;
        for (int i = 0; i < (int)inShape.size(); ++i) {
          if (i < axis)
            n_before *= inShape[i];
          else if (i > axis)
            n_after *= inShape[i];
        }
        // Output shape: same as input but axis dim = k
        SmallVector<int64_t> outShape(inShape.begin(), inShape.end());
        outShape[axis] = k;
        auto valTy = RankedTensorType::get(outShape, inTy.getElementType());
        auto idxTy =
            RankedTensorType::get(outShape, f32Ty); // indices always i32/f32

        auto inMem = rx.toMemRef(genLoc, in);
        auto valMem = rx.alloc(genLoc, valTy);
        auto idxMem = rx.alloc(genLoc, idxTy);

        rx.callTopK(genLoc, valMem, idxMem, inMem, k, n_before, axis_size,
                    n_after, inTy.getElementType());

        if (topkOp.getValues() && !topkOp.getValues().getType().isa<NoneType>())
          valueMap[topkOp.getValues()] = rx.toTensor(genLoc, valMem, valTy);
        if (topkOp.getIndices() &&
            !topkOp.getIndices().getType().isa<NoneType>())
          valueMap[topkOp.getIndices()] = rx.toTensor(genLoc, idxMem, idxTy);
        continue;
      }

      // ── Arg (argmax / topk) ──────────────────────────────────────────────
      if (auto argOp = dyn_cast<tpu::ArgOp>(op)) {
        auto in = valueMap.lookup(argOp.getInput());
        if (!in)
          return argOp.emitError("convert-tpu-to-rxops cannot map Arg input");
        auto inTy = dyn_cast<RankedTensorType>(in.getType());
        if (!inTy || failed(requireStaticShape(argOp, inTy, "Arg input")))
          return failure();
        int64_t axis = argOp.getAxis();
        auto inShape = inTy.getShape();
        int64_t axis_size = inShape[axis];
        int64_t n_before = 1, n_after = 1;
        for (int i = 0; i < (int)inShape.size(); ++i) {
          if (i < axis)
            n_before *= inShape[i];
          else if (i > axis)
            n_after *= inShape[i];
        }

        // Output shape: same as input but axis dim = 1 (keepdims) or removed
        SmallVector<int64_t> outShape;
        if (argOp.getKeepdims()) {
          for (int i = 0; i < (int)inShape.size(); ++i)
            outShape.push_back(i == axis ? 1 : inShape[i]);
        } else {
          for (int i = 0; i < (int)inShape.size(); ++i)
            if (i != axis)
              outShape.push_back(inShape[i]);
        }
        auto idxTy = RankedTensorType::get(outShape, f32Ty);
        auto valTy = RankedTensorType::get(outShape, f32Ty);

        auto inMem = rx.toMemRef(genLoc, in);
        auto idxMem = rx.alloc(genLoc, idxTy);
        auto valMem = rx.alloc(genLoc, valTy);

        // k=1 for argmax
        rx.callTopK(genLoc, valMem, idxMem, inMem,
                    /*k=*/1, n_before, axis_size, n_after,
                    inTy.getElementType());
        // Arg returns indices as primary result, values as secondary
        valueMap[argOp.getIndices()] = rx.toTensor(genLoc, idxMem, idxTy);
        if (argOp.getValues() && !argOp.getValues().getType().isa<NoneType>())
          valueMap[argOp.getValues()] = rx.toTensor(genLoc, valMem, valTy);
        continue;
      }

      return op->emitError("convert-tpu-to-rxops unsupported op in lowering: ")
             << op->getName().getStringRef();
    }

    oldBlock.dropAllReferences();
    oldBlock.erase();
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // ── Auto-pipeline: if state is not yet TPU_ADDRESSED, run the upstream
    // passes inline so callers can pass a TOP_F32 module directly.
    auto state = module::getState();
    if (state != module::State::TPU_ADDRESSED) {
      if (state != module::State::TOP_F32 &&
          state != module::State::TPU_LOWERED) {
        module.emitError("convert-tpu-to-rxops: unexpected module.state; "
                         "expected TOP_F32, TPU_LOWERED, or TPU_ADDRESSED");
        signalPassFailure();
        return;
      }

      // Build a textual pipeline string and parse it into a pass manager.
      std::string pipelineStr;
      if (state == module::State::TOP_F32) {
        // Need to assign the chip/mode, initialise the module internals,
        // lower TOP → TPU, then assign buffer addresses.
        pipelineStr = llvm::formatv("processor-assign{{chip={0} mode={1}}},"
                                    "init{{freq=0 level=0}},"
                                    "convert-top-to-tpu,"
                                    "address-assign{{reuse_addr=false}}",
                                    chip, mode)
                          .str();
      } else {
        // Already TPU_LOWERED: only address assignment is missing.
        pipelineStr = "address-assign{reuse_addr=false}";
      }

      OpPassManager pm;
      if (failed(mlir::parsePassPipeline(pipelineStr, pm))) {
        module.emitError(
            "convert-tpu-to-rxops: failed to parse auto-pipeline: ")
            << pipelineStr;
        signalPassFailure();
        return;
      }
      if (failed(runPipeline(pm, module))) {
        // Diagnostic already emitted by the nested pass.
        signalPassFailure();
        return;
      }
    }

    // After auto-pipeline, the chip must be Ada300.
    if (!module::isChip(module::Chip::Ada300) && !module::isTarget("ada300")) {
      module.emitError("convert-tpu-to-rxops: chip is not Ada300 "
                       "(pass chip=ada300 or set module.chip before running)");
      signalPassFailure();
      return;
    }

    // Count supported vs unsupported ops for diagnostics.
    int64_t supported = 0;
    int64_t placeholders = 0;
    module.walk([&](Operation *op) {
      auto ns = op->getName().getDialectNamespace();
      if (ns.equals("top")) {
        if (isa<top::InputOp, top::NoneOp>(op))
          ++supported;
        else
          ++placeholders;
        return;
      }
      if (!ns.equals("tpu"))
        return;

      if (isa<tpu::MatMulOp, tpu::AddOp, tpu::MulOp, tpu::CastOp, tpu::ActiveOp,
              tpu::RMSNormOp, tpu::A16MatMulOp, tpu::ReshapeOp, tpu::SliceOp,
              tpu::GatherOp, tpu::PermuteOp, tpu::ConcatOp, tpu::RopeOp,
              tpu::FAttentionOp, tpu::ArgOp, tpu::TopKOp, tpu::LoadOp,
              tpu::StoreOp>(op))
        ++supported;
      else
        ++placeholders;
    });

    auto *ctx = module.getContext();
    module->setAttr("ada300.rxops_codegen", StringAttr::get(ctx, "full"));
    module->setAttr(
        "ada300.rxops_supported_ops",
        IntegerAttr::get(mlir::IntegerType::get(ctx, 64), supported));
    module->setAttr(
        "ada300.rxops_placeholder_ops",
        IntegerAttr::get(mlir::IntegerType::get(ctx, 64), placeholders));

    if (placeholders != 0 && !allowStubs) {
      module.emitError("convert-tpu-to-rxops found unsupported TPU/top ops");
      signalPassFailure();
      return;
    }

    SmallVector<func::FuncOp> funcs;
    module.walk([&](func::FuncOp func) { funcs.push_back(func); });
    for (auto func : funcs) {
      if (failed(lowerFunc(module, func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTpuToRxOps() {
  return std::make_unique<ConvertTpuToRxOps>();
}

} // namespace tpu_mlir
