//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/Device.h"
#include "mlir/IR/Builders.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace backend {

Device::Device(llvm::StringRef target, int64_t gmemBytes, int64_t sramBankCount,
               int64_t sramBankBytes, int64_t weightMemoryBytes,
               int64_t alignmentBytes, int64_t gmemStartAddr,
               int64_t weightStartAddr, int64_t sramStartAddr)
    : target(target), gmemBytes(gmemBytes), sramBankCount(sramBankCount),
      sramBankBytes(sramBankBytes), weightMemoryBytes(weightMemoryBytes),
      alignmentBytes(alignmentBytes), gmemStartAddr(gmemStartAddr),
      weightStartAddr(weightStartAddr), sramStartAddr(sramStartAddr) {}

const Device &getDevice(mlir::ModuleOp moduleOp) {
  (void)moduleOp;
  static const Device ada300("ada300", 512LL * 1024 * 1024, 5, 256LL * 1024,
                             256LL * 1024 * 1024, 64, 0, 0, 0);
  static const Device bmFallback(
      "bm168x", 0, 0, 0, 0, BM168x::ALIGNMENT, BM168x::CTX_START_ADDR,
      BM168x::COEFF_START_ADDR, BM168x::L2_SRAM_START_ADDR);

  if (module::isChip(module::Chip::Ada300) || module::isTarget("ada300")) {
    return ada300;
  }
  return bmFallback;
}

void attachDeviceAttrs(mlir::ModuleOp moduleOp) {
  const auto &device = getDevice(moduleOp);
  mlir::OpBuilder builder(moduleOp.getContext());
  moduleOp->setAttr("tpu.target", builder.getStringAttr(device.getTarget()));
  moduleOp->setAttr("tpu.gmem_bytes",
                    builder.getI64IntegerAttr(device.getGmemBytes()));
  moduleOp->setAttr("tpu.sram_bank_count",
                    builder.getI64IntegerAttr(device.getSramBankCount()));
  moduleOp->setAttr("tpu.sram_bank_bytes",
                    builder.getI64IntegerAttr(device.getSramBankBytes()));
  moduleOp->setAttr("tpu.weight_memory_bytes",
                    builder.getI64IntegerAttr(device.getWeightMemoryBytes()));
}

} // namespace backend
} // namespace tpu_mlir
