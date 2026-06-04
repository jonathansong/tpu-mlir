//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace tpu_mlir {
namespace backend {

class Device {
public:
  Device(llvm::StringRef target, int64_t gmemBytes, int64_t sramBankCount,
         int64_t sramBankBytes, int64_t weightMemoryBytes,
         int64_t alignmentBytes, int64_t gmemStartAddr, int64_t weightStartAddr,
         int64_t sramStartAddr);

  llvm::StringRef getTarget() const { return target; }
  int64_t getGmemBytes() const { return gmemBytes; }
  int64_t getSramBankCount() const { return sramBankCount; }
  int64_t getSramBankBytes() const { return sramBankBytes; }
  int64_t getSramBytes() const { return sramBankCount * sramBankBytes; }
  int64_t getWeightMemoryBytes() const { return weightMemoryBytes; }
  int64_t getAlignmentBytes() const { return alignmentBytes; }
  int64_t getGmemStartAddr() const { return gmemStartAddr; }
  int64_t getWeightStartAddr() const { return weightStartAddr; }
  int64_t getSramStartAddr() const { return sramStartAddr; }

private:
  llvm::StringRef target;
  int64_t gmemBytes;
  int64_t sramBankCount;
  int64_t sramBankBytes;
  int64_t weightMemoryBytes;
  int64_t alignmentBytes;
  int64_t gmemStartAddr;
  int64_t weightStartAddr;
  int64_t sramStartAddr;
};

const Device &getDevice(mlir::ModuleOp moduleOp);
void attachDeviceAttrs(mlir::ModuleOp moduleOp);

} // namespace backend
} // namespace tpu_mlir
