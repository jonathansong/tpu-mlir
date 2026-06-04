//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Ada300Codegen.hpp – minimal codegen driver declaration for the Ada300 NPU.
//
// M1 scope: the pass runs, validates the TPU IR, and writes a placeholder
// artifact.  Real instruction emission is deferred to a later milestone.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace tpu_mlir {
namespace tpu {

class Ada300Codegen {
public:
  Ada300Codegen() = default;

  /// Entry point called from Codegen.cpp.
  void run(mlir::ModuleOp mOp, const std::string &filename);
};

} // namespace tpu
} // namespace tpu_mlir
