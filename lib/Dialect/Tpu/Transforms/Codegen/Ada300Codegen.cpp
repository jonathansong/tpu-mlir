//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Ada300 codegen driver (M1 placeholder).
//
// Walks the TPU-dialect module, prints a summary of each op to stderr for
// debugging, and writes a minimal text-format artifact so the pipeline
// completes end-to-end without crashing into BMCodegen.
//
// Real instruction encoding is left for a subsequent milestone.
//
//===----------------------------------------------------------------------===//

#include "Ada300Codegen.hpp"

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "ada300_codegen"

namespace tpu_mlir {
namespace tpu {

void Ada300Codegen::run(mlir::ModuleOp mOp, const std::string &filename) {
  // ------------------------------------------------------------------ //
  // 1. Walk every function / op in the module and emit a debug log.
  // ------------------------------------------------------------------ //
  LLVM_DEBUG(llvm::dbgs() << "[Ada300Codegen] begin module: "
                          << module::getName(mOp) << "\n");

  mOp.walk([&](mlir::Operation *op) {
    LLVM_DEBUG(llvm::dbgs()
               << "  op: " << op->getName().getStringRef() << "\n");
  });

  // ------------------------------------------------------------------ //
  // 2. Write a placeholder artifact.
  //    The file records the chip target and op count so that downstream
  //    tooling (or FileCheck tests) can verify the pipeline ran.
  // ------------------------------------------------------------------ //
  std::error_code ec;
  llvm::raw_fd_ostream out(filename, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    mOp.emitError("Ada300Codegen: cannot open output file '")
        << filename << "': " << ec.message();
    return;
  }

  out << "# Ada300 placeholder artifact\n";
  out << "chip: " << module::getChipStr() << "\n";

  unsigned op_count = 0;
  mOp.walk([&](mlir::Operation *) { ++op_count; });
  out << "op_count: " << op_count << "\n";

  LLVM_DEBUG(llvm::dbgs() << "[Ada300Codegen] wrote placeholder to: "
                          << filename << "\n");
}

} // namespace tpu
} // namespace tpu_mlir
