// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" --init="freq=0 level=0" %s | FileCheck %s
// RUN: not tpuc-opt --processor-assign="chip=unknown mode=F16" %s 2>&1 | FileCheck %s --check-prefix=UNKNOWN

// CHECK: module @ada300_target attributes
// CHECK-SAME: module.chip = "ada300"
// CHECK-SAME: tpu.gmem_bytes = 536870912 : i64
// CHECK-SAME: tpu.sram_bank_bytes = 262144 : i64
// CHECK-SAME: tpu.sram_bank_count = 5 : i64
// CHECK-SAME: tpu.target = "ada300"
// CHECK-SAME: tpu.weight_memory_bytes = 268435456 : i64
// UNKNOWN: unsupported chip: unknown
module @ada300_target attributes {module.chip = "ALL", module.platform = "ONNX", module.state = "TOP_F32", module.weight_file = ""} {
  func.func @main(%arg0: tensor<1x4xf32> loc("input")) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
