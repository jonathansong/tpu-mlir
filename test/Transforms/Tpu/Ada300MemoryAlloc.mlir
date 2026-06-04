// RUN: tpuc-opt --init="freq=0 level=0" --address-assign="reuse_addr=false" %s | FileCheck %s

// CHECK: module @ada300_memory attributes
// CHECK-SAME: module.coeff_addr = 0 : i64
// CHECK-SAME: module.coeff_size = 0 : i64
// CHECK-SAME: module.neuron_addr = 0 : i64
// CHECK-SAME: module.neuron_size
// CHECK-SAME: tpu.target = "ada300"
module @ada300_memory attributes {module.chip = "ada300", module.cores = 1 : i64, module.devices = 1 : i64, module.mode = "F16", module.platform = "ONNX", module.state = "TPU_DIVIDED", module.weight_file = "", tpu.target = "ada300"} {
  func.func @main(%arg0: tensor<1x4xf16> loc("input")) -> tensor<1x4xf16> {
    %0 = "tpu.Add"(%arg0, %arg0) {do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<1x4xf16>, tensor<1x4xf16>) -> tensor<1x4xf16> loc("add")
    return %0 : tensor<1x4xf16>
  }
}
