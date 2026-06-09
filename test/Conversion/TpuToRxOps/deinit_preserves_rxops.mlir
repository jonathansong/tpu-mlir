// RUN: tpuc-opt --convert-tpu-to-rxops %s -o - | FileCheck %s

// CHECK-LABEL: module @rxops_deinit_keep
// CHECK: call @rxops_bridge_add_f32
// CHECK-SAME: ada300.rxops.keep = true
// CHECK: return

module @rxops_deinit_keep attributes {
  module.chip = "ada300",
  module.platform = "ONNX",
  module.state = "TPU_ADDRESSED",
  tpu.target = "ada300"} {
  func.func @main(%arg0: tensor<1x8xf32, 0 : i64> loc("lhs"),
                  %arg1: tensor<1x8xf32, 64 : i64> loc("rhs"))
      -> tensor<1x8xf32, 256 : i64> {
    %lhs = "top.Input"(%arg0) {do_preprocess = false}
      : (tensor<1x8xf32, 0 : i64>) -> tensor<1x8xf32, 128 : i64> loc("lhs")
    %rhs = "top.Input"(%arg1) {do_preprocess = false}
      : (tensor<1x8xf32, 64 : i64>) -> tensor<1x8xf32, 192 : i64> loc("rhs")
    %add = "tpu.Add"(%lhs, %rhs) {
      coeff = [],
      do_relu = false,
      is_scalar = false,
      relu_limit = -1.000000e+00 : f64
    } : (tensor<1x8xf32, 128 : i64>, tensor<1x8xf32, 192 : i64>)
      -> tensor<1x8xf32, 256 : i64> loc("add")
    return %add : tensor<1x8xf32, 256 : i64>
  } loc("main")
}
