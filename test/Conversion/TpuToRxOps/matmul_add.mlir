// RUN: tpuc-opt --convert-tpu-to-rxops %s | FileCheck %s

// CHECK-LABEL: module @rxops_matmul_add
// CHECK: func.func private @rxops_bridge_add_f32
// CHECK: func.func private @rxops_bridge_matmul_f32
// CHECK-LABEL: func.func @main
// CHECK: call @rxops_bridge_matmul_f32
// CHECK-SAME: ada300.rxops.keep = true
// CHECK: call @rxops_bridge_add_f32
// CHECK-SAME: ada300.rxops.keep = true
// CHECK: return

module @rxops_matmul_add attributes {
  module.chip = "ada300",
  module.platform = "ONNX",
  module.state = "TPU_ADDRESSED",
  tpu.target = "ada300"} {
  func.func @main(%arg0: tensor<1x8xf32, 0 : i64> loc("hidden"),
                  %arg1: tensor<8x8xf32, 64 : i64> loc("weight"))
      -> tensor<1x8xf32, 512 : i64> {
    %none = "top.None"() : () -> none loc("none")
    %in = "top.Input"(%arg0) {do_preprocess = false}
      : (tensor<1x8xf32, 0 : i64>) -> tensor<1x8xf32, 128 : i64> loc("input")
    %w = "top.Input"(%arg1) {do_preprocess = false}
      : (tensor<8x8xf32, 64 : i64>) -> tensor<8x8xf32, 192 : i64> loc("weight")
    %mm = "tpu.MatMul"(%in, %w, %none, %none, %none) {
      do_relu = false,
      dq_type = "NONE",
      fuse_rq = false,
      hdim_is_batch = false,
      input_zp = 0 : i64,
      is_lora = false,
      keep_dims = true,
      left_reuse = 1 : i64,
      left_transpose = false,
      multipliers = [1],
      output_transpose = false,
      q_group_size = 0,
      quant_mode = #tpu<rq_mode MultiplierShift>,
      relu_limit = -1.000000e+00 : f64,
      right_transpose = false,
      right_zp = 0 : i64,
      round_mode = #tpu<round_mode HalfAwayFromZero>,
      rshifts = [0]
    } : (tensor<1x8xf32, 128 : i64>, tensor<8x8xf32, 192 : i64>, none, none, none)
      -> tensor<1x8xf32, 320 : i64> loc("matmul")
    %add = "tpu.Add"(%mm, %in) {
      coeff = [],
      do_relu = false,
      is_scalar = false,
      relu_limit = -1.000000e+00 : f64
    } : (tensor<1x8xf32, 320 : i64>, tensor<1x8xf32, 128 : i64>)
      -> tensor<1x8xf32, 512 : i64> loc("add")
    return %add : tensor<1x8xf32, 512 : i64>
  } loc("main")
}
