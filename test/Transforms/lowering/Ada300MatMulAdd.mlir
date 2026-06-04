// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify that top.MatMul and top.Add are lowered to their tpu-dialect
// counterparts for the ada300 target in F16 mode.

// CHECK: func.func @main
// CHECK:   tpu.MatMul
// CHECK:   tpu.Add

module @ada300_matmul_add attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0: tensor<1x4x8xf32> loc("input"),
      %arg1: tensor<8x4xf32>   loc("weight"))
      -> tensor<1x4x4xf32> {
    %none = "top.None"() : () -> none loc("none")
    %in0  = "top.Input"(%arg0) {} : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32> loc("input")
    %wt   = "top.Input"(%arg1) {} : (tensor<8x4xf32>)   -> tensor<8x4xf32>   loc("weight")
    // MatMul: [1,4,8] x [8,4] -> [1,4,4]
    %mm = "top.MatMul"(%in0, %wt, %none) {
        do_relu = false,
        keep_dims = false,
        left_transpose  = false,
        right_transpose = false,
        output_transpose = false,
        relu_limit = -1.0 : f64,
        hdim_is_batch = false
    } : (tensor<1x4x8xf32>, tensor<8x4xf32>, none) -> tensor<1x4x4xf32>
        loc("matmul")

    // Add: element-wise add with itself (same shape)
    %add = "top.Add"(%mm, %mm) {
        do_relu = false,
        relu_limit = -1.0 : f64,
        coeff = []
    } : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
        loc("add")

    return %add : tensor<1x4x4xf32>
  }
}
