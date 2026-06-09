module @tiny_block7 attributes {
  module.chip = "ALL",
  module.platform = "ONNX",
  module.state = "TOP_F32",
  module.weight_file = ""} {
  func.func @main(%arg0: tensor<1x8xf32> loc("hidden"),
                  %arg1: tensor<8x8xf32> loc("block7_w")) -> tensor<1x8xf32> {
    %none = "top.None"() : () -> none loc("none")
    %hidden = "top.Input"(%arg0) {} : (tensor<1x8xf32>) -> tensor<1x8xf32> loc("block7.hidden")
    %w = "top.Input"(%arg1) {} : (tensor<8x8xf32>) -> tensor<8x8xf32> loc("block7.w")
    %mm = "top.MatMul"(%hidden, %w, %none) {
      do_relu = false, keep_dims = true, left_transpose = false,
      right_transpose = false, output_transpose = false,
      relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8xf32>, tensor<8x8xf32>, none) -> tensor<1x8xf32> loc("block7.matmul")
    %out = "top.Add"(%mm, %hidden) {do_relu = false, is_scalar = false, relu_limit = -1.0 : f64, coeff = []}
      : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<1x8xf32> loc("block7.residual")
    return %out : tensor<1x8xf32>
  }
}
