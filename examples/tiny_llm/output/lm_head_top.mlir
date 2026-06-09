module @tiny_lm_head attributes {
  module.chip = "ALL",
  module.platform = "ONNX",
  module.state = "TOP_F32",
  module.weight_file = ""} {
  func.func @main(%arg0: tensor<1x8xf32> loc("hidden"),
                  %arg1: tensor<8x1xf32> loc("lm_w")) -> tensor<1x1xf32> {
    %none = "top.None"() : () -> none loc("none")
    %hidden = "top.Input"(%arg0) {} : (tensor<1x8xf32>) -> tensor<1x8xf32> loc("lm.hidden")
    %w = "top.Input"(%arg1) {} : (tensor<8x1xf32>) -> tensor<8x1xf32> loc("lm.w")
    %out = "top.MatMul"(%hidden, %w, %none) {
      do_relu = false, keep_dims = true, left_transpose = false,
      right_transpose = false, output_transpose = false,
      relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8xf32>, tensor<8x1xf32>, none) -> tensor<1x1xf32> loc("lm.matmul")
    return %out : tensor<1x1xf32>
  }
}
