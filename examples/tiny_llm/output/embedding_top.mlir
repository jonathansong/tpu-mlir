module @tiny_embedding attributes {
  module.chip = "ALL",
  module.platform = "ONNX",
  module.state = "TOP_F32",
  module.weight_file = ""} {
  func.func @main(%arg0: tensor<1x8xf32> loc("input_ids")) -> tensor<1x8xf32> {
    %in = "top.Input"(%arg0) {} : (tensor<1x8xf32>) -> tensor<1x8xf32> loc("embedding.input")
    %out = "top.Add"(%in, %in) {do_relu = false, is_scalar = false, relu_limit = -1.0 : f64, coeff = []}
      : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<1x8xf32> loc("embedding.add")
    return %out : tensor<1x8xf32>
  }
}
