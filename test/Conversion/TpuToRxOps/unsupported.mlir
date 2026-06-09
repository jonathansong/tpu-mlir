// RUN: not tpuc-opt --convert-tpu-to-rxops %s 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: convert-tpu-to-rxops unsupported op in minimal lowering
// ERR: arith.constant

module @rxops_unsupported attributes {
  module.chip = "ada300",
  module.platform = "ONNX",
  module.state = "TPU_ADDRESSED",
  tpu.target = "ada300"} {
  func.func @main(%arg0: tensor<1x8xf32, 0 : i64> loc("input"))
      -> tensor<1x8xf32> {
    %unused = arith.constant 0 : i32
    %in = "top.Input"(%arg0) {do_preprocess = false}
      : (tensor<1x8xf32, 0 : i64>) -> tensor<1x8xf32> loc("input")
    return %in : tensor<1x8xf32>
  } loc("main")
}
