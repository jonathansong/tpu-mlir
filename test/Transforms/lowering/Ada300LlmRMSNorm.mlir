// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify that top.RMSNorm is lowered to tpu.RMSNorm in F16 mode for ada300.

// CHECK: func.func @main
// CHECK:   tpu.RMSNorm

module @ada300_rmsnorm attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0: tensor<1x8x64xf32> loc("input"),
      %arg1: tensor<64xf32>     loc("gamma"))
      -> tensor<1x8x64xf32> {

    %in0  = "top.Input"(%arg0) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("input")
    %gamma = "top.Input"(%arg1) {} : (tensor<64xf32>) -> tensor<64xf32> loc("gamma")

    %out = "top.RMSNorm"(%in0, %gamma) {
        eps = 1.0e-06 : f64,
        axis = -1 : i64,
        accumulate_dtype = "NONE",
        weight_keep_f32 = false
    } : (tensor<1x8x64xf32>, tensor<64xf32>) -> tensor<1x8x64xf32>
        loc("rmsnorm")

    return %out : tensor<1x8x64xf32>
  }
}
