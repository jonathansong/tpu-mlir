// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify that top.Rope is lowered to tpu.Rope in F16 mode for ada300.
// The top round-mode attributes are removed and rope_mode is translated from
// a string to the tpu::RopeMode enum.

// CHECK: func.func @main
// CHECK:   tpu.Rope

module @ada300_rope attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0: tensor<1x8x8xf32> loc("in1"),
      %arg1: tensor<1x8x8xf32> loc("in2"),
      %arg2: tensor<1x8x8xf32> loc("in3"))
      -> tensor<1x8x8xf32> {

    %in1 = "top.Input"(%arg0) {} : (tensor<1x8x8xf32>) -> tensor<1x8x8xf32> loc("in1")
    %in2 = "top.Input"(%arg1) {} : (tensor<1x8x8xf32>) -> tensor<1x8x8xf32> loc("in2")
    %in3 = "top.Input"(%arg2) {} : (tensor<1x8x8xf32>) -> tensor<1x8x8xf32> loc("in3")

    // No optional operands (pos / rope_cache absent => operandSegmentSizes 0,0)
    %out = "top.Rope"(%in1, %in2, %in3) {
        rope_mode = "interleaved_pairs",
        freq_base = 10000.0 : f64,
        freq_scale = 1.0 : f64,
        xpos_base = 0.0 : f64,
        xpos_down = 0 : si32,
        n_dims = 0 : si32,
        use_rope_cache = false,
        is_permute_optimize = false,
        mul1_round_mode = "HalfAwayFromZero",
        mul2_round_mode = "HalfAwayFromZero",
        add_round_mode = "HalfAwayFromZero",
        mul1_shift = 0 : si32,
        mul2_shift = 0 : si32,
        add_shift = 0 : si32,
        mul1_saturation = true,
        mul2_saturation = true,
        add_saturation = true,
        force_f32 = false,
        operand_segment_sizes = array<i32: 1, 1, 1, 0, 0>
    } : (tensor<1x8x8xf32>, tensor<1x8x8xf32>, tensor<1x8x8xf32>)
          -> tensor<1x8x8xf32>
        loc("rope")

    return %out : tensor<1x8x8xf32>
  }
}
