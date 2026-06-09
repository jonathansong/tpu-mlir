// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// FP16-only Ada300 prefill block. Projections use top.MatMul with FP16
// lowering; quantized A16MatMul is intentionally out of scope for M2.

// CHECK: func.func @main
// CHECK:   tpu.RMSNorm
// CHECK:   tpu.MatMul
// CHECK:   tpu.Rope
// CHECK:   tpu.FAttention
// CHECK:   tpu.Add
// CHECK:   tpu.Active
// CHECK:   tpu.Mul
// CHECK-NOT: tpu.A16MatMul

module @ada300_prefill_block attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0:  tensor<1x8x64xf32> loc("input"),
      %arg1:  tensor<64xf32> loc("gamma1"),
      %arg2:  tensor<64x64xf32> loc("q_w"),
      %arg3:  tensor<64x64xf32> loc("k_w"),
      %arg4:  tensor<64x64xf32> loc("v_w"),
      %arg5:  tensor<1x8x64xf32> loc("cos"),
      %arg6:  tensor<1x8x64xf32> loc("sin"),
      %arg7:  tensor<64x64xf32> loc("o_w"),
      %arg8:  tensor<64xf32> loc("gamma2"),
      %arg9:  tensor<64x128xf32> loc("gate_w"),
      %arg10: tensor<64x128xf32> loc("up_w"),
      %arg11: tensor<128x64xf32> loc("down_w"))
      -> (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>) {
    %none = "top.None"() : () -> none loc("none")
    %hidden = "top.Input"(%arg0) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("input")
    %gamma1 = "top.Input"(%arg1) {} : (tensor<64xf32>) -> tensor<64xf32> loc("gamma1")
    %norm1 = "top.RMSNorm"(%hidden, %gamma1) {
        eps = 1.0e-06 : f64,
        axis = -1 : i64,
        accumulate_dtype = "NONE",
        weight_keep_f32 = false
    } : (tensor<1x8x64xf32>, tensor<64xf32>) -> tensor<1x8x64xf32> loc("rmsnorm1")

    %q_w = "top.Input"(%arg2) {} : (tensor<64x64xf32>) -> tensor<64x64xf32> loc("q_w")
    %q = "top.MatMul"(%norm1, %q_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x64xf32>, tensor<64x64xf32>, none) -> tensor<1x8x64xf32> loc("q_proj")

    %k_w = "top.Input"(%arg3) {} : (tensor<64x64xf32>) -> tensor<64x64xf32> loc("k_w")
    %k = "top.MatMul"(%norm1, %k_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x64xf32>, tensor<64x64xf32>, none) -> tensor<1x8x64xf32> loc("k_proj")

    %v_w = "top.Input"(%arg4) {} : (tensor<64x64xf32>) -> tensor<64x64xf32> loc("v_w")
    %v = "top.MatMul"(%norm1, %v_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x64xf32>, tensor<64x64xf32>, none) -> tensor<1x8x64xf32> loc("v_proj")

    %cos = "top.Input"(%arg5) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("cos")
    %sin = "top.Input"(%arg6) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("sin")
    %q_rope = "top.Rope"(%q, %cos, %sin) {
        rope_mode = "interleaved_pairs", freq_base = 10000.0 : f64,
        freq_scale = 1.0 : f64, xpos_base = 0.0 : f64, xpos_down = 0 : si32,
        n_dims = 0 : si32, use_rope_cache = false, is_permute_optimize = false,
        mul1_round_mode = "HalfAwayFromZero", mul2_round_mode = "HalfAwayFromZero",
        add_round_mode = "HalfAwayFromZero", mul1_shift = 0 : si32,
        mul2_shift = 0 : si32, add_shift = 0 : si32, mul1_saturation = true,
        mul2_saturation = true, add_saturation = true, force_f32 = false,
        operand_segment_sizes = array<i32: 1, 1, 1, 0, 0>
    } : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("q_rope")
    %k_rope = "top.Rope"(%k, %cos, %sin) {
        rope_mode = "interleaved_pairs", freq_base = 10000.0 : f64,
        freq_scale = 1.0 : f64, xpos_base = 0.0 : f64, xpos_down = 0 : si32,
        n_dims = 0 : si32, use_rope_cache = false, is_permute_optimize = false,
        mul1_round_mode = "HalfAwayFromZero", mul2_round_mode = "HalfAwayFromZero",
        add_round_mode = "HalfAwayFromZero", mul1_shift = 0 : si32,
        mul2_shift = 0 : si32, add_shift = 0 : si32, mul1_saturation = true,
        mul2_saturation = true, add_saturation = true, force_f32 = false,
        operand_segment_sizes = array<i32: 1, 1, 1, 0, 0>
    } : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("k_rope")

    %attn = "top.FAttention"(%q_rope, %k_rope, %v, %none, %none) {
        scale = 0.35355338 : f64, batch = 1 : i64, q_head = 8 : i64,
        kv_head = 8 : i64, dim = 8 : i64, mq = 8 : i64, mk = 8 : i64,
        keep_dims = true, causal = true, has_mask = false,
        block_size = 0 : i64, max_blocks_per_seq = 0 : i64,
        kv_cache_mode = "paged",
        operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0>
    } : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>, none, none) -> tensor<1x8x64xf32> loc("fattention")

    %o_w = "top.Input"(%arg7) {} : (tensor<64x64xf32>) -> tensor<64x64xf32> loc("o_w")
    %o = "top.MatMul"(%attn, %o_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x64xf32>, tensor<64x64xf32>, none) -> tensor<1x8x64xf32> loc("o_proj")
    %res1 = "top.Add"(%hidden, %o) {do_relu = false, relu_limit = -1.0 : f64, coeff = []}
        : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("res_attn")

    %gamma2 = "top.Input"(%arg8) {} : (tensor<64xf32>) -> tensor<64xf32> loc("gamma2")
    %norm2 = "top.RMSNorm"(%res1, %gamma2) {
        eps = 1.0e-06 : f64, axis = -1 : i64,
        accumulate_dtype = "NONE", weight_keep_f32 = false
    } : (tensor<1x8x64xf32>, tensor<64xf32>) -> tensor<1x8x64xf32> loc("rmsnorm2")

    %gate_w = "top.Input"(%arg9) {} : (tensor<64x128xf32>) -> tensor<64x128xf32> loc("gate_w")
    %gate = "top.MatMul"(%norm2, %gate_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x64xf32>, tensor<64x128xf32>, none) -> tensor<1x8x128xf32> loc("gate_proj")
    %gate_act = "top.SiLU"(%gate) {} : (tensor<1x8x128xf32>) -> tensor<1x8x128xf32> loc("silu")

    %up_w = "top.Input"(%arg10) {} : (tensor<64x128xf32>) -> tensor<64x128xf32> loc("up_w")
    %up = "top.MatMul"(%norm2, %up_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x64xf32>, tensor<64x128xf32>, none) -> tensor<1x8x128xf32> loc("up_proj")
    %swiglu = "top.Mul"(%gate_act, %up) {do_relu = false, relu_limit = -1.0 : f64, coeff = []}
        : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<1x8x128xf32> loc("swiglu")

    %down_w = "top.Input"(%arg11) {} : (tensor<128x64xf32>) -> tensor<128x64xf32> loc("down_w")
    %mlp = "top.MatMul"(%swiglu, %down_w, %none) {
        do_relu = false, keep_dims = false, left_transpose = false,
        right_transpose = false, output_transpose = false,
        relu_limit = -1.0 : f64, hdim_is_batch = false
    } : (tensor<1x8x128xf32>, tensor<128x64xf32>, none) -> tensor<1x8x64xf32> loc("down_proj")
    %res2 = "top.Add"(%res1, %mlp) {do_relu = false, relu_limit = -1.0 : f64, coeff = []}
        : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("res_mlp")

    return %res2, %k_rope, %v : tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>
  }
}
