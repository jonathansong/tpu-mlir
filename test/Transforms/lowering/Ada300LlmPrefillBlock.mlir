// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify the core Transformer prefill sub-graph for ada300:
//   RMSNorm -> A16MatMul -> SiLU -> Mul
// All ops must be lowered to their tpu-dialect counterparts.

// CHECK: func.func @main
// CHECK:   tpu.RMSNorm
// CHECK:   tpu.A16MatMul
// CHECK:   tpu.Active
// CHECK:   tpu.Mul

module @ada300_prefill_block attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0:  tensor<1x8x64xf32>  loc("input"),
      %arg1:  tensor<64xf32>      loc("gamma"),
      %arg2:  tensor<64x128xi8>   loc("gate_w"),
      %arg3:  tensor<128x1xf32>   loc("gate_scale"),
      %arg4:  tensor<128x1xf32>   loc("gate_zp"),
      %arg5:  tensor<128xf32>     loc("gate_bias"),
      %arg6:  tensor<64x128xi8>   loc("up_w"),
      %arg7:  tensor<128x1xf32>   loc("up_scale"),
      %arg8:  tensor<128x1xf32>   loc("up_zp"),
      %arg9:  tensor<128xf32>     loc("up_bias"),
      %arg10: tensor<128x64xi8>   loc("down_w"),
      %arg11: tensor<64x1xf32>    loc("down_scale"),
      %arg12: tensor<64x1xf32>    loc("down_zp"),
      %arg13: tensor<64xf32>      loc("down_bias"))
      -> tensor<1x8x64xf32> {

    %in0  = "top.Input"(%arg0)  {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("input")

    // --- RMSNorm ---
    %gamma = "top.Input"(%arg1) {} : (tensor<64xf32>)     -> tensor<64xf32>     loc("gamma")
    %norm = "top.RMSNorm"(%in0, %gamma) {
        eps = 1.0e-06 : f64,
        axis = -1 : i64,
        accumulate_dtype = "NONE",
        weight_keep_f32 = false
    } : (tensor<1x8x64xf32>, tensor<64xf32>) -> tensor<1x8x64xf32>
        loc("rmsnorm")

    // --- A16MatMul (gate projection, int8 weight) ---
    %w    = "top.Input"(%arg2)  {} : (tensor<64x128xi8>)  -> tensor<64x128xi8>  loc("gate_w")
    %scl  = "top.Input"(%arg3)  {} : (tensor<128x1xf32>)  -> tensor<128x1xf32>  loc("gate_scale")
    %zp   = "top.Input"(%arg4)  {} : (tensor<128x1xf32>)  -> tensor<128x1xf32>  loc("gate_zp")
    %bias = "top.Input"(%arg5)  {} : (tensor<128xf32>)    -> tensor<128xf32>    loc("gate_bias")
    %gate = "top.A16MatMul"(%norm, %w, %scl, %zp, %bias) {
        weight_bits = 8 : i64,
        q_group_size = 64 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x8x64xf32>, tensor<64x128xi8>,
         tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>)
          -> tensor<1x8x128xf32>
        loc("gate_proj")

    // --- SiLU activation (gate path) ---
    %silu = "top.SiLU"(%gate) {
    } : (tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
        loc("silu")

    // --- Up projection (second A16MatMul, same weight layout) ---
    %wu   = "top.Input"(%arg6)  {} : (tensor<64x128xi8>)  -> tensor<64x128xi8>  loc("up_w")
    %sclu = "top.Input"(%arg7)  {} : (tensor<128x1xf32>)  -> tensor<128x1xf32>  loc("up_scale")
    %zpu  = "top.Input"(%arg8)  {} : (tensor<128x1xf32>)  -> tensor<128x1xf32>  loc("up_zp")
    %bbu  = "top.Input"(%arg9)  {} : (tensor<128xf32>)    -> tensor<128xf32>    loc("up_bias")
    %up   = "top.A16MatMul"(%norm, %wu, %sclu, %zpu, %bbu) {
        weight_bits = 8 : i64,
        q_group_size = 64 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x8x64xf32>, tensor<64x128xi8>,
         tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>)
          -> tensor<1x8x128xf32>
        loc("up_proj")

    // --- SwiGLU: gate * up ---
    %swiglu = "top.Mul"(%silu, %up) {
        do_relu = false,
        relu_limit = -1.0 : f64,
        coeff = []
    } : (tensor<1x8x128xf32>, tensor<1x8x128xf32>) -> tensor<1x8x128xf32>
        loc("swiglu")

    // --- Down projection ---
    %wd   = "top.Input"(%arg10) {} : (tensor<128x64xi8>)  -> tensor<128x64xi8>  loc("down_w")
    %scld = "top.Input"(%arg11) {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("down_scale")
    %zpd  = "top.Input"(%arg12) {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("down_zp")
    %bbd  = "top.Input"(%arg13) {} : (tensor<64xf32>)     -> tensor<64xf32>     loc("down_bias")
    %out  = "top.A16MatMul"(%swiglu, %wd, %scld, %zpd, %bbd) {
        weight_bits = 8 : i64,
        q_group_size = 128 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x8x128xf32>, tensor<128x64xi8>,
         tensor<64x1xf32>, tensor<64x1xf32>, tensor<64xf32>)
          -> tensor<1x8x64xf32>
        loc("down_proj")

    return %out : tensor<1x8x64xf32>
  }
}
