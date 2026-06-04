// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify the core Transformer decode sub-graph for ada300:
//   RMSNorm -> A16MatMul (Q proj) -> Rope -> KVCacheUpdate
//   -> FAttention -> A16MatMul (O proj) -> Add (residual)
//   -> RMSNorm -> SwiGLU MLP -> Add (residual)
// All ops must be lowered to their tpu-dialect counterparts.

// CHECK: func.func @main
// CHECK:   tpu.RMSNorm
// CHECK:   tpu.A16MatMul
// CHECK:   tpu.Rope
// CHECK:   tpu.KVCacheUpdate
// CHECK:   tpu.FAttention
// CHECK:   tpu.Add
// CHECK:   tpu.Active
// CHECK:   tpu.Mul

module @ada300_decode_block attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      // Hidden state: single decode token
      %arg0:  tensor<1x1x64xf32>   loc("hidden"),
      // Attention pre-norm
      %arg1:  tensor<64xf32>        loc("gamma1"),
      // Q projection weights (weight: [K×N]=[64×64], scale/zp 2-D: [N×1])
      %arg2:  tensor<64x64xi8>      loc("q_w"),
      %arg3:  tensor<64x1xf32>      loc("q_scl"),
      %arg4:  tensor<64x1xf32>      loc("q_zp"),
      %arg5:  tensor<64xf32>        loc("q_bias"),
      // RoPE tables
      %arg6:  tensor<1x1x64xf32>   loc("cos"),
      %arg7:  tensor<1x1x64xf32>   loc("sin"),
      // New K/V tokens fed directly into KVCacheUpdate
      %arg8:  tensor<1x1x64xf32>   loc("key"),
      %arg9:  tensor<1x1x64xf32>   loc("value"),
      // KV cache buffers (history_len=8)
      %arg10: tensor<1x8x64xf32>   loc("cache_k"),
      %arg11: tensor<1x8x64xf32>   loc("cache_v"),
      %arg12: tensor<1x8xf32>       loc("block_table"),
      %arg13: tensor<1xf32>         loc("seq_lens"),
      // O projection weights
      %arg14: tensor<64x64xi8>      loc("o_w"),
      %arg15: tensor<64x1xf32>      loc("o_scl"),
      %arg16: tensor<64x1xf32>      loc("o_zp"),
      %arg17: tensor<64xf32>        loc("o_bias"),
      // MLP pre-norm
      %arg18: tensor<64xf32>        loc("gamma2"),
      // MLP gate projection [64×128]
      %arg19: tensor<64x128xi8>     loc("gate_w"),
      %arg20: tensor<128x1xf32>     loc("gate_scl"),
      %arg21: tensor<128x1xf32>     loc("gate_zp"),
      %arg22: tensor<128xf32>       loc("gate_bias"),
      // MLP up projection [64×128]
      %arg23: tensor<64x128xi8>     loc("up_w"),
      %arg24: tensor<128x1xf32>     loc("up_scl"),
      %arg25: tensor<128x1xf32>     loc("up_zp"),
      %arg26: tensor<128xf32>       loc("up_bias"),
      // MLP down projection [128×64]
      %arg27: tensor<128x64xi8>     loc("down_w"),
      %arg28: tensor<64x1xf32>      loc("down_scl"),
      %arg29: tensor<64x1xf32>      loc("down_zp"),
      %arg30: tensor<64xf32>        loc("down_bias"))
      -> tensor<1x1x64xf32> {

    %hidden = "top.Input"(%arg0)  {} : (tensor<1x1x64xf32>)  -> tensor<1x1x64xf32>  loc("hidden")

    // -------------------------------------------------------------------------
    // Attention pre-norm
    // -------------------------------------------------------------------------
    %gamma1 = "top.Input"(%arg1)  {} : (tensor<64xf32>)       -> tensor<64xf32>       loc("gamma1")
    %norm1 = "top.RMSNorm"(%hidden, %gamma1) {
        eps = 1.0e-06 : f64,
        axis = -1 : i64,
        accumulate_dtype = "NONE",
        weight_keep_f32 = false
    } : (tensor<1x1x64xf32>, tensor<64xf32>) -> tensor<1x1x64xf32>
        loc("rmsnorm1")

    // -------------------------------------------------------------------------
    // Q projection (A16MatMul, int8 weight, 2-D scale/zp)
    // -------------------------------------------------------------------------
    %q_w   = "top.Input"(%arg2)  {} : (tensor<64x64xi8>)   -> tensor<64x64xi8>   loc("q_w")
    %q_scl = "top.Input"(%arg3)  {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("q_scl")
    %q_zp  = "top.Input"(%arg4)  {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("q_zp")
    %q_b   = "top.Input"(%arg5)  {} : (tensor<64xf32>)     -> tensor<64xf32>     loc("q_bias")
    %q = "top.A16MatMul"(%norm1, %q_w, %q_scl, %q_zp, %q_b) {
        weight_bits = 8 : i64,
        q_group_size = 64 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x1x64xf32>, tensor<64x64xi8>,
         tensor<64x1xf32>, tensor<64x1xf32>, tensor<64xf32>)
          -> tensor<1x1x64xf32>
        loc("q_proj")

    // -------------------------------------------------------------------------
    // RoPE on Q
    // -------------------------------------------------------------------------
    %cos = "top.Input"(%arg6)  {} : (tensor<1x1x64xf32>)  -> tensor<1x1x64xf32>  loc("cos")
    %sin = "top.Input"(%arg7)  {} : (tensor<1x1x64xf32>)  -> tensor<1x1x64xf32>  loc("sin")
    %q_rope = "top.Rope"(%q, %cos, %sin) {
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
    } : (tensor<1x1x64xf32>, tensor<1x1x64xf32>, tensor<1x1x64xf32>)
          -> tensor<1x1x64xf32>
        loc("q_rope")

    // -------------------------------------------------------------------------
    // KV cache update: write new K/V token into paged cache
    // -------------------------------------------------------------------------
    %key  = "top.Input"(%arg8)  {} : (tensor<1x1x64xf32>)  -> tensor<1x1x64xf32>  loc("key")
    %val  = "top.Input"(%arg9)  {} : (tensor<1x1x64xf32>)  -> tensor<1x1x64xf32>  loc("value")
    %ck   = "top.Input"(%arg10) {} : (tensor<1x8x64xf32>)  -> tensor<1x8x64xf32>  loc("cache_k")
    %cv   = "top.Input"(%arg11) {} : (tensor<1x8x64xf32>)  -> tensor<1x8x64xf32>  loc("cache_v")
    %bt   = "top.Input"(%arg12) {} : (tensor<1x8xf32>)     -> tensor<1x8xf32>     loc("block_table")
    %sl   = "top.Input"(%arg13) {} : (tensor<1xf32>)       -> tensor<1xf32>       loc("seq_lens")

    %uk, %uv, %us = "top.KVCacheUpdate"(%ck, %cv, %key, %val, %bt, %sl) {
        kv_format = "KV8",
        mode = "paged",
        block_size = 8 : i64,
        max_blocks_per_seq = 1 : i64,
        axis = 1 : i64
    } : (tensor<1x8x64xf32>, tensor<1x8x64xf32>,
         tensor<1x1x64xf32>, tensor<1x1x64xf32>,
         tensor<1x8xf32>, tensor<1xf32>)
          -> (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1xf32>)
        loc("kvcache_update")

    // -------------------------------------------------------------------------
    // Flash attention (decode: mq=1, mk=8 = full history cache)
    // -------------------------------------------------------------------------
    %none = "top.None"() : () -> none loc("none")
    %attn = "top.FAttention"(%q_rope, %uk, %uv, %none, %none) {
        scale = 0.35355338 : f64,
        batch = 1 : i64,
        q_head = 8 : i64,
        kv_head = 8 : i64,
        dim = 8 : i64,
        mq = 1 : i64,
        mk = 8 : i64,
        keep_dims = true,
        causal = false,
        has_mask = false,
        block_size = 0 : i64,
        max_blocks_per_seq = 0 : i64,
        kv_cache_mode = "paged",
        operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0>
    } : (tensor<1x1x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>,
         none, none) -> tensor<1x1x64xf32>
        loc("fattention")

    // -------------------------------------------------------------------------
    // O projection
    // -------------------------------------------------------------------------
    %o_w   = "top.Input"(%arg14) {} : (tensor<64x64xi8>)   -> tensor<64x64xi8>   loc("o_w")
    %o_scl = "top.Input"(%arg15) {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("o_scl")
    %o_zp  = "top.Input"(%arg16) {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("o_zp")
    %o_b   = "top.Input"(%arg17) {} : (tensor<64xf32>)     -> tensor<64xf32>     loc("o_bias")
    %o_out = "top.A16MatMul"(%attn, %o_w, %o_scl, %o_zp, %o_b) {
        weight_bits = 8 : i64,
        q_group_size = 64 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x1x64xf32>, tensor<64x64xi8>,
         tensor<64x1xf32>, tensor<64x1xf32>, tensor<64xf32>)
          -> tensor<1x1x64xf32>
        loc("o_proj")

    // -------------------------------------------------------------------------
    // Attention residual
    // -------------------------------------------------------------------------
    %res1 = "top.Add"(%hidden, %o_out) {
        do_relu = false,
        relu_limit = -1.0 : f64,
        coeff = []
    } : (tensor<1x1x64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
        loc("res_attn")

    // -------------------------------------------------------------------------
    // MLP pre-norm
    // -------------------------------------------------------------------------
    %gamma2 = "top.Input"(%arg18) {} : (tensor<64xf32>)    -> tensor<64xf32>     loc("gamma2")
    %norm2 = "top.RMSNorm"(%res1, %gamma2) {
        eps = 1.0e-06 : f64,
        axis = -1 : i64,
        accumulate_dtype = "NONE",
        weight_keep_f32 = false
    } : (tensor<1x1x64xf32>, tensor<64xf32>) -> tensor<1x1x64xf32>
        loc("rmsnorm2")

    // -------------------------------------------------------------------------
    // SwiGLU MLP
    // -------------------------------------------------------------------------
    %gate_w   = "top.Input"(%arg19) {} : (tensor<64x128xi8>)  -> tensor<64x128xi8>  loc("gate_w")
    %gate_scl = "top.Input"(%arg20) {} : (tensor<128x1xf32>)  -> tensor<128x1xf32>  loc("gate_scl")
    %gate_zp  = "top.Input"(%arg21) {} : (tensor<128x1xf32>)  -> tensor<128x1xf32>  loc("gate_zp")
    %gate_b   = "top.Input"(%arg22) {} : (tensor<128xf32>)    -> tensor<128xf32>    loc("gate_bias")
    %gate = "top.A16MatMul"(%norm2, %gate_w, %gate_scl, %gate_zp, %gate_b) {
        weight_bits = 8 : i64,
        q_group_size = 64 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x1x64xf32>, tensor<64x128xi8>,
         tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>)
          -> tensor<1x1x128xf32>
        loc("gate_proj")

    %up_w   = "top.Input"(%arg23) {} : (tensor<64x128xi8>)   -> tensor<64x128xi8>   loc("up_w")
    %up_scl = "top.Input"(%arg24) {} : (tensor<128x1xf32>)   -> tensor<128x1xf32>   loc("up_scl")
    %up_zp  = "top.Input"(%arg25) {} : (tensor<128x1xf32>)   -> tensor<128x1xf32>   loc("up_zp")
    %up_b   = "top.Input"(%arg26) {} : (tensor<128xf32>)     -> tensor<128xf32>     loc("up_bias")
    %up = "top.A16MatMul"(%norm2, %up_w, %up_scl, %up_zp, %up_b) {
        weight_bits = 8 : i64,
        q_group_size = 64 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x1x64xf32>, tensor<64x128xi8>,
         tensor<128x1xf32>, tensor<128x1xf32>, tensor<128xf32>)
          -> tensor<1x1x128xf32>
        loc("up_proj")

    %gate_act = "top.SiLU"(%gate) {
    } : (tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
        loc("silu")

    %swiglu = "top.Mul"(%gate_act, %up) {
        do_relu = false,
        relu_limit = -1.0 : f64,
        coeff = []
    } : (tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
        loc("swiglu")

    %down_w   = "top.Input"(%arg27) {} : (tensor<128x64xi8>)  -> tensor<128x64xi8>  loc("down_w")
    %down_scl = "top.Input"(%arg28) {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("down_scl")
    %down_zp  = "top.Input"(%arg29) {} : (tensor<64x1xf32>)   -> tensor<64x1xf32>   loc("down_zp")
    %down_b   = "top.Input"(%arg30) {} : (tensor<64xf32>)     -> tensor<64xf32>     loc("down_bias")
    %mlp_out = "top.A16MatMul"(%swiglu, %down_w, %down_scl, %down_zp, %down_b) {
        weight_bits = 8 : i64,
        q_group_size = 128 : i64,
        has_bias = true,
        do_relu = false,
        relu_limit = -1.0 : f64,
        output_transpose = false,
        right_transpose = false,
        dq_type = "NONE"
    } : (tensor<1x1x128xf32>, tensor<128x64xi8>,
         tensor<64x1xf32>, tensor<64x1xf32>, tensor<64xf32>)
          -> tensor<1x1x64xf32>
        loc("down_proj")

    // -------------------------------------------------------------------------
    // MLP residual
    // -------------------------------------------------------------------------
    %res2 = "top.Add"(%res1, %mlp_out) {
        do_relu = false,
        relu_limit = -1.0 : f64,
        coeff = []
    } : (tensor<1x1x64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
        loc("res_mlp")

    return %res2 : tensor<1x1x64xf32>
  }
}
