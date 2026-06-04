// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify that top.FAttention is lowered to tpu.FAttention in F16 mode for ada300.

// CHECK: func.func @main
// CHECK:   tpu.FAttention

module @ada300_fattention attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0: tensor<1x8x64xf32>  loc("q"),
      %arg1: tensor<1x8x64xf32>  loc("k"),
      %arg2: tensor<1x8x64xf32>  loc("v"))
      -> tensor<1x8x64xf32> {

    %q = "top.Input"(%arg0) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("q")
    %k = "top.Input"(%arg1) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("k")
    %v = "top.Input"(%arg2) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32> loc("v")
    // mask, buffer are none; cache_k/v/block_table/seq_lens are absent
    %none = "top.None"() : () -> none loc("none")

    %out = "top.FAttention"(%q, %k, %v, %none, %none) {
        scale = 0.125 : f64,
        batch = 1 : i64,
        q_head = 8 : i64,
        kv_head = 8 : i64,
        dim = 64 : i64,
        mq = 8 : i64,
        mk = 8 : i64,
        keep_dims = true,
        causal = true,
        has_mask = false,
        block_size = 0 : i64,
        max_blocks_per_seq = 0 : i64,
        kv_cache_mode = "paged",
        operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 0, 0>
    } : (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1x8x64xf32>, none, none)
          -> tensor<1x8x64xf32>
        loc("fattention")

    return %out : tensor<1x8x64xf32>
  }
}
