// RUN: tpuc-opt --processor-assign="chip=ada300 mode=F16" \
// RUN:   --init="freq=0 level=0" \
// RUN:   --convert-top-to-tpu \
// RUN:   %s | FileCheck %s

// Verify that top.KVCacheUpdate is lowered to tpu.KVCacheUpdate in F16 mode
// for ada300.  All operands are wrapped as top.Input (no weight files needed).

// CHECK: func.func @main
// CHECK:   tpu.KVCacheUpdate

module @ada300_kvcache_update attributes {
    module.chip = "ALL",
    module.platform = "ONNX",
    module.state = "TOP_F32",
    module.weight_file = ""} {
  func.func @main(
      %arg0: tensor<1x8x64xf32>  loc("cache_k"),
      %arg1: tensor<1x8x64xf32>  loc("cache_v"),
      %arg2: tensor<1x1x64xf32>  loc("key"),
      %arg3: tensor<1x1x64xf32>  loc("value"),
      %arg4: tensor<1x8xi32>     loc("block_table"),
      %arg5: tensor<1xi32>       loc("seq_lens"))
      -> tensor<1xi32> {

    %ck  = "top.Input"(%arg0) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
        loc("cache_k")
    %cv  = "top.Input"(%arg1) {} : (tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
        loc("cache_v")
    %key = "top.Input"(%arg2) {} : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
        loc("key")
    %val = "top.Input"(%arg3) {} : (tensor<1x1x64xf32>) -> tensor<1x1x64xf32>
        loc("value")
    %bt  = "top.Input"(%arg4) {} : (tensor<1x8xi32>)    -> tensor<1x8xi32>
        loc("block_table")
    %sl  = "top.Input"(%arg5) {} : (tensor<1xi32>)      -> tensor<1xi32>
        loc("seq_lens")

    %uk, %uv, %us = "top.KVCacheUpdate"(%ck, %cv, %key, %val, %bt, %sl) {
        kv_format = "KV8",
        mode = "paged",
        block_size = 8 : i64,
        max_blocks_per_seq = 1 : i64,
        axis = 1 : i64
    } : (tensor<1x8x64xf32>, tensor<1x8x64xf32>,
         tensor<1x1x64xf32>, tensor<1x1x64xf32>,
         tensor<1x8xi32>, tensor<1xi32>)
          -> (tensor<1x8x64xf32>, tensor<1x8x64xf32>, tensor<1xi32>)
        loc("kvcache_update")

    return %us : tensor<1xi32>
  }
}
