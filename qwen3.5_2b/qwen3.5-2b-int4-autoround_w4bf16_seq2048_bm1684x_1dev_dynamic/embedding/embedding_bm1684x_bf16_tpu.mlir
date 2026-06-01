#loc = loc(unknown)
module @embedding attributes {module.FLOPs = 0 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "bm1684x", module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "BF16", module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_LOWERED", module.top_run_mode = "STATIC", module.weight_file = "embedding_bm1684x_bf16_tpu_weights.npz"} {
  func.func @main(%arg0: tensor<1x1024xsi32> loc(unknown)) -> tensor<1x1024x2048xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x1024xsi32>) -> tensor<1x1024xsi32> loc(#loc1)
    %2 = "top.Weight"() : () -> tensor<248320x2048xbf16> loc(#loc2)
    %3 = "tpu.Gather"(%2, %1, %0) {axis = 0 : si32, if_neg_index = true, is_lora = false, is_scalar = false, keepdims = true} : (tensor<248320x2048xbf16>, tensor<1x1024xsi32>, none) -> tensor<1x1024x2048xbf16> loc(#loc3)
    %4 = "tpu.Cast"(%3) {round_mode = #tpu<round_mode HalfAwayFromZero>, with_scale = true} : (tensor<1x1024x2048xbf16>) -> tensor<1x1024x2048xf32> loc(#loc4)
    return %4 : tensor<1x1024x2048xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input_ids")
#loc2 = loc("model.language_model.embed_tokens.weight_bf16")
#loc3 = loc("embedding")
#loc4 = loc("embedding_f32")

