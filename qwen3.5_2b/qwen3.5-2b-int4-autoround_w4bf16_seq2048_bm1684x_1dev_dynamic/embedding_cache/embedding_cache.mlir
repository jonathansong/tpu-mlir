#loc = loc(unknown)
module @embedding_cache attributes {module.chip = "ALL", module.platform = "LLM_QUANTIZED", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "../embedding_top_weights.npz"} {
  func.func @main(%arg0: tensor<1x1xsi32> loc(unknown)) -> tensor<1x1x2048xf32> {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x1xsi32>) -> tensor<1x1xf32> loc(#loc1)
    %2 = "top.Weight"() : () -> tensor<248320x2048xf32> loc(#loc2)
    %3 = "top.Gather"(%2, %1) {axis = 0 : si32, is_lora = false, is_scalar = false, keepdims = true} : (tensor<248320x2048xf32>, tensor<1x1xf32>) -> tensor<1x1x2048xf32> loc(#loc3)
    return %3 : tensor<1x1x2048xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input_ids")
#loc2 = loc("model.language_model.embed_tokens.weight")
#loc3 = loc("embedding_cache")
