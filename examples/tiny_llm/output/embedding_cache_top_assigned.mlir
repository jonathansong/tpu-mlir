#loc = loc(unknown)
module @embedding_cache attributes {module.addr_mode = "basic", module.chip = "ada300", module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.platform = "LLM_QUANTIZED", module.state = "TOP_F32", module.top_run_mode = "STATIC", module.weight_file = "embedding_cache_top_f32_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  func.func @main(%arg0: tensor<1x1xsi32> loc(unknown)) -> tensor<1x1x2048xf32> {
    %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x1xsi32>) -> tensor<1x1xf32> loc(#loc1)
    %1 = "top.Weight"() : () -> tensor<248320x2048xf32> loc(#loc2)
    %2 = "top.Gather"(%1, %0) {axis = 0 : si32, is_lora = false, is_scalar = false, keepdims = true} : (tensor<248320x2048xf32>, tensor<1x1xf32>) -> tensor<1x1x2048xf32> loc(#loc3)
    return %2 : tensor<1x1x2048xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("input_ids")
#loc2 = loc("model.language_model.embed_tokens.weight")
#loc3 = loc("embedding_cache")

