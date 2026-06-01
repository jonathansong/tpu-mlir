#loc = loc(unknown)
#loc1 = loc("input_ids")
module @embedding attributes {module.FLOPs = 0 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "bm1684x", module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.inputs = ["input_ids"], module.mode = "BF16", module.outputs = ["embedding"], module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "embedding_tpu_addressed_bm1684x_bf16_weight.npz"} {
  module @embedding attributes {module.coeff_addr = 4294967296 : i64, module.coeff_size = 1017118720 : i64, module.device_id = 0 : i64, module.dynamic_coeff_offset = 1017118720 : i64, module.neuron_addr = 5312086016 : i64, module.neuron_size = 4198400 : i64, module.step = 0 : i64} {
    func.func @main(%arg0: tensor<1x1024xsi32> loc(unknown)) -> tensor<1x1024x2048xbf16, 5312090112 : i64> {
      %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x1024xsi32>) -> tensor<1x1024xsi32, 5312086016 : i64> loc(#loc1)
      %1 = call @subfunc_0(%0) : (tensor<1x1024xsi32, 5312086016 : i64>) -> tensor<1x1024x2048xbf16, 5312090112 : i64> loc(#loc)
      return %1 : tensor<1x1024x2048xbf16, 5312090112 : i64> loc(#loc)
    } loc(#loc)
    func.func @subfunc_0(%arg0: tensor<1x1024xsi32, 5312086016 : i64> loc("input_ids")) -> tensor<1x1024x2048xbf16, 5312090112 : i64> attributes {id = 0 : i64, mode = #tpu<run_mode TPU_STATIC>, next_index = array<i32: -1>} {
      %0 = "top.None"() : () -> none loc(#loc)
      %1 = "top.Weight"() {path = "tpu.Gather.source"} : () -> tensor<248320x2048xbf16, 4294967296 : i64> loc(#loc2)
      %2 = "tpu.Gather"(%1, %arg0, %0) {axis = 0 : si32, if_neg_index = true, is_lora = false, is_scalar = false, keepdims = true} : (tensor<248320x2048xbf16, 4294967296 : i64>, tensor<1x1024xsi32, 5312086016 : i64>, none) -> tensor<1x1024x2048xbf16, 5312090112 : i64> loc(#loc3)
      return %2 : tensor<1x1024x2048xbf16, 5312090112 : i64> loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("model.language_model.embed_tokens.weight_bf16")
#loc3 = loc("embedding")

