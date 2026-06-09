#loc2 = loc("input_ids")
module @tiny_embedding attributes {module.FLOPs = 8 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 0 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 320 : i64, module.platform = "ONNX", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.weight_file = "tiny_embedding_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  func.func @main(%arg0: tensor<1x8xf32, 0 : i64> loc("input_ids")) -> tensor<1x8xf32, 256 : i64> {
    %0 = "top.Input"(%arg0) {do_preprocess = false} : (tensor<1x8xf32, 0 : i64>) -> tensor<1x8xf32, 64 : i64> loc(#loc3)
    %1 = "tpu.Cast"(%0) {round_mode = #tpu<round_mode HalfAwayFromZero>, with_scale = true} : (tensor<1x8xf32, 64 : i64>) -> tensor<1x8xf16, 128 : i64> loc(#loc4)
    %2 = "tpu.Add"(%1, %1) {coeff = [], do_relu = false, is_scalar = false, relu_limit = -1.000000e+00 : f64} : (tensor<1x8xf16, 128 : i64>, tensor<1x8xf16, 128 : i64>) -> tensor<1x8xf16, 192 : i64> loc(#loc5)
    %3 = "tpu.Cast"(%2) {round_mode = #tpu<round_mode HalfAwayFromZero>, with_scale = true} : (tensor<1x8xf16, 192 : i64>) -> tensor<1x8xf32, 256 : i64> loc(#loc6)
    return %3 : tensor<1x8xf32, 256 : i64> loc(#loc7)
  } loc(#loc1)
} loc(#loc)
#loc = loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":1:1)
#loc1 = loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":6:3)
#loc3 = loc("embedding.input")
#loc4 = loc("embedding.inputembedding.add_f16")
#loc5 = loc("embedding.add")
#loc6 = loc("embedding.add_f32")
#loc7 = loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":10:5)

