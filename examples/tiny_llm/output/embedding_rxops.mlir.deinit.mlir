#loc1 = loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":6:3)
module @tiny_embedding attributes {ada300.rxops_codegen = "minimal", ada300.rxops_placeholder_ops = 0 : i64, ada300.rxops_supported_ops = 4 : i64, module.FLOPs = 8 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 0 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 320 : i64, module.platform = "ONNX", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.weight_file = "tiny_embedding_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  func.func private @rxops_bridge_add_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32 loc(#loc)
  func.func @main(%arg0: tensor<1x8xf32> loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":6:3)) -> tensor<1x8xf32> {
    %alloc = memref.alloc() : memref<1x8xf32> loc(#loc2)
    %0 = bufferization.to_tensor %alloc restrict : memref<1x8xf32> loc(#loc2)
    return %0 : tensor<1x8xf32> loc(#loc3)
  } loc(#loc1)
} loc(#loc)
#loc = loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":1:1)
#loc2 = loc(unknown)
#loc3 = loc("/workspace/tpu-mlir/examples/tiny_llm/output/embedding_top.mlir":10:5)

