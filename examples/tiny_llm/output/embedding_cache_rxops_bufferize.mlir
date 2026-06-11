module @embedding_cache attributes {ada300.rxops_codegen = "full", ada300.rxops_placeholder_ops = 1 : i64, ada300.rxops_supported_ops = 6 : i64, module.FLOPs = 0 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 1017118720 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 12544 : i64, module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "embedding_cache_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  func.func private @rxops_bridge_gather_nd(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func @main(%arg0: memref<1x1xsi32, strided<[?, ?], offset: ?>>) -> memref<1x1x2048xf16> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(2048 : i64) : i64
    %2 = llvm.mlir.constant(248320 : i64) : i64
    %3 = llvm.mlir.constant(2 : i64) : i64
    %4 = llvm.mlir.constant(0 : i64) : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<248320x2048xf16>
    %alloc_0 = memref.alloc() : memref<1x1xsi32>
    memref.copy %arg0, %alloc_0 : memref<1x1xsi32, strided<[?, ?], offset: ?>> to memref<1x1xsi32>
    %alloc_1 = memref.alloc() : memref<1x1x2048xf16>
    %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1x2048xf16> -> index
    %5 = arith.index_cast %intptr : index to i64
    %6 = llvm.inttoptr %5 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %alloc : memref<248320x2048xf16> -> index
    %7 = arith.index_cast %intptr_2 : index to i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<1x1xsi32> -> index
    %9 = arith.index_cast %intptr_3 : index to i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %11 = call @rxops_bridge_gather_nd(%6, %8, %10, %4, %3, %2, %1, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %0, %1, %0, %0, %0) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %cast = memref.cast %alloc_1 : memref<1x1x2048xf16> to memref<1x1x2048xf16, strided<[?, ?, ?], offset: ?>>
    memref.dealloc %alloc : memref<248320x2048xf16>
    return %alloc_1 : memref<1x1x2048xf16>
  }
}

