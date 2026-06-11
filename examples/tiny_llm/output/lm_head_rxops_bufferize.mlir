module @lm_head attributes {ada300.rxops_codegen = "full", ada300.rxops_placeholder_ops = 1 : i64, ada300.rxops_supported_ops = 7 : i64, module.FLOPs = 1017118720 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 1017118720 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 2007168 : i64, module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "lm_head_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  func.func private @rxops_bridge_topk_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_reshape_bytes(!llvm.ptr, !llvm.ptr, i64) -> i32
  func.func private @rxops_bridge_matmul_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @main(%arg0: memref<1x2048xf32, strided<[?, ?], offset: ?>>) -> memref<1x1xf32> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(496640 : i64) : i64
    %2 = llvm.mlir.constant(2048 : i64) : i64
    %3 = llvm.mlir.constant(248320 : i64) : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<248320x2048xf16>
    %alloc_0 = memref.alloc() : memref<1x2048xf32>
    memref.copy %arg0, %alloc_0 : memref<1x2048xf32, strided<[?, ?], offset: ?>> to memref<1x2048xf32>
    %alloc_1 = memref.alloc() : memref<248320x1xf16>
    %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<248320x1xf16> -> index
    %4 = arith.index_cast %intptr : index to i64
    %5 = llvm.inttoptr %4 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %alloc : memref<248320x2048xf16> -> index
    %6 = arith.index_cast %intptr_2 : index to i64
    %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<1x2048xf32> -> index
    %8 = arith.index_cast %intptr_3 : index to i64
    %9 = llvm.inttoptr %8 : i64 to !llvm.ptr
    %10 = call @rxops_bridge_matmul_f16(%5, %7, %9, %3, %2, %2) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_4 = memref.alloc() : memref<1x248320xf16>
    %intptr_5 = memref.extract_aligned_pointer_as_index %alloc_4 : memref<1x248320xf16> -> index
    %11 = arith.index_cast %intptr_5 : index to i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %intptr_6 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<248320x1xf16> -> index
    %13 = arith.index_cast %intptr_6 : index to i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %15 = call @rxops_bridge_reshape_bytes(%12, %14, %1) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_7 = memref.alloc() : memref<1x1xf16>
    %alloc_8 = memref.alloc() : memref<1x1xf32>
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_7 : memref<1x1xf16> -> index
    %16 = arith.index_cast %intptr_9 : index to i64
    %17 = llvm.inttoptr %16 : i64 to !llvm.ptr
    %intptr_10 = memref.extract_aligned_pointer_as_index %alloc_8 : memref<1x1xf32> -> index
    %18 = arith.index_cast %intptr_10 : index to i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %intptr_11 = memref.extract_aligned_pointer_as_index %alloc_4 : memref<1x248320xf16> -> index
    %20 = arith.index_cast %intptr_11 : index to i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %22 = call @rxops_bridge_topk_f16(%17, %19, %21, %0, %0, %3, %0) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    %cast = memref.cast %alloc_8 : memref<1x1xf32> to memref<1x1xf32, strided<[?, ?], offset: ?>>
    memref.dealloc %alloc : memref<248320x2048xf16>
    return %alloc_8 : memref<1x1xf32>
  }
}

