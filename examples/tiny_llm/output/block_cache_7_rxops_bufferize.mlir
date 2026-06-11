module @block_cache_7 attributes {ada300.rxops_codegen = "full", ada300.rxops_placeholder_ops = 29 : i64, ada300.rxops_supported_ops = 59 : i64, module.FLOPs = 121716736 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 27715072 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 25358656 : i64, module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "block_cache_7_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  func.func private @rxops_bridge_silu_f16(!llvm.ptr, !llvm.ptr, i64) -> i32
  func.func private @rxops_bridge_add_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  func.func private @rxops_bridge_mul_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  func.func private @rxops_bridge_fattention_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_concat2_nd(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_rope_contiguous_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_transpose_nd(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_gather_nd(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_rms_norm_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func private @rxops_bridge_sigmoid_f16(!llvm.ptr, !llvm.ptr, i64) -> i32
  func.func private @rxops_bridge_slice_nd(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_reshape_bytes(!llvm.ptr, !llvm.ptr, i64) -> i32
  func.func private @rxops_bridge_a16matmul_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
  func.func private @rxops_bridge_rms_norm_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
  func.func @main(%arg0: memref<1x1x2048xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<3x1xsi32, strided<[?, ?], offset: ?>>, %arg2: memref<1x1x1x2049xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg3: memref<1x2048x2x256xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg4: memref<1x2048x2x256xf32, strided<[?, ?, ?, ?], offset: ?>>) -> (memref<1x1x2048xf16>, memref<1x1x2x256xf16>, memref<1x1x2x256xf16>) {
    %0 = llvm.mlir.constant(6144 : i64) : i64
    %1 = llvm.mlir.constant(4589168020290535424 : i64) : i64
    %2 = llvm.mlir.constant(2049 : i64) : i64
    %3 = llvm.mlir.constant(64 : i64) : i64
    %4 = llvm.mlir.constant(96 : i64) : i64
    %5 = llvm.mlir.constant(192 : i64) : i64
    %6 = llvm.mlir.constant(3 : i64) : i64
    %7 = llvm.mlir.constant(32 : i64) : i64
    %8 = llvm.mlir.constant(1024 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(512 : i64) : i64
    %11 = llvm.mlir.constant(256 : i64) : i64
    %12 = llvm.mlir.constant(8 : i64) : i64
    %13 = llvm.mlir.constant(2 : i64) : i64
    %14 = llvm.mlir.constant(8192 : i64) : i64
    %15 = llvm.mlir.constant(4 : i64) : i64
    %16 = llvm.mlir.constant(128 : i64) : i64
    %17 = llvm.mlir.constant(4096 : i64) : i64
    %18 = llvm.mlir.constant(4517329193108106637 : i64) : i64
    %19 = llvm.mlir.constant(2048 : i64) : i64
    %20 = llvm.mlir.constant(1 : i64) : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2048xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2048xf32>
    memref.copy %arg0, %alloc_0 : memref<1x1x2048xf32, strided<[?, ?, ?], offset: ?>> to memref<1x1x2048xf32>
    %alloc_1 = memref.alloc() : memref<1x1x2048xf16>
    %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1x2048xf16> -> index
    %21 = arith.index_cast %intptr : index to i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<1x1x2048xf32> -> index
    %23 = arith.index_cast %intptr_2 : index to i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc : memref<1x1x2048xf16> -> index
    %25 = arith.index_cast %intptr_3 : index to i64
    %26 = llvm.inttoptr %25 : i64 to !llvm.ptr
    %27 = call @rxops_bridge_rms_norm_f32(%22, %24, %26, %20, %19, %18) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4096x1024xui8>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4096x16xui8>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4096x16xf16>
    %alloc_7 = memref.alloc() : memref<1x1x4096xf16>
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloc_7 : memref<1x1x4096xf16> -> index
    %28 = arith.index_cast %intptr_8 : index to i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1x2048xf16> -> index
    %30 = arith.index_cast %intptr_9 : index to i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %intptr_10 = memref.extract_aligned_pointer_as_index %alloc_4 : memref<4096x1024xui8> -> index
    %32 = arith.index_cast %intptr_10 : index to i64
    %33 = llvm.inttoptr %32 : i64 to !llvm.ptr
    %intptr_11 = memref.extract_aligned_pointer_as_index %alloc_6 : memref<4096x16xf16> -> index
    %34 = arith.index_cast %intptr_11 : index to i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %intptr_12 = memref.extract_aligned_pointer_as_index %alloc_5 : memref<4096x16xui8> -> index
    %36 = arith.index_cast %intptr_12 : index to i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr
    %38 = call @rxops_bridge_a16matmul_f16(%29, %31, %33, %35, %37, %20, %19, %17, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_13 = memref.alloc() : memref<1x1x8x512xf16>
    %intptr_14 = memref.extract_aligned_pointer_as_index %alloc_13 : memref<1x1x8x512xf16> -> index
    %39 = arith.index_cast %intptr_14 : index to i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_7 : memref<1x1x4096xf16> -> index
    %41 = arith.index_cast %intptr_15 : index to i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    %43 = call @rxops_bridge_reshape_bytes(%40, %42, %14) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_16 = memref.alloc() : memref<1x1x8x256xf16>
    %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<1x1x8x256xf16> -> index
    %44 = arith.index_cast %intptr_17 : index to i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %alloc_13 : memref<1x1x8x512xf16> -> index
    %46 = arith.index_cast %intptr_18 : index to i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    %48 = call @rxops_bridge_slice_nd(%45, %47, %13, %15, %20, %20, %12, %11, %20, %20, %20, %20, %12, %10, %20, %20, %9, %9, %9, %9, %20, %20, %20, %20, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_19 = memref.alloc() : memref<1x1x8x256xf16>
    %intptr_20 = memref.extract_aligned_pointer_as_index %alloc_19 : memref<1x1x8x256xf16> -> index
    %49 = arith.index_cast %intptr_20 : index to i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_13 : memref<1x1x8x512xf16> -> index
    %51 = arith.index_cast %intptr_21 : index to i64
    %52 = llvm.inttoptr %51 : i64 to !llvm.ptr
    %53 = call @rxops_bridge_slice_nd(%50, %52, %13, %15, %20, %20, %12, %11, %20, %20, %20, %20, %12, %10, %20, %20, %9, %9, %9, %11, %20, %20, %20, %20, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_22 = memref.alloc() : memref<1x1x8x256xf16>
    %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<1x1x8x256xf16> -> index
    %54 = arith.index_cast %intptr_23 : index to i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    %intptr_24 = memref.extract_aligned_pointer_as_index %alloc_19 : memref<1x1x8x256xf16> -> index
    %56 = arith.index_cast %intptr_24 : index to i64
    %57 = llvm.inttoptr %56 : i64 to !llvm.ptr
    %58 = call @rxops_bridge_sigmoid_f16(%55, %57, %19) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xui8>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<512x16xui8>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<512x16xf16>
    %alloc_28 = memref.alloc() : memref<1x1x512xf16>
    %intptr_29 = memref.extract_aligned_pointer_as_index %alloc_28 : memref<1x1x512xf16> -> index
    %59 = arith.index_cast %intptr_29 : index to i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %intptr_30 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1x2048xf16> -> index
    %61 = arith.index_cast %intptr_30 : index to i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %intptr_31 = memref.extract_aligned_pointer_as_index %alloc_25 : memref<512x1024xui8> -> index
    %63 = arith.index_cast %intptr_31 : index to i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %intptr_32 = memref.extract_aligned_pointer_as_index %alloc_27 : memref<512x16xf16> -> index
    %65 = arith.index_cast %intptr_32 : index to i64
    %66 = llvm.inttoptr %65 : i64 to !llvm.ptr
    %intptr_33 = memref.extract_aligned_pointer_as_index %alloc_26 : memref<512x16xui8> -> index
    %67 = arith.index_cast %intptr_33 : index to i64
    %68 = llvm.inttoptr %67 : i64 to !llvm.ptr
    %69 = call @rxops_bridge_a16matmul_f16(%60, %62, %64, %66, %68, %20, %19, %10, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xui8>
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<512x16xui8>
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<512x16xf16>
    %alloc_37 = memref.alloc() : memref<1x1x512xf16>
    %intptr_38 = memref.extract_aligned_pointer_as_index %alloc_37 : memref<1x1x512xf16> -> index
    %70 = arith.index_cast %intptr_38 : index to i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    %intptr_39 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1x2048xf16> -> index
    %72 = arith.index_cast %intptr_39 : index to i64
    %73 = llvm.inttoptr %72 : i64 to !llvm.ptr
    %intptr_40 = memref.extract_aligned_pointer_as_index %alloc_34 : memref<512x1024xui8> -> index
    %74 = arith.index_cast %intptr_40 : index to i64
    %75 = llvm.inttoptr %74 : i64 to !llvm.ptr
    %intptr_41 = memref.extract_aligned_pointer_as_index %alloc_36 : memref<512x16xf16> -> index
    %76 = arith.index_cast %intptr_41 : index to i64
    %77 = llvm.inttoptr %76 : i64 to !llvm.ptr
    %intptr_42 = memref.extract_aligned_pointer_as_index %alloc_35 : memref<512x16xui8> -> index
    %78 = arith.index_cast %intptr_42 : index to i64
    %79 = llvm.inttoptr %78 : i64 to !llvm.ptr
    %80 = call @rxops_bridge_a16matmul_f16(%71, %73, %75, %77, %79, %20, %19, %10, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_43 = memref.alloc() : memref<1x1x2x256xf16>
    %intptr_44 = memref.extract_aligned_pointer_as_index %alloc_43 : memref<1x1x2x256xf16> -> index
    %81 = arith.index_cast %intptr_44 : index to i64
    %82 = llvm.inttoptr %81 : i64 to !llvm.ptr
    %intptr_45 = memref.extract_aligned_pointer_as_index %alloc_28 : memref<1x1x512xf16> -> index
    %83 = arith.index_cast %intptr_45 : index to i64
    %84 = llvm.inttoptr %83 : i64 to !llvm.ptr
    %85 = call @rxops_bridge_reshape_bytes(%82, %84, %8) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_46 = memref.alloc() : memref<1x1x2x256xf16>
    %intptr_47 = memref.extract_aligned_pointer_as_index %alloc_46 : memref<1x1x2x256xf16> -> index
    %86 = arith.index_cast %intptr_47 : index to i64
    %87 = llvm.inttoptr %86 : i64 to !llvm.ptr
    %intptr_48 = memref.extract_aligned_pointer_as_index %alloc_37 : memref<1x1x512xf16> -> index
    %88 = arith.index_cast %intptr_48 : index to i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    %90 = call @rxops_bridge_reshape_bytes(%87, %89, %8) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x256xf16>
    %alloc_50 = memref.alloc() : memref<1x1x8x256xf16>
    %intptr_51 = memref.extract_aligned_pointer_as_index %alloc_50 : memref<1x1x8x256xf16> -> index
    %91 = arith.index_cast %intptr_51 : index to i64
    %92 = llvm.inttoptr %91 : i64 to !llvm.ptr
    %intptr_52 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<1x1x8x256xf16> -> index
    %93 = arith.index_cast %intptr_52 : index to i64
    %94 = llvm.inttoptr %93 : i64 to !llvm.ptr
    %intptr_53 = memref.extract_aligned_pointer_as_index %alloc_49 : memref<1x1x1x256xf16> -> index
    %95 = arith.index_cast %intptr_53 : index to i64
    %96 = llvm.inttoptr %95 : i64 to !llvm.ptr
    %97 = call @rxops_bridge_rms_norm_f16(%92, %94, %96, %12, %11, %18) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x256xf16>
    %alloc_55 = memref.alloc() : memref<1x1x2x256xf16>
    %intptr_56 = memref.extract_aligned_pointer_as_index %alloc_55 : memref<1x1x2x256xf16> -> index
    %98 = arith.index_cast %intptr_56 : index to i64
    %99 = llvm.inttoptr %98 : i64 to !llvm.ptr
    %intptr_57 = memref.extract_aligned_pointer_as_index %alloc_43 : memref<1x1x2x256xf16> -> index
    %100 = arith.index_cast %intptr_57 : index to i64
    %101 = llvm.inttoptr %100 : i64 to !llvm.ptr
    %intptr_58 = memref.extract_aligned_pointer_as_index %alloc_54 : memref<1x1x1x256xf16> -> index
    %102 = arith.index_cast %intptr_58 : index to i64
    %103 = llvm.inttoptr %102 : i64 to !llvm.ptr
    %104 = call @rxops_bridge_rms_norm_f16(%99, %101, %103, %13, %11, %18) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<2048x1x32xf16>
    %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<3x1xsi32>
    memref.copy %arg1, %alloc_60 : memref<3x1xsi32, strided<[?, ?], offset: ?>> to memref<3x1xsi32>
    %alloc_61 = memref.alloc() : memref<3x1x1x32xf16>
    %intptr_62 = memref.extract_aligned_pointer_as_index %alloc_61 : memref<3x1x1x32xf16> -> index
    %105 = arith.index_cast %intptr_62 : index to i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %intptr_63 = memref.extract_aligned_pointer_as_index %alloc_59 : memref<2048x1x32xf16> -> index
    %107 = arith.index_cast %intptr_63 : index to i64
    %108 = llvm.inttoptr %107 : i64 to !llvm.ptr
    %intptr_64 = memref.extract_aligned_pointer_as_index %alloc_60 : memref<3x1xsi32> -> index
    %109 = arith.index_cast %intptr_64 : index to i64
    %110 = llvm.inttoptr %109 : i64 to !llvm.ptr
    %111 = call @rxops_bridge_gather_nd(%106, %108, %110, %9, %13, %19, %20, %7, %20, %20, %20, %6, %20, %20, %20, %20, %20, %6, %20, %20, %7, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_65 = memref.alloc() : memref<3x32x1x1xf16>
    %intptr_66 = memref.extract_aligned_pointer_as_index %alloc_65 : memref<3x32x1x1xf16> -> index
    %112 = arith.index_cast %intptr_66 : index to i64
    %113 = llvm.inttoptr %112 : i64 to !llvm.ptr
    %intptr_67 = memref.extract_aligned_pointer_as_index %alloc_61 : memref<3x1x1x32xf16> -> index
    %114 = arith.index_cast %intptr_67 : index to i64
    %115 = llvm.inttoptr %114 : i64 to !llvm.ptr
    %116 = call @rxops_bridge_transpose_nd(%113, %115, %13, %15, %9, %6, %13, %20, %20, %20, %6, %20, %20, %7, %20, %20, %6, %7, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_68 = memref.alloc() : memref<96x1x1xf16>
    %intptr_69 = memref.extract_aligned_pointer_as_index %alloc_68 : memref<96x1x1xf16> -> index
    %117 = arith.index_cast %intptr_69 : index to i64
    %118 = llvm.inttoptr %117 : i64 to !llvm.ptr
    %intptr_70 = memref.extract_aligned_pointer_as_index %alloc_65 : memref<3x32x1x1xf16> -> index
    %119 = arith.index_cast %intptr_70 : index to i64
    %120 = llvm.inttoptr %119 : i64 to !llvm.ptr
    %121 = call @rxops_bridge_reshape_bytes(%118, %120, %5) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<1x64xi32>
    %alloc_72 = memref.alloc() : memref<1x64x1x1xf16>
    %intptr_73 = memref.extract_aligned_pointer_as_index %alloc_72 : memref<1x64x1x1xf16> -> index
    %122 = arith.index_cast %intptr_73 : index to i64
    %123 = llvm.inttoptr %122 : i64 to !llvm.ptr
    %intptr_74 = memref.extract_aligned_pointer_as_index %alloc_68 : memref<96x1x1xf16> -> index
    %124 = arith.index_cast %intptr_74 : index to i64
    %125 = llvm.inttoptr %124 : i64 to !llvm.ptr
    %intptr_75 = memref.extract_aligned_pointer_as_index %alloc_71 : memref<1x64xi32> -> index
    %126 = arith.index_cast %intptr_75 : index to i64
    %127 = llvm.inttoptr %126 : i64 to !llvm.ptr
    %128 = call @rxops_bridge_gather_nd(%123, %125, %127, %9, %13, %4, %20, %20, %20, %20, %20, %20, %3, %20, %20, %20, %20, %20, %3, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_76 = memref.alloc() : memref<1x1x1x64xf16>
    %intptr_77 = memref.extract_aligned_pointer_as_index %alloc_76 : memref<1x1x1x64xf16> -> index
    %129 = arith.index_cast %intptr_77 : index to i64
    %130 = llvm.inttoptr %129 : i64 to !llvm.ptr
    %intptr_78 = memref.extract_aligned_pointer_as_index %alloc_72 : memref<1x64x1x1xf16> -> index
    %131 = arith.index_cast %intptr_78 : index to i64
    %132 = llvm.inttoptr %131 : i64 to !llvm.ptr
    %133 = call @rxops_bridge_transpose_nd(%130, %132, %13, %15, %9, %6, %13, %20, %20, %20, %20, %3, %20, %20, %20, %20, %20, %20, %20, %3, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<2048x1x32xf16>
    %alloc_80 = memref.alloc() : memref<3x1xsi32>
    memref.copy %arg1, %alloc_80 : memref<3x1xsi32, strided<[?, ?], offset: ?>> to memref<3x1xsi32>
    %alloc_81 = memref.alloc() : memref<3x1x1x32xf16>
    %intptr_82 = memref.extract_aligned_pointer_as_index %alloc_81 : memref<3x1x1x32xf16> -> index
    %134 = arith.index_cast %intptr_82 : index to i64
    %135 = llvm.inttoptr %134 : i64 to !llvm.ptr
    %intptr_83 = memref.extract_aligned_pointer_as_index %alloc_79 : memref<2048x1x32xf16> -> index
    %136 = arith.index_cast %intptr_83 : index to i64
    %137 = llvm.inttoptr %136 : i64 to !llvm.ptr
    %intptr_84 = memref.extract_aligned_pointer_as_index %alloc_80 : memref<3x1xsi32> -> index
    %138 = arith.index_cast %intptr_84 : index to i64
    %139 = llvm.inttoptr %138 : i64 to !llvm.ptr
    %140 = call @rxops_bridge_gather_nd(%135, %137, %139, %9, %13, %19, %20, %7, %20, %20, %20, %6, %20, %20, %20, %20, %20, %6, %20, %20, %7, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_85 = memref.alloc() : memref<3x32x1x1xf16>
    %intptr_86 = memref.extract_aligned_pointer_as_index %alloc_85 : memref<3x32x1x1xf16> -> index
    %141 = arith.index_cast %intptr_86 : index to i64
    %142 = llvm.inttoptr %141 : i64 to !llvm.ptr
    %intptr_87 = memref.extract_aligned_pointer_as_index %alloc_81 : memref<3x1x1x32xf16> -> index
    %143 = arith.index_cast %intptr_87 : index to i64
    %144 = llvm.inttoptr %143 : i64 to !llvm.ptr
    %145 = call @rxops_bridge_transpose_nd(%142, %144, %13, %15, %9, %6, %13, %20, %20, %20, %6, %20, %20, %7, %20, %20, %6, %7, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_88 = memref.alloc() : memref<96x1x1xf16>
    %intptr_89 = memref.extract_aligned_pointer_as_index %alloc_88 : memref<96x1x1xf16> -> index
    %146 = arith.index_cast %intptr_89 : index to i64
    %147 = llvm.inttoptr %146 : i64 to !llvm.ptr
    %intptr_90 = memref.extract_aligned_pointer_as_index %alloc_85 : memref<3x32x1x1xf16> -> index
    %148 = arith.index_cast %intptr_90 : index to i64
    %149 = llvm.inttoptr %148 : i64 to !llvm.ptr
    %150 = call @rxops_bridge_reshape_bytes(%147, %149, %5) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<1x64xi32>
    %alloc_92 = memref.alloc() : memref<1x64x1x1xf16>
    %intptr_93 = memref.extract_aligned_pointer_as_index %alloc_92 : memref<1x64x1x1xf16> -> index
    %151 = arith.index_cast %intptr_93 : index to i64
    %152 = llvm.inttoptr %151 : i64 to !llvm.ptr
    %intptr_94 = memref.extract_aligned_pointer_as_index %alloc_88 : memref<96x1x1xf16> -> index
    %153 = arith.index_cast %intptr_94 : index to i64
    %154 = llvm.inttoptr %153 : i64 to !llvm.ptr
    %intptr_95 = memref.extract_aligned_pointer_as_index %alloc_91 : memref<1x64xi32> -> index
    %155 = arith.index_cast %intptr_95 : index to i64
    %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
    %157 = call @rxops_bridge_gather_nd(%152, %154, %156, %9, %13, %4, %20, %20, %20, %20, %20, %20, %3, %20, %20, %20, %20, %20, %3, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_96 = memref.alloc() : memref<1x1x1x64xf16>
    %intptr_97 = memref.extract_aligned_pointer_as_index %alloc_96 : memref<1x1x1x64xf16> -> index
    %158 = arith.index_cast %intptr_97 : index to i64
    %159 = llvm.inttoptr %158 : i64 to !llvm.ptr
    %intptr_98 = memref.extract_aligned_pointer_as_index %alloc_92 : memref<1x64x1x1xf16> -> index
    %160 = arith.index_cast %intptr_98 : index to i64
    %161 = llvm.inttoptr %160 : i64 to !llvm.ptr
    %162 = call @rxops_bridge_transpose_nd(%159, %161, %13, %15, %9, %6, %13, %20, %20, %20, %20, %3, %20, %20, %20, %20, %20, %20, %20, %3, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_99 = memref.alloc() : memref<1x1x8x64xf16>
    %intptr_100 = memref.extract_aligned_pointer_as_index %alloc_99 : memref<1x1x8x64xf16> -> index
    %163 = arith.index_cast %intptr_100 : index to i64
    %164 = llvm.inttoptr %163 : i64 to !llvm.ptr
    %intptr_101 = memref.extract_aligned_pointer_as_index %alloc_50 : memref<1x1x8x256xf16> -> index
    %165 = arith.index_cast %intptr_101 : index to i64
    %166 = llvm.inttoptr %165 : i64 to !llvm.ptr
    %167 = call @rxops_bridge_slice_nd(%164, %166, %13, %15, %20, %20, %12, %3, %20, %20, %20, %20, %12, %11, %20, %20, %9, %9, %9, %9, %20, %20, %20, %20, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_102 = memref.alloc() : memref<1x1x8x192xf16>
    %intptr_103 = memref.extract_aligned_pointer_as_index %alloc_102 : memref<1x1x8x192xf16> -> index
    %168 = arith.index_cast %intptr_103 : index to i64
    %169 = llvm.inttoptr %168 : i64 to !llvm.ptr
    %intptr_104 = memref.extract_aligned_pointer_as_index %alloc_50 : memref<1x1x8x256xf16> -> index
    %170 = arith.index_cast %intptr_104 : index to i64
    %171 = llvm.inttoptr %170 : i64 to !llvm.ptr
    %172 = call @rxops_bridge_slice_nd(%169, %171, %13, %15, %20, %20, %12, %5, %20, %20, %20, %20, %12, %11, %20, %20, %9, %9, %9, %3, %20, %20, %20, %20, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_105 = memref.alloc() : memref<1x1x8x64xf16>
    %intptr_106 = memref.extract_aligned_pointer_as_index %alloc_105 : memref<1x1x8x64xf16> -> index
    %173 = arith.index_cast %intptr_106 : index to i64
    %174 = llvm.inttoptr %173 : i64 to !llvm.ptr
    %intptr_107 = memref.extract_aligned_pointer_as_index %alloc_99 : memref<1x1x8x64xf16> -> index
    %175 = arith.index_cast %intptr_107 : index to i64
    %176 = llvm.inttoptr %175 : i64 to !llvm.ptr
    %intptr_108 = memref.extract_aligned_pointer_as_index %alloc_76 : memref<1x1x1x64xf16> -> index
    %177 = arith.index_cast %intptr_108 : index to i64
    %178 = llvm.inttoptr %177 : i64 to !llvm.ptr
    %intptr_109 = memref.extract_aligned_pointer_as_index %alloc_96 : memref<1x1x1x64xf16> -> index
    %179 = arith.index_cast %intptr_109 : index to i64
    %180 = llvm.inttoptr %179 : i64 to !llvm.ptr
    %181 = call @rxops_bridge_rope_contiguous_f16(%174, %176, %178, %180, %20, %12, %3, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    %alloc_110 = memref.alloc() : memref<1x1x8x256xf16>
    %intptr_111 = memref.extract_aligned_pointer_as_index %alloc_110 : memref<1x1x8x256xf16> -> index
    %182 = arith.index_cast %intptr_111 : index to i64
    %183 = llvm.inttoptr %182 : i64 to !llvm.ptr
    %intptr_112 = memref.extract_aligned_pointer_as_index %alloc_105 : memref<1x1x8x64xf16> -> index
    %184 = arith.index_cast %intptr_112 : index to i64
    %185 = llvm.inttoptr %184 : i64 to !llvm.ptr
    %intptr_113 = memref.extract_aligned_pointer_as_index %alloc_102 : memref<1x1x8x192xf16> -> index
    %186 = arith.index_cast %intptr_113 : index to i64
    %187 = llvm.inttoptr %186 : i64 to !llvm.ptr
    %188 = call @rxops_bridge_concat2_nd(%183, %185, %187, %13, %15, %6, %20, %20, %12, %11, %20, %20, %20, %20, %12, %3, %20, %20, %20, %20, %12, %5, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_114 = memref.alloc() : memref<1x1x2x64xf16>
    %intptr_115 = memref.extract_aligned_pointer_as_index %alloc_114 : memref<1x1x2x64xf16> -> index
    %189 = arith.index_cast %intptr_115 : index to i64
    %190 = llvm.inttoptr %189 : i64 to !llvm.ptr
    %intptr_116 = memref.extract_aligned_pointer_as_index %alloc_55 : memref<1x1x2x256xf16> -> index
    %191 = arith.index_cast %intptr_116 : index to i64
    %192 = llvm.inttoptr %191 : i64 to !llvm.ptr
    %193 = call @rxops_bridge_slice_nd(%190, %192, %13, %15, %20, %20, %13, %3, %20, %20, %20, %20, %13, %11, %20, %20, %9, %9, %9, %9, %20, %20, %20, %20, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_117 = memref.alloc() : memref<1x1x2x192xf16>
    %intptr_118 = memref.extract_aligned_pointer_as_index %alloc_117 : memref<1x1x2x192xf16> -> index
    %194 = arith.index_cast %intptr_118 : index to i64
    %195 = llvm.inttoptr %194 : i64 to !llvm.ptr
    %intptr_119 = memref.extract_aligned_pointer_as_index %alloc_55 : memref<1x1x2x256xf16> -> index
    %196 = arith.index_cast %intptr_119 : index to i64
    %197 = llvm.inttoptr %196 : i64 to !llvm.ptr
    %198 = call @rxops_bridge_slice_nd(%195, %197, %13, %15, %20, %20, %13, %5, %20, %20, %20, %20, %13, %11, %20, %20, %9, %9, %9, %3, %20, %20, %20, %20, %20, %20, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_120 = memref.alloc() : memref<1x1x2x64xf16>
    %intptr_121 = memref.extract_aligned_pointer_as_index %alloc_120 : memref<1x1x2x64xf16> -> index
    %199 = arith.index_cast %intptr_121 : index to i64
    %200 = llvm.inttoptr %199 : i64 to !llvm.ptr
    %intptr_122 = memref.extract_aligned_pointer_as_index %alloc_114 : memref<1x1x2x64xf16> -> index
    %201 = arith.index_cast %intptr_122 : index to i64
    %202 = llvm.inttoptr %201 : i64 to !llvm.ptr
    %intptr_123 = memref.extract_aligned_pointer_as_index %alloc_76 : memref<1x1x1x64xf16> -> index
    %203 = arith.index_cast %intptr_123 : index to i64
    %204 = llvm.inttoptr %203 : i64 to !llvm.ptr
    %intptr_124 = memref.extract_aligned_pointer_as_index %alloc_96 : memref<1x1x1x64xf16> -> index
    %205 = arith.index_cast %intptr_124 : index to i64
    %206 = llvm.inttoptr %205 : i64 to !llvm.ptr
    %207 = call @rxops_bridge_rope_contiguous_f16(%200, %202, %204, %206, %20, %13, %3, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    %alloc_125 = memref.alloc() : memref<1x1x2x256xf16>
    %intptr_126 = memref.extract_aligned_pointer_as_index %alloc_125 : memref<1x1x2x256xf16> -> index
    %208 = arith.index_cast %intptr_126 : index to i64
    %209 = llvm.inttoptr %208 : i64 to !llvm.ptr
    %intptr_127 = memref.extract_aligned_pointer_as_index %alloc_120 : memref<1x1x2x64xf16> -> index
    %210 = arith.index_cast %intptr_127 : index to i64
    %211 = llvm.inttoptr %210 : i64 to !llvm.ptr
    %intptr_128 = memref.extract_aligned_pointer_as_index %alloc_117 : memref<1x1x2x192xf16> -> index
    %212 = arith.index_cast %intptr_128 : index to i64
    %213 = llvm.inttoptr %212 : i64 to !llvm.ptr
    %214 = call @rxops_bridge_concat2_nd(%209, %211, %213, %13, %15, %6, %20, %20, %13, %11, %20, %20, %20, %20, %13, %3, %20, %20, %20, %20, %13, %5, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_129 = memref.alloc() : memref<1x2048x2x256xf32>
    memref.copy %arg3, %alloc_129 : memref<1x2048x2x256xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x2048x2x256xf32>
    %alloc_130 = memref.alloc() : memref<1x2049x2x256xf32>
    %intptr_131 = memref.extract_aligned_pointer_as_index %alloc_130 : memref<1x2049x2x256xf32> -> index
    %215 = arith.index_cast %intptr_131 : index to i64
    %216 = llvm.inttoptr %215 : i64 to !llvm.ptr
    %intptr_132 = memref.extract_aligned_pointer_as_index %alloc_129 : memref<1x2048x2x256xf32> -> index
    %217 = arith.index_cast %intptr_132 : index to i64
    %218 = llvm.inttoptr %217 : i64 to !llvm.ptr
    %intptr_133 = memref.extract_aligned_pointer_as_index %alloc_125 : memref<1x1x2x256xf16> -> index
    %219 = arith.index_cast %intptr_133 : index to i64
    %220 = llvm.inttoptr %219 : i64 to !llvm.ptr
    %221 = call @rxops_bridge_concat2_nd(%216, %218, %220, %15, %15, %20, %20, %2, %13, %11, %20, %20, %20, %19, %13, %11, %20, %20, %20, %20, %13, %11, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_134 = memref.alloc() : memref<1x2048x2x256xf32>
    memref.copy %arg4, %alloc_134 : memref<1x2048x2x256xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x2048x2x256xf32>
    %alloc_135 = memref.alloc() : memref<1x2049x2x256xf32>
    %intptr_136 = memref.extract_aligned_pointer_as_index %alloc_135 : memref<1x2049x2x256xf32> -> index
    %222 = arith.index_cast %intptr_136 : index to i64
    %223 = llvm.inttoptr %222 : i64 to !llvm.ptr
    %intptr_137 = memref.extract_aligned_pointer_as_index %alloc_134 : memref<1x2048x2x256xf32> -> index
    %224 = arith.index_cast %intptr_137 : index to i64
    %225 = llvm.inttoptr %224 : i64 to !llvm.ptr
    %intptr_138 = memref.extract_aligned_pointer_as_index %alloc_46 : memref<1x1x2x256xf16> -> index
    %226 = arith.index_cast %intptr_138 : index to i64
    %227 = llvm.inttoptr %226 : i64 to !llvm.ptr
    %228 = call @rxops_bridge_concat2_nd(%223, %225, %227, %15, %15, %20, %20, %2, %13, %11, %20, %20, %20, %19, %13, %11, %20, %20, %20, %20, %13, %11, %20, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_139 = memref.alloc() : memref<1x1x1x2049xf32>
    memref.copy %arg2, %alloc_139 : memref<1x1x1x2049xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x1x2049xf32>
    %alloc_140 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_141 = memref.extract_aligned_pointer_as_index %alloc_140 : memref<1x1x2048xf16> -> index
    %229 = arith.index_cast %intptr_141 : index to i64
    %230 = llvm.inttoptr %229 : i64 to !llvm.ptr
    %intptr_142 = memref.extract_aligned_pointer_as_index %alloc_110 : memref<1x1x8x256xf16> -> index
    %231 = arith.index_cast %intptr_142 : index to i64
    %232 = llvm.inttoptr %231 : i64 to !llvm.ptr
    %intptr_143 = memref.extract_aligned_pointer_as_index %alloc_130 : memref<1x2049x2x256xf32> -> index
    %233 = arith.index_cast %intptr_143 : index to i64
    %234 = llvm.inttoptr %233 : i64 to !llvm.ptr
    %intptr_144 = memref.extract_aligned_pointer_as_index %alloc_135 : memref<1x2049x2x256xf32> -> index
    %235 = arith.index_cast %intptr_144 : index to i64
    %236 = llvm.inttoptr %235 : i64 to !llvm.ptr
    %intptr_145 = memref.extract_aligned_pointer_as_index %alloc_139 : memref<1x1x1x2049xf32> -> index
    %237 = arith.index_cast %intptr_145 : index to i64
    %238 = llvm.inttoptr %237 : i64 to !llvm.ptr
    %239 = call @rxops_bridge_fattention_f16(%230, %232, %234, %236, %238, %20, %20, %2, %12, %13, %11, %1, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_146 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_147 = memref.extract_aligned_pointer_as_index %alloc_146 : memref<1x1x2048xf16> -> index
    %240 = arith.index_cast %intptr_147 : index to i64
    %241 = llvm.inttoptr %240 : i64 to !llvm.ptr
    %intptr_148 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<1x1x8x256xf16> -> index
    %242 = arith.index_cast %intptr_148 : index to i64
    %243 = llvm.inttoptr %242 : i64 to !llvm.ptr
    %244 = call @rxops_bridge_reshape_bytes(%241, %243, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_149 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_150 = memref.extract_aligned_pointer_as_index %alloc_149 : memref<1x1x2048xf16> -> index
    %245 = arith.index_cast %intptr_150 : index to i64
    %246 = llvm.inttoptr %245 : i64 to !llvm.ptr
    %intptr_151 = memref.extract_aligned_pointer_as_index %alloc_140 : memref<1x1x2048xf16> -> index
    %247 = arith.index_cast %intptr_151 : index to i64
    %248 = llvm.inttoptr %247 : i64 to !llvm.ptr
    %intptr_152 = memref.extract_aligned_pointer_as_index %alloc_146 : memref<1x1x2048xf16> -> index
    %249 = arith.index_cast %intptr_152 : index to i64
    %250 = llvm.inttoptr %249 : i64 to !llvm.ptr
    %251 = call @rxops_bridge_mul_f16(%246, %248, %250, %19) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_153 = memref.alloc() {alignment = 64 : i64} : memref<2048x1024xui8>
    %alloc_154 = memref.alloc() {alignment = 64 : i64} : memref<2048x16xui8>
    %alloc_155 = memref.alloc() {alignment = 64 : i64} : memref<2048x16xf16>
    %alloc_156 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_157 = memref.extract_aligned_pointer_as_index %alloc_156 : memref<1x1x2048xf16> -> index
    %252 = arith.index_cast %intptr_157 : index to i64
    %253 = llvm.inttoptr %252 : i64 to !llvm.ptr
    %intptr_158 = memref.extract_aligned_pointer_as_index %alloc_149 : memref<1x1x2048xf16> -> index
    %254 = arith.index_cast %intptr_158 : index to i64
    %255 = llvm.inttoptr %254 : i64 to !llvm.ptr
    %intptr_159 = memref.extract_aligned_pointer_as_index %alloc_153 : memref<2048x1024xui8> -> index
    %256 = arith.index_cast %intptr_159 : index to i64
    %257 = llvm.inttoptr %256 : i64 to !llvm.ptr
    %intptr_160 = memref.extract_aligned_pointer_as_index %alloc_155 : memref<2048x16xf16> -> index
    %258 = arith.index_cast %intptr_160 : index to i64
    %259 = llvm.inttoptr %258 : i64 to !llvm.ptr
    %intptr_161 = memref.extract_aligned_pointer_as_index %alloc_154 : memref<2048x16xui8> -> index
    %260 = arith.index_cast %intptr_161 : index to i64
    %261 = llvm.inttoptr %260 : i64 to !llvm.ptr
    %262 = call @rxops_bridge_a16matmul_f16(%253, %255, %257, %259, %261, %20, %19, %19, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_162 = memref.alloc() : memref<1x1x2048xf32>
    memref.copy %arg0, %alloc_162 : memref<1x1x2048xf32, strided<[?, ?, ?], offset: ?>> to memref<1x1x2048xf32>
    %alloc_163 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_164 = memref.extract_aligned_pointer_as_index %alloc_163 : memref<1x1x2048xf16> -> index
    %263 = arith.index_cast %intptr_164 : index to i64
    %264 = llvm.inttoptr %263 : i64 to !llvm.ptr
    %intptr_165 = memref.extract_aligned_pointer_as_index %alloc_162 : memref<1x1x2048xf32> -> index
    %265 = arith.index_cast %intptr_165 : index to i64
    %266 = llvm.inttoptr %265 : i64 to !llvm.ptr
    %intptr_166 = memref.extract_aligned_pointer_as_index %alloc_156 : memref<1x1x2048xf16> -> index
    %267 = arith.index_cast %intptr_166 : index to i64
    %268 = llvm.inttoptr %267 : i64 to !llvm.ptr
    %269 = call @rxops_bridge_add_f16(%264, %266, %268, %19) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_167 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2048xf16>
    %alloc_168 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_169 = memref.extract_aligned_pointer_as_index %alloc_168 : memref<1x1x2048xf16> -> index
    %270 = arith.index_cast %intptr_169 : index to i64
    %271 = llvm.inttoptr %270 : i64 to !llvm.ptr
    %intptr_170 = memref.extract_aligned_pointer_as_index %alloc_163 : memref<1x1x2048xf16> -> index
    %272 = arith.index_cast %intptr_170 : index to i64
    %273 = llvm.inttoptr %272 : i64 to !llvm.ptr
    %intptr_171 = memref.extract_aligned_pointer_as_index %alloc_167 : memref<1x1x2048xf16> -> index
    %274 = arith.index_cast %intptr_171 : index to i64
    %275 = llvm.inttoptr %274 : i64 to !llvm.ptr
    %276 = call @rxops_bridge_rms_norm_f16(%271, %273, %275, %20, %19, %18) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_172 = memref.alloc() {alignment = 64 : i64} : memref<6144x1024xui8>
    %alloc_173 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xui8>
    %alloc_174 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xf16>
    %alloc_175 = memref.alloc() : memref<1x1x6144xf16>
    %intptr_176 = memref.extract_aligned_pointer_as_index %alloc_175 : memref<1x1x6144xf16> -> index
    %277 = arith.index_cast %intptr_176 : index to i64
    %278 = llvm.inttoptr %277 : i64 to !llvm.ptr
    %intptr_177 = memref.extract_aligned_pointer_as_index %alloc_168 : memref<1x1x2048xf16> -> index
    %279 = arith.index_cast %intptr_177 : index to i64
    %280 = llvm.inttoptr %279 : i64 to !llvm.ptr
    %intptr_178 = memref.extract_aligned_pointer_as_index %alloc_172 : memref<6144x1024xui8> -> index
    %281 = arith.index_cast %intptr_178 : index to i64
    %282 = llvm.inttoptr %281 : i64 to !llvm.ptr
    %intptr_179 = memref.extract_aligned_pointer_as_index %alloc_174 : memref<6144x16xf16> -> index
    %283 = arith.index_cast %intptr_179 : index to i64
    %284 = llvm.inttoptr %283 : i64 to !llvm.ptr
    %intptr_180 = memref.extract_aligned_pointer_as_index %alloc_173 : memref<6144x16xui8> -> index
    %285 = arith.index_cast %intptr_180 : index to i64
    %286 = llvm.inttoptr %285 : i64 to !llvm.ptr
    %287 = call @rxops_bridge_a16matmul_f16(%278, %280, %282, %284, %286, %20, %19, %0, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_181 = memref.alloc() : memref<1x1x6144xf16>
    %intptr_182 = memref.extract_aligned_pointer_as_index %alloc_181 : memref<1x1x6144xf16> -> index
    %288 = arith.index_cast %intptr_182 : index to i64
    %289 = llvm.inttoptr %288 : i64 to !llvm.ptr
    %intptr_183 = memref.extract_aligned_pointer_as_index %alloc_175 : memref<1x1x6144xf16> -> index
    %290 = arith.index_cast %intptr_183 : index to i64
    %291 = llvm.inttoptr %290 : i64 to !llvm.ptr
    %292 = call @rxops_bridge_silu_f16(%289, %291, %0) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_184 = memref.alloc() {alignment = 64 : i64} : memref<6144x1024xui8>
    %alloc_185 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xui8>
    %alloc_186 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xf16>
    %alloc_187 = memref.alloc() : memref<1x1x6144xf16>
    %intptr_188 = memref.extract_aligned_pointer_as_index %alloc_187 : memref<1x1x6144xf16> -> index
    %293 = arith.index_cast %intptr_188 : index to i64
    %294 = llvm.inttoptr %293 : i64 to !llvm.ptr
    %intptr_189 = memref.extract_aligned_pointer_as_index %alloc_168 : memref<1x1x2048xf16> -> index
    %295 = arith.index_cast %intptr_189 : index to i64
    %296 = llvm.inttoptr %295 : i64 to !llvm.ptr
    %intptr_190 = memref.extract_aligned_pointer_as_index %alloc_184 : memref<6144x1024xui8> -> index
    %297 = arith.index_cast %intptr_190 : index to i64
    %298 = llvm.inttoptr %297 : i64 to !llvm.ptr
    %intptr_191 = memref.extract_aligned_pointer_as_index %alloc_186 : memref<6144x16xf16> -> index
    %299 = arith.index_cast %intptr_191 : index to i64
    %300 = llvm.inttoptr %299 : i64 to !llvm.ptr
    %intptr_192 = memref.extract_aligned_pointer_as_index %alloc_185 : memref<6144x16xui8> -> index
    %301 = arith.index_cast %intptr_192 : index to i64
    %302 = llvm.inttoptr %301 : i64 to !llvm.ptr
    %303 = call @rxops_bridge_a16matmul_f16(%294, %296, %298, %300, %302, %20, %19, %0, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_193 = memref.alloc() : memref<1x1x6144xf16>
    %intptr_194 = memref.extract_aligned_pointer_as_index %alloc_193 : memref<1x1x6144xf16> -> index
    %304 = arith.index_cast %intptr_194 : index to i64
    %305 = llvm.inttoptr %304 : i64 to !llvm.ptr
    %intptr_195 = memref.extract_aligned_pointer_as_index %alloc_181 : memref<1x1x6144xf16> -> index
    %306 = arith.index_cast %intptr_195 : index to i64
    %307 = llvm.inttoptr %306 : i64 to !llvm.ptr
    %intptr_196 = memref.extract_aligned_pointer_as_index %alloc_187 : memref<1x1x6144xf16> -> index
    %308 = arith.index_cast %intptr_196 : index to i64
    %309 = llvm.inttoptr %308 : i64 to !llvm.ptr
    %310 = call @rxops_bridge_mul_f16(%305, %307, %309, %0) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_197 = memref.alloc() {alignment = 64 : i64} : memref<2048x3072xui8>
    %alloc_198 = memref.alloc() {alignment = 64 : i64} : memref<2048x48xui8>
    %alloc_199 = memref.alloc() {alignment = 64 : i64} : memref<2048x48xf16>
    %alloc_200 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_201 = memref.extract_aligned_pointer_as_index %alloc_200 : memref<1x1x2048xf16> -> index
    %311 = arith.index_cast %intptr_201 : index to i64
    %312 = llvm.inttoptr %311 : i64 to !llvm.ptr
    %intptr_202 = memref.extract_aligned_pointer_as_index %alloc_193 : memref<1x1x6144xf16> -> index
    %313 = arith.index_cast %intptr_202 : index to i64
    %314 = llvm.inttoptr %313 : i64 to !llvm.ptr
    %intptr_203 = memref.extract_aligned_pointer_as_index %alloc_197 : memref<2048x3072xui8> -> index
    %315 = arith.index_cast %intptr_203 : index to i64
    %316 = llvm.inttoptr %315 : i64 to !llvm.ptr
    %intptr_204 = memref.extract_aligned_pointer_as_index %alloc_199 : memref<2048x48xf16> -> index
    %317 = arith.index_cast %intptr_204 : index to i64
    %318 = llvm.inttoptr %317 : i64 to !llvm.ptr
    %intptr_205 = memref.extract_aligned_pointer_as_index %alloc_198 : memref<2048x48xui8> -> index
    %319 = arith.index_cast %intptr_205 : index to i64
    %320 = llvm.inttoptr %319 : i64 to !llvm.ptr
    %321 = call @rxops_bridge_a16matmul_f16(%312, %314, %316, %318, %320, %20, %0, %19, %16, %15) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_206 = memref.alloc() : memref<1x1x2048xf16>
    %intptr_207 = memref.extract_aligned_pointer_as_index %alloc_206 : memref<1x1x2048xf16> -> index
    %322 = arith.index_cast %intptr_207 : index to i64
    %323 = llvm.inttoptr %322 : i64 to !llvm.ptr
    %intptr_208 = memref.extract_aligned_pointer_as_index %alloc_163 : memref<1x1x2048xf16> -> index
    %324 = arith.index_cast %intptr_208 : index to i64
    %325 = llvm.inttoptr %324 : i64 to !llvm.ptr
    %intptr_209 = memref.extract_aligned_pointer_as_index %alloc_200 : memref<1x1x2048xf16> -> index
    %326 = arith.index_cast %intptr_209 : index to i64
    %327 = llvm.inttoptr %326 : i64 to !llvm.ptr
    %328 = call @rxops_bridge_add_f16(%323, %325, %327, %19) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %cast = memref.cast %alloc_206 : memref<1x1x2048xf16> to memref<1x1x2048xf16, strided<[?, ?, ?], offset: ?>>
    %cast_210 = memref.cast %alloc_125 : memref<1x1x2x256xf16> to memref<1x1x2x256xf16, strided<[?, ?, ?, ?], offset: ?>>
    %cast_211 = memref.cast %alloc_46 : memref<1x1x2x256xf16> to memref<1x1x2x256xf16, strided<[?, ?, ?, ?], offset: ?>>
    memref.dealloc %alloc : memref<1x1x2048xf16>
    memref.dealloc %alloc_0 : memref<1x1x2048xf32>
    memref.dealloc %alloc_4 : memref<4096x1024xui8>
    memref.dealloc %alloc_5 : memref<4096x16xui8>
    memref.dealloc %alloc_6 : memref<4096x16xf16>
    memref.dealloc %alloc_25 : memref<512x1024xui8>
    memref.dealloc %alloc_26 : memref<512x16xui8>
    memref.dealloc %alloc_27 : memref<512x16xf16>
    memref.dealloc %alloc_34 : memref<512x1024xui8>
    memref.dealloc %alloc_35 : memref<512x16xui8>
    memref.dealloc %alloc_36 : memref<512x16xf16>
    memref.dealloc %alloc_49 : memref<1x1x1x256xf16>
    memref.dealloc %alloc_54 : memref<1x1x1x256xf16>
    memref.dealloc %alloc_59 : memref<2048x1x32xf16>
    memref.dealloc %alloc_60 : memref<3x1xsi32>
    memref.dealloc %alloc_71 : memref<1x64xi32>
    memref.dealloc %alloc_79 : memref<2048x1x32xf16>
    memref.dealloc %alloc_91 : memref<1x64xi32>
    memref.dealloc %alloc_153 : memref<2048x1024xui8>
    memref.dealloc %alloc_154 : memref<2048x16xui8>
    memref.dealloc %alloc_155 : memref<2048x16xf16>
    memref.dealloc %alloc_167 : memref<1x1x2048xf16>
    memref.dealloc %alloc_172 : memref<6144x1024xui8>
    memref.dealloc %alloc_173 : memref<6144x16xui8>
    memref.dealloc %alloc_174 : memref<6144x16xf16>
    memref.dealloc %alloc_184 : memref<6144x1024xui8>
    memref.dealloc %alloc_185 : memref<6144x16xui8>
    memref.dealloc %alloc_186 : memref<6144x16xf16>
    memref.dealloc %alloc_197 : memref<2048x3072xui8>
    memref.dealloc %alloc_198 : memref<2048x48xui8>
    memref.dealloc %alloc_199 : memref<2048x48xf16>
    return %alloc_206, %alloc_125, %alloc_46 : memref<1x1x2048xf16>, memref<1x1x2x256xf16>, memref<1x1x2x256xf16>
  }
}

