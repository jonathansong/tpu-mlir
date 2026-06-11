module @block_7 attributes {ada300.rxops_codegen = "full", ada300.rxops_placeholder_ops = 29 : i64, ada300.rxops_supported_ops = 53 : i64, module.FLOPs = 116039614464 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 27715072 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 184467456 : i64, module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "block_7_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
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
  func.func @main(%arg0: memref<1x1024x2048xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<3x1024xsi32, strided<[?, ?], offset: ?>>, %arg2: memref<1x1x1024x1024xf32, strided<[?, ?, ?, ?], offset: ?>>) -> (memref<1x1024x2048xf16>, memref<1x1024x2x256xf16>, memref<1x1024x2x256xf16>) {
    %0 = llvm.mlir.constant(6291456 : i64) : i64
    %1 = llvm.mlir.constant(6144 : i64) : i64
    %2 = llvm.mlir.constant(4194304 : i64) : i64
    %3 = llvm.mlir.constant(4589168020290535424 : i64) : i64
    %4 = llvm.mlir.constant(192 : i64) : i64
    %5 = llvm.mlir.constant(64 : i64) : i64
    %6 = llvm.mlir.constant(96 : i64) : i64
    %7 = llvm.mlir.constant(196608 : i64) : i64
    %8 = llvm.mlir.constant(3 : i64) : i64
    %9 = llvm.mlir.constant(32 : i64) : i64
    %10 = llvm.mlir.constant(8192 : i64) : i64
    %11 = llvm.mlir.constant(1048576 : i64) : i64
    %12 = llvm.mlir.constant(2097152 : i64) : i64
    %13 = llvm.mlir.constant(0 : i64) : i64
    %14 = llvm.mlir.constant(512 : i64) : i64
    %15 = llvm.mlir.constant(256 : i64) : i64
    %16 = llvm.mlir.constant(8 : i64) : i64
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.mlir.constant(2 : i64) : i64
    %19 = llvm.mlir.constant(8388608 : i64) : i64
    %20 = llvm.mlir.constant(4 : i64) : i64
    %21 = llvm.mlir.constant(128 : i64) : i64
    %22 = llvm.mlir.constant(4096 : i64) : i64
    %23 = llvm.mlir.constant(4517329193108106637 : i64) : i64
    %24 = llvm.mlir.constant(2048 : i64) : i64
    %25 = llvm.mlir.constant(1024 : i64) : i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2048xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x2048xf32>
    memref.copy %arg0, %alloc_0 : memref<1x1024x2048xf32, strided<[?, ?, ?], offset: ?>> to memref<1x1024x2048xf32>
    %alloc_1 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1024x2048xf16> -> index
    %26 = arith.index_cast %intptr : index to i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    %intptr_2 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<1x1024x2048xf32> -> index
    %28 = arith.index_cast %intptr_2 : index to i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloc : memref<1x1x2048xf16> -> index
    %30 = arith.index_cast %intptr_3 : index to i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = call @rxops_bridge_rms_norm_f32(%27, %29, %31, %25, %24, %23) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4096x1024xui8>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4096x16xui8>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<4096x16xf16>
    %alloc_7 = memref.alloc() : memref<1x1024x4096xf16>
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloc_7 : memref<1x1024x4096xf16> -> index
    %33 = arith.index_cast %intptr_8 : index to i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1024x2048xf16> -> index
    %35 = arith.index_cast %intptr_9 : index to i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %intptr_10 = memref.extract_aligned_pointer_as_index %alloc_4 : memref<4096x1024xui8> -> index
    %37 = arith.index_cast %intptr_10 : index to i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %intptr_11 = memref.extract_aligned_pointer_as_index %alloc_6 : memref<4096x16xf16> -> index
    %39 = arith.index_cast %intptr_11 : index to i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %intptr_12 = memref.extract_aligned_pointer_as_index %alloc_5 : memref<4096x16xui8> -> index
    %41 = arith.index_cast %intptr_12 : index to i64
    %42 = llvm.inttoptr %41 : i64 to !llvm.ptr
    %43 = call @rxops_bridge_a16matmul_f16(%34, %36, %38, %40, %42, %25, %24, %22, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_13 = memref.alloc() : memref<1x1024x8x512xf16>
    %intptr_14 = memref.extract_aligned_pointer_as_index %alloc_13 : memref<1x1024x8x512xf16> -> index
    %44 = arith.index_cast %intptr_14 : index to i64
    %45 = llvm.inttoptr %44 : i64 to !llvm.ptr
    %intptr_15 = memref.extract_aligned_pointer_as_index %alloc_7 : memref<1x1024x4096xf16> -> index
    %46 = arith.index_cast %intptr_15 : index to i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    %48 = call @rxops_bridge_reshape_bytes(%45, %47, %19) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_16 = memref.alloc() : memref<1x1024x8x256xf16>
    %intptr_17 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<1x1024x8x256xf16> -> index
    %49 = arith.index_cast %intptr_17 : index to i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %intptr_18 = memref.extract_aligned_pointer_as_index %alloc_13 : memref<1x1024x8x512xf16> -> index
    %51 = arith.index_cast %intptr_18 : index to i64
    %52 = llvm.inttoptr %51 : i64 to !llvm.ptr
    %53 = call @rxops_bridge_slice_nd(%50, %52, %18, %20, %17, %25, %16, %15, %17, %17, %17, %25, %16, %14, %17, %17, %13, %13, %13, %13, %17, %17, %17, %17, %17, %17, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_19 = memref.alloc() : memref<1x1024x8x256xf16>
    %intptr_20 = memref.extract_aligned_pointer_as_index %alloc_19 : memref<1x1024x8x256xf16> -> index
    %54 = arith.index_cast %intptr_20 : index to i64
    %55 = llvm.inttoptr %54 : i64 to !llvm.ptr
    %intptr_21 = memref.extract_aligned_pointer_as_index %alloc_13 : memref<1x1024x8x512xf16> -> index
    %56 = arith.index_cast %intptr_21 : index to i64
    %57 = llvm.inttoptr %56 : i64 to !llvm.ptr
    %58 = call @rxops_bridge_slice_nd(%55, %57, %18, %20, %17, %25, %16, %15, %17, %17, %17, %25, %16, %14, %17, %17, %13, %13, %13, %15, %17, %17, %17, %17, %17, %17, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_22 = memref.alloc() : memref<1x1024x8x256xf16>
    %intptr_23 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<1x1024x8x256xf16> -> index
    %59 = arith.index_cast %intptr_23 : index to i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %intptr_24 = memref.extract_aligned_pointer_as_index %alloc_19 : memref<1x1024x8x256xf16> -> index
    %61 = arith.index_cast %intptr_24 : index to i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = call @rxops_bridge_sigmoid_f16(%60, %62, %12) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xui8>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<512x16xui8>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<512x16xf16>
    %alloc_28 = memref.alloc() : memref<1x1024x512xf16>
    %intptr_29 = memref.extract_aligned_pointer_as_index %alloc_28 : memref<1x1024x512xf16> -> index
    %64 = arith.index_cast %intptr_29 : index to i64
    %65 = llvm.inttoptr %64 : i64 to !llvm.ptr
    %intptr_30 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1024x2048xf16> -> index
    %66 = arith.index_cast %intptr_30 : index to i64
    %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
    %intptr_31 = memref.extract_aligned_pointer_as_index %alloc_25 : memref<512x1024xui8> -> index
    %68 = arith.index_cast %intptr_31 : index to i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    %intptr_32 = memref.extract_aligned_pointer_as_index %alloc_27 : memref<512x16xf16> -> index
    %70 = arith.index_cast %intptr_32 : index to i64
    %71 = llvm.inttoptr %70 : i64 to !llvm.ptr
    %intptr_33 = memref.extract_aligned_pointer_as_index %alloc_26 : memref<512x16xui8> -> index
    %72 = arith.index_cast %intptr_33 : index to i64
    %73 = llvm.inttoptr %72 : i64 to !llvm.ptr
    %74 = call @rxops_bridge_a16matmul_f16(%65, %67, %69, %71, %73, %25, %24, %14, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xui8>
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<512x16xui8>
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<512x16xf16>
    %alloc_37 = memref.alloc() : memref<1x1024x512xf16>
    %intptr_38 = memref.extract_aligned_pointer_as_index %alloc_37 : memref<1x1024x512xf16> -> index
    %75 = arith.index_cast %intptr_38 : index to i64
    %76 = llvm.inttoptr %75 : i64 to !llvm.ptr
    %intptr_39 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<1x1024x2048xf16> -> index
    %77 = arith.index_cast %intptr_39 : index to i64
    %78 = llvm.inttoptr %77 : i64 to !llvm.ptr
    %intptr_40 = memref.extract_aligned_pointer_as_index %alloc_34 : memref<512x1024xui8> -> index
    %79 = arith.index_cast %intptr_40 : index to i64
    %80 = llvm.inttoptr %79 : i64 to !llvm.ptr
    %intptr_41 = memref.extract_aligned_pointer_as_index %alloc_36 : memref<512x16xf16> -> index
    %81 = arith.index_cast %intptr_41 : index to i64
    %82 = llvm.inttoptr %81 : i64 to !llvm.ptr
    %intptr_42 = memref.extract_aligned_pointer_as_index %alloc_35 : memref<512x16xui8> -> index
    %83 = arith.index_cast %intptr_42 : index to i64
    %84 = llvm.inttoptr %83 : i64 to !llvm.ptr
    %85 = call @rxops_bridge_a16matmul_f16(%76, %78, %80, %82, %84, %25, %24, %14, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_43 = memref.alloc() : memref<1x1024x2x256xf16>
    %intptr_44 = memref.extract_aligned_pointer_as_index %alloc_43 : memref<1x1024x2x256xf16> -> index
    %86 = arith.index_cast %intptr_44 : index to i64
    %87 = llvm.inttoptr %86 : i64 to !llvm.ptr
    %intptr_45 = memref.extract_aligned_pointer_as_index %alloc_28 : memref<1x1024x512xf16> -> index
    %88 = arith.index_cast %intptr_45 : index to i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    %90 = call @rxops_bridge_reshape_bytes(%87, %89, %11) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_46 = memref.alloc() : memref<1x1024x2x256xf16>
    %intptr_47 = memref.extract_aligned_pointer_as_index %alloc_46 : memref<1x1024x2x256xf16> -> index
    %91 = arith.index_cast %intptr_47 : index to i64
    %92 = llvm.inttoptr %91 : i64 to !llvm.ptr
    %intptr_48 = memref.extract_aligned_pointer_as_index %alloc_37 : memref<1x1024x512xf16> -> index
    %93 = arith.index_cast %intptr_48 : index to i64
    %94 = llvm.inttoptr %93 : i64 to !llvm.ptr
    %95 = call @rxops_bridge_reshape_bytes(%92, %94, %11) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x256xf16>
    %alloc_50 = memref.alloc() : memref<1x1024x8x256xf16>
    %intptr_51 = memref.extract_aligned_pointer_as_index %alloc_50 : memref<1x1024x8x256xf16> -> index
    %96 = arith.index_cast %intptr_51 : index to i64
    %97 = llvm.inttoptr %96 : i64 to !llvm.ptr
    %intptr_52 = memref.extract_aligned_pointer_as_index %alloc_16 : memref<1x1024x8x256xf16> -> index
    %98 = arith.index_cast %intptr_52 : index to i64
    %99 = llvm.inttoptr %98 : i64 to !llvm.ptr
    %intptr_53 = memref.extract_aligned_pointer_as_index %alloc_49 : memref<1x1x1x256xf16> -> index
    %100 = arith.index_cast %intptr_53 : index to i64
    %101 = llvm.inttoptr %100 : i64 to !llvm.ptr
    %102 = call @rxops_bridge_rms_norm_f16(%97, %99, %101, %10, %15, %23) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1x256xf16>
    %alloc_55 = memref.alloc() : memref<1x1024x2x256xf16>
    %intptr_56 = memref.extract_aligned_pointer_as_index %alloc_55 : memref<1x1024x2x256xf16> -> index
    %103 = arith.index_cast %intptr_56 : index to i64
    %104 = llvm.inttoptr %103 : i64 to !llvm.ptr
    %intptr_57 = memref.extract_aligned_pointer_as_index %alloc_43 : memref<1x1024x2x256xf16> -> index
    %105 = arith.index_cast %intptr_57 : index to i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %intptr_58 = memref.extract_aligned_pointer_as_index %alloc_54 : memref<1x1x1x256xf16> -> index
    %107 = arith.index_cast %intptr_58 : index to i64
    %108 = llvm.inttoptr %107 : i64 to !llvm.ptr
    %109 = call @rxops_bridge_rms_norm_f16(%104, %106, %108, %24, %15, %23) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<2048x1x32xf16>
    %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<3x1024xsi32>
    memref.copy %arg1, %alloc_60 : memref<3x1024xsi32, strided<[?, ?], offset: ?>> to memref<3x1024xsi32>
    %alloc_61 = memref.alloc() : memref<3x1024x1x32xf16>
    %intptr_62 = memref.extract_aligned_pointer_as_index %alloc_61 : memref<3x1024x1x32xf16> -> index
    %110 = arith.index_cast %intptr_62 : index to i64
    %111 = llvm.inttoptr %110 : i64 to !llvm.ptr
    %intptr_63 = memref.extract_aligned_pointer_as_index %alloc_59 : memref<2048x1x32xf16> -> index
    %112 = arith.index_cast %intptr_63 : index to i64
    %113 = llvm.inttoptr %112 : i64 to !llvm.ptr
    %intptr_64 = memref.extract_aligned_pointer_as_index %alloc_60 : memref<3x1024xsi32> -> index
    %114 = arith.index_cast %intptr_64 : index to i64
    %115 = llvm.inttoptr %114 : i64 to !llvm.ptr
    %116 = call @rxops_bridge_gather_nd(%111, %113, %115, %13, %18, %24, %17, %9, %17, %17, %17, %8, %25, %17, %17, %17, %17, %8, %25, %17, %9, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_65 = memref.alloc() : memref<3x32x1x1024xf16>
    %intptr_66 = memref.extract_aligned_pointer_as_index %alloc_65 : memref<3x32x1x1024xf16> -> index
    %117 = arith.index_cast %intptr_66 : index to i64
    %118 = llvm.inttoptr %117 : i64 to !llvm.ptr
    %intptr_67 = memref.extract_aligned_pointer_as_index %alloc_61 : memref<3x1024x1x32xf16> -> index
    %119 = arith.index_cast %intptr_67 : index to i64
    %120 = llvm.inttoptr %119 : i64 to !llvm.ptr
    %121 = call @rxops_bridge_transpose_nd(%118, %120, %18, %20, %13, %8, %18, %17, %17, %17, %8, %25, %17, %9, %17, %17, %8, %9, %17, %25, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_68 = memref.alloc() : memref<96x1x1024xf16>
    %intptr_69 = memref.extract_aligned_pointer_as_index %alloc_68 : memref<96x1x1024xf16> -> index
    %122 = arith.index_cast %intptr_69 : index to i64
    %123 = llvm.inttoptr %122 : i64 to !llvm.ptr
    %intptr_70 = memref.extract_aligned_pointer_as_index %alloc_65 : memref<3x32x1x1024xf16> -> index
    %124 = arith.index_cast %intptr_70 : index to i64
    %125 = llvm.inttoptr %124 : i64 to !llvm.ptr
    %126 = call @rxops_bridge_reshape_bytes(%123, %125, %7) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_71 = memref.alloc() {alignment = 64 : i64} : memref<1x64xi32>
    %alloc_72 = memref.alloc() : memref<1x64x1x1024xf16>
    %intptr_73 = memref.extract_aligned_pointer_as_index %alloc_72 : memref<1x64x1x1024xf16> -> index
    %127 = arith.index_cast %intptr_73 : index to i64
    %128 = llvm.inttoptr %127 : i64 to !llvm.ptr
    %intptr_74 = memref.extract_aligned_pointer_as_index %alloc_68 : memref<96x1x1024xf16> -> index
    %129 = arith.index_cast %intptr_74 : index to i64
    %130 = llvm.inttoptr %129 : i64 to !llvm.ptr
    %intptr_75 = memref.extract_aligned_pointer_as_index %alloc_71 : memref<1x64xi32> -> index
    %131 = arith.index_cast %intptr_75 : index to i64
    %132 = llvm.inttoptr %131 : i64 to !llvm.ptr
    %133 = call @rxops_bridge_gather_nd(%128, %130, %132, %13, %18, %6, %17, %25, %17, %17, %17, %17, %5, %17, %17, %17, %17, %17, %5, %17, %25, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_76 = memref.alloc() : memref<1x1024x1x64xf16>
    %intptr_77 = memref.extract_aligned_pointer_as_index %alloc_76 : memref<1x1024x1x64xf16> -> index
    %134 = arith.index_cast %intptr_77 : index to i64
    %135 = llvm.inttoptr %134 : i64 to !llvm.ptr
    %intptr_78 = memref.extract_aligned_pointer_as_index %alloc_72 : memref<1x64x1x1024xf16> -> index
    %136 = arith.index_cast %intptr_78 : index to i64
    %137 = llvm.inttoptr %136 : i64 to !llvm.ptr
    %138 = call @rxops_bridge_transpose_nd(%135, %137, %18, %20, %13, %8, %18, %17, %17, %17, %17, %5, %17, %25, %17, %17, %17, %25, %17, %5, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_79 = memref.alloc() {alignment = 64 : i64} : memref<2048x1x32xf16>
    %alloc_80 = memref.alloc() : memref<3x1024xsi32>
    memref.copy %arg1, %alloc_80 : memref<3x1024xsi32, strided<[?, ?], offset: ?>> to memref<3x1024xsi32>
    %alloc_81 = memref.alloc() : memref<3x1024x1x32xf16>
    %intptr_82 = memref.extract_aligned_pointer_as_index %alloc_81 : memref<3x1024x1x32xf16> -> index
    %139 = arith.index_cast %intptr_82 : index to i64
    %140 = llvm.inttoptr %139 : i64 to !llvm.ptr
    %intptr_83 = memref.extract_aligned_pointer_as_index %alloc_79 : memref<2048x1x32xf16> -> index
    %141 = arith.index_cast %intptr_83 : index to i64
    %142 = llvm.inttoptr %141 : i64 to !llvm.ptr
    %intptr_84 = memref.extract_aligned_pointer_as_index %alloc_80 : memref<3x1024xsi32> -> index
    %143 = arith.index_cast %intptr_84 : index to i64
    %144 = llvm.inttoptr %143 : i64 to !llvm.ptr
    %145 = call @rxops_bridge_gather_nd(%140, %142, %144, %13, %18, %24, %17, %9, %17, %17, %17, %8, %25, %17, %17, %17, %17, %8, %25, %17, %9, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_85 = memref.alloc() : memref<3x32x1x1024xf16>
    %intptr_86 = memref.extract_aligned_pointer_as_index %alloc_85 : memref<3x32x1x1024xf16> -> index
    %146 = arith.index_cast %intptr_86 : index to i64
    %147 = llvm.inttoptr %146 : i64 to !llvm.ptr
    %intptr_87 = memref.extract_aligned_pointer_as_index %alloc_81 : memref<3x1024x1x32xf16> -> index
    %148 = arith.index_cast %intptr_87 : index to i64
    %149 = llvm.inttoptr %148 : i64 to !llvm.ptr
    %150 = call @rxops_bridge_transpose_nd(%147, %149, %18, %20, %13, %8, %18, %17, %17, %17, %8, %25, %17, %9, %17, %17, %8, %9, %17, %25, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_88 = memref.alloc() : memref<96x1x1024xf16>
    %intptr_89 = memref.extract_aligned_pointer_as_index %alloc_88 : memref<96x1x1024xf16> -> index
    %151 = arith.index_cast %intptr_89 : index to i64
    %152 = llvm.inttoptr %151 : i64 to !llvm.ptr
    %intptr_90 = memref.extract_aligned_pointer_as_index %alloc_85 : memref<3x32x1x1024xf16> -> index
    %153 = arith.index_cast %intptr_90 : index to i64
    %154 = llvm.inttoptr %153 : i64 to !llvm.ptr
    %155 = call @rxops_bridge_reshape_bytes(%152, %154, %7) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_91 = memref.alloc() {alignment = 64 : i64} : memref<1x64xi32>
    %alloc_92 = memref.alloc() : memref<1x64x1x1024xf16>
    %intptr_93 = memref.extract_aligned_pointer_as_index %alloc_92 : memref<1x64x1x1024xf16> -> index
    %156 = arith.index_cast %intptr_93 : index to i64
    %157 = llvm.inttoptr %156 : i64 to !llvm.ptr
    %intptr_94 = memref.extract_aligned_pointer_as_index %alloc_88 : memref<96x1x1024xf16> -> index
    %158 = arith.index_cast %intptr_94 : index to i64
    %159 = llvm.inttoptr %158 : i64 to !llvm.ptr
    %intptr_95 = memref.extract_aligned_pointer_as_index %alloc_91 : memref<1x64xi32> -> index
    %160 = arith.index_cast %intptr_95 : index to i64
    %161 = llvm.inttoptr %160 : i64 to !llvm.ptr
    %162 = call @rxops_bridge_gather_nd(%157, %159, %161, %13, %18, %6, %17, %25, %17, %17, %17, %17, %5, %17, %17, %17, %17, %17, %5, %17, %25, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_96 = memref.alloc() : memref<1x1024x1x64xf16>
    %intptr_97 = memref.extract_aligned_pointer_as_index %alloc_96 : memref<1x1024x1x64xf16> -> index
    %163 = arith.index_cast %intptr_97 : index to i64
    %164 = llvm.inttoptr %163 : i64 to !llvm.ptr
    %intptr_98 = memref.extract_aligned_pointer_as_index %alloc_92 : memref<1x64x1x1024xf16> -> index
    %165 = arith.index_cast %intptr_98 : index to i64
    %166 = llvm.inttoptr %165 : i64 to !llvm.ptr
    %167 = call @rxops_bridge_transpose_nd(%164, %166, %18, %20, %13, %8, %18, %17, %17, %17, %17, %5, %17, %25, %17, %17, %17, %25, %17, %5, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_99 = memref.alloc() : memref<1x1024x8x64xf16>
    %intptr_100 = memref.extract_aligned_pointer_as_index %alloc_99 : memref<1x1024x8x64xf16> -> index
    %168 = arith.index_cast %intptr_100 : index to i64
    %169 = llvm.inttoptr %168 : i64 to !llvm.ptr
    %intptr_101 = memref.extract_aligned_pointer_as_index %alloc_50 : memref<1x1024x8x256xf16> -> index
    %170 = arith.index_cast %intptr_101 : index to i64
    %171 = llvm.inttoptr %170 : i64 to !llvm.ptr
    %172 = call @rxops_bridge_slice_nd(%169, %171, %18, %20, %17, %25, %16, %5, %17, %17, %17, %25, %16, %15, %17, %17, %13, %13, %13, %13, %17, %17, %17, %17, %17, %17, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_102 = memref.alloc() : memref<1x1024x8x192xf16>
    %intptr_103 = memref.extract_aligned_pointer_as_index %alloc_102 : memref<1x1024x8x192xf16> -> index
    %173 = arith.index_cast %intptr_103 : index to i64
    %174 = llvm.inttoptr %173 : i64 to !llvm.ptr
    %intptr_104 = memref.extract_aligned_pointer_as_index %alloc_50 : memref<1x1024x8x256xf16> -> index
    %175 = arith.index_cast %intptr_104 : index to i64
    %176 = llvm.inttoptr %175 : i64 to !llvm.ptr
    %177 = call @rxops_bridge_slice_nd(%174, %176, %18, %20, %17, %25, %16, %4, %17, %17, %17, %25, %16, %15, %17, %17, %13, %13, %13, %5, %17, %17, %17, %17, %17, %17, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_105 = memref.alloc() : memref<1x1024x8x64xf16>
    %intptr_106 = memref.extract_aligned_pointer_as_index %alloc_105 : memref<1x1024x8x64xf16> -> index
    %178 = arith.index_cast %intptr_106 : index to i64
    %179 = llvm.inttoptr %178 : i64 to !llvm.ptr
    %intptr_107 = memref.extract_aligned_pointer_as_index %alloc_99 : memref<1x1024x8x64xf16> -> index
    %180 = arith.index_cast %intptr_107 : index to i64
    %181 = llvm.inttoptr %180 : i64 to !llvm.ptr
    %intptr_108 = memref.extract_aligned_pointer_as_index %alloc_76 : memref<1x1024x1x64xf16> -> index
    %182 = arith.index_cast %intptr_108 : index to i64
    %183 = llvm.inttoptr %182 : i64 to !llvm.ptr
    %intptr_109 = memref.extract_aligned_pointer_as_index %alloc_96 : memref<1x1024x1x64xf16> -> index
    %184 = arith.index_cast %intptr_109 : index to i64
    %185 = llvm.inttoptr %184 : i64 to !llvm.ptr
    %186 = call @rxops_bridge_rope_contiguous_f16(%179, %181, %183, %185, %25, %16, %5, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    %alloc_110 = memref.alloc() : memref<1x1024x8x256xf16>
    %intptr_111 = memref.extract_aligned_pointer_as_index %alloc_110 : memref<1x1024x8x256xf16> -> index
    %187 = arith.index_cast %intptr_111 : index to i64
    %188 = llvm.inttoptr %187 : i64 to !llvm.ptr
    %intptr_112 = memref.extract_aligned_pointer_as_index %alloc_105 : memref<1x1024x8x64xf16> -> index
    %189 = arith.index_cast %intptr_112 : index to i64
    %190 = llvm.inttoptr %189 : i64 to !llvm.ptr
    %intptr_113 = memref.extract_aligned_pointer_as_index %alloc_102 : memref<1x1024x8x192xf16> -> index
    %191 = arith.index_cast %intptr_113 : index to i64
    %192 = llvm.inttoptr %191 : i64 to !llvm.ptr
    %193 = call @rxops_bridge_concat2_nd(%188, %190, %192, %18, %20, %8, %17, %25, %16, %15, %17, %17, %17, %25, %16, %5, %17, %17, %17, %25, %16, %4, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_114 = memref.alloc() : memref<1x1024x2x64xf16>
    %intptr_115 = memref.extract_aligned_pointer_as_index %alloc_114 : memref<1x1024x2x64xf16> -> index
    %194 = arith.index_cast %intptr_115 : index to i64
    %195 = llvm.inttoptr %194 : i64 to !llvm.ptr
    %intptr_116 = memref.extract_aligned_pointer_as_index %alloc_55 : memref<1x1024x2x256xf16> -> index
    %196 = arith.index_cast %intptr_116 : index to i64
    %197 = llvm.inttoptr %196 : i64 to !llvm.ptr
    %198 = call @rxops_bridge_slice_nd(%195, %197, %18, %20, %17, %25, %18, %5, %17, %17, %17, %25, %18, %15, %17, %17, %13, %13, %13, %13, %17, %17, %17, %17, %17, %17, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_117 = memref.alloc() : memref<1x1024x2x192xf16>
    %intptr_118 = memref.extract_aligned_pointer_as_index %alloc_117 : memref<1x1024x2x192xf16> -> index
    %199 = arith.index_cast %intptr_118 : index to i64
    %200 = llvm.inttoptr %199 : i64 to !llvm.ptr
    %intptr_119 = memref.extract_aligned_pointer_as_index %alloc_55 : memref<1x1024x2x256xf16> -> index
    %201 = arith.index_cast %intptr_119 : index to i64
    %202 = llvm.inttoptr %201 : i64 to !llvm.ptr
    %203 = call @rxops_bridge_slice_nd(%200, %202, %18, %20, %17, %25, %18, %4, %17, %17, %17, %25, %18, %15, %17, %17, %13, %13, %13, %5, %17, %17, %17, %17, %17, %17, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_120 = memref.alloc() : memref<1x1024x2x64xf16>
    %intptr_121 = memref.extract_aligned_pointer_as_index %alloc_120 : memref<1x1024x2x64xf16> -> index
    %204 = arith.index_cast %intptr_121 : index to i64
    %205 = llvm.inttoptr %204 : i64 to !llvm.ptr
    %intptr_122 = memref.extract_aligned_pointer_as_index %alloc_114 : memref<1x1024x2x64xf16> -> index
    %206 = arith.index_cast %intptr_122 : index to i64
    %207 = llvm.inttoptr %206 : i64 to !llvm.ptr
    %intptr_123 = memref.extract_aligned_pointer_as_index %alloc_76 : memref<1x1024x1x64xf16> -> index
    %208 = arith.index_cast %intptr_123 : index to i64
    %209 = llvm.inttoptr %208 : i64 to !llvm.ptr
    %intptr_124 = memref.extract_aligned_pointer_as_index %alloc_96 : memref<1x1024x1x64xf16> -> index
    %210 = arith.index_cast %intptr_124 : index to i64
    %211 = llvm.inttoptr %210 : i64 to !llvm.ptr
    %212 = call @rxops_bridge_rope_contiguous_f16(%205, %207, %209, %211, %25, %18, %5, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    %alloc_125 = memref.alloc() : memref<1x1024x2x256xf16>
    %intptr_126 = memref.extract_aligned_pointer_as_index %alloc_125 : memref<1x1024x2x256xf16> -> index
    %213 = arith.index_cast %intptr_126 : index to i64
    %214 = llvm.inttoptr %213 : i64 to !llvm.ptr
    %intptr_127 = memref.extract_aligned_pointer_as_index %alloc_120 : memref<1x1024x2x64xf16> -> index
    %215 = arith.index_cast %intptr_127 : index to i64
    %216 = llvm.inttoptr %215 : i64 to !llvm.ptr
    %intptr_128 = memref.extract_aligned_pointer_as_index %alloc_117 : memref<1x1024x2x192xf16> -> index
    %217 = arith.index_cast %intptr_128 : index to i64
    %218 = llvm.inttoptr %217 : i64 to !llvm.ptr
    %219 = call @rxops_bridge_concat2_nd(%214, %216, %218, %18, %20, %8, %17, %25, %18, %15, %17, %17, %17, %25, %18, %5, %17, %17, %17, %25, %18, %4, %17, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_129 = memref.alloc() : memref<1x1x1024x1024xf32>
    memref.copy %arg2, %alloc_129 : memref<1x1x1024x1024xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<1x1x1024x1024xf32>
    %alloc_130 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_131 = memref.extract_aligned_pointer_as_index %alloc_130 : memref<1x1024x2048xf16> -> index
    %220 = arith.index_cast %intptr_131 : index to i64
    %221 = llvm.inttoptr %220 : i64 to !llvm.ptr
    %intptr_132 = memref.extract_aligned_pointer_as_index %alloc_110 : memref<1x1024x8x256xf16> -> index
    %222 = arith.index_cast %intptr_132 : index to i64
    %223 = llvm.inttoptr %222 : i64 to !llvm.ptr
    %intptr_133 = memref.extract_aligned_pointer_as_index %alloc_125 : memref<1x1024x2x256xf16> -> index
    %224 = arith.index_cast %intptr_133 : index to i64
    %225 = llvm.inttoptr %224 : i64 to !llvm.ptr
    %intptr_134 = memref.extract_aligned_pointer_as_index %alloc_46 : memref<1x1024x2x256xf16> -> index
    %226 = arith.index_cast %intptr_134 : index to i64
    %227 = llvm.inttoptr %226 : i64 to !llvm.ptr
    %intptr_135 = memref.extract_aligned_pointer_as_index %alloc_129 : memref<1x1x1024x1024xf32> -> index
    %228 = arith.index_cast %intptr_135 : index to i64
    %229 = llvm.inttoptr %228 : i64 to !llvm.ptr
    %230 = call @rxops_bridge_fattention_f16(%221, %223, %225, %227, %229, %17, %25, %25, %16, %18, %15, %3, %17) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    %alloc_136 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_137 = memref.extract_aligned_pointer_as_index %alloc_136 : memref<1x1024x2048xf16> -> index
    %231 = arith.index_cast %intptr_137 : index to i64
    %232 = llvm.inttoptr %231 : i64 to !llvm.ptr
    %intptr_138 = memref.extract_aligned_pointer_as_index %alloc_22 : memref<1x1024x8x256xf16> -> index
    %233 = arith.index_cast %intptr_138 : index to i64
    %234 = llvm.inttoptr %233 : i64 to !llvm.ptr
    %235 = call @rxops_bridge_reshape_bytes(%232, %234, %2) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_139 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_140 = memref.extract_aligned_pointer_as_index %alloc_139 : memref<1x1024x2048xf16> -> index
    %236 = arith.index_cast %intptr_140 : index to i64
    %237 = llvm.inttoptr %236 : i64 to !llvm.ptr
    %intptr_141 = memref.extract_aligned_pointer_as_index %alloc_130 : memref<1x1024x2048xf16> -> index
    %238 = arith.index_cast %intptr_141 : index to i64
    %239 = llvm.inttoptr %238 : i64 to !llvm.ptr
    %intptr_142 = memref.extract_aligned_pointer_as_index %alloc_136 : memref<1x1024x2048xf16> -> index
    %240 = arith.index_cast %intptr_142 : index to i64
    %241 = llvm.inttoptr %240 : i64 to !llvm.ptr
    %242 = call @rxops_bridge_mul_f16(%237, %239, %241, %12) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_143 = memref.alloc() {alignment = 64 : i64} : memref<2048x1024xui8>
    %alloc_144 = memref.alloc() {alignment = 64 : i64} : memref<2048x16xui8>
    %alloc_145 = memref.alloc() {alignment = 64 : i64} : memref<2048x16xf16>
    %alloc_146 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_147 = memref.extract_aligned_pointer_as_index %alloc_146 : memref<1x1024x2048xf16> -> index
    %243 = arith.index_cast %intptr_147 : index to i64
    %244 = llvm.inttoptr %243 : i64 to !llvm.ptr
    %intptr_148 = memref.extract_aligned_pointer_as_index %alloc_139 : memref<1x1024x2048xf16> -> index
    %245 = arith.index_cast %intptr_148 : index to i64
    %246 = llvm.inttoptr %245 : i64 to !llvm.ptr
    %intptr_149 = memref.extract_aligned_pointer_as_index %alloc_143 : memref<2048x1024xui8> -> index
    %247 = arith.index_cast %intptr_149 : index to i64
    %248 = llvm.inttoptr %247 : i64 to !llvm.ptr
    %intptr_150 = memref.extract_aligned_pointer_as_index %alloc_145 : memref<2048x16xf16> -> index
    %249 = arith.index_cast %intptr_150 : index to i64
    %250 = llvm.inttoptr %249 : i64 to !llvm.ptr
    %intptr_151 = memref.extract_aligned_pointer_as_index %alloc_144 : memref<2048x16xui8> -> index
    %251 = arith.index_cast %intptr_151 : index to i64
    %252 = llvm.inttoptr %251 : i64 to !llvm.ptr
    %253 = call @rxops_bridge_a16matmul_f16(%244, %246, %248, %250, %252, %25, %24, %24, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_152 = memref.alloc() : memref<1x1024x2048xf32>
    memref.copy %arg0, %alloc_152 : memref<1x1024x2048xf32, strided<[?, ?, ?], offset: ?>> to memref<1x1024x2048xf32>
    %alloc_153 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_154 = memref.extract_aligned_pointer_as_index %alloc_153 : memref<1x1024x2048xf16> -> index
    %254 = arith.index_cast %intptr_154 : index to i64
    %255 = llvm.inttoptr %254 : i64 to !llvm.ptr
    %intptr_155 = memref.extract_aligned_pointer_as_index %alloc_152 : memref<1x1024x2048xf32> -> index
    %256 = arith.index_cast %intptr_155 : index to i64
    %257 = llvm.inttoptr %256 : i64 to !llvm.ptr
    %intptr_156 = memref.extract_aligned_pointer_as_index %alloc_146 : memref<1x1024x2048xf16> -> index
    %258 = arith.index_cast %intptr_156 : index to i64
    %259 = llvm.inttoptr %258 : i64 to !llvm.ptr
    %260 = call @rxops_bridge_add_f16(%255, %257, %259, %12) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_157 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2048xf16>
    %alloc_158 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_159 = memref.extract_aligned_pointer_as_index %alloc_158 : memref<1x1024x2048xf16> -> index
    %261 = arith.index_cast %intptr_159 : index to i64
    %262 = llvm.inttoptr %261 : i64 to !llvm.ptr
    %intptr_160 = memref.extract_aligned_pointer_as_index %alloc_153 : memref<1x1024x2048xf16> -> index
    %263 = arith.index_cast %intptr_160 : index to i64
    %264 = llvm.inttoptr %263 : i64 to !llvm.ptr
    %intptr_161 = memref.extract_aligned_pointer_as_index %alloc_157 : memref<1x1x2048xf16> -> index
    %265 = arith.index_cast %intptr_161 : index to i64
    %266 = llvm.inttoptr %265 : i64 to !llvm.ptr
    %267 = call @rxops_bridge_rms_norm_f16(%262, %264, %266, %25, %24, %23) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %alloc_162 = memref.alloc() {alignment = 64 : i64} : memref<6144x1024xui8>
    %alloc_163 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xui8>
    %alloc_164 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xf16>
    %alloc_165 = memref.alloc() : memref<1x1024x6144xf16>
    %intptr_166 = memref.extract_aligned_pointer_as_index %alloc_165 : memref<1x1024x6144xf16> -> index
    %268 = arith.index_cast %intptr_166 : index to i64
    %269 = llvm.inttoptr %268 : i64 to !llvm.ptr
    %intptr_167 = memref.extract_aligned_pointer_as_index %alloc_158 : memref<1x1024x2048xf16> -> index
    %270 = arith.index_cast %intptr_167 : index to i64
    %271 = llvm.inttoptr %270 : i64 to !llvm.ptr
    %intptr_168 = memref.extract_aligned_pointer_as_index %alloc_162 : memref<6144x1024xui8> -> index
    %272 = arith.index_cast %intptr_168 : index to i64
    %273 = llvm.inttoptr %272 : i64 to !llvm.ptr
    %intptr_169 = memref.extract_aligned_pointer_as_index %alloc_164 : memref<6144x16xf16> -> index
    %274 = arith.index_cast %intptr_169 : index to i64
    %275 = llvm.inttoptr %274 : i64 to !llvm.ptr
    %intptr_170 = memref.extract_aligned_pointer_as_index %alloc_163 : memref<6144x16xui8> -> index
    %276 = arith.index_cast %intptr_170 : index to i64
    %277 = llvm.inttoptr %276 : i64 to !llvm.ptr
    %278 = call @rxops_bridge_a16matmul_f16(%269, %271, %273, %275, %277, %25, %24, %1, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_171 = memref.alloc() : memref<1x1024x6144xf16>
    %intptr_172 = memref.extract_aligned_pointer_as_index %alloc_171 : memref<1x1024x6144xf16> -> index
    %279 = arith.index_cast %intptr_172 : index to i64
    %280 = llvm.inttoptr %279 : i64 to !llvm.ptr
    %intptr_173 = memref.extract_aligned_pointer_as_index %alloc_165 : memref<1x1024x6144xf16> -> index
    %281 = arith.index_cast %intptr_173 : index to i64
    %282 = llvm.inttoptr %281 : i64 to !llvm.ptr
    %283 = call @rxops_bridge_silu_f16(%280, %282, %0) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_174 = memref.alloc() {alignment = 64 : i64} : memref<6144x1024xui8>
    %alloc_175 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xui8>
    %alloc_176 = memref.alloc() {alignment = 64 : i64} : memref<6144x16xf16>
    %alloc_177 = memref.alloc() : memref<1x1024x6144xf16>
    %intptr_178 = memref.extract_aligned_pointer_as_index %alloc_177 : memref<1x1024x6144xf16> -> index
    %284 = arith.index_cast %intptr_178 : index to i64
    %285 = llvm.inttoptr %284 : i64 to !llvm.ptr
    %intptr_179 = memref.extract_aligned_pointer_as_index %alloc_158 : memref<1x1024x2048xf16> -> index
    %286 = arith.index_cast %intptr_179 : index to i64
    %287 = llvm.inttoptr %286 : i64 to !llvm.ptr
    %intptr_180 = memref.extract_aligned_pointer_as_index %alloc_174 : memref<6144x1024xui8> -> index
    %288 = arith.index_cast %intptr_180 : index to i64
    %289 = llvm.inttoptr %288 : i64 to !llvm.ptr
    %intptr_181 = memref.extract_aligned_pointer_as_index %alloc_176 : memref<6144x16xf16> -> index
    %290 = arith.index_cast %intptr_181 : index to i64
    %291 = llvm.inttoptr %290 : i64 to !llvm.ptr
    %intptr_182 = memref.extract_aligned_pointer_as_index %alloc_175 : memref<6144x16xui8> -> index
    %292 = arith.index_cast %intptr_182 : index to i64
    %293 = llvm.inttoptr %292 : i64 to !llvm.ptr
    %294 = call @rxops_bridge_a16matmul_f16(%285, %287, %289, %291, %293, %25, %24, %1, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_183 = memref.alloc() : memref<1x1024x6144xf16>
    %intptr_184 = memref.extract_aligned_pointer_as_index %alloc_183 : memref<1x1024x6144xf16> -> index
    %295 = arith.index_cast %intptr_184 : index to i64
    %296 = llvm.inttoptr %295 : i64 to !llvm.ptr
    %intptr_185 = memref.extract_aligned_pointer_as_index %alloc_171 : memref<1x1024x6144xf16> -> index
    %297 = arith.index_cast %intptr_185 : index to i64
    %298 = llvm.inttoptr %297 : i64 to !llvm.ptr
    %intptr_186 = memref.extract_aligned_pointer_as_index %alloc_177 : memref<1x1024x6144xf16> -> index
    %299 = arith.index_cast %intptr_186 : index to i64
    %300 = llvm.inttoptr %299 : i64 to !llvm.ptr
    %301 = call @rxops_bridge_mul_f16(%296, %298, %300, %0) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %alloc_187 = memref.alloc() {alignment = 64 : i64} : memref<2048x3072xui8>
    %alloc_188 = memref.alloc() {alignment = 64 : i64} : memref<2048x48xui8>
    %alloc_189 = memref.alloc() {alignment = 64 : i64} : memref<2048x48xf16>
    %alloc_190 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_191 = memref.extract_aligned_pointer_as_index %alloc_190 : memref<1x1024x2048xf16> -> index
    %302 = arith.index_cast %intptr_191 : index to i64
    %303 = llvm.inttoptr %302 : i64 to !llvm.ptr
    %intptr_192 = memref.extract_aligned_pointer_as_index %alloc_183 : memref<1x1024x6144xf16> -> index
    %304 = arith.index_cast %intptr_192 : index to i64
    %305 = llvm.inttoptr %304 : i64 to !llvm.ptr
    %intptr_193 = memref.extract_aligned_pointer_as_index %alloc_187 : memref<2048x3072xui8> -> index
    %306 = arith.index_cast %intptr_193 : index to i64
    %307 = llvm.inttoptr %306 : i64 to !llvm.ptr
    %intptr_194 = memref.extract_aligned_pointer_as_index %alloc_189 : memref<2048x48xf16> -> index
    %308 = arith.index_cast %intptr_194 : index to i64
    %309 = llvm.inttoptr %308 : i64 to !llvm.ptr
    %intptr_195 = memref.extract_aligned_pointer_as_index %alloc_188 : memref<2048x48xui8> -> index
    %310 = arith.index_cast %intptr_195 : index to i64
    %311 = llvm.inttoptr %310 : i64 to !llvm.ptr
    %312 = call @rxops_bridge_a16matmul_f16(%303, %305, %307, %309, %311, %25, %1, %24, %21, %20) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
    %alloc_196 = memref.alloc() : memref<1x1024x2048xf16>
    %intptr_197 = memref.extract_aligned_pointer_as_index %alloc_196 : memref<1x1024x2048xf16> -> index
    %313 = arith.index_cast %intptr_197 : index to i64
    %314 = llvm.inttoptr %313 : i64 to !llvm.ptr
    %intptr_198 = memref.extract_aligned_pointer_as_index %alloc_153 : memref<1x1024x2048xf16> -> index
    %315 = arith.index_cast %intptr_198 : index to i64
    %316 = llvm.inttoptr %315 : i64 to !llvm.ptr
    %intptr_199 = memref.extract_aligned_pointer_as_index %alloc_190 : memref<1x1024x2048xf16> -> index
    %317 = arith.index_cast %intptr_199 : index to i64
    %318 = llvm.inttoptr %317 : i64 to !llvm.ptr
    %319 = call @rxops_bridge_add_f16(%314, %316, %318, %12) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    %cast = memref.cast %alloc_196 : memref<1x1024x2048xf16> to memref<1x1024x2048xf16, strided<[?, ?, ?], offset: ?>>
    %cast_200 = memref.cast %alloc_125 : memref<1x1024x2x256xf16> to memref<1x1024x2x256xf16, strided<[?, ?, ?, ?], offset: ?>>
    %cast_201 = memref.cast %alloc_46 : memref<1x1024x2x256xf16> to memref<1x1024x2x256xf16, strided<[?, ?, ?, ?], offset: ?>>
    memref.dealloc %alloc : memref<1x1x2048xf16>
    memref.dealloc %alloc_0 : memref<1x1024x2048xf32>
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
    memref.dealloc %alloc_60 : memref<3x1024xsi32>
    memref.dealloc %alloc_71 : memref<1x64xi32>
    memref.dealloc %alloc_79 : memref<2048x1x32xf16>
    memref.dealloc %alloc_91 : memref<1x64xi32>
    memref.dealloc %alloc_143 : memref<2048x1024xui8>
    memref.dealloc %alloc_144 : memref<2048x16xui8>
    memref.dealloc %alloc_145 : memref<2048x16xf16>
    memref.dealloc %alloc_157 : memref<1x1x2048xf16>
    memref.dealloc %alloc_162 : memref<6144x1024xui8>
    memref.dealloc %alloc_163 : memref<6144x16xui8>
    memref.dealloc %alloc_164 : memref<6144x16xf16>
    memref.dealloc %alloc_174 : memref<6144x1024xui8>
    memref.dealloc %alloc_175 : memref<6144x16xui8>
    memref.dealloc %alloc_176 : memref<6144x16xf16>
    memref.dealloc %alloc_187 : memref<2048x3072xui8>
    memref.dealloc %alloc_188 : memref<2048x48xui8>
    memref.dealloc %alloc_189 : memref<2048x48xf16>
    return %alloc_196, %alloc_125, %alloc_46 : memref<1x1024x2048xf16>, memref<1x1024x2x256xf16>, memref<1x1024x2x256xf16>
  }
}

