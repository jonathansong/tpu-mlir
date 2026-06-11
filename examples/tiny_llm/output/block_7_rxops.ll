; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define private i32 @rxops_bridge_silu_f16(ptr %0, ptr %1, i64 %2) {
  %4 = call i32 @_mlir_ciface_rxops_bridge_silu_f16(ptr %0, ptr %1, i64 %2)
  ret i32 %4
}

declare i32 @_mlir_ciface_rxops_bridge_silu_f16(ptr, ptr, i64)

define private i32 @rxops_bridge_add_f16(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = call i32 @_mlir_ciface_rxops_bridge_add_f16(ptr %0, ptr %1, ptr %2, i64 %3)
  ret i32 %5
}

declare i32 @_mlir_ciface_rxops_bridge_add_f16(ptr, ptr, ptr, i64)

define private i32 @rxops_bridge_mul_f16(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = call i32 @_mlir_ciface_rxops_bridge_mul_f16(ptr %0, ptr %1, ptr %2, i64 %3)
  ret i32 %5
}

declare i32 @_mlir_ciface_rxops_bridge_mul_f16(ptr, ptr, ptr, i64)

define private i32 @rxops_bridge_fattention_f16(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12) {
  %14 = call i32 @_mlir_ciface_rxops_bridge_fattention_f16(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12)
  ret i32 %14
}

declare i32 @_mlir_ciface_rxops_bridge_fattention_f16(ptr, ptr, ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64)

define private i32 @rxops_bridge_concat2_nd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23) {
  %25 = call i32 @_mlir_ciface_rxops_bridge_concat2_nd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23)
  ret i32 %25
}

declare i32 @_mlir_ciface_rxops_bridge_concat2_nd(ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define private i32 @rxops_bridge_rope_contiguous_f16(ptr %0, ptr %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7) {
  %9 = call i32 @_mlir_ciface_rxops_bridge_rope_contiguous_f16(ptr %0, ptr %1, ptr %2, ptr %3, i64 %4, i64 %5, i64 %6, i64 %7)
  ret i32 %9
}

declare i32 @_mlir_ciface_rxops_bridge_rope_contiguous_f16(ptr, ptr, ptr, ptr, i64, i64, i64, i64)

define private i32 @rxops_bridge_transpose_nd(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21) {
  %23 = call i32 @_mlir_ciface_rxops_bridge_transpose_nd(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21)
  ret i32 %23
}

declare i32 @_mlir_ciface_rxops_bridge_transpose_nd(ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define private i32 @rxops_bridge_gather_nd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
  %24 = call i32 @_mlir_ciface_rxops_bridge_gather_nd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22)
  ret i32 %24
}

declare i32 @_mlir_ciface_rxops_bridge_gather_nd(ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define private i32 @rxops_bridge_rms_norm_f16(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5) {
  %7 = call i32 @_mlir_ciface_rxops_bridge_rms_norm_f16(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5)
  ret i32 %7
}

declare i32 @_mlir_ciface_rxops_bridge_rms_norm_f16(ptr, ptr, ptr, i64, i64, i64)

define private i32 @rxops_bridge_sigmoid_f16(ptr %0, ptr %1, i64 %2) {
  %4 = call i32 @_mlir_ciface_rxops_bridge_sigmoid_f16(ptr %0, ptr %1, i64 %2)
  ret i32 %4
}

declare i32 @_mlir_ciface_rxops_bridge_sigmoid_f16(ptr, ptr, i64)

define private i32 @rxops_bridge_slice_nd(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27) {
  %29 = call i32 @_mlir_ciface_rxops_bridge_slice_nd(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27)
  ret i32 %29
}

declare i32 @_mlir_ciface_rxops_bridge_slice_nd(ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define private i32 @rxops_bridge_reshape_bytes(ptr %0, ptr %1, i64 %2) {
  %4 = call i32 @_mlir_ciface_rxops_bridge_reshape_bytes(ptr %0, ptr %1, i64 %2)
  ret i32 %4
}

declare i32 @_mlir_ciface_rxops_bridge_reshape_bytes(ptr, ptr, i64)

define private i32 @rxops_bridge_a16matmul_f16(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9) {
  %11 = call i32 @_mlir_ciface_rxops_bridge_a16matmul_f16(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9)
  ret i32 %11
}

declare i32 @_mlir_ciface_rxops_bridge_a16matmul_f16(ptr, ptr, ptr, ptr, ptr, i64, i64, i64, i64, i64)

define private i32 @rxops_bridge_rms_norm_f32(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5) {
  %7 = call i32 @_mlir_ciface_rxops_bridge_rms_norm_f32(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5)
  ret i32 %7
}

declare i32 @_mlir_ciface_rxops_bridge_rms_norm_f32(ptr, ptr, ptr, i64, i64, i64)

define { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } @block7_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26) {
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %0, 0
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, ptr %1, 1
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 %2, 2
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 %3, 3, 0
  %32 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, i64 %6, 4, 0
  %33 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %32, i64 %4, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, i64 %7, 4, 1
  %35 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, i64 %5, 3, 2
  %36 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %35, i64 %8, 4, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %9, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, ptr %10, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %11, 2
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %12, 3, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %14, 4, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %13, 3, 1
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %15, 4, 1
  %44 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %16, 0
  %45 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %44, ptr %17, 1
  %46 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %45, i64 %18, 2
  %47 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %46, i64 %19, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %47, i64 %23, 4, 0
  %49 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %48, i64 %20, 3, 1
  %50 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, i64 %24, 4, 1
  %51 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %50, i64 %21, 3, 2
  %52 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %51, i64 %25, 4, 2
  %53 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %52, i64 %22, 3, 3
  %54 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %53, i64 %26, 4, 3
  %55 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64), i64 64))
  %56 = ptrtoint ptr %55 to i64
  %57 = add i64 %56, 63
  %58 = urem i64 %57, 64
  %59 = sub i64 %57, %58
  %60 = inttoptr i64 %59 to ptr
  %61 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %55, 0
  %62 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %61, ptr %60, 1
  %63 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %62, i64 0, 2
  %64 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %63, i64 1, 3, 0
  %65 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %64, i64 1, 3, 1
  %66 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %65, i64 2048, 3, 2
  %67 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %66, i64 2048, 4, 0
  %68 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %67, i64 2048, 4, 1
  %69 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %68, i64 1, 4, 2
  %70 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2097152) to i64), i64 64))
  %71 = ptrtoint ptr %70 to i64
  %72 = add i64 %71, 63
  %73 = urem i64 %72, 64
  %74 = sub i64 %72, %73
  %75 = inttoptr i64 %74 to ptr
  %76 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %70, 0
  %77 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %76, ptr %75, 1
  %78 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %77, i64 0, 2
  %79 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %78, i64 1, 3, 0
  %80 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, i64 1024, 3, 1
  %81 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %80, i64 2048, 3, 2
  %82 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %81, i64 2097152, 4, 0
  %83 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %82, i64 2048, 4, 1
  %84 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %83, i64 1, 4, 2
  %85 = call ptr @llvm.stacksave()
  %86 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, ptr %86, align 8
  %87 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %86, 1
  %88 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %84, ptr %88, align 8
  %89 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %88, 1
  %90 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %87, ptr %90, align 8
  %91 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %89, ptr %91, align 8
  call void @memrefCopy(i64 4, ptr %90, ptr %91)
  call void @llvm.stackrestore(ptr %85)
  %92 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %93 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %92, 0
  %94 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %93, ptr %92, 1
  %95 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %94, i64 0, 2
  %96 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %95, i64 1, 3, 0
  %97 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %96, i64 1024, 3, 1
  %98 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %97, i64 2048, 3, 2
  %99 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %98, i64 2097152, 4, 0
  %100 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, i64 2048, 4, 1
  %101 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %100, i64 1, 4, 2
  %102 = ptrtoint ptr %92 to i64
  %103 = inttoptr i64 %102 to ptr
  %104 = ptrtoint ptr %75 to i64
  %105 = inttoptr i64 %104 to ptr
  %106 = ptrtoint ptr %60 to i64
  %107 = inttoptr i64 %106 to ptr
  %108 = call i32 @rxops_bridge_rms_norm_f32(ptr %103, ptr %105, ptr %107, i64 1024, i64 2048, i64 4517329193108106637)
  %109 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 4194304) to i64), i64 64))
  %110 = ptrtoint ptr %109 to i64
  %111 = add i64 %110, 63
  %112 = urem i64 %111, 64
  %113 = sub i64 %111, %112
  %114 = inttoptr i64 %113 to ptr
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %109, 0
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, ptr %114, 1
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 0, 2
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 4096, 3, 0
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 1024, 3, 1
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 1024, 4, 0
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 1, 4, 1
  %122 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 65536) to i64), i64 64))
  %123 = ptrtoint ptr %122 to i64
  %124 = add i64 %123, 63
  %125 = urem i64 %124, 64
  %126 = sub i64 %124, %125
  %127 = inttoptr i64 %126 to ptr
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %122, 0
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, ptr %127, 1
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 0, 2
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %130, i64 4096, 3, 0
  %132 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, i64 16, 3, 1
  %133 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %132, i64 16, 4, 0
  %134 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %133, i64 1, 4, 1
  %135 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64), i64 64))
  %136 = ptrtoint ptr %135 to i64
  %137 = add i64 %136, 63
  %138 = urem i64 %137, 64
  %139 = sub i64 %137, %138
  %140 = inttoptr i64 %139 to ptr
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %135, 0
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, ptr %140, 1
  %143 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, i64 0, 2
  %144 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %143, i64 4096, 3, 0
  %145 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %144, i64 16, 3, 1
  %146 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %145, i64 16, 4, 0
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %146, i64 1, 4, 1
  %148 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 4194304) to i64))
  %149 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %148, 0
  %150 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %149, ptr %148, 1
  %151 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %150, i64 0, 2
  %152 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %151, i64 1, 3, 0
  %153 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %152, i64 1024, 3, 1
  %154 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %153, i64 4096, 3, 2
  %155 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %154, i64 4194304, 4, 0
  %156 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %155, i64 4096, 4, 1
  %157 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %156, i64 1, 4, 2
  %158 = ptrtoint ptr %148 to i64
  %159 = inttoptr i64 %158 to ptr
  %160 = ptrtoint ptr %92 to i64
  %161 = inttoptr i64 %160 to ptr
  %162 = ptrtoint ptr %114 to i64
  %163 = inttoptr i64 %162 to ptr
  %164 = ptrtoint ptr %140 to i64
  %165 = inttoptr i64 %164 to ptr
  %166 = ptrtoint ptr %127 to i64
  %167 = inttoptr i64 %166 to ptr
  %168 = call i32 @rxops_bridge_a16matmul_f16(ptr %159, ptr %161, ptr %163, ptr %165, ptr %167, i64 1024, i64 2048, i64 4096, i64 128, i64 4)
  %169 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 4194304) to i64))
  %170 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %169, 0
  %171 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %170, ptr %169, 1
  %172 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %171, i64 0, 2
  %173 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %172, i64 1, 3, 0
  %174 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %173, i64 1024, 3, 1
  %175 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %174, i64 8, 3, 2
  %176 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %175, i64 512, 3, 3
  %177 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %176, i64 4194304, 4, 0
  %178 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %177, i64 4096, 4, 1
  %179 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %178, i64 512, 4, 2
  %180 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %179, i64 1, 4, 3
  %181 = ptrtoint ptr %169 to i64
  %182 = inttoptr i64 %181 to ptr
  %183 = ptrtoint ptr %148 to i64
  %184 = inttoptr i64 %183 to ptr
  %185 = call i32 @rxops_bridge_reshape_bytes(ptr %182, ptr %184, i64 8388608)
  %186 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %187 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %186, 0
  %188 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %187, ptr %186, 1
  %189 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %188, i64 0, 2
  %190 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %189, i64 1, 3, 0
  %191 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %190, i64 1024, 3, 1
  %192 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %191, i64 8, 3, 2
  %193 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %192, i64 256, 3, 3
  %194 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %193, i64 2097152, 4, 0
  %195 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %194, i64 2048, 4, 1
  %196 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %195, i64 256, 4, 2
  %197 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %196, i64 1, 4, 3
  %198 = ptrtoint ptr %186 to i64
  %199 = inttoptr i64 %198 to ptr
  %200 = ptrtoint ptr %169 to i64
  %201 = inttoptr i64 %200 to ptr
  %202 = call i32 @rxops_bridge_slice_nd(ptr %199, ptr %201, i64 2, i64 4, i64 1, i64 1024, i64 8, i64 256, i64 1, i64 1, i64 1, i64 1024, i64 8, i64 512, i64 1, i64 1, i64 0, i64 0, i64 0, i64 0, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %203 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %204 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %203, 0
  %205 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %204, ptr %203, 1
  %206 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %205, i64 0, 2
  %207 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %206, i64 1, 3, 0
  %208 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %207, i64 1024, 3, 1
  %209 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %208, i64 8, 3, 2
  %210 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %209, i64 256, 3, 3
  %211 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %210, i64 2097152, 4, 0
  %212 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %211, i64 2048, 4, 1
  %213 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %212, i64 256, 4, 2
  %214 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %213, i64 1, 4, 3
  %215 = ptrtoint ptr %203 to i64
  %216 = inttoptr i64 %215 to ptr
  %217 = ptrtoint ptr %169 to i64
  %218 = inttoptr i64 %217 to ptr
  %219 = call i32 @rxops_bridge_slice_nd(ptr %216, ptr %218, i64 2, i64 4, i64 1, i64 1024, i64 8, i64 256, i64 1, i64 1, i64 1, i64 1024, i64 8, i64 512, i64 1, i64 1, i64 0, i64 0, i64 0, i64 256, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %220 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %221 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %220, 0
  %222 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %221, ptr %220, 1
  %223 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %222, i64 0, 2
  %224 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %223, i64 1, 3, 0
  %225 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %224, i64 1024, 3, 1
  %226 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %225, i64 8, 3, 2
  %227 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %226, i64 256, 3, 3
  %228 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %227, i64 2097152, 4, 0
  %229 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %228, i64 2048, 4, 1
  %230 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %229, i64 256, 4, 2
  %231 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %230, i64 1, 4, 3
  %232 = ptrtoint ptr %220 to i64
  %233 = inttoptr i64 %232 to ptr
  %234 = ptrtoint ptr %203 to i64
  %235 = inttoptr i64 %234 to ptr
  %236 = call i32 @rxops_bridge_sigmoid_f16(ptr %233, ptr %235, i64 2097152)
  %237 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 524288) to i64), i64 64))
  %238 = ptrtoint ptr %237 to i64
  %239 = add i64 %238, 63
  %240 = urem i64 %239, 64
  %241 = sub i64 %239, %240
  %242 = inttoptr i64 %241 to ptr
  %243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %237, 0
  %244 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %243, ptr %242, 1
  %245 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %244, i64 0, 2
  %246 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %245, i64 512, 3, 0
  %247 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %246, i64 1024, 3, 1
  %248 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %247, i64 1024, 4, 0
  %249 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %248, i64 1, 4, 1
  %250 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 8192) to i64), i64 64))
  %251 = ptrtoint ptr %250 to i64
  %252 = add i64 %251, 63
  %253 = urem i64 %252, 64
  %254 = sub i64 %252, %253
  %255 = inttoptr i64 %254 to ptr
  %256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %250, 0
  %257 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %256, ptr %255, 1
  %258 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %257, i64 0, 2
  %259 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %258, i64 512, 3, 0
  %260 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %259, i64 16, 3, 1
  %261 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %260, i64 16, 4, 0
  %262 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %261, i64 1, 4, 1
  %263 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 8192) to i64), i64 64))
  %264 = ptrtoint ptr %263 to i64
  %265 = add i64 %264, 63
  %266 = urem i64 %265, 64
  %267 = sub i64 %265, %266
  %268 = inttoptr i64 %267 to ptr
  %269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %263, 0
  %270 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %269, ptr %268, 1
  %271 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, i64 0, 2
  %272 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %271, i64 512, 3, 0
  %273 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %272, i64 16, 3, 1
  %274 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %273, i64 16, 4, 0
  %275 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %274, i64 1, 4, 1
  %276 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %277 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %276, 0
  %278 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %277, ptr %276, 1
  %279 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %278, i64 0, 2
  %280 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %279, i64 1, 3, 0
  %281 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %280, i64 1024, 3, 1
  %282 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %281, i64 512, 3, 2
  %283 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %282, i64 524288, 4, 0
  %284 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %283, i64 512, 4, 1
  %285 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %284, i64 1, 4, 2
  %286 = ptrtoint ptr %276 to i64
  %287 = inttoptr i64 %286 to ptr
  %288 = ptrtoint ptr %92 to i64
  %289 = inttoptr i64 %288 to ptr
  %290 = ptrtoint ptr %242 to i64
  %291 = inttoptr i64 %290 to ptr
  %292 = ptrtoint ptr %268 to i64
  %293 = inttoptr i64 %292 to ptr
  %294 = ptrtoint ptr %255 to i64
  %295 = inttoptr i64 %294 to ptr
  %296 = call i32 @rxops_bridge_a16matmul_f16(ptr %287, ptr %289, ptr %291, ptr %293, ptr %295, i64 1024, i64 2048, i64 512, i64 128, i64 4)
  %297 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 524288) to i64), i64 64))
  %298 = ptrtoint ptr %297 to i64
  %299 = add i64 %298, 63
  %300 = urem i64 %299, 64
  %301 = sub i64 %299, %300
  %302 = inttoptr i64 %301 to ptr
  %303 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %297, 0
  %304 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, ptr %302, 1
  %305 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %304, i64 0, 2
  %306 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, i64 512, 3, 0
  %307 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %306, i64 1024, 3, 1
  %308 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %307, i64 1024, 4, 0
  %309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %308, i64 1, 4, 1
  %310 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 8192) to i64), i64 64))
  %311 = ptrtoint ptr %310 to i64
  %312 = add i64 %311, 63
  %313 = urem i64 %312, 64
  %314 = sub i64 %312, %313
  %315 = inttoptr i64 %314 to ptr
  %316 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %310, 0
  %317 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %316, ptr %315, 1
  %318 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %317, i64 0, 2
  %319 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %318, i64 512, 3, 0
  %320 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %319, i64 16, 3, 1
  %321 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %320, i64 16, 4, 0
  %322 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %321, i64 1, 4, 1
  %323 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 8192) to i64), i64 64))
  %324 = ptrtoint ptr %323 to i64
  %325 = add i64 %324, 63
  %326 = urem i64 %325, 64
  %327 = sub i64 %325, %326
  %328 = inttoptr i64 %327 to ptr
  %329 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %323, 0
  %330 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %329, ptr %328, 1
  %331 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %330, i64 0, 2
  %332 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %331, i64 512, 3, 0
  %333 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %332, i64 16, 3, 1
  %334 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %333, i64 16, 4, 0
  %335 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %334, i64 1, 4, 1
  %336 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %337 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %336, 0
  %338 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %337, ptr %336, 1
  %339 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %338, i64 0, 2
  %340 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %339, i64 1, 3, 0
  %341 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %340, i64 1024, 3, 1
  %342 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %341, i64 512, 3, 2
  %343 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %342, i64 524288, 4, 0
  %344 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %343, i64 512, 4, 1
  %345 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %344, i64 1, 4, 2
  %346 = ptrtoint ptr %336 to i64
  %347 = inttoptr i64 %346 to ptr
  %348 = ptrtoint ptr %92 to i64
  %349 = inttoptr i64 %348 to ptr
  %350 = ptrtoint ptr %302 to i64
  %351 = inttoptr i64 %350 to ptr
  %352 = ptrtoint ptr %328 to i64
  %353 = inttoptr i64 %352 to ptr
  %354 = ptrtoint ptr %315 to i64
  %355 = inttoptr i64 %354 to ptr
  %356 = call i32 @rxops_bridge_a16matmul_f16(ptr %347, ptr %349, ptr %351, ptr %353, ptr %355, i64 1024, i64 2048, i64 512, i64 128, i64 4)
  %357 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %358 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %357, 0
  %359 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %358, ptr %357, 1
  %360 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %359, i64 0, 2
  %361 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %360, i64 1, 3, 0
  %362 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %361, i64 1024, 3, 1
  %363 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %362, i64 2, 3, 2
  %364 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %363, i64 256, 3, 3
  %365 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %364, i64 524288, 4, 0
  %366 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %365, i64 512, 4, 1
  %367 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %366, i64 256, 4, 2
  %368 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %367, i64 1, 4, 3
  %369 = ptrtoint ptr %357 to i64
  %370 = inttoptr i64 %369 to ptr
  %371 = ptrtoint ptr %276 to i64
  %372 = inttoptr i64 %371 to ptr
  %373 = call i32 @rxops_bridge_reshape_bytes(ptr %370, ptr %372, i64 1048576)
  %374 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %375 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %374, 0
  %376 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %375, ptr %374, 1
  %377 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %376, i64 0, 2
  %378 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %377, i64 1, 3, 0
  %379 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %378, i64 1024, 3, 1
  %380 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %379, i64 2, 3, 2
  %381 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %380, i64 256, 3, 3
  %382 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %381, i64 524288, 4, 0
  %383 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %382, i64 512, 4, 1
  %384 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %383, i64 256, 4, 2
  %385 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %384, i64 1, 4, 3
  %386 = ptrtoint ptr %374 to i64
  %387 = inttoptr i64 %386 to ptr
  %388 = ptrtoint ptr %336 to i64
  %389 = inttoptr i64 %388 to ptr
  %390 = call i32 @rxops_bridge_reshape_bytes(ptr %387, ptr %389, i64 1048576)
  %391 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 256) to i64), i64 64))
  %392 = ptrtoint ptr %391 to i64
  %393 = add i64 %392, 63
  %394 = urem i64 %393, 64
  %395 = sub i64 %393, %394
  %396 = inttoptr i64 %395 to ptr
  %397 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %391, 0
  %398 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %397, ptr %396, 1
  %399 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %398, i64 0, 2
  %400 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %399, i64 1, 3, 0
  %401 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %400, i64 1, 3, 1
  %402 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %401, i64 1, 3, 2
  %403 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %402, i64 256, 3, 3
  %404 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %403, i64 256, 4, 0
  %405 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %404, i64 256, 4, 1
  %406 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %405, i64 256, 4, 2
  %407 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %406, i64 1, 4, 3
  %408 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %409 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %408, 0
  %410 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %409, ptr %408, 1
  %411 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %410, i64 0, 2
  %412 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %411, i64 1, 3, 0
  %413 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %412, i64 1024, 3, 1
  %414 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %413, i64 8, 3, 2
  %415 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %414, i64 256, 3, 3
  %416 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %415, i64 2097152, 4, 0
  %417 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %416, i64 2048, 4, 1
  %418 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %417, i64 256, 4, 2
  %419 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %418, i64 1, 4, 3
  %420 = ptrtoint ptr %408 to i64
  %421 = inttoptr i64 %420 to ptr
  %422 = ptrtoint ptr %186 to i64
  %423 = inttoptr i64 %422 to ptr
  %424 = ptrtoint ptr %396 to i64
  %425 = inttoptr i64 %424 to ptr
  %426 = call i32 @rxops_bridge_rms_norm_f16(ptr %421, ptr %423, ptr %425, i64 8192, i64 256, i64 4517329193108106637)
  %427 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 256) to i64), i64 64))
  %428 = ptrtoint ptr %427 to i64
  %429 = add i64 %428, 63
  %430 = urem i64 %429, 64
  %431 = sub i64 %429, %430
  %432 = inttoptr i64 %431 to ptr
  %433 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %427, 0
  %434 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %433, ptr %432, 1
  %435 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %434, i64 0, 2
  %436 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %435, i64 1, 3, 0
  %437 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %436, i64 1, 3, 1
  %438 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %437, i64 1, 3, 2
  %439 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %438, i64 256, 3, 3
  %440 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %439, i64 256, 4, 0
  %441 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %440, i64 256, 4, 1
  %442 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %441, i64 256, 4, 2
  %443 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %442, i64 1, 4, 3
  %444 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %445 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %444, 0
  %446 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %445, ptr %444, 1
  %447 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %446, i64 0, 2
  %448 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %447, i64 1, 3, 0
  %449 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %448, i64 1024, 3, 1
  %450 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %449, i64 2, 3, 2
  %451 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %450, i64 256, 3, 3
  %452 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %451, i64 524288, 4, 0
  %453 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %452, i64 512, 4, 1
  %454 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %453, i64 256, 4, 2
  %455 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %454, i64 1, 4, 3
  %456 = ptrtoint ptr %444 to i64
  %457 = inttoptr i64 %456 to ptr
  %458 = ptrtoint ptr %357 to i64
  %459 = inttoptr i64 %458 to ptr
  %460 = ptrtoint ptr %432 to i64
  %461 = inttoptr i64 %460 to ptr
  %462 = call i32 @rxops_bridge_rms_norm_f16(ptr %457, ptr %459, ptr %461, i64 2048, i64 256, i64 4517329193108106637)
  %463 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64), i64 64))
  %464 = ptrtoint ptr %463 to i64
  %465 = add i64 %464, 63
  %466 = urem i64 %465, 64
  %467 = sub i64 %465, %466
  %468 = inttoptr i64 %467 to ptr
  %469 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %463, 0
  %470 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %469, ptr %468, 1
  %471 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %470, i64 0, 2
  %472 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %471, i64 2048, 3, 0
  %473 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %472, i64 1, 3, 1
  %474 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %473, i64 32, 3, 2
  %475 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %474, i64 32, 4, 0
  %476 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %475, i64 32, 4, 1
  %477 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %476, i64 1, 4, 2
  %478 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 3072) to i64), i64 64))
  %479 = ptrtoint ptr %478 to i64
  %480 = add i64 %479, 63
  %481 = urem i64 %480, 64
  %482 = sub i64 %480, %481
  %483 = inttoptr i64 %482 to ptr
  %484 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %478, 0
  %485 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %484, ptr %483, 1
  %486 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %485, i64 0, 2
  %487 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %486, i64 3, 3, 0
  %488 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %487, i64 1024, 3, 1
  %489 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %488, i64 1024, 4, 0
  %490 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %489, i64 1, 4, 1
  %491 = call ptr @llvm.stacksave()
  %492 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, ptr %492, align 8
  %493 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %492, 1
  %494 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %490, ptr %494, align 8
  %495 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %494, 1
  %496 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %493, ptr %496, align 8
  %497 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %495, ptr %497, align 8
  call void @memrefCopy(i64 4, ptr %496, ptr %497)
  call void @llvm.stackrestore(ptr %491)
  %498 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64))
  %499 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %498, 0
  %500 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %499, ptr %498, 1
  %501 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %500, i64 0, 2
  %502 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %501, i64 3, 3, 0
  %503 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %502, i64 1024, 3, 1
  %504 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %503, i64 1, 3, 2
  %505 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %504, i64 32, 3, 3
  %506 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %505, i64 32768, 4, 0
  %507 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %506, i64 32, 4, 1
  %508 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %507, i64 32, 4, 2
  %509 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %508, i64 1, 4, 3
  %510 = ptrtoint ptr %498 to i64
  %511 = inttoptr i64 %510 to ptr
  %512 = ptrtoint ptr %468 to i64
  %513 = inttoptr i64 %512 to ptr
  %514 = ptrtoint ptr %483 to i64
  %515 = inttoptr i64 %514 to ptr
  %516 = call i32 @rxops_bridge_gather_nd(ptr %511, ptr %513, ptr %515, i64 0, i64 2, i64 2048, i64 1, i64 32, i64 1, i64 1, i64 1, i64 3, i64 1024, i64 1, i64 1, i64 1, i64 1, i64 3, i64 1024, i64 1, i64 32, i64 1, i64 1)
  %517 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64))
  %518 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %517, 0
  %519 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %518, ptr %517, 1
  %520 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %519, i64 0, 2
  %521 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %520, i64 3, 3, 0
  %522 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %521, i64 32, 3, 1
  %523 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %522, i64 1, 3, 2
  %524 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %523, i64 1024, 3, 3
  %525 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %524, i64 32768, 4, 0
  %526 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %525, i64 1024, 4, 1
  %527 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %526, i64 1024, 4, 2
  %528 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %527, i64 1, 4, 3
  %529 = ptrtoint ptr %517 to i64
  %530 = inttoptr i64 %529 to ptr
  %531 = ptrtoint ptr %498 to i64
  %532 = inttoptr i64 %531 to ptr
  %533 = call i32 @rxops_bridge_transpose_nd(ptr %530, ptr %532, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 3, i64 1024, i64 1, i64 32, i64 1, i64 1, i64 3, i64 32, i64 1, i64 1024, i64 1, i64 1)
  %534 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64))
  %535 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %534, 0
  %536 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %535, ptr %534, 1
  %537 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %536, i64 0, 2
  %538 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %537, i64 96, 3, 0
  %539 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %538, i64 1, 3, 1
  %540 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %539, i64 1024, 3, 2
  %541 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %540, i64 1024, 4, 0
  %542 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %541, i64 1024, 4, 1
  %543 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %542, i64 1, 4, 2
  %544 = ptrtoint ptr %534 to i64
  %545 = inttoptr i64 %544 to ptr
  %546 = ptrtoint ptr %517 to i64
  %547 = inttoptr i64 %546 to ptr
  %548 = call i32 @rxops_bridge_reshape_bytes(ptr %545, ptr %547, i64 196608)
  %549 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 64) to i64), i64 64))
  %550 = ptrtoint ptr %549 to i64
  %551 = add i64 %550, 63
  %552 = urem i64 %551, 64
  %553 = sub i64 %551, %552
  %554 = inttoptr i64 %553 to ptr
  %555 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %549, 0
  %556 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %555, ptr %554, 1
  %557 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %556, i64 0, 2
  %558 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %557, i64 1, 3, 0
  %559 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %558, i64 64, 3, 1
  %560 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %559, i64 64, 4, 0
  %561 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %560, i64 1, 4, 1
  %562 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64))
  %563 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %562, 0
  %564 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %563, ptr %562, 1
  %565 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %564, i64 0, 2
  %566 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %565, i64 1, 3, 0
  %567 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %566, i64 64, 3, 1
  %568 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %567, i64 1, 3, 2
  %569 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %568, i64 1024, 3, 3
  %570 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %569, i64 65536, 4, 0
  %571 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %570, i64 1024, 4, 1
  %572 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %571, i64 1024, 4, 2
  %573 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %572, i64 1, 4, 3
  %574 = ptrtoint ptr %562 to i64
  %575 = inttoptr i64 %574 to ptr
  %576 = ptrtoint ptr %534 to i64
  %577 = inttoptr i64 %576 to ptr
  %578 = ptrtoint ptr %554 to i64
  %579 = inttoptr i64 %578 to ptr
  %580 = call i32 @rxops_bridge_gather_nd(ptr %575, ptr %577, ptr %579, i64 0, i64 2, i64 96, i64 1, i64 1024, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1024, i64 1, i64 1)
  %581 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64))
  %582 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %581, 0
  %583 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %582, ptr %581, 1
  %584 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %583, i64 0, 2
  %585 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %584, i64 1, 3, 0
  %586 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %585, i64 1024, 3, 1
  %587 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %586, i64 1, 3, 2
  %588 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %587, i64 64, 3, 3
  %589 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %588, i64 65536, 4, 0
  %590 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %589, i64 64, 4, 1
  %591 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %590, i64 64, 4, 2
  %592 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %591, i64 1, 4, 3
  %593 = ptrtoint ptr %581 to i64
  %594 = inttoptr i64 %593 to ptr
  %595 = ptrtoint ptr %562 to i64
  %596 = inttoptr i64 %595 to ptr
  %597 = call i32 @rxops_bridge_transpose_nd(ptr %594, ptr %596, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1024, i64 1, i64 1, i64 1, i64 1024, i64 1, i64 64, i64 1, i64 1)
  %598 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64), i64 64))
  %599 = ptrtoint ptr %598 to i64
  %600 = add i64 %599, 63
  %601 = urem i64 %600, 64
  %602 = sub i64 %600, %601
  %603 = inttoptr i64 %602 to ptr
  %604 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %598, 0
  %605 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %604, ptr %603, 1
  %606 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %605, i64 0, 2
  %607 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %606, i64 2048, 3, 0
  %608 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %607, i64 1, 3, 1
  %609 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %608, i64 32, 3, 2
  %610 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %609, i64 32, 4, 0
  %611 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %610, i64 32, 4, 1
  %612 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %611, i64 1, 4, 2
  %613 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 3072) to i64), i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)))
  %614 = ptrtoint ptr %613 to i64
  %615 = add i64 %614, sub (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 1)
  %616 = urem i64 %615, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %617 = sub i64 %615, %616
  %618 = inttoptr i64 %617 to ptr
  %619 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %613, 0
  %620 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %619, ptr %618, 1
  %621 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %620, i64 0, 2
  %622 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %621, i64 3, 3, 0
  %623 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %622, i64 1024, 3, 1
  %624 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %623, i64 1024, 4, 0
  %625 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %624, i64 1, 4, 1
  %626 = call ptr @llvm.stacksave()
  %627 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, ptr %627, align 8
  %628 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %627, 1
  %629 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %625, ptr %629, align 8
  %630 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %629, 1
  %631 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %628, ptr %631, align 8
  %632 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %630, ptr %632, align 8
  call void @memrefCopy(i64 4, ptr %631, ptr %632)
  call void @llvm.stackrestore(ptr %626)
  %633 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64))
  %634 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %633, 0
  %635 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %634, ptr %633, 1
  %636 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %635, i64 0, 2
  %637 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %636, i64 3, 3, 0
  %638 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %637, i64 1024, 3, 1
  %639 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %638, i64 1, 3, 2
  %640 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %639, i64 32, 3, 3
  %641 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %640, i64 32768, 4, 0
  %642 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %641, i64 32, 4, 1
  %643 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %642, i64 32, 4, 2
  %644 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %643, i64 1, 4, 3
  %645 = ptrtoint ptr %633 to i64
  %646 = inttoptr i64 %645 to ptr
  %647 = ptrtoint ptr %603 to i64
  %648 = inttoptr i64 %647 to ptr
  %649 = ptrtoint ptr %618 to i64
  %650 = inttoptr i64 %649 to ptr
  %651 = call i32 @rxops_bridge_gather_nd(ptr %646, ptr %648, ptr %650, i64 0, i64 2, i64 2048, i64 1, i64 32, i64 1, i64 1, i64 1, i64 3, i64 1024, i64 1, i64 1, i64 1, i64 1, i64 3, i64 1024, i64 1, i64 32, i64 1, i64 1)
  %652 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64))
  %653 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %652, 0
  %654 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %653, ptr %652, 1
  %655 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %654, i64 0, 2
  %656 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %655, i64 3, 3, 0
  %657 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %656, i64 32, 3, 1
  %658 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %657, i64 1, 3, 2
  %659 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %658, i64 1024, 3, 3
  %660 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %659, i64 32768, 4, 0
  %661 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %660, i64 1024, 4, 1
  %662 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %661, i64 1024, 4, 2
  %663 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %662, i64 1, 4, 3
  %664 = ptrtoint ptr %652 to i64
  %665 = inttoptr i64 %664 to ptr
  %666 = ptrtoint ptr %633 to i64
  %667 = inttoptr i64 %666 to ptr
  %668 = call i32 @rxops_bridge_transpose_nd(ptr %665, ptr %667, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 3, i64 1024, i64 1, i64 32, i64 1, i64 1, i64 3, i64 32, i64 1, i64 1024, i64 1, i64 1)
  %669 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64))
  %670 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %669, 0
  %671 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %670, ptr %669, 1
  %672 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %671, i64 0, 2
  %673 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %672, i64 96, 3, 0
  %674 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %673, i64 1, 3, 1
  %675 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %674, i64 1024, 3, 2
  %676 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %675, i64 1024, 4, 0
  %677 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %676, i64 1024, 4, 1
  %678 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %677, i64 1, 4, 2
  %679 = ptrtoint ptr %669 to i64
  %680 = inttoptr i64 %679 to ptr
  %681 = ptrtoint ptr %652 to i64
  %682 = inttoptr i64 %681 to ptr
  %683 = call i32 @rxops_bridge_reshape_bytes(ptr %680, ptr %682, i64 196608)
  %684 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 64) to i64), i64 64))
  %685 = ptrtoint ptr %684 to i64
  %686 = add i64 %685, 63
  %687 = urem i64 %686, 64
  %688 = sub i64 %686, %687
  %689 = inttoptr i64 %688 to ptr
  %690 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %684, 0
  %691 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %690, ptr %689, 1
  %692 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %691, i64 0, 2
  %693 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %692, i64 1, 3, 0
  %694 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %693, i64 64, 3, 1
  %695 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %694, i64 64, 4, 0
  %696 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %695, i64 1, 4, 1
  %697 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64))
  %698 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %697, 0
  %699 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %698, ptr %697, 1
  %700 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %699, i64 0, 2
  %701 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %700, i64 1, 3, 0
  %702 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %701, i64 64, 3, 1
  %703 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %702, i64 1, 3, 2
  %704 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %703, i64 1024, 3, 3
  %705 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %704, i64 65536, 4, 0
  %706 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %705, i64 1024, 4, 1
  %707 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %706, i64 1024, 4, 2
  %708 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %707, i64 1, 4, 3
  %709 = ptrtoint ptr %697 to i64
  %710 = inttoptr i64 %709 to ptr
  %711 = ptrtoint ptr %669 to i64
  %712 = inttoptr i64 %711 to ptr
  %713 = ptrtoint ptr %689 to i64
  %714 = inttoptr i64 %713 to ptr
  %715 = call i32 @rxops_bridge_gather_nd(ptr %710, ptr %712, ptr %714, i64 0, i64 2, i64 96, i64 1, i64 1024, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1024, i64 1, i64 1)
  %716 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64))
  %717 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %716, 0
  %718 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %717, ptr %716, 1
  %719 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %718, i64 0, 2
  %720 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %719, i64 1, 3, 0
  %721 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %720, i64 1024, 3, 1
  %722 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %721, i64 1, 3, 2
  %723 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %722, i64 64, 3, 3
  %724 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %723, i64 65536, 4, 0
  %725 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %724, i64 64, 4, 1
  %726 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %725, i64 64, 4, 2
  %727 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %726, i64 1, 4, 3
  %728 = ptrtoint ptr %716 to i64
  %729 = inttoptr i64 %728 to ptr
  %730 = ptrtoint ptr %697 to i64
  %731 = inttoptr i64 %730 to ptr
  %732 = call i32 @rxops_bridge_transpose_nd(ptr %729, ptr %731, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1024, i64 1, i64 1, i64 1, i64 1024, i64 1, i64 64, i64 1, i64 1)
  %733 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %734 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %733, 0
  %735 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %734, ptr %733, 1
  %736 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %735, i64 0, 2
  %737 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %736, i64 1, 3, 0
  %738 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %737, i64 1024, 3, 1
  %739 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %738, i64 8, 3, 2
  %740 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %739, i64 64, 3, 3
  %741 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %740, i64 524288, 4, 0
  %742 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %741, i64 512, 4, 1
  %743 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %742, i64 64, 4, 2
  %744 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %743, i64 1, 4, 3
  %745 = ptrtoint ptr %733 to i64
  %746 = inttoptr i64 %745 to ptr
  %747 = ptrtoint ptr %408 to i64
  %748 = inttoptr i64 %747 to ptr
  %749 = call i32 @rxops_bridge_slice_nd(ptr %746, ptr %748, i64 2, i64 4, i64 1, i64 1024, i64 8, i64 64, i64 1, i64 1, i64 1, i64 1024, i64 8, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 0, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %750 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 1572864) to i64))
  %751 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %750, 0
  %752 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %751, ptr %750, 1
  %753 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %752, i64 0, 2
  %754 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %753, i64 1, 3, 0
  %755 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %754, i64 1024, 3, 1
  %756 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %755, i64 8, 3, 2
  %757 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %756, i64 192, 3, 3
  %758 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %757, i64 1572864, 4, 0
  %759 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %758, i64 1536, 4, 1
  %760 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %759, i64 192, 4, 2
  %761 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %760, i64 1, 4, 3
  %762 = ptrtoint ptr %750 to i64
  %763 = inttoptr i64 %762 to ptr
  %764 = ptrtoint ptr %408 to i64
  %765 = inttoptr i64 %764 to ptr
  %766 = call i32 @rxops_bridge_slice_nd(ptr %763, ptr %765, i64 2, i64 4, i64 1, i64 1024, i64 8, i64 192, i64 1, i64 1, i64 1, i64 1024, i64 8, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %767 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %768 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %767, 0
  %769 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %768, ptr %767, 1
  %770 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %769, i64 0, 2
  %771 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %770, i64 1, 3, 0
  %772 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %771, i64 1024, 3, 1
  %773 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %772, i64 8, 3, 2
  %774 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %773, i64 64, 3, 3
  %775 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %774, i64 524288, 4, 0
  %776 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %775, i64 512, 4, 1
  %777 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %776, i64 64, 4, 2
  %778 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %777, i64 1, 4, 3
  %779 = ptrtoint ptr %767 to i64
  %780 = inttoptr i64 %779 to ptr
  %781 = ptrtoint ptr %733 to i64
  %782 = inttoptr i64 %781 to ptr
  %783 = ptrtoint ptr %581 to i64
  %784 = inttoptr i64 %783 to ptr
  %785 = ptrtoint ptr %716 to i64
  %786 = inttoptr i64 %785 to ptr
  %787 = call i32 @rxops_bridge_rope_contiguous_f16(ptr %780, ptr %782, ptr %784, ptr %786, i64 1024, i64 8, i64 64, i64 1)
  %788 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %789 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %788, 0
  %790 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %789, ptr %788, 1
  %791 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %790, i64 0, 2
  %792 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %791, i64 1, 3, 0
  %793 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %792, i64 1024, 3, 1
  %794 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %793, i64 8, 3, 2
  %795 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %794, i64 256, 3, 3
  %796 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %795, i64 2097152, 4, 0
  %797 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %796, i64 2048, 4, 1
  %798 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %797, i64 256, 4, 2
  %799 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %798, i64 1, 4, 3
  %800 = ptrtoint ptr %788 to i64
  %801 = inttoptr i64 %800 to ptr
  %802 = ptrtoint ptr %767 to i64
  %803 = inttoptr i64 %802 to ptr
  %804 = ptrtoint ptr %750 to i64
  %805 = inttoptr i64 %804 to ptr
  %806 = call i32 @rxops_bridge_concat2_nd(ptr %801, ptr %803, ptr %805, i64 2, i64 4, i64 3, i64 1, i64 1024, i64 8, i64 256, i64 1, i64 1, i64 1, i64 1024, i64 8, i64 64, i64 1, i64 1, i64 1, i64 1024, i64 8, i64 192, i64 1, i64 1)
  %807 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 131072) to i64))
  %808 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %807, 0
  %809 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %808, ptr %807, 1
  %810 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %809, i64 0, 2
  %811 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %810, i64 1, 3, 0
  %812 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %811, i64 1024, 3, 1
  %813 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %812, i64 2, 3, 2
  %814 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %813, i64 64, 3, 3
  %815 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %814, i64 131072, 4, 0
  %816 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %815, i64 128, 4, 1
  %817 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %816, i64 64, 4, 2
  %818 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %817, i64 1, 4, 3
  %819 = ptrtoint ptr %807 to i64
  %820 = inttoptr i64 %819 to ptr
  %821 = ptrtoint ptr %444 to i64
  %822 = inttoptr i64 %821 to ptr
  %823 = call i32 @rxops_bridge_slice_nd(ptr %820, ptr %822, i64 2, i64 4, i64 1, i64 1024, i64 2, i64 64, i64 1, i64 1, i64 1, i64 1024, i64 2, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 0, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %824 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 393216) to i64))
  %825 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %824, 0
  %826 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %825, ptr %824, 1
  %827 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %826, i64 0, 2
  %828 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %827, i64 1, 3, 0
  %829 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %828, i64 1024, 3, 1
  %830 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %829, i64 2, 3, 2
  %831 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %830, i64 192, 3, 3
  %832 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %831, i64 393216, 4, 0
  %833 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %832, i64 384, 4, 1
  %834 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %833, i64 192, 4, 2
  %835 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %834, i64 1, 4, 3
  %836 = ptrtoint ptr %824 to i64
  %837 = inttoptr i64 %836 to ptr
  %838 = ptrtoint ptr %444 to i64
  %839 = inttoptr i64 %838 to ptr
  %840 = call i32 @rxops_bridge_slice_nd(ptr %837, ptr %839, i64 2, i64 4, i64 1, i64 1024, i64 2, i64 192, i64 1, i64 1, i64 1, i64 1024, i64 2, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %841 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 131072) to i64))
  %842 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %841, 0
  %843 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %842, ptr %841, 1
  %844 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %843, i64 0, 2
  %845 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %844, i64 1, 3, 0
  %846 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %845, i64 1024, 3, 1
  %847 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %846, i64 2, 3, 2
  %848 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %847, i64 64, 3, 3
  %849 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %848, i64 131072, 4, 0
  %850 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %849, i64 128, 4, 1
  %851 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %850, i64 64, 4, 2
  %852 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %851, i64 1, 4, 3
  %853 = ptrtoint ptr %841 to i64
  %854 = inttoptr i64 %853 to ptr
  %855 = ptrtoint ptr %807 to i64
  %856 = inttoptr i64 %855 to ptr
  %857 = ptrtoint ptr %581 to i64
  %858 = inttoptr i64 %857 to ptr
  %859 = ptrtoint ptr %716 to i64
  %860 = inttoptr i64 %859 to ptr
  %861 = call i32 @rxops_bridge_rope_contiguous_f16(ptr %854, ptr %856, ptr %858, ptr %860, i64 1024, i64 2, i64 64, i64 1)
  %862 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 524288) to i64))
  %863 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %862, 0
  %864 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %863, ptr %862, 1
  %865 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %864, i64 0, 2
  %866 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %865, i64 1, 3, 0
  %867 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %866, i64 1024, 3, 1
  %868 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %867, i64 2, 3, 2
  %869 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %868, i64 256, 3, 3
  %870 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %869, i64 524288, 4, 0
  %871 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %870, i64 512, 4, 1
  %872 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %871, i64 256, 4, 2
  %873 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %872, i64 1, 4, 3
  %874 = ptrtoint ptr %862 to i64
  %875 = inttoptr i64 %874 to ptr
  %876 = ptrtoint ptr %841 to i64
  %877 = inttoptr i64 %876 to ptr
  %878 = ptrtoint ptr %824 to i64
  %879 = inttoptr i64 %878 to ptr
  %880 = call i32 @rxops_bridge_concat2_nd(ptr %875, ptr %877, ptr %879, i64 2, i64 4, i64 3, i64 1, i64 1024, i64 2, i64 256, i64 1, i64 1, i64 1, i64 1024, i64 2, i64 64, i64 1, i64 1, i64 1, i64 1024, i64 2, i64 192, i64 1, i64 1)
  %881 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  %882 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %881, 0
  %883 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %882, ptr %881, 1
  %884 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %883, i64 0, 2
  %885 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %884, i64 1, 3, 0
  %886 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %885, i64 1, 3, 1
  %887 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %886, i64 1024, 3, 2
  %888 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %887, i64 1024, 3, 3
  %889 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %888, i64 1048576, 4, 0
  %890 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %889, i64 1048576, 4, 1
  %891 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %890, i64 1024, 4, 2
  %892 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %891, i64 1, 4, 3
  %893 = call ptr @llvm.stacksave()
  %894 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %54, ptr %894, align 8
  %895 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %894, 1
  %896 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %892, ptr %896, align 8
  %897 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %896, 1
  %898 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %895, ptr %898, align 8
  %899 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %897, ptr %899, align 8
  call void @memrefCopy(i64 4, ptr %898, ptr %899)
  call void @llvm.stackrestore(ptr %893)
  %900 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %901 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %900, 0
  %902 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %901, ptr %900, 1
  %903 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %902, i64 0, 2
  %904 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %903, i64 1, 3, 0
  %905 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %904, i64 1024, 3, 1
  %906 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %905, i64 2048, 3, 2
  %907 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %906, i64 2097152, 4, 0
  %908 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %907, i64 2048, 4, 1
  %909 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %908, i64 1, 4, 2
  %910 = ptrtoint ptr %900 to i64
  %911 = inttoptr i64 %910 to ptr
  %912 = ptrtoint ptr %788 to i64
  %913 = inttoptr i64 %912 to ptr
  %914 = ptrtoint ptr %862 to i64
  %915 = inttoptr i64 %914 to ptr
  %916 = ptrtoint ptr %374 to i64
  %917 = inttoptr i64 %916 to ptr
  %918 = ptrtoint ptr %881 to i64
  %919 = inttoptr i64 %918 to ptr
  %920 = call i32 @rxops_bridge_fattention_f16(ptr %911, ptr %913, ptr %915, ptr %917, ptr %919, i64 1, i64 1024, i64 1024, i64 8, i64 2, i64 256, i64 4589168020290535424, i64 1)
  %921 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %922 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %921, 0
  %923 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %922, ptr %921, 1
  %924 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %923, i64 0, 2
  %925 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %924, i64 1, 3, 0
  %926 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %925, i64 1024, 3, 1
  %927 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %926, i64 2048, 3, 2
  %928 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %927, i64 2097152, 4, 0
  %929 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %928, i64 2048, 4, 1
  %930 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %929, i64 1, 4, 2
  %931 = ptrtoint ptr %921 to i64
  %932 = inttoptr i64 %931 to ptr
  %933 = ptrtoint ptr %220 to i64
  %934 = inttoptr i64 %933 to ptr
  %935 = call i32 @rxops_bridge_reshape_bytes(ptr %932, ptr %934, i64 4194304)
  %936 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %937 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %936, 0
  %938 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %937, ptr %936, 1
  %939 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %938, i64 0, 2
  %940 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %939, i64 1, 3, 0
  %941 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %940, i64 1024, 3, 1
  %942 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %941, i64 2048, 3, 2
  %943 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %942, i64 2097152, 4, 0
  %944 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %943, i64 2048, 4, 1
  %945 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %944, i64 1, 4, 2
  %946 = ptrtoint ptr %936 to i64
  %947 = inttoptr i64 %946 to ptr
  %948 = ptrtoint ptr %900 to i64
  %949 = inttoptr i64 %948 to ptr
  %950 = ptrtoint ptr %921 to i64
  %951 = inttoptr i64 %950 to ptr
  %952 = call i32 @rxops_bridge_mul_f16(ptr %947, ptr %949, ptr %951, i64 2097152)
  %953 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 2097152) to i64), i64 64))
  %954 = ptrtoint ptr %953 to i64
  %955 = add i64 %954, 63
  %956 = urem i64 %955, 64
  %957 = sub i64 %955, %956
  %958 = inttoptr i64 %957 to ptr
  %959 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %953, 0
  %960 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %959, ptr %958, 1
  %961 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %960, i64 0, 2
  %962 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %961, i64 2048, 3, 0
  %963 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %962, i64 1024, 3, 1
  %964 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %963, i64 1024, 4, 0
  %965 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %964, i64 1, 4, 1
  %966 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 32768) to i64), i64 64))
  %967 = ptrtoint ptr %966 to i64
  %968 = add i64 %967, 63
  %969 = urem i64 %968, 64
  %970 = sub i64 %968, %969
  %971 = inttoptr i64 %970 to ptr
  %972 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %966, 0
  %973 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %972, ptr %971, 1
  %974 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %973, i64 0, 2
  %975 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %974, i64 2048, 3, 0
  %976 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %975, i64 16, 3, 1
  %977 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %976, i64 16, 4, 0
  %978 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %977, i64 1, 4, 1
  %979 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 32768) to i64), i64 64))
  %980 = ptrtoint ptr %979 to i64
  %981 = add i64 %980, 63
  %982 = urem i64 %981, 64
  %983 = sub i64 %981, %982
  %984 = inttoptr i64 %983 to ptr
  %985 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %979, 0
  %986 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %985, ptr %984, 1
  %987 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %986, i64 0, 2
  %988 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %987, i64 2048, 3, 0
  %989 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %988, i64 16, 3, 1
  %990 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %989, i64 16, 4, 0
  %991 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %990, i64 1, 4, 1
  %992 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %993 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %992, 0
  %994 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %993, ptr %992, 1
  %995 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %994, i64 0, 2
  %996 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %995, i64 1, 3, 0
  %997 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %996, i64 1024, 3, 1
  %998 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %997, i64 2048, 3, 2
  %999 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %998, i64 2097152, 4, 0
  %1000 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %999, i64 2048, 4, 1
  %1001 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1000, i64 1, 4, 2
  %1002 = ptrtoint ptr %992 to i64
  %1003 = inttoptr i64 %1002 to ptr
  %1004 = ptrtoint ptr %936 to i64
  %1005 = inttoptr i64 %1004 to ptr
  %1006 = ptrtoint ptr %958 to i64
  %1007 = inttoptr i64 %1006 to ptr
  %1008 = ptrtoint ptr %984 to i64
  %1009 = inttoptr i64 %1008 to ptr
  %1010 = ptrtoint ptr %971 to i64
  %1011 = inttoptr i64 %1010 to ptr
  %1012 = call i32 @rxops_bridge_a16matmul_f16(ptr %1003, ptr %1005, ptr %1007, ptr %1009, ptr %1011, i64 1024, i64 2048, i64 2048, i64 128, i64 4)
  %1013 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2097152) to i64))
  %1014 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1013, 0
  %1015 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1014, ptr %1013, 1
  %1016 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1015, i64 0, 2
  %1017 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1016, i64 1, 3, 0
  %1018 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1017, i64 1024, 3, 1
  %1019 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1018, i64 2048, 3, 2
  %1020 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1019, i64 2097152, 4, 0
  %1021 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1020, i64 2048, 4, 1
  %1022 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1021, i64 1, 4, 2
  %1023 = call ptr @llvm.stacksave()
  %1024 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, ptr %1024, align 8
  %1025 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %1024, 1
  %1026 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %1022, ptr %1026, align 8
  %1027 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %1026, 1
  %1028 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1025, ptr %1028, align 8
  %1029 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1027, ptr %1029, align 8
  call void @memrefCopy(i64 4, ptr %1028, ptr %1029)
  call void @llvm.stackrestore(ptr %1023)
  %1030 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %1031 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1030, 0
  %1032 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1031, ptr %1030, 1
  %1033 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1032, i64 0, 2
  %1034 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1033, i64 1, 3, 0
  %1035 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1034, i64 1024, 3, 1
  %1036 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1035, i64 2048, 3, 2
  %1037 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1036, i64 2097152, 4, 0
  %1038 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1037, i64 2048, 4, 1
  %1039 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1038, i64 1, 4, 2
  %1040 = ptrtoint ptr %1030 to i64
  %1041 = inttoptr i64 %1040 to ptr
  %1042 = ptrtoint ptr %1013 to i64
  %1043 = inttoptr i64 %1042 to ptr
  %1044 = ptrtoint ptr %992 to i64
  %1045 = inttoptr i64 %1044 to ptr
  %1046 = call i32 @rxops_bridge_add_f16(ptr %1041, ptr %1043, ptr %1045, i64 2097152)
  %1047 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64), i64 64))
  %1048 = ptrtoint ptr %1047 to i64
  %1049 = add i64 %1048, 63
  %1050 = urem i64 %1049, 64
  %1051 = sub i64 %1049, %1050
  %1052 = inttoptr i64 %1051 to ptr
  %1053 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1047, 0
  %1054 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1053, ptr %1052, 1
  %1055 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1054, i64 0, 2
  %1056 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1055, i64 1, 3, 0
  %1057 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1056, i64 1, 3, 1
  %1058 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1057, i64 2048, 3, 2
  %1059 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1058, i64 2048, 4, 0
  %1060 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1059, i64 2048, 4, 1
  %1061 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1060, i64 1, 4, 2
  %1062 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %1063 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1062, 0
  %1064 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1063, ptr %1062, 1
  %1065 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1064, i64 0, 2
  %1066 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1065, i64 1, 3, 0
  %1067 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1066, i64 1024, 3, 1
  %1068 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1067, i64 2048, 3, 2
  %1069 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1068, i64 2097152, 4, 0
  %1070 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1069, i64 2048, 4, 1
  %1071 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1070, i64 1, 4, 2
  %1072 = ptrtoint ptr %1062 to i64
  %1073 = inttoptr i64 %1072 to ptr
  %1074 = ptrtoint ptr %1030 to i64
  %1075 = inttoptr i64 %1074 to ptr
  %1076 = ptrtoint ptr %1052 to i64
  %1077 = inttoptr i64 %1076 to ptr
  %1078 = call i32 @rxops_bridge_rms_norm_f16(ptr %1073, ptr %1075, ptr %1077, i64 1024, i64 2048, i64 4517329193108106637)
  %1079 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 6291456) to i64), i64 64))
  %1080 = ptrtoint ptr %1079 to i64
  %1081 = add i64 %1080, 63
  %1082 = urem i64 %1081, 64
  %1083 = sub i64 %1081, %1082
  %1084 = inttoptr i64 %1083 to ptr
  %1085 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1079, 0
  %1086 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1085, ptr %1084, 1
  %1087 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1086, i64 0, 2
  %1088 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1087, i64 6144, 3, 0
  %1089 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1088, i64 1024, 3, 1
  %1090 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1089, i64 1024, 4, 0
  %1091 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1090, i64 1, 4, 1
  %1092 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 98304) to i64), i64 64))
  %1093 = ptrtoint ptr %1092 to i64
  %1094 = add i64 %1093, 63
  %1095 = urem i64 %1094, 64
  %1096 = sub i64 %1094, %1095
  %1097 = inttoptr i64 %1096 to ptr
  %1098 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1092, 0
  %1099 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1098, ptr %1097, 1
  %1100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1099, i64 0, 2
  %1101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1100, i64 6144, 3, 0
  %1102 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1101, i64 16, 3, 1
  %1103 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1102, i64 16, 4, 0
  %1104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1103, i64 1, 4, 1
  %1105 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64), i64 64))
  %1106 = ptrtoint ptr %1105 to i64
  %1107 = add i64 %1106, 63
  %1108 = urem i64 %1107, 64
  %1109 = sub i64 %1107, %1108
  %1110 = inttoptr i64 %1109 to ptr
  %1111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1105, 0
  %1112 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1111, ptr %1110, 1
  %1113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1112, i64 0, 2
  %1114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1113, i64 6144, 3, 0
  %1115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1114, i64 16, 3, 1
  %1116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1115, i64 16, 4, 0
  %1117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1116, i64 1, 4, 1
  %1118 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6291456) to i64))
  %1119 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1118, 0
  %1120 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1119, ptr %1118, 1
  %1121 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1120, i64 0, 2
  %1122 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1121, i64 1, 3, 0
  %1123 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1122, i64 1024, 3, 1
  %1124 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1123, i64 6144, 3, 2
  %1125 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1124, i64 6291456, 4, 0
  %1126 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1125, i64 6144, 4, 1
  %1127 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1126, i64 1, 4, 2
  %1128 = ptrtoint ptr %1118 to i64
  %1129 = inttoptr i64 %1128 to ptr
  %1130 = ptrtoint ptr %1062 to i64
  %1131 = inttoptr i64 %1130 to ptr
  %1132 = ptrtoint ptr %1084 to i64
  %1133 = inttoptr i64 %1132 to ptr
  %1134 = ptrtoint ptr %1110 to i64
  %1135 = inttoptr i64 %1134 to ptr
  %1136 = ptrtoint ptr %1097 to i64
  %1137 = inttoptr i64 %1136 to ptr
  %1138 = call i32 @rxops_bridge_a16matmul_f16(ptr %1129, ptr %1131, ptr %1133, ptr %1135, ptr %1137, i64 1024, i64 2048, i64 6144, i64 128, i64 4)
  %1139 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6291456) to i64))
  %1140 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1139, 0
  %1141 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1140, ptr %1139, 1
  %1142 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1141, i64 0, 2
  %1143 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1142, i64 1, 3, 0
  %1144 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1143, i64 1024, 3, 1
  %1145 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1144, i64 6144, 3, 2
  %1146 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1145, i64 6291456, 4, 0
  %1147 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1146, i64 6144, 4, 1
  %1148 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1147, i64 1, 4, 2
  %1149 = ptrtoint ptr %1139 to i64
  %1150 = inttoptr i64 %1149 to ptr
  %1151 = ptrtoint ptr %1118 to i64
  %1152 = inttoptr i64 %1151 to ptr
  %1153 = call i32 @rxops_bridge_silu_f16(ptr %1150, ptr %1152, i64 6291456)
  %1154 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 6291456) to i64), i64 64))
  %1155 = ptrtoint ptr %1154 to i64
  %1156 = add i64 %1155, 63
  %1157 = urem i64 %1156, 64
  %1158 = sub i64 %1156, %1157
  %1159 = inttoptr i64 %1158 to ptr
  %1160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1154, 0
  %1161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1160, ptr %1159, 1
  %1162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1161, i64 0, 2
  %1163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1162, i64 6144, 3, 0
  %1164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1163, i64 1024, 3, 1
  %1165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1164, i64 1024, 4, 0
  %1166 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1165, i64 1, 4, 1
  %1167 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 98304) to i64), i64 64))
  %1168 = ptrtoint ptr %1167 to i64
  %1169 = add i64 %1168, 63
  %1170 = urem i64 %1169, 64
  %1171 = sub i64 %1169, %1170
  %1172 = inttoptr i64 %1171 to ptr
  %1173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1167, 0
  %1174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1173, ptr %1172, 1
  %1175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1174, i64 0, 2
  %1176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1175, i64 6144, 3, 0
  %1177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1176, i64 16, 3, 1
  %1178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1177, i64 16, 4, 0
  %1179 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1178, i64 1, 4, 1
  %1180 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64), i64 64))
  %1181 = ptrtoint ptr %1180 to i64
  %1182 = add i64 %1181, 63
  %1183 = urem i64 %1182, 64
  %1184 = sub i64 %1182, %1183
  %1185 = inttoptr i64 %1184 to ptr
  %1186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1180, 0
  %1187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1186, ptr %1185, 1
  %1188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1187, i64 0, 2
  %1189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1188, i64 6144, 3, 0
  %1190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1189, i64 16, 3, 1
  %1191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1190, i64 16, 4, 0
  %1192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1191, i64 1, 4, 1
  %1193 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6291456) to i64))
  %1194 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1193, 0
  %1195 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1194, ptr %1193, 1
  %1196 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1195, i64 0, 2
  %1197 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1196, i64 1, 3, 0
  %1198 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1197, i64 1024, 3, 1
  %1199 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1198, i64 6144, 3, 2
  %1200 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1199, i64 6291456, 4, 0
  %1201 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1200, i64 6144, 4, 1
  %1202 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1201, i64 1, 4, 2
  %1203 = ptrtoint ptr %1193 to i64
  %1204 = inttoptr i64 %1203 to ptr
  %1205 = ptrtoint ptr %1062 to i64
  %1206 = inttoptr i64 %1205 to ptr
  %1207 = ptrtoint ptr %1159 to i64
  %1208 = inttoptr i64 %1207 to ptr
  %1209 = ptrtoint ptr %1185 to i64
  %1210 = inttoptr i64 %1209 to ptr
  %1211 = ptrtoint ptr %1172 to i64
  %1212 = inttoptr i64 %1211 to ptr
  %1213 = call i32 @rxops_bridge_a16matmul_f16(ptr %1204, ptr %1206, ptr %1208, ptr %1210, ptr %1212, i64 1024, i64 2048, i64 6144, i64 128, i64 4)
  %1214 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6291456) to i64))
  %1215 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1214, 0
  %1216 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1215, ptr %1214, 1
  %1217 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1216, i64 0, 2
  %1218 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1217, i64 1, 3, 0
  %1219 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1218, i64 1024, 3, 1
  %1220 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1219, i64 6144, 3, 2
  %1221 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1220, i64 6291456, 4, 0
  %1222 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1221, i64 6144, 4, 1
  %1223 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1222, i64 1, 4, 2
  %1224 = ptrtoint ptr %1214 to i64
  %1225 = inttoptr i64 %1224 to ptr
  %1226 = ptrtoint ptr %1139 to i64
  %1227 = inttoptr i64 %1226 to ptr
  %1228 = ptrtoint ptr %1193 to i64
  %1229 = inttoptr i64 %1228 to ptr
  %1230 = call i32 @rxops_bridge_mul_f16(ptr %1225, ptr %1227, ptr %1229, i64 6291456)
  %1231 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 6291456) to i64), i64 64))
  %1232 = ptrtoint ptr %1231 to i64
  %1233 = add i64 %1232, 63
  %1234 = urem i64 %1233, 64
  %1235 = sub i64 %1233, %1234
  %1236 = inttoptr i64 %1235 to ptr
  %1237 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1231, 0
  %1238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1237, ptr %1236, 1
  %1239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1238, i64 0, 2
  %1240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1239, i64 2048, 3, 0
  %1241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1240, i64 3072, 3, 1
  %1242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1241, i64 3072, 4, 0
  %1243 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1242, i64 1, 4, 1
  %1244 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 98304) to i64), i64 64))
  %1245 = ptrtoint ptr %1244 to i64
  %1246 = add i64 %1245, 63
  %1247 = urem i64 %1246, 64
  %1248 = sub i64 %1246, %1247
  %1249 = inttoptr i64 %1248 to ptr
  %1250 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1244, 0
  %1251 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1250, ptr %1249, 1
  %1252 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1251, i64 0, 2
  %1253 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1252, i64 2048, 3, 0
  %1254 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1253, i64 48, 3, 1
  %1255 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1254, i64 48, 4, 0
  %1256 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1255, i64 1, 4, 1
  %1257 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64), i64 64))
  %1258 = ptrtoint ptr %1257 to i64
  %1259 = add i64 %1258, 63
  %1260 = urem i64 %1259, 64
  %1261 = sub i64 %1259, %1260
  %1262 = inttoptr i64 %1261 to ptr
  %1263 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1257, 0
  %1264 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1263, ptr %1262, 1
  %1265 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1264, i64 0, 2
  %1266 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1265, i64 2048, 3, 0
  %1267 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1266, i64 48, 3, 1
  %1268 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1267, i64 48, 4, 0
  %1269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1268, i64 1, 4, 1
  %1270 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %1271 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1270, 0
  %1272 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1271, ptr %1270, 1
  %1273 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1272, i64 0, 2
  %1274 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1273, i64 1, 3, 0
  %1275 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1274, i64 1024, 3, 1
  %1276 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1275, i64 2048, 3, 2
  %1277 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1276, i64 2097152, 4, 0
  %1278 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1277, i64 2048, 4, 1
  %1279 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1278, i64 1, 4, 2
  %1280 = ptrtoint ptr %1270 to i64
  %1281 = inttoptr i64 %1280 to ptr
  %1282 = ptrtoint ptr %1214 to i64
  %1283 = inttoptr i64 %1282 to ptr
  %1284 = ptrtoint ptr %1236 to i64
  %1285 = inttoptr i64 %1284 to ptr
  %1286 = ptrtoint ptr %1262 to i64
  %1287 = inttoptr i64 %1286 to ptr
  %1288 = ptrtoint ptr %1249 to i64
  %1289 = inttoptr i64 %1288 to ptr
  %1290 = call i32 @rxops_bridge_a16matmul_f16(ptr %1281, ptr %1283, ptr %1285, ptr %1287, ptr %1289, i64 1024, i64 6144, i64 2048, i64 128, i64 4)
  %1291 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2097152) to i64))
  %1292 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1291, 0
  %1293 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1292, ptr %1291, 1
  %1294 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1293, i64 0, 2
  %1295 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1294, i64 1, 3, 0
  %1296 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1295, i64 1024, 3, 1
  %1297 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1296, i64 2048, 3, 2
  %1298 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1297, i64 2097152, 4, 0
  %1299 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1298, i64 2048, 4, 1
  %1300 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1299, i64 1, 4, 2
  %1301 = ptrtoint ptr %1291 to i64
  %1302 = inttoptr i64 %1301 to ptr
  %1303 = ptrtoint ptr %1030 to i64
  %1304 = inttoptr i64 %1303 to ptr
  %1305 = ptrtoint ptr %1270 to i64
  %1306 = inttoptr i64 %1305 to ptr
  %1307 = call i32 @rxops_bridge_add_f16(ptr %1302, ptr %1304, ptr %1306, i64 2097152)
  call void @free(ptr %55)
  call void @free(ptr %70)
  call void @free(ptr %109)
  call void @free(ptr %122)
  call void @free(ptr %135)
  call void @free(ptr %237)
  call void @free(ptr %250)
  call void @free(ptr %263)
  call void @free(ptr %297)
  call void @free(ptr %310)
  call void @free(ptr %323)
  call void @free(ptr %391)
  call void @free(ptr %427)
  call void @free(ptr %463)
  call void @free(ptr %478)
  call void @free(ptr %549)
  call void @free(ptr %598)
  call void @free(ptr %684)
  call void @free(ptr %953)
  call void @free(ptr %966)
  call void @free(ptr %979)
  call void @free(ptr %1047)
  call void @free(ptr %1079)
  call void @free(ptr %1092)
  call void @free(ptr %1105)
  call void @free(ptr %1154)
  call void @free(ptr %1167)
  call void @free(ptr %1180)
  call void @free(ptr %1231)
  call void @free(ptr %1244)
  call void @free(ptr %1257)
  %1308 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } undef, { ptr, ptr, i64, [3 x i64], [3 x i64] } %1300, 0
  %1309 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %1308, { ptr, ptr, i64, [4 x i64], [4 x i64] } %873, 1
  %1310 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %1309, { ptr, ptr, i64, [4 x i64], [4 x i64] } %385, 2
  ret { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %1310
}

define void @_mlir_ciface_block7_main(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 3, 1
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 3, 2
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 4, 0
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 4, 1
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %5, 4, 2
  %15 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 2
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 3, 1
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 0
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, 4, 1
  %23 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %3, align 8
  %24 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 0
  %25 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 1
  %26 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 2
  %27 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 3, 0
  %28 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 3, 1
  %29 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 3, 2
  %30 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 3, 3
  %31 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 4, 0
  %32 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 4, 1
  %33 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 4, 2
  %34 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %23, 4, 3
  %35 = call { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } @block7_main(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, ptr %24, ptr %25, i64 %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34)
  store { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %35, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
