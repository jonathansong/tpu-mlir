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

define { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } @block_cache7_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, ptr %27, ptr %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, ptr %38, ptr %39, i64 %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47, i64 %48) {
  %50 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %0, 0
  %51 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %50, ptr %1, 1
  %52 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %51, i64 %2, 2
  %53 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %52, i64 %3, 3, 0
  %54 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %53, i64 %6, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %54, i64 %4, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, i64 %7, 4, 1
  %57 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %56, i64 %5, 3, 2
  %58 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %57, i64 %8, 4, 2
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %9, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, ptr %10, 1
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 %11, 2
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 %12, 3, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 %14, 4, 0
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 %13, 3, 1
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 %15, 4, 1
  %66 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %16, 0
  %67 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %66, ptr %17, 1
  %68 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %67, i64 %18, 2
  %69 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %68, i64 %19, 3, 0
  %70 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %69, i64 %23, 4, 0
  %71 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %70, i64 %20, 3, 1
  %72 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %71, i64 %24, 4, 1
  %73 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %72, i64 %21, 3, 2
  %74 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %73, i64 %25, 4, 2
  %75 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %74, i64 %22, 3, 3
  %76 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %75, i64 %26, 4, 3
  %77 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %27, 0
  %78 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %77, ptr %28, 1
  %79 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %78, i64 %29, 2
  %80 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %79, i64 %30, 3, 0
  %81 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %80, i64 %34, 4, 0
  %82 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %81, i64 %31, 3, 1
  %83 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %82, i64 %35, 4, 1
  %84 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %83, i64 %32, 3, 2
  %85 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %84, i64 %36, 4, 2
  %86 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %85, i64 %33, 3, 3
  %87 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %86, i64 %37, 4, 3
  %88 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %38, 0
  %89 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %88, ptr %39, 1
  %90 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %89, i64 %40, 2
  %91 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %90, i64 %41, 3, 0
  %92 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %91, i64 %45, 4, 0
  %93 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %92, i64 %42, 3, 1
  %94 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %93, i64 %46, 4, 1
  %95 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %94, i64 %43, 3, 2
  %96 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %95, i64 %47, 4, 2
  %97 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %96, i64 %44, 3, 3
  %98 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %97, i64 %48, 4, 3
  %99 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64), i64 64))
  %100 = ptrtoint ptr %99 to i64
  %101 = add i64 %100, 63
  %102 = urem i64 %101, 64
  %103 = sub i64 %101, %102
  %104 = inttoptr i64 %103 to ptr
  %105 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %99, 0
  %106 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %105, ptr %104, 1
  %107 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, i64 0, 2
  %108 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, i64 1, 3, 0
  %109 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %108, i64 1, 3, 1
  %110 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %109, i64 2048, 3, 2
  %111 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %110, i64 2048, 4, 0
  %112 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %111, i64 2048, 4, 1
  %113 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %112, i64 1, 4, 2
  %114 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64), i64 64))
  %115 = ptrtoint ptr %114 to i64
  %116 = add i64 %115, 63
  %117 = urem i64 %116, 64
  %118 = sub i64 %116, %117
  %119 = inttoptr i64 %118 to ptr
  %120 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %114, 0
  %121 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, ptr %119, 1
  %122 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %121, i64 0, 2
  %123 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %122, i64 1, 3, 0
  %124 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %123, i64 1, 3, 1
  %125 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %124, i64 2048, 3, 2
  %126 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %125, i64 2048, 4, 0
  %127 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, i64 2048, 4, 1
  %128 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %127, i64 1, 4, 2
  %129 = call ptr @llvm.stacksave()
  %130 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %58, ptr %130, align 8
  %131 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %130, 1
  %132 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %128, ptr %132, align 8
  %133 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %132, 1
  %134 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %131, ptr %134, align 8
  %135 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %133, ptr %135, align 8
  call void @memrefCopy(i64 4, ptr %134, ptr %135)
  call void @llvm.stackrestore(ptr %129)
  %136 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %137 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %136, 0
  %138 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %137, ptr %136, 1
  %139 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %138, i64 0, 2
  %140 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %139, i64 1, 3, 0
  %141 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %140, i64 1, 3, 1
  %142 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %141, i64 2048, 3, 2
  %143 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %142, i64 2048, 4, 0
  %144 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %143, i64 2048, 4, 1
  %145 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %144, i64 1, 4, 2
  %146 = ptrtoint ptr %136 to i64
  %147 = inttoptr i64 %146 to ptr
  %148 = ptrtoint ptr %119 to i64
  %149 = inttoptr i64 %148 to ptr
  %150 = ptrtoint ptr %104 to i64
  %151 = inttoptr i64 %150 to ptr
  %152 = call i32 @rxops_bridge_rms_norm_f32(ptr %147, ptr %149, ptr %151, i64 1, i64 2048, i64 4517329193108106637)
  %153 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 4194304) to i64), i64 64))
  %154 = ptrtoint ptr %153 to i64
  %155 = add i64 %154, 63
  %156 = urem i64 %155, 64
  %157 = sub i64 %155, %156
  %158 = inttoptr i64 %157 to ptr
  %159 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %153, 0
  %160 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %159, ptr %158, 1
  %161 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %160, i64 0, 2
  %162 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %161, i64 4096, 3, 0
  %163 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %162, i64 1024, 3, 1
  %164 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %163, i64 1024, 4, 0
  %165 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %164, i64 1, 4, 1
  %166 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 65536) to i64), i64 64))
  %167 = ptrtoint ptr %166 to i64
  %168 = add i64 %167, 63
  %169 = urem i64 %168, 64
  %170 = sub i64 %168, %169
  %171 = inttoptr i64 %170 to ptr
  %172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %166, 0
  %173 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %172, ptr %171, 1
  %174 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %173, i64 0, 2
  %175 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %174, i64 4096, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %175, i64 16, 3, 1
  %177 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %176, i64 16, 4, 0
  %178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %177, i64 1, 4, 1
  %179 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64), i64 64))
  %180 = ptrtoint ptr %179 to i64
  %181 = add i64 %180, 63
  %182 = urem i64 %181, 64
  %183 = sub i64 %181, %182
  %184 = inttoptr i64 %183 to ptr
  %185 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %179, 0
  %186 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %185, ptr %184, 1
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %186, i64 0, 2
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, i64 4096, 3, 0
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, i64 16, 3, 1
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 16, 4, 0
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 1, 4, 1
  %192 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 4096) to i64))
  %193 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %192, 0
  %194 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %193, ptr %192, 1
  %195 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %194, i64 0, 2
  %196 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %195, i64 1, 3, 0
  %197 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %196, i64 1, 3, 1
  %198 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %197, i64 4096, 3, 2
  %199 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %198, i64 4096, 4, 0
  %200 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %199, i64 4096, 4, 1
  %201 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %200, i64 1, 4, 2
  %202 = ptrtoint ptr %192 to i64
  %203 = inttoptr i64 %202 to ptr
  %204 = ptrtoint ptr %136 to i64
  %205 = inttoptr i64 %204 to ptr
  %206 = ptrtoint ptr %158 to i64
  %207 = inttoptr i64 %206 to ptr
  %208 = ptrtoint ptr %184 to i64
  %209 = inttoptr i64 %208 to ptr
  %210 = ptrtoint ptr %171 to i64
  %211 = inttoptr i64 %210 to ptr
  %212 = call i32 @rxops_bridge_a16matmul_f16(ptr %203, ptr %205, ptr %207, ptr %209, ptr %211, i64 1, i64 2048, i64 4096, i64 128, i64 4)
  %213 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 4096) to i64))
  %214 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %213, 0
  %215 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %214, ptr %213, 1
  %216 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %215, i64 0, 2
  %217 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %216, i64 1, 3, 0
  %218 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %217, i64 1, 3, 1
  %219 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %218, i64 8, 3, 2
  %220 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %219, i64 512, 3, 3
  %221 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %220, i64 4096, 4, 0
  %222 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %221, i64 4096, 4, 1
  %223 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %222, i64 512, 4, 2
  %224 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %223, i64 1, 4, 3
  %225 = ptrtoint ptr %213 to i64
  %226 = inttoptr i64 %225 to ptr
  %227 = ptrtoint ptr %192 to i64
  %228 = inttoptr i64 %227 to ptr
  %229 = call i32 @rxops_bridge_reshape_bytes(ptr %226, ptr %228, i64 8192)
  %230 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %231 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %230, 0
  %232 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %231, ptr %230, 1
  %233 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %232, i64 0, 2
  %234 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %233, i64 1, 3, 0
  %235 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %234, i64 1, 3, 1
  %236 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %235, i64 8, 3, 2
  %237 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %236, i64 256, 3, 3
  %238 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %237, i64 2048, 4, 0
  %239 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %238, i64 2048, 4, 1
  %240 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %239, i64 256, 4, 2
  %241 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %240, i64 1, 4, 3
  %242 = ptrtoint ptr %230 to i64
  %243 = inttoptr i64 %242 to ptr
  %244 = ptrtoint ptr %213 to i64
  %245 = inttoptr i64 %244 to ptr
  %246 = call i32 @rxops_bridge_slice_nd(ptr %243, ptr %245, i64 2, i64 4, i64 1, i64 1, i64 8, i64 256, i64 1, i64 1, i64 1, i64 1, i64 8, i64 512, i64 1, i64 1, i64 0, i64 0, i64 0, i64 0, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %247 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %248 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %247, 0
  %249 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %248, ptr %247, 1
  %250 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %249, i64 0, 2
  %251 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %250, i64 1, 3, 0
  %252 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %251, i64 1, 3, 1
  %253 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %252, i64 8, 3, 2
  %254 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %253, i64 256, 3, 3
  %255 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %254, i64 2048, 4, 0
  %256 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %255, i64 2048, 4, 1
  %257 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %256, i64 256, 4, 2
  %258 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %257, i64 1, 4, 3
  %259 = ptrtoint ptr %247 to i64
  %260 = inttoptr i64 %259 to ptr
  %261 = ptrtoint ptr %213 to i64
  %262 = inttoptr i64 %261 to ptr
  %263 = call i32 @rxops_bridge_slice_nd(ptr %260, ptr %262, i64 2, i64 4, i64 1, i64 1, i64 8, i64 256, i64 1, i64 1, i64 1, i64 1, i64 8, i64 512, i64 1, i64 1, i64 0, i64 0, i64 0, i64 256, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %264 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %265 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %264, 0
  %266 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %265, ptr %264, 1
  %267 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %266, i64 0, 2
  %268 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %267, i64 1, 3, 0
  %269 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %268, i64 1, 3, 1
  %270 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %269, i64 8, 3, 2
  %271 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %270, i64 256, 3, 3
  %272 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %271, i64 2048, 4, 0
  %273 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %272, i64 2048, 4, 1
  %274 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %273, i64 256, 4, 2
  %275 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %274, i64 1, 4, 3
  %276 = ptrtoint ptr %264 to i64
  %277 = inttoptr i64 %276 to ptr
  %278 = ptrtoint ptr %247 to i64
  %279 = inttoptr i64 %278 to ptr
  %280 = call i32 @rxops_bridge_sigmoid_f16(ptr %277, ptr %279, i64 2048)
  %281 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 524288) to i64), i64 64))
  %282 = ptrtoint ptr %281 to i64
  %283 = add i64 %282, 63
  %284 = urem i64 %283, 64
  %285 = sub i64 %283, %284
  %286 = inttoptr i64 %285 to ptr
  %287 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %281, 0
  %288 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %287, ptr %286, 1
  %289 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %288, i64 0, 2
  %290 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %289, i64 512, 3, 0
  %291 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %290, i64 1024, 3, 1
  %292 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %291, i64 1024, 4, 0
  %293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %292, i64 1, 4, 1
  %294 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 8192) to i64), i64 64))
  %295 = ptrtoint ptr %294 to i64
  %296 = add i64 %295, 63
  %297 = urem i64 %296, 64
  %298 = sub i64 %296, %297
  %299 = inttoptr i64 %298 to ptr
  %300 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %294, 0
  %301 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %300, ptr %299, 1
  %302 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %301, i64 0, 2
  %303 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %302, i64 512, 3, 0
  %304 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %303, i64 16, 3, 1
  %305 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %304, i64 16, 4, 0
  %306 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %305, i64 1, 4, 1
  %307 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 8192) to i64), i64 64))
  %308 = ptrtoint ptr %307 to i64
  %309 = add i64 %308, 63
  %310 = urem i64 %309, 64
  %311 = sub i64 %309, %310
  %312 = inttoptr i64 %311 to ptr
  %313 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %307, 0
  %314 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, ptr %312, 1
  %315 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %314, i64 0, 2
  %316 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %315, i64 512, 3, 0
  %317 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %316, i64 16, 3, 1
  %318 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %317, i64 16, 4, 0
  %319 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %318, i64 1, 4, 1
  %320 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %321 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %320, 0
  %322 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %321, ptr %320, 1
  %323 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %322, i64 0, 2
  %324 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %323, i64 1, 3, 0
  %325 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %324, i64 1, 3, 1
  %326 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %325, i64 512, 3, 2
  %327 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %326, i64 512, 4, 0
  %328 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %327, i64 512, 4, 1
  %329 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %328, i64 1, 4, 2
  %330 = ptrtoint ptr %320 to i64
  %331 = inttoptr i64 %330 to ptr
  %332 = ptrtoint ptr %136 to i64
  %333 = inttoptr i64 %332 to ptr
  %334 = ptrtoint ptr %286 to i64
  %335 = inttoptr i64 %334 to ptr
  %336 = ptrtoint ptr %312 to i64
  %337 = inttoptr i64 %336 to ptr
  %338 = ptrtoint ptr %299 to i64
  %339 = inttoptr i64 %338 to ptr
  %340 = call i32 @rxops_bridge_a16matmul_f16(ptr %331, ptr %333, ptr %335, ptr %337, ptr %339, i64 1, i64 2048, i64 512, i64 128, i64 4)
  %341 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 524288) to i64), i64 64))
  %342 = ptrtoint ptr %341 to i64
  %343 = add i64 %342, 63
  %344 = urem i64 %343, 64
  %345 = sub i64 %343, %344
  %346 = inttoptr i64 %345 to ptr
  %347 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %341, 0
  %348 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %347, ptr %346, 1
  %349 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %348, i64 0, 2
  %350 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %349, i64 512, 3, 0
  %351 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %350, i64 1024, 3, 1
  %352 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %351, i64 1024, 4, 0
  %353 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %352, i64 1, 4, 1
  %354 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 8192) to i64), i64 64))
  %355 = ptrtoint ptr %354 to i64
  %356 = add i64 %355, 63
  %357 = urem i64 %356, 64
  %358 = sub i64 %356, %357
  %359 = inttoptr i64 %358 to ptr
  %360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %354, 0
  %361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %360, ptr %359, 1
  %362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %361, i64 0, 2
  %363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %362, i64 512, 3, 0
  %364 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %363, i64 16, 3, 1
  %365 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %364, i64 16, 4, 0
  %366 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %365, i64 1, 4, 1
  %367 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 8192) to i64), i64 64))
  %368 = ptrtoint ptr %367 to i64
  %369 = add i64 %368, 63
  %370 = urem i64 %369, 64
  %371 = sub i64 %369, %370
  %372 = inttoptr i64 %371 to ptr
  %373 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %367, 0
  %374 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, ptr %372, 1
  %375 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %374, i64 0, 2
  %376 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %375, i64 512, 3, 0
  %377 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %376, i64 16, 3, 1
  %378 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %377, i64 16, 4, 0
  %379 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %378, i64 1, 4, 1
  %380 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %381 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %380, 0
  %382 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %381, ptr %380, 1
  %383 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %382, i64 0, 2
  %384 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %383, i64 1, 3, 0
  %385 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %384, i64 1, 3, 1
  %386 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %385, i64 512, 3, 2
  %387 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %386, i64 512, 4, 0
  %388 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %387, i64 512, 4, 1
  %389 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %388, i64 1, 4, 2
  %390 = ptrtoint ptr %380 to i64
  %391 = inttoptr i64 %390 to ptr
  %392 = ptrtoint ptr %136 to i64
  %393 = inttoptr i64 %392 to ptr
  %394 = ptrtoint ptr %346 to i64
  %395 = inttoptr i64 %394 to ptr
  %396 = ptrtoint ptr %372 to i64
  %397 = inttoptr i64 %396 to ptr
  %398 = ptrtoint ptr %359 to i64
  %399 = inttoptr i64 %398 to ptr
  %400 = call i32 @rxops_bridge_a16matmul_f16(ptr %391, ptr %393, ptr %395, ptr %397, ptr %399, i64 1, i64 2048, i64 512, i64 128, i64 4)
  %401 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %402 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %401, 0
  %403 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %402, ptr %401, 1
  %404 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %403, i64 0, 2
  %405 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %404, i64 1, 3, 0
  %406 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %405, i64 1, 3, 1
  %407 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %406, i64 2, 3, 2
  %408 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %407, i64 256, 3, 3
  %409 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %408, i64 512, 4, 0
  %410 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %409, i64 512, 4, 1
  %411 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %410, i64 256, 4, 2
  %412 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %411, i64 1, 4, 3
  %413 = ptrtoint ptr %401 to i64
  %414 = inttoptr i64 %413 to ptr
  %415 = ptrtoint ptr %320 to i64
  %416 = inttoptr i64 %415 to ptr
  %417 = call i32 @rxops_bridge_reshape_bytes(ptr %414, ptr %416, i64 1024)
  %418 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %419 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %418, 0
  %420 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %419, ptr %418, 1
  %421 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %420, i64 0, 2
  %422 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %421, i64 1, 3, 0
  %423 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %422, i64 1, 3, 1
  %424 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %423, i64 2, 3, 2
  %425 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %424, i64 256, 3, 3
  %426 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %425, i64 512, 4, 0
  %427 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %426, i64 512, 4, 1
  %428 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %427, i64 256, 4, 2
  %429 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %428, i64 1, 4, 3
  %430 = ptrtoint ptr %418 to i64
  %431 = inttoptr i64 %430 to ptr
  %432 = ptrtoint ptr %380 to i64
  %433 = inttoptr i64 %432 to ptr
  %434 = call i32 @rxops_bridge_reshape_bytes(ptr %431, ptr %433, i64 1024)
  %435 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 256) to i64), i64 64))
  %436 = ptrtoint ptr %435 to i64
  %437 = add i64 %436, 63
  %438 = urem i64 %437, 64
  %439 = sub i64 %437, %438
  %440 = inttoptr i64 %439 to ptr
  %441 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %435, 0
  %442 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %441, ptr %440, 1
  %443 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %442, i64 0, 2
  %444 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %443, i64 1, 3, 0
  %445 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %444, i64 1, 3, 1
  %446 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %445, i64 1, 3, 2
  %447 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %446, i64 256, 3, 3
  %448 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %447, i64 256, 4, 0
  %449 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %448, i64 256, 4, 1
  %450 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %449, i64 256, 4, 2
  %451 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %450, i64 1, 4, 3
  %452 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %453 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %452, 0
  %454 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %453, ptr %452, 1
  %455 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %454, i64 0, 2
  %456 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %455, i64 1, 3, 0
  %457 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %456, i64 1, 3, 1
  %458 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %457, i64 8, 3, 2
  %459 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %458, i64 256, 3, 3
  %460 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %459, i64 2048, 4, 0
  %461 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %460, i64 2048, 4, 1
  %462 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %461, i64 256, 4, 2
  %463 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %462, i64 1, 4, 3
  %464 = ptrtoint ptr %452 to i64
  %465 = inttoptr i64 %464 to ptr
  %466 = ptrtoint ptr %230 to i64
  %467 = inttoptr i64 %466 to ptr
  %468 = ptrtoint ptr %440 to i64
  %469 = inttoptr i64 %468 to ptr
  %470 = call i32 @rxops_bridge_rms_norm_f16(ptr %465, ptr %467, ptr %469, i64 8, i64 256, i64 4517329193108106637)
  %471 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 256) to i64), i64 64))
  %472 = ptrtoint ptr %471 to i64
  %473 = add i64 %472, 63
  %474 = urem i64 %473, 64
  %475 = sub i64 %473, %474
  %476 = inttoptr i64 %475 to ptr
  %477 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %471, 0
  %478 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %477, ptr %476, 1
  %479 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %478, i64 0, 2
  %480 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %479, i64 1, 3, 0
  %481 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %480, i64 1, 3, 1
  %482 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %481, i64 1, 3, 2
  %483 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %482, i64 256, 3, 3
  %484 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %483, i64 256, 4, 0
  %485 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %484, i64 256, 4, 1
  %486 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %485, i64 256, 4, 2
  %487 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %486, i64 1, 4, 3
  %488 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %489 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %488, 0
  %490 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %489, ptr %488, 1
  %491 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %490, i64 0, 2
  %492 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %491, i64 1, 3, 0
  %493 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %492, i64 1, 3, 1
  %494 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %493, i64 2, 3, 2
  %495 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %494, i64 256, 3, 3
  %496 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %495, i64 512, 4, 0
  %497 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %496, i64 512, 4, 1
  %498 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %497, i64 256, 4, 2
  %499 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %498, i64 1, 4, 3
  %500 = ptrtoint ptr %488 to i64
  %501 = inttoptr i64 %500 to ptr
  %502 = ptrtoint ptr %401 to i64
  %503 = inttoptr i64 %502 to ptr
  %504 = ptrtoint ptr %476 to i64
  %505 = inttoptr i64 %504 to ptr
  %506 = call i32 @rxops_bridge_rms_norm_f16(ptr %501, ptr %503, ptr %505, i64 2, i64 256, i64 4517329193108106637)
  %507 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64), i64 64))
  %508 = ptrtoint ptr %507 to i64
  %509 = add i64 %508, 63
  %510 = urem i64 %509, 64
  %511 = sub i64 %509, %510
  %512 = inttoptr i64 %511 to ptr
  %513 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %507, 0
  %514 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %513, ptr %512, 1
  %515 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %514, i64 0, 2
  %516 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %515, i64 2048, 3, 0
  %517 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %516, i64 1, 3, 1
  %518 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %517, i64 32, 3, 2
  %519 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %518, i64 32, 4, 0
  %520 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %519, i64 32, 4, 1
  %521 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %520, i64 1, 4, 2
  %522 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 3) to i64), i64 64))
  %523 = ptrtoint ptr %522 to i64
  %524 = add i64 %523, 63
  %525 = urem i64 %524, 64
  %526 = sub i64 %524, %525
  %527 = inttoptr i64 %526 to ptr
  %528 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %522, 0
  %529 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %528, ptr %527, 1
  %530 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %529, i64 0, 2
  %531 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %530, i64 3, 3, 0
  %532 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %531, i64 1, 3, 1
  %533 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %532, i64 1, 4, 0
  %534 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %533, i64 1, 4, 1
  %535 = call ptr @llvm.stacksave()
  %536 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, ptr %536, align 8
  %537 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %536, 1
  %538 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %534, ptr %538, align 8
  %539 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %538, 1
  %540 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %537, ptr %540, align 8
  %541 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %539, ptr %541, align 8
  call void @memrefCopy(i64 4, ptr %540, ptr %541)
  call void @llvm.stackrestore(ptr %535)
  %542 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 96) to i64))
  %543 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %542, 0
  %544 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %543, ptr %542, 1
  %545 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %544, i64 0, 2
  %546 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %545, i64 3, 3, 0
  %547 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %546, i64 1, 3, 1
  %548 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %547, i64 1, 3, 2
  %549 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %548, i64 32, 3, 3
  %550 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %549, i64 32, 4, 0
  %551 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %550, i64 32, 4, 1
  %552 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %551, i64 32, 4, 2
  %553 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %552, i64 1, 4, 3
  %554 = ptrtoint ptr %542 to i64
  %555 = inttoptr i64 %554 to ptr
  %556 = ptrtoint ptr %512 to i64
  %557 = inttoptr i64 %556 to ptr
  %558 = ptrtoint ptr %527 to i64
  %559 = inttoptr i64 %558 to ptr
  %560 = call i32 @rxops_bridge_gather_nd(ptr %555, ptr %557, ptr %559, i64 0, i64 2, i64 2048, i64 1, i64 32, i64 1, i64 1, i64 1, i64 3, i64 1, i64 1, i64 1, i64 1, i64 1, i64 3, i64 1, i64 1, i64 32, i64 1, i64 1)
  %561 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 96) to i64))
  %562 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %561, 0
  %563 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %562, ptr %561, 1
  %564 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %563, i64 0, 2
  %565 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %564, i64 3, 3, 0
  %566 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %565, i64 32, 3, 1
  %567 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %566, i64 1, 3, 2
  %568 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %567, i64 1, 3, 3
  %569 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %568, i64 32, 4, 0
  %570 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %569, i64 1, 4, 1
  %571 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %570, i64 1, 4, 2
  %572 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %571, i64 1, 4, 3
  %573 = ptrtoint ptr %561 to i64
  %574 = inttoptr i64 %573 to ptr
  %575 = ptrtoint ptr %542 to i64
  %576 = inttoptr i64 %575 to ptr
  %577 = call i32 @rxops_bridge_transpose_nd(ptr %574, ptr %576, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 3, i64 1, i64 1, i64 32, i64 1, i64 1, i64 3, i64 32, i64 1, i64 1, i64 1, i64 1)
  %578 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 96) to i64))
  %579 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %578, 0
  %580 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %579, ptr %578, 1
  %581 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %580, i64 0, 2
  %582 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %581, i64 96, 3, 0
  %583 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %582, i64 1, 3, 1
  %584 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %583, i64 1, 3, 2
  %585 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %584, i64 1, 4, 0
  %586 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %585, i64 1, 4, 1
  %587 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %586, i64 1, 4, 2
  %588 = ptrtoint ptr %578 to i64
  %589 = inttoptr i64 %588 to ptr
  %590 = ptrtoint ptr %561 to i64
  %591 = inttoptr i64 %590 to ptr
  %592 = call i32 @rxops_bridge_reshape_bytes(ptr %589, ptr %591, i64 192)
  %593 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 64) to i64), i64 64))
  %594 = ptrtoint ptr %593 to i64
  %595 = add i64 %594, 63
  %596 = urem i64 %595, 64
  %597 = sub i64 %595, %596
  %598 = inttoptr i64 %597 to ptr
  %599 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %593, 0
  %600 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %599, ptr %598, 1
  %601 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %600, i64 0, 2
  %602 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %601, i64 1, 3, 0
  %603 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %602, i64 64, 3, 1
  %604 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %603, i64 64, 4, 0
  %605 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %604, i64 1, 4, 1
  %606 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 64) to i64))
  %607 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %606, 0
  %608 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %607, ptr %606, 1
  %609 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %608, i64 0, 2
  %610 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %609, i64 1, 3, 0
  %611 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %610, i64 64, 3, 1
  %612 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %611, i64 1, 3, 2
  %613 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %612, i64 1, 3, 3
  %614 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %613, i64 64, 4, 0
  %615 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %614, i64 1, 4, 1
  %616 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %615, i64 1, 4, 2
  %617 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %616, i64 1, 4, 3
  %618 = ptrtoint ptr %606 to i64
  %619 = inttoptr i64 %618 to ptr
  %620 = ptrtoint ptr %578 to i64
  %621 = inttoptr i64 %620 to ptr
  %622 = ptrtoint ptr %598 to i64
  %623 = inttoptr i64 %622 to ptr
  %624 = call i32 @rxops_bridge_gather_nd(ptr %619, ptr %621, ptr %623, i64 0, i64 2, i64 96, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1)
  %625 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 64) to i64))
  %626 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %625, 0
  %627 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %626, ptr %625, 1
  %628 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %627, i64 0, 2
  %629 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %628, i64 1, 3, 0
  %630 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %629, i64 1, 3, 1
  %631 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %630, i64 1, 3, 2
  %632 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %631, i64 64, 3, 3
  %633 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %632, i64 64, 4, 0
  %634 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %633, i64 64, 4, 1
  %635 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %634, i64 64, 4, 2
  %636 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %635, i64 1, 4, 3
  %637 = ptrtoint ptr %625 to i64
  %638 = inttoptr i64 %637 to ptr
  %639 = ptrtoint ptr %606 to i64
  %640 = inttoptr i64 %639 to ptr
  %641 = call i32 @rxops_bridge_transpose_nd(ptr %638, ptr %640, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1)
  %642 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 65536) to i64), i64 64))
  %643 = ptrtoint ptr %642 to i64
  %644 = add i64 %643, 63
  %645 = urem i64 %644, 64
  %646 = sub i64 %644, %645
  %647 = inttoptr i64 %646 to ptr
  %648 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %642, 0
  %649 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %648, ptr %647, 1
  %650 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %649, i64 0, 2
  %651 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %650, i64 2048, 3, 0
  %652 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %651, i64 1, 3, 1
  %653 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %652, i64 32, 3, 2
  %654 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %653, i64 32, 4, 0
  %655 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %654, i64 32, 4, 1
  %656 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %655, i64 1, 4, 2
  %657 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 3) to i64), i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)))
  %658 = ptrtoint ptr %657 to i64
  %659 = add i64 %658, sub (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 1)
  %660 = urem i64 %659, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %661 = sub i64 %659, %660
  %662 = inttoptr i64 %661 to ptr
  %663 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %657, 0
  %664 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %663, ptr %662, 1
  %665 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %664, i64 0, 2
  %666 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %665, i64 3, 3, 0
  %667 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %666, i64 1, 3, 1
  %668 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %667, i64 1, 4, 0
  %669 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %668, i64 1, 4, 1
  %670 = call ptr @llvm.stacksave()
  %671 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, ptr %671, align 8
  %672 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %671, 1
  %673 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %669, ptr %673, align 8
  %674 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %673, 1
  %675 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %672, ptr %675, align 8
  %676 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %674, ptr %676, align 8
  call void @memrefCopy(i64 4, ptr %675, ptr %676)
  call void @llvm.stackrestore(ptr %670)
  %677 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 96) to i64))
  %678 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %677, 0
  %679 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %678, ptr %677, 1
  %680 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %679, i64 0, 2
  %681 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %680, i64 3, 3, 0
  %682 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %681, i64 1, 3, 1
  %683 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %682, i64 1, 3, 2
  %684 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %683, i64 32, 3, 3
  %685 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %684, i64 32, 4, 0
  %686 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %685, i64 32, 4, 1
  %687 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %686, i64 32, 4, 2
  %688 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %687, i64 1, 4, 3
  %689 = ptrtoint ptr %677 to i64
  %690 = inttoptr i64 %689 to ptr
  %691 = ptrtoint ptr %647 to i64
  %692 = inttoptr i64 %691 to ptr
  %693 = ptrtoint ptr %662 to i64
  %694 = inttoptr i64 %693 to ptr
  %695 = call i32 @rxops_bridge_gather_nd(ptr %690, ptr %692, ptr %694, i64 0, i64 2, i64 2048, i64 1, i64 32, i64 1, i64 1, i64 1, i64 3, i64 1, i64 1, i64 1, i64 1, i64 1, i64 3, i64 1, i64 1, i64 32, i64 1, i64 1)
  %696 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 96) to i64))
  %697 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %696, 0
  %698 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %697, ptr %696, 1
  %699 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %698, i64 0, 2
  %700 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %699, i64 3, 3, 0
  %701 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %700, i64 32, 3, 1
  %702 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %701, i64 1, 3, 2
  %703 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %702, i64 1, 3, 3
  %704 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %703, i64 32, 4, 0
  %705 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %704, i64 1, 4, 1
  %706 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %705, i64 1, 4, 2
  %707 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %706, i64 1, 4, 3
  %708 = ptrtoint ptr %696 to i64
  %709 = inttoptr i64 %708 to ptr
  %710 = ptrtoint ptr %677 to i64
  %711 = inttoptr i64 %710 to ptr
  %712 = call i32 @rxops_bridge_transpose_nd(ptr %709, ptr %711, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 3, i64 1, i64 1, i64 32, i64 1, i64 1, i64 3, i64 32, i64 1, i64 1, i64 1, i64 1)
  %713 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 96) to i64))
  %714 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %713, 0
  %715 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %714, ptr %713, 1
  %716 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %715, i64 0, 2
  %717 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %716, i64 96, 3, 0
  %718 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %717, i64 1, 3, 1
  %719 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %718, i64 1, 3, 2
  %720 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %719, i64 1, 4, 0
  %721 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %720, i64 1, 4, 1
  %722 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %721, i64 1, 4, 2
  %723 = ptrtoint ptr %713 to i64
  %724 = inttoptr i64 %723 to ptr
  %725 = ptrtoint ptr %696 to i64
  %726 = inttoptr i64 %725 to ptr
  %727 = call i32 @rxops_bridge_reshape_bytes(ptr %724, ptr %726, i64 192)
  %728 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 64) to i64), i64 64))
  %729 = ptrtoint ptr %728 to i64
  %730 = add i64 %729, 63
  %731 = urem i64 %730, 64
  %732 = sub i64 %730, %731
  %733 = inttoptr i64 %732 to ptr
  %734 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %728, 0
  %735 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %734, ptr %733, 1
  %736 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %735, i64 0, 2
  %737 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %736, i64 1, 3, 0
  %738 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %737, i64 64, 3, 1
  %739 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %738, i64 64, 4, 0
  %740 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %739, i64 1, 4, 1
  %741 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 64) to i64))
  %742 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %741, 0
  %743 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %742, ptr %741, 1
  %744 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %743, i64 0, 2
  %745 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %744, i64 1, 3, 0
  %746 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %745, i64 64, 3, 1
  %747 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %746, i64 1, 3, 2
  %748 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %747, i64 1, 3, 3
  %749 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %748, i64 64, 4, 0
  %750 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %749, i64 1, 4, 1
  %751 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %750, i64 1, 4, 2
  %752 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %751, i64 1, 4, 3
  %753 = ptrtoint ptr %741 to i64
  %754 = inttoptr i64 %753 to ptr
  %755 = ptrtoint ptr %713 to i64
  %756 = inttoptr i64 %755 to ptr
  %757 = ptrtoint ptr %733 to i64
  %758 = inttoptr i64 %757 to ptr
  %759 = call i32 @rxops_bridge_gather_nd(ptr %754, ptr %756, ptr %758, i64 0, i64 2, i64 96, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1)
  %760 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 64) to i64))
  %761 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %760, 0
  %762 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %761, ptr %760, 1
  %763 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %762, i64 0, 2
  %764 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %763, i64 1, 3, 0
  %765 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %764, i64 1, 3, 1
  %766 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %765, i64 1, 3, 2
  %767 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %766, i64 64, 3, 3
  %768 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %767, i64 64, 4, 0
  %769 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %768, i64 64, 4, 1
  %770 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %769, i64 64, 4, 2
  %771 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %770, i64 1, 4, 3
  %772 = ptrtoint ptr %760 to i64
  %773 = inttoptr i64 %772 to ptr
  %774 = ptrtoint ptr %741 to i64
  %775 = inttoptr i64 %774 to ptr
  %776 = call i32 @rxops_bridge_transpose_nd(ptr %773, ptr %775, i64 2, i64 4, i64 0, i64 3, i64 2, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 64, i64 1, i64 1)
  %777 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %778 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %777, 0
  %779 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %778, ptr %777, 1
  %780 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %779, i64 0, 2
  %781 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %780, i64 1, 3, 0
  %782 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %781, i64 1, 3, 1
  %783 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %782, i64 8, 3, 2
  %784 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %783, i64 64, 3, 3
  %785 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %784, i64 512, 4, 0
  %786 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %785, i64 512, 4, 1
  %787 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %786, i64 64, 4, 2
  %788 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %787, i64 1, 4, 3
  %789 = ptrtoint ptr %777 to i64
  %790 = inttoptr i64 %789 to ptr
  %791 = ptrtoint ptr %452 to i64
  %792 = inttoptr i64 %791 to ptr
  %793 = call i32 @rxops_bridge_slice_nd(ptr %790, ptr %792, i64 2, i64 4, i64 1, i64 1, i64 8, i64 64, i64 1, i64 1, i64 1, i64 1, i64 8, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 0, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %794 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 1536) to i64))
  %795 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %794, 0
  %796 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %795, ptr %794, 1
  %797 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %796, i64 0, 2
  %798 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %797, i64 1, 3, 0
  %799 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %798, i64 1, 3, 1
  %800 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %799, i64 8, 3, 2
  %801 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %800, i64 192, 3, 3
  %802 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %801, i64 1536, 4, 0
  %803 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %802, i64 1536, 4, 1
  %804 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %803, i64 192, 4, 2
  %805 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %804, i64 1, 4, 3
  %806 = ptrtoint ptr %794 to i64
  %807 = inttoptr i64 %806 to ptr
  %808 = ptrtoint ptr %452 to i64
  %809 = inttoptr i64 %808 to ptr
  %810 = call i32 @rxops_bridge_slice_nd(ptr %807, ptr %809, i64 2, i64 4, i64 1, i64 1, i64 8, i64 192, i64 1, i64 1, i64 1, i64 1, i64 8, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %811 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %812 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %811, 0
  %813 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %812, ptr %811, 1
  %814 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %813, i64 0, 2
  %815 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %814, i64 1, 3, 0
  %816 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %815, i64 1, 3, 1
  %817 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %816, i64 8, 3, 2
  %818 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %817, i64 64, 3, 3
  %819 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %818, i64 512, 4, 0
  %820 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %819, i64 512, 4, 1
  %821 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %820, i64 64, 4, 2
  %822 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %821, i64 1, 4, 3
  %823 = ptrtoint ptr %811 to i64
  %824 = inttoptr i64 %823 to ptr
  %825 = ptrtoint ptr %777 to i64
  %826 = inttoptr i64 %825 to ptr
  %827 = ptrtoint ptr %625 to i64
  %828 = inttoptr i64 %827 to ptr
  %829 = ptrtoint ptr %760 to i64
  %830 = inttoptr i64 %829 to ptr
  %831 = call i32 @rxops_bridge_rope_contiguous_f16(ptr %824, ptr %826, ptr %828, ptr %830, i64 1, i64 8, i64 64, i64 1)
  %832 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %833 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %832, 0
  %834 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %833, ptr %832, 1
  %835 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %834, i64 0, 2
  %836 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %835, i64 1, 3, 0
  %837 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %836, i64 1, 3, 1
  %838 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %837, i64 8, 3, 2
  %839 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %838, i64 256, 3, 3
  %840 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %839, i64 2048, 4, 0
  %841 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %840, i64 2048, 4, 1
  %842 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %841, i64 256, 4, 2
  %843 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %842, i64 1, 4, 3
  %844 = ptrtoint ptr %832 to i64
  %845 = inttoptr i64 %844 to ptr
  %846 = ptrtoint ptr %811 to i64
  %847 = inttoptr i64 %846 to ptr
  %848 = ptrtoint ptr %794 to i64
  %849 = inttoptr i64 %848 to ptr
  %850 = call i32 @rxops_bridge_concat2_nd(ptr %845, ptr %847, ptr %849, i64 2, i64 4, i64 3, i64 1, i64 1, i64 8, i64 256, i64 1, i64 1, i64 1, i64 1, i64 8, i64 64, i64 1, i64 1, i64 1, i64 1, i64 8, i64 192, i64 1, i64 1)
  %851 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 128) to i64))
  %852 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %851, 0
  %853 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %852, ptr %851, 1
  %854 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %853, i64 0, 2
  %855 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %854, i64 1, 3, 0
  %856 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %855, i64 1, 3, 1
  %857 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %856, i64 2, 3, 2
  %858 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %857, i64 64, 3, 3
  %859 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %858, i64 128, 4, 0
  %860 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %859, i64 128, 4, 1
  %861 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %860, i64 64, 4, 2
  %862 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %861, i64 1, 4, 3
  %863 = ptrtoint ptr %851 to i64
  %864 = inttoptr i64 %863 to ptr
  %865 = ptrtoint ptr %488 to i64
  %866 = inttoptr i64 %865 to ptr
  %867 = call i32 @rxops_bridge_slice_nd(ptr %864, ptr %866, i64 2, i64 4, i64 1, i64 1, i64 2, i64 64, i64 1, i64 1, i64 1, i64 1, i64 2, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 0, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %868 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 384) to i64))
  %869 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %868, 0
  %870 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %869, ptr %868, 1
  %871 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %870, i64 0, 2
  %872 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %871, i64 1, 3, 0
  %873 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %872, i64 1, 3, 1
  %874 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %873, i64 2, 3, 2
  %875 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %874, i64 192, 3, 3
  %876 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %875, i64 384, 4, 0
  %877 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %876, i64 384, 4, 1
  %878 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %877, i64 192, 4, 2
  %879 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %878, i64 1, 4, 3
  %880 = ptrtoint ptr %868 to i64
  %881 = inttoptr i64 %880 to ptr
  %882 = ptrtoint ptr %488 to i64
  %883 = inttoptr i64 %882 to ptr
  %884 = call i32 @rxops_bridge_slice_nd(ptr %881, ptr %883, i64 2, i64 4, i64 1, i64 1, i64 2, i64 192, i64 1, i64 1, i64 1, i64 1, i64 2, i64 256, i64 1, i64 1, i64 0, i64 0, i64 0, i64 64, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %885 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 128) to i64))
  %886 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %885, 0
  %887 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %886, ptr %885, 1
  %888 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %887, i64 0, 2
  %889 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %888, i64 1, 3, 0
  %890 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %889, i64 1, 3, 1
  %891 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %890, i64 2, 3, 2
  %892 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %891, i64 64, 3, 3
  %893 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %892, i64 128, 4, 0
  %894 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %893, i64 128, 4, 1
  %895 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %894, i64 64, 4, 2
  %896 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %895, i64 1, 4, 3
  %897 = ptrtoint ptr %885 to i64
  %898 = inttoptr i64 %897 to ptr
  %899 = ptrtoint ptr %851 to i64
  %900 = inttoptr i64 %899 to ptr
  %901 = ptrtoint ptr %625 to i64
  %902 = inttoptr i64 %901 to ptr
  %903 = ptrtoint ptr %760 to i64
  %904 = inttoptr i64 %903 to ptr
  %905 = call i32 @rxops_bridge_rope_contiguous_f16(ptr %898, ptr %900, ptr %902, ptr %904, i64 1, i64 2, i64 64, i64 1)
  %906 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 512) to i64))
  %907 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %906, 0
  %908 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %907, ptr %906, 1
  %909 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %908, i64 0, 2
  %910 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %909, i64 1, 3, 0
  %911 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %910, i64 1, 3, 1
  %912 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %911, i64 2, 3, 2
  %913 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %912, i64 256, 3, 3
  %914 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %913, i64 512, 4, 0
  %915 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %914, i64 512, 4, 1
  %916 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %915, i64 256, 4, 2
  %917 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %916, i64 1, 4, 3
  %918 = ptrtoint ptr %906 to i64
  %919 = inttoptr i64 %918 to ptr
  %920 = ptrtoint ptr %885 to i64
  %921 = inttoptr i64 %920 to ptr
  %922 = ptrtoint ptr %868 to i64
  %923 = inttoptr i64 %922 to ptr
  %924 = call i32 @rxops_bridge_concat2_nd(ptr %919, ptr %921, ptr %923, i64 2, i64 4, i64 3, i64 1, i64 1, i64 2, i64 256, i64 1, i64 1, i64 1, i64 1, i64 2, i64 64, i64 1, i64 1, i64 1, i64 1, i64 2, i64 192, i64 1, i64 1)
  %925 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  %926 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %925, 0
  %927 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %926, ptr %925, 1
  %928 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %927, i64 0, 2
  %929 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %928, i64 1, 3, 0
  %930 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %929, i64 2048, 3, 1
  %931 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %930, i64 2, 3, 2
  %932 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %931, i64 256, 3, 3
  %933 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %932, i64 1048576, 4, 0
  %934 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %933, i64 512, 4, 1
  %935 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %934, i64 256, 4, 2
  %936 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %935, i64 1, 4, 3
  %937 = call ptr @llvm.stacksave()
  %938 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %87, ptr %938, align 8
  %939 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %938, 1
  %940 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %936, ptr %940, align 8
  %941 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %940, 1
  %942 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %939, ptr %942, align 8
  %943 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %941, ptr %943, align 8
  call void @memrefCopy(i64 4, ptr %942, ptr %943)
  call void @llvm.stackrestore(ptr %937)
  %944 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1049088) to i64))
  %945 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %944, 0
  %946 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %945, ptr %944, 1
  %947 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %946, i64 0, 2
  %948 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %947, i64 1, 3, 0
  %949 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %948, i64 2049, 3, 1
  %950 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %949, i64 2, 3, 2
  %951 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %950, i64 256, 3, 3
  %952 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %951, i64 1049088, 4, 0
  %953 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %952, i64 512, 4, 1
  %954 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %953, i64 256, 4, 2
  %955 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %954, i64 1, 4, 3
  %956 = ptrtoint ptr %944 to i64
  %957 = inttoptr i64 %956 to ptr
  %958 = ptrtoint ptr %925 to i64
  %959 = inttoptr i64 %958 to ptr
  %960 = ptrtoint ptr %906 to i64
  %961 = inttoptr i64 %960 to ptr
  %962 = call i32 @rxops_bridge_concat2_nd(ptr %957, ptr %959, ptr %961, i64 4, i64 4, i64 1, i64 1, i64 2049, i64 2, i64 256, i64 1, i64 1, i64 1, i64 2048, i64 2, i64 256, i64 1, i64 1, i64 1, i64 1, i64 2, i64 256, i64 1, i64 1)
  %963 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1048576) to i64))
  %964 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %963, 0
  %965 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %964, ptr %963, 1
  %966 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %965, i64 0, 2
  %967 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %966, i64 1, 3, 0
  %968 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %967, i64 2048, 3, 1
  %969 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %968, i64 2, 3, 2
  %970 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %969, i64 256, 3, 3
  %971 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %970, i64 1048576, 4, 0
  %972 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %971, i64 512, 4, 1
  %973 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %972, i64 256, 4, 2
  %974 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %973, i64 1, 4, 3
  %975 = call ptr @llvm.stacksave()
  %976 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %98, ptr %976, align 8
  %977 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %976, 1
  %978 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %974, ptr %978, align 8
  %979 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %978, 1
  %980 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %977, ptr %980, align 8
  %981 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %979, ptr %981, align 8
  call void @memrefCopy(i64 4, ptr %980, ptr %981)
  call void @llvm.stackrestore(ptr %975)
  %982 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1049088) to i64))
  %983 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %982, 0
  %984 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %983, ptr %982, 1
  %985 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %984, i64 0, 2
  %986 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %985, i64 1, 3, 0
  %987 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %986, i64 2049, 3, 1
  %988 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %987, i64 2, 3, 2
  %989 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %988, i64 256, 3, 3
  %990 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %989, i64 1049088, 4, 0
  %991 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %990, i64 512, 4, 1
  %992 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %991, i64 256, 4, 2
  %993 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %992, i64 1, 4, 3
  %994 = ptrtoint ptr %982 to i64
  %995 = inttoptr i64 %994 to ptr
  %996 = ptrtoint ptr %963 to i64
  %997 = inttoptr i64 %996 to ptr
  %998 = ptrtoint ptr %418 to i64
  %999 = inttoptr i64 %998 to ptr
  %1000 = call i32 @rxops_bridge_concat2_nd(ptr %995, ptr %997, ptr %999, i64 4, i64 4, i64 1, i64 1, i64 2049, i64 2, i64 256, i64 1, i64 1, i64 1, i64 2048, i64 2, i64 256, i64 1, i64 1, i64 1, i64 1, i64 2, i64 256, i64 1, i64 1)
  %1001 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2049) to i64))
  %1002 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } undef, ptr %1001, 0
  %1003 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1002, ptr %1001, 1
  %1004 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1003, i64 0, 2
  %1005 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1004, i64 1, 3, 0
  %1006 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1005, i64 1, 3, 1
  %1007 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1006, i64 1, 3, 2
  %1008 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1007, i64 2049, 3, 3
  %1009 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1008, i64 2049, 4, 0
  %1010 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1009, i64 2049, 4, 1
  %1011 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1010, i64 2049, 4, 2
  %1012 = insertvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %1011, i64 1, 4, 3
  %1013 = call ptr @llvm.stacksave()
  %1014 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %76, ptr %1014, align 8
  %1015 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1014, 1
  %1016 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } %1012, ptr %1016, align 8
  %1017 = insertvalue { i64, ptr } { i64 4, ptr undef }, ptr %1016, 1
  %1018 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1015, ptr %1018, align 8
  %1019 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1017, ptr %1019, align 8
  call void @memrefCopy(i64 4, ptr %1018, ptr %1019)
  call void @llvm.stackrestore(ptr %1013)
  %1020 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1021 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1020, 0
  %1022 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1021, ptr %1020, 1
  %1023 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1022, i64 0, 2
  %1024 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1023, i64 1, 3, 0
  %1025 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1024, i64 1, 3, 1
  %1026 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1025, i64 2048, 3, 2
  %1027 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1026, i64 2048, 4, 0
  %1028 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1027, i64 2048, 4, 1
  %1029 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1028, i64 1, 4, 2
  %1030 = ptrtoint ptr %1020 to i64
  %1031 = inttoptr i64 %1030 to ptr
  %1032 = ptrtoint ptr %832 to i64
  %1033 = inttoptr i64 %1032 to ptr
  %1034 = ptrtoint ptr %944 to i64
  %1035 = inttoptr i64 %1034 to ptr
  %1036 = ptrtoint ptr %982 to i64
  %1037 = inttoptr i64 %1036 to ptr
  %1038 = ptrtoint ptr %1001 to i64
  %1039 = inttoptr i64 %1038 to ptr
  %1040 = call i32 @rxops_bridge_fattention_f16(ptr %1031, ptr %1033, ptr %1035, ptr %1037, ptr %1039, i64 1, i64 1, i64 2049, i64 8, i64 2, i64 256, i64 4589168020290535424, i64 1)
  %1041 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1042 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1041, 0
  %1043 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1042, ptr %1041, 1
  %1044 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1043, i64 0, 2
  %1045 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1044, i64 1, 3, 0
  %1046 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1045, i64 1, 3, 1
  %1047 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1046, i64 2048, 3, 2
  %1048 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1047, i64 2048, 4, 0
  %1049 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1048, i64 2048, 4, 1
  %1050 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1049, i64 1, 4, 2
  %1051 = ptrtoint ptr %1041 to i64
  %1052 = inttoptr i64 %1051 to ptr
  %1053 = ptrtoint ptr %264 to i64
  %1054 = inttoptr i64 %1053 to ptr
  %1055 = call i32 @rxops_bridge_reshape_bytes(ptr %1052, ptr %1054, i64 4096)
  %1056 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1057 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1056, 0
  %1058 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1057, ptr %1056, 1
  %1059 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1058, i64 0, 2
  %1060 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1059, i64 1, 3, 0
  %1061 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1060, i64 1, 3, 1
  %1062 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1061, i64 2048, 3, 2
  %1063 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1062, i64 2048, 4, 0
  %1064 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1063, i64 2048, 4, 1
  %1065 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1064, i64 1, 4, 2
  %1066 = ptrtoint ptr %1056 to i64
  %1067 = inttoptr i64 %1066 to ptr
  %1068 = ptrtoint ptr %1020 to i64
  %1069 = inttoptr i64 %1068 to ptr
  %1070 = ptrtoint ptr %1041 to i64
  %1071 = inttoptr i64 %1070 to ptr
  %1072 = call i32 @rxops_bridge_mul_f16(ptr %1067, ptr %1069, ptr %1071, i64 2048)
  %1073 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 2097152) to i64), i64 64))
  %1074 = ptrtoint ptr %1073 to i64
  %1075 = add i64 %1074, 63
  %1076 = urem i64 %1075, 64
  %1077 = sub i64 %1075, %1076
  %1078 = inttoptr i64 %1077 to ptr
  %1079 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1073, 0
  %1080 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1079, ptr %1078, 1
  %1081 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1080, i64 0, 2
  %1082 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1081, i64 2048, 3, 0
  %1083 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1082, i64 1024, 3, 1
  %1084 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1083, i64 1024, 4, 0
  %1085 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1084, i64 1, 4, 1
  %1086 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 32768) to i64), i64 64))
  %1087 = ptrtoint ptr %1086 to i64
  %1088 = add i64 %1087, 63
  %1089 = urem i64 %1088, 64
  %1090 = sub i64 %1088, %1089
  %1091 = inttoptr i64 %1090 to ptr
  %1092 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1086, 0
  %1093 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1092, ptr %1091, 1
  %1094 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1093, i64 0, 2
  %1095 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1094, i64 2048, 3, 0
  %1096 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1095, i64 16, 3, 1
  %1097 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1096, i64 16, 4, 0
  %1098 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1097, i64 1, 4, 1
  %1099 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 32768) to i64), i64 64))
  %1100 = ptrtoint ptr %1099 to i64
  %1101 = add i64 %1100, 63
  %1102 = urem i64 %1101, 64
  %1103 = sub i64 %1101, %1102
  %1104 = inttoptr i64 %1103 to ptr
  %1105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1099, 0
  %1106 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1105, ptr %1104, 1
  %1107 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1106, i64 0, 2
  %1108 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1107, i64 2048, 3, 0
  %1109 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1108, i64 16, 3, 1
  %1110 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1109, i64 16, 4, 0
  %1111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1110, i64 1, 4, 1
  %1112 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1113 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1112, 0
  %1114 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1113, ptr %1112, 1
  %1115 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1114, i64 0, 2
  %1116 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1115, i64 1, 3, 0
  %1117 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1116, i64 1, 3, 1
  %1118 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1117, i64 2048, 3, 2
  %1119 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1118, i64 2048, 4, 0
  %1120 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1119, i64 2048, 4, 1
  %1121 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1120, i64 1, 4, 2
  %1122 = ptrtoint ptr %1112 to i64
  %1123 = inttoptr i64 %1122 to ptr
  %1124 = ptrtoint ptr %1056 to i64
  %1125 = inttoptr i64 %1124 to ptr
  %1126 = ptrtoint ptr %1078 to i64
  %1127 = inttoptr i64 %1126 to ptr
  %1128 = ptrtoint ptr %1104 to i64
  %1129 = inttoptr i64 %1128 to ptr
  %1130 = ptrtoint ptr %1091 to i64
  %1131 = inttoptr i64 %1130 to ptr
  %1132 = call i32 @rxops_bridge_a16matmul_f16(ptr %1123, ptr %1125, ptr %1127, ptr %1129, ptr %1131, i64 1, i64 2048, i64 2048, i64 128, i64 4)
  %1133 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64))
  %1134 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1133, 0
  %1135 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1134, ptr %1133, 1
  %1136 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1135, i64 0, 2
  %1137 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1136, i64 1, 3, 0
  %1138 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1137, i64 1, 3, 1
  %1139 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1138, i64 2048, 3, 2
  %1140 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1139, i64 2048, 4, 0
  %1141 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1140, i64 2048, 4, 1
  %1142 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1141, i64 1, 4, 2
  %1143 = call ptr @llvm.stacksave()
  %1144 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %58, ptr %1144, align 8
  %1145 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %1144, 1
  %1146 = alloca { ptr, ptr, i64, [3 x i64], [3 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %1142, ptr %1146, align 8
  %1147 = insertvalue { i64, ptr } { i64 3, ptr undef }, ptr %1146, 1
  %1148 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1145, ptr %1148, align 8
  %1149 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %1147, ptr %1149, align 8
  call void @memrefCopy(i64 4, ptr %1148, ptr %1149)
  call void @llvm.stackrestore(ptr %1143)
  %1150 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1151 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1150, 0
  %1152 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1151, ptr %1150, 1
  %1153 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1152, i64 0, 2
  %1154 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1153, i64 1, 3, 0
  %1155 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1154, i64 1, 3, 1
  %1156 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1155, i64 2048, 3, 2
  %1157 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1156, i64 2048, 4, 0
  %1158 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1157, i64 2048, 4, 1
  %1159 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1158, i64 1, 4, 2
  %1160 = ptrtoint ptr %1150 to i64
  %1161 = inttoptr i64 %1160 to ptr
  %1162 = ptrtoint ptr %1133 to i64
  %1163 = inttoptr i64 %1162 to ptr
  %1164 = ptrtoint ptr %1112 to i64
  %1165 = inttoptr i64 %1164 to ptr
  %1166 = call i32 @rxops_bridge_add_f16(ptr %1161, ptr %1163, ptr %1165, i64 2048)
  %1167 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64), i64 64))
  %1168 = ptrtoint ptr %1167 to i64
  %1169 = add i64 %1168, 63
  %1170 = urem i64 %1169, 64
  %1171 = sub i64 %1169, %1170
  %1172 = inttoptr i64 %1171 to ptr
  %1173 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1167, 0
  %1174 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1173, ptr %1172, 1
  %1175 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1174, i64 0, 2
  %1176 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1175, i64 1, 3, 0
  %1177 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1176, i64 1, 3, 1
  %1178 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1177, i64 2048, 3, 2
  %1179 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1178, i64 2048, 4, 0
  %1180 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1179, i64 2048, 4, 1
  %1181 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1180, i64 1, 4, 2
  %1182 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1183 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1182, 0
  %1184 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1183, ptr %1182, 1
  %1185 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1184, i64 0, 2
  %1186 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1185, i64 1, 3, 0
  %1187 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1186, i64 1, 3, 1
  %1188 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1187, i64 2048, 3, 2
  %1189 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1188, i64 2048, 4, 0
  %1190 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1189, i64 2048, 4, 1
  %1191 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1190, i64 1, 4, 2
  %1192 = ptrtoint ptr %1182 to i64
  %1193 = inttoptr i64 %1192 to ptr
  %1194 = ptrtoint ptr %1150 to i64
  %1195 = inttoptr i64 %1194 to ptr
  %1196 = ptrtoint ptr %1172 to i64
  %1197 = inttoptr i64 %1196 to ptr
  %1198 = call i32 @rxops_bridge_rms_norm_f16(ptr %1193, ptr %1195, ptr %1197, i64 1, i64 2048, i64 4517329193108106637)
  %1199 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 6291456) to i64), i64 64))
  %1200 = ptrtoint ptr %1199 to i64
  %1201 = add i64 %1200, 63
  %1202 = urem i64 %1201, 64
  %1203 = sub i64 %1201, %1202
  %1204 = inttoptr i64 %1203 to ptr
  %1205 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1199, 0
  %1206 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1205, ptr %1204, 1
  %1207 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1206, i64 0, 2
  %1208 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1207, i64 6144, 3, 0
  %1209 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1208, i64 1024, 3, 1
  %1210 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1209, i64 1024, 4, 0
  %1211 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1210, i64 1, 4, 1
  %1212 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 98304) to i64), i64 64))
  %1213 = ptrtoint ptr %1212 to i64
  %1214 = add i64 %1213, 63
  %1215 = urem i64 %1214, 64
  %1216 = sub i64 %1214, %1215
  %1217 = inttoptr i64 %1216 to ptr
  %1218 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1212, 0
  %1219 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1218, ptr %1217, 1
  %1220 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1219, i64 0, 2
  %1221 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1220, i64 6144, 3, 0
  %1222 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1221, i64 16, 3, 1
  %1223 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1222, i64 16, 4, 0
  %1224 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1223, i64 1, 4, 1
  %1225 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64), i64 64))
  %1226 = ptrtoint ptr %1225 to i64
  %1227 = add i64 %1226, 63
  %1228 = urem i64 %1227, 64
  %1229 = sub i64 %1227, %1228
  %1230 = inttoptr i64 %1229 to ptr
  %1231 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1225, 0
  %1232 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1231, ptr %1230, 1
  %1233 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1232, i64 0, 2
  %1234 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1233, i64 6144, 3, 0
  %1235 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1234, i64 16, 3, 1
  %1236 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1235, i64 16, 4, 0
  %1237 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1236, i64 1, 4, 1
  %1238 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6144) to i64))
  %1239 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1238, 0
  %1240 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1239, ptr %1238, 1
  %1241 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1240, i64 0, 2
  %1242 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1241, i64 1, 3, 0
  %1243 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1242, i64 1, 3, 1
  %1244 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1243, i64 6144, 3, 2
  %1245 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1244, i64 6144, 4, 0
  %1246 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1245, i64 6144, 4, 1
  %1247 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1246, i64 1, 4, 2
  %1248 = ptrtoint ptr %1238 to i64
  %1249 = inttoptr i64 %1248 to ptr
  %1250 = ptrtoint ptr %1182 to i64
  %1251 = inttoptr i64 %1250 to ptr
  %1252 = ptrtoint ptr %1204 to i64
  %1253 = inttoptr i64 %1252 to ptr
  %1254 = ptrtoint ptr %1230 to i64
  %1255 = inttoptr i64 %1254 to ptr
  %1256 = ptrtoint ptr %1217 to i64
  %1257 = inttoptr i64 %1256 to ptr
  %1258 = call i32 @rxops_bridge_a16matmul_f16(ptr %1249, ptr %1251, ptr %1253, ptr %1255, ptr %1257, i64 1, i64 2048, i64 6144, i64 128, i64 4)
  %1259 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6144) to i64))
  %1260 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1259, 0
  %1261 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1260, ptr %1259, 1
  %1262 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1261, i64 0, 2
  %1263 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1262, i64 1, 3, 0
  %1264 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1263, i64 1, 3, 1
  %1265 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1264, i64 6144, 3, 2
  %1266 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1265, i64 6144, 4, 0
  %1267 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1266, i64 6144, 4, 1
  %1268 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1267, i64 1, 4, 2
  %1269 = ptrtoint ptr %1259 to i64
  %1270 = inttoptr i64 %1269 to ptr
  %1271 = ptrtoint ptr %1238 to i64
  %1272 = inttoptr i64 %1271 to ptr
  %1273 = call i32 @rxops_bridge_silu_f16(ptr %1270, ptr %1272, i64 6144)
  %1274 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 6291456) to i64), i64 64))
  %1275 = ptrtoint ptr %1274 to i64
  %1276 = add i64 %1275, 63
  %1277 = urem i64 %1276, 64
  %1278 = sub i64 %1276, %1277
  %1279 = inttoptr i64 %1278 to ptr
  %1280 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1274, 0
  %1281 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1280, ptr %1279, 1
  %1282 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1281, i64 0, 2
  %1283 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1282, i64 6144, 3, 0
  %1284 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1283, i64 1024, 3, 1
  %1285 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1284, i64 1024, 4, 0
  %1286 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1285, i64 1, 4, 1
  %1287 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 98304) to i64), i64 64))
  %1288 = ptrtoint ptr %1287 to i64
  %1289 = add i64 %1288, 63
  %1290 = urem i64 %1289, 64
  %1291 = sub i64 %1289, %1290
  %1292 = inttoptr i64 %1291 to ptr
  %1293 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1287, 0
  %1294 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1293, ptr %1292, 1
  %1295 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1294, i64 0, 2
  %1296 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1295, i64 6144, 3, 0
  %1297 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1296, i64 16, 3, 1
  %1298 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1297, i64 16, 4, 0
  %1299 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1298, i64 1, 4, 1
  %1300 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64), i64 64))
  %1301 = ptrtoint ptr %1300 to i64
  %1302 = add i64 %1301, 63
  %1303 = urem i64 %1302, 64
  %1304 = sub i64 %1302, %1303
  %1305 = inttoptr i64 %1304 to ptr
  %1306 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1300, 0
  %1307 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1306, ptr %1305, 1
  %1308 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1307, i64 0, 2
  %1309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1308, i64 6144, 3, 0
  %1310 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1309, i64 16, 3, 1
  %1311 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1310, i64 16, 4, 0
  %1312 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1311, i64 1, 4, 1
  %1313 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6144) to i64))
  %1314 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1313, 0
  %1315 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1314, ptr %1313, 1
  %1316 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1315, i64 0, 2
  %1317 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1316, i64 1, 3, 0
  %1318 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1317, i64 1, 3, 1
  %1319 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1318, i64 6144, 3, 2
  %1320 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1319, i64 6144, 4, 0
  %1321 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1320, i64 6144, 4, 1
  %1322 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1321, i64 1, 4, 2
  %1323 = ptrtoint ptr %1313 to i64
  %1324 = inttoptr i64 %1323 to ptr
  %1325 = ptrtoint ptr %1182 to i64
  %1326 = inttoptr i64 %1325 to ptr
  %1327 = ptrtoint ptr %1279 to i64
  %1328 = inttoptr i64 %1327 to ptr
  %1329 = ptrtoint ptr %1305 to i64
  %1330 = inttoptr i64 %1329 to ptr
  %1331 = ptrtoint ptr %1292 to i64
  %1332 = inttoptr i64 %1331 to ptr
  %1333 = call i32 @rxops_bridge_a16matmul_f16(ptr %1324, ptr %1326, ptr %1328, ptr %1330, ptr %1332, i64 1, i64 2048, i64 6144, i64 128, i64 4)
  %1334 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 6144) to i64))
  %1335 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1334, 0
  %1336 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1335, ptr %1334, 1
  %1337 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1336, i64 0, 2
  %1338 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1337, i64 1, 3, 0
  %1339 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1338, i64 1, 3, 1
  %1340 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1339, i64 6144, 3, 2
  %1341 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1340, i64 6144, 4, 0
  %1342 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1341, i64 6144, 4, 1
  %1343 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1342, i64 1, 4, 2
  %1344 = ptrtoint ptr %1334 to i64
  %1345 = inttoptr i64 %1344 to ptr
  %1346 = ptrtoint ptr %1259 to i64
  %1347 = inttoptr i64 %1346 to ptr
  %1348 = ptrtoint ptr %1313 to i64
  %1349 = inttoptr i64 %1348 to ptr
  %1350 = call i32 @rxops_bridge_mul_f16(ptr %1345, ptr %1347, ptr %1349, i64 6144)
  %1351 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 6291456) to i64), i64 64))
  %1352 = ptrtoint ptr %1351 to i64
  %1353 = add i64 %1352, 63
  %1354 = urem i64 %1353, 64
  %1355 = sub i64 %1353, %1354
  %1356 = inttoptr i64 %1355 to ptr
  %1357 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1351, 0
  %1358 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1357, ptr %1356, 1
  %1359 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1358, i64 0, 2
  %1360 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1359, i64 2048, 3, 0
  %1361 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1360, i64 3072, 3, 1
  %1362 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1361, i64 3072, 4, 0
  %1363 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1362, i64 1, 4, 1
  %1364 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 98304) to i64), i64 64))
  %1365 = ptrtoint ptr %1364 to i64
  %1366 = add i64 %1365, 63
  %1367 = urem i64 %1366, 64
  %1368 = sub i64 %1366, %1367
  %1369 = inttoptr i64 %1368 to ptr
  %1370 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1364, 0
  %1371 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1370, ptr %1369, 1
  %1372 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1371, i64 0, 2
  %1373 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1372, i64 2048, 3, 0
  %1374 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1373, i64 48, 3, 1
  %1375 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1374, i64 48, 4, 0
  %1376 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1375, i64 1, 4, 1
  %1377 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i32 98304) to i64), i64 64))
  %1378 = ptrtoint ptr %1377 to i64
  %1379 = add i64 %1378, 63
  %1380 = urem i64 %1379, 64
  %1381 = sub i64 %1379, %1380
  %1382 = inttoptr i64 %1381 to ptr
  %1383 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1377, 0
  %1384 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1383, ptr %1382, 1
  %1385 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1384, i64 0, 2
  %1386 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1385, i64 2048, 3, 0
  %1387 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1386, i64 48, 3, 1
  %1388 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1387, i64 48, 4, 0
  %1389 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %1388, i64 1, 4, 1
  %1390 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1391 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1390, 0
  %1392 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1391, ptr %1390, 1
  %1393 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1392, i64 0, 2
  %1394 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1393, i64 1, 3, 0
  %1395 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1394, i64 1, 3, 1
  %1396 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1395, i64 2048, 3, 2
  %1397 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1396, i64 2048, 4, 0
  %1398 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1397, i64 2048, 4, 1
  %1399 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1398, i64 1, 4, 2
  %1400 = ptrtoint ptr %1390 to i64
  %1401 = inttoptr i64 %1400 to ptr
  %1402 = ptrtoint ptr %1334 to i64
  %1403 = inttoptr i64 %1402 to ptr
  %1404 = ptrtoint ptr %1356 to i64
  %1405 = inttoptr i64 %1404 to ptr
  %1406 = ptrtoint ptr %1382 to i64
  %1407 = inttoptr i64 %1406 to ptr
  %1408 = ptrtoint ptr %1369 to i64
  %1409 = inttoptr i64 %1408 to ptr
  %1410 = call i32 @rxops_bridge_a16matmul_f16(ptr %1401, ptr %1403, ptr %1405, ptr %1407, ptr %1409, i64 1, i64 6144, i64 2048, i64 128, i64 4)
  %1411 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %1412 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %1411, 0
  %1413 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1412, ptr %1411, 1
  %1414 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1413, i64 0, 2
  %1415 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1414, i64 1, 3, 0
  %1416 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1415, i64 1, 3, 1
  %1417 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1416, i64 2048, 3, 2
  %1418 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1417, i64 2048, 4, 0
  %1419 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1418, i64 2048, 4, 1
  %1420 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %1419, i64 1, 4, 2
  %1421 = ptrtoint ptr %1411 to i64
  %1422 = inttoptr i64 %1421 to ptr
  %1423 = ptrtoint ptr %1150 to i64
  %1424 = inttoptr i64 %1423 to ptr
  %1425 = ptrtoint ptr %1390 to i64
  %1426 = inttoptr i64 %1425 to ptr
  %1427 = call i32 @rxops_bridge_add_f16(ptr %1422, ptr %1424, ptr %1426, i64 2048)
  call void @free(ptr %99)
  call void @free(ptr %114)
  call void @free(ptr %153)
  call void @free(ptr %166)
  call void @free(ptr %179)
  call void @free(ptr %281)
  call void @free(ptr %294)
  call void @free(ptr %307)
  call void @free(ptr %341)
  call void @free(ptr %354)
  call void @free(ptr %367)
  call void @free(ptr %435)
  call void @free(ptr %471)
  call void @free(ptr %507)
  call void @free(ptr %522)
  call void @free(ptr %593)
  call void @free(ptr %642)
  call void @free(ptr %728)
  call void @free(ptr %1073)
  call void @free(ptr %1086)
  call void @free(ptr %1099)
  call void @free(ptr %1167)
  call void @free(ptr %1199)
  call void @free(ptr %1212)
  call void @free(ptr %1225)
  call void @free(ptr %1274)
  call void @free(ptr %1287)
  call void @free(ptr %1300)
  call void @free(ptr %1351)
  call void @free(ptr %1364)
  call void @free(ptr %1377)
  %1428 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } undef, { ptr, ptr, i64, [3 x i64], [3 x i64] } %1420, 0
  %1429 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %1428, { ptr, ptr, i64, [4 x i64], [4 x i64] } %917, 1
  %1430 = insertvalue { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %1429, { ptr, ptr, i64, [4 x i64], [4 x i64] } %429, 2
  ret { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %1430
}

define void @_mlir_ciface_block_cache7_main(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) {
  %7 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 0
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 1
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 2
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 3, 0
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 3, 1
  %13 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 3, 2
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 4, 0
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 4, 1
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %7, 4, 2
  %17 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 0
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 1
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 2
  %21 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 3, 0
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 3, 1
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 4, 0
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, 4, 1
  %25 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %3, align 8
  %26 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 0
  %27 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 1
  %28 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 2
  %29 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 3, 0
  %30 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 3, 1
  %31 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 3, 2
  %32 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 3, 3
  %33 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 4, 0
  %34 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 4, 1
  %35 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 4, 2
  %36 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %25, 4, 3
  %37 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %4, align 8
  %38 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 0
  %39 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 1
  %40 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 2
  %41 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 3, 0
  %42 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 3, 1
  %43 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 3, 2
  %44 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 3, 3
  %45 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 4, 0
  %46 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 4, 1
  %47 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 4, 2
  %48 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %37, 4, 3
  %49 = load { ptr, ptr, i64, [4 x i64], [4 x i64] }, ptr %5, align 8
  %50 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 0
  %51 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 1
  %52 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 2
  %53 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 3, 0
  %54 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 3, 1
  %55 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 3, 2
  %56 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 3, 3
  %57 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 4, 0
  %58 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 4, 1
  %59 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 4, 2
  %60 = extractvalue { ptr, ptr, i64, [4 x i64], [4 x i64] } %49, 4, 3
  %61 = call { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } @block_cache7_main(ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, ptr %38, ptr %39, i64 %40, i64 %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47, i64 %48, ptr %50, ptr %51, i64 %52, i64 %53, i64 %54, i64 %55, i64 %56, i64 %57, i64 %58, i64 %59, i64 %60)
  store { { ptr, ptr, i64, [3 x i64], [3 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] }, { ptr, ptr, i64, [4 x i64], [4 x i64] } } %61, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
