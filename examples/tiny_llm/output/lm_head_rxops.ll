; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define private i32 @rxops_bridge_topk_f16(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = call i32 @_mlir_ciface_rxops_bridge_topk_f16(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6)
  ret i32 %8
}

declare i32 @_mlir_ciface_rxops_bridge_topk_f16(ptr, ptr, ptr, i64, i64, i64, i64)

define private i32 @rxops_bridge_reshape_bytes(ptr %0, ptr %1, i64 %2) {
  %4 = call i32 @_mlir_ciface_rxops_bridge_reshape_bytes(ptr %0, ptr %1, i64 %2)
  ret i32 %4
}

declare i32 @_mlir_ciface_rxops_bridge_reshape_bytes(ptr, ptr, i64)

define private i32 @rxops_bridge_matmul_f16(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5) {
  %7 = call i32 @_mlir_ciface_rxops_bridge_matmul_f16(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5)
  ret i32 %7
}

declare i32 @_mlir_ciface_rxops_bridge_matmul_f16(ptr, ptr, ptr, i64, i64, i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @lm_head_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %1, 1
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %5, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %4, 3, 1
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %6, 4, 1
  %15 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (half, ptr null, i64 508559360) to i64), i64 64))
  %16 = ptrtoint ptr %15 to i64
  %17 = add i64 %16, 63
  %18 = urem i64 %17, 64
  %19 = sub i64 %17, %18
  %20 = inttoptr i64 %19 to ptr
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %15, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %20, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 0, 2
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 248320, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 2048, 3, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 2048, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 1, 4, 1
  %28 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2048) to i64))
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %28, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %28, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 0, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 1, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 2048, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 2048, 4, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 1, 4, 1
  %36 = call ptr @llvm.stacksave()
  %37 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, ptr %37, align 8
  %38 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %37, 1
  %39 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, ptr %39, align 8
  %40 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %39, 1
  %41 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %38, ptr %41, align 8
  %42 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %40, ptr %42, align 8
  call void @memrefCopy(i64 4, ptr %41, ptr %42)
  call void @llvm.stackrestore(ptr %36)
  %43 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 248320) to i64))
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %43, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, ptr %43, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 0, 2
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 248320, 3, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 1, 3, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 1, 4, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 1, 4, 1
  %51 = ptrtoint ptr %43 to i64
  %52 = inttoptr i64 %51 to ptr
  %53 = ptrtoint ptr %20 to i64
  %54 = inttoptr i64 %53 to ptr
  %55 = ptrtoint ptr %28 to i64
  %56 = inttoptr i64 %55 to ptr
  %57 = call i32 @rxops_bridge_matmul_f16(ptr %52, ptr %54, ptr %56, i64 248320, i64 2048, i64 2048)
  %58 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 248320) to i64))
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %58, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, ptr %58, 1
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 0, 2
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 1, 3, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 248320, 3, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 248320, 4, 0
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 1, 4, 1
  %66 = ptrtoint ptr %58 to i64
  %67 = inttoptr i64 %66 to ptr
  %68 = ptrtoint ptr %43 to i64
  %69 = inttoptr i64 %68 to ptr
  %70 = call i32 @rxops_bridge_reshape_bytes(ptr %67, ptr %69, i64 496640)
  %71 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 1) to i64))
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %71, 0
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, ptr %71, 1
  %74 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, i64 0, 2
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %74, i64 1, 3, 0
  %76 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, i64 1, 3, 1
  %77 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, i64 1, 4, 0
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, i64 1, 4, 1
  %79 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %79, 0
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, ptr %79, 1
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, i64 0, 2
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 1, 3, 0
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, i64 1, 3, 1
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, i64 1, 4, 0
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, i64 1, 4, 1
  %87 = ptrtoint ptr %71 to i64
  %88 = inttoptr i64 %87 to ptr
  %89 = ptrtoint ptr %79 to i64
  %90 = inttoptr i64 %89 to ptr
  %91 = ptrtoint ptr %58 to i64
  %92 = inttoptr i64 %91 to ptr
  %93 = call i32 @rxops_bridge_topk_f16(ptr %88, ptr %90, ptr %92, i64 1, i64 1, i64 248320, i64 1)
  call void @free(ptr %15)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %86
}

define void @_mlir_ciface_lm_head_main(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 1
  %11 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @lm_head_main(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
