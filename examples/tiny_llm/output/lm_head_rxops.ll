; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define private i32 @rxops_bridge_matmul_f32(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5) {
  %7 = call i32 @_mlir_ciface_rxops_bridge_matmul_f32(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5)
  ret i32 %7
}

declare i32 @_mlir_ciface_rxops_bridge_matmul_f32(ptr, ptr, ptr, i64, i64, i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @lm_head_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %1, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %7, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %8, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %9, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %10, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %12, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %11, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 4, 1
  %29 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, ptr %29, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 0, 2
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 1, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 8, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 8, 4, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 1, 4, 1
  %37 = call ptr @llvm.stacksave()
  %38 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %38, align 8
  %39 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %38, 1
  %40 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %40, align 8
  %41 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %40, 1
  %42 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %39, ptr %42, align 8
  %43 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %41, ptr %43, align 8
  call void @memrefCopy(i64 4, ptr %42, ptr %43)
  call void @llvm.stackrestore(ptr %37)
  %44 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %44, 0
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, ptr %44, 1
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 0, 2
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 8, 3, 0
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 1, 3, 1
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 1, 4, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 1, 4, 1
  %52 = call ptr @llvm.stacksave()
  %53 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, ptr %53, align 8
  %54 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %53, 1
  %55 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, ptr %55, align 8
  %56 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %55, 1
  %57 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %54, ptr %57, align 8
  %58 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %56, ptr %58, align 8
  call void @memrefCopy(i64 4, ptr %57, ptr %58)
  call void @llvm.stackrestore(ptr %52)
  %59 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64))
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %59, 0
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, ptr %59, 1
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 0, 2
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 1, 3, 0
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 1, 3, 1
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 1, 4, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 1, 4, 1
  %67 = ptrtoint ptr %59 to i64
  %68 = inttoptr i64 %67 to ptr
  %69 = ptrtoint ptr %29 to i64
  %70 = inttoptr i64 %69 to ptr
  %71 = ptrtoint ptr %44 to i64
  %72 = inttoptr i64 %71 to ptr
  %73 = call i32 @rxops_bridge_matmul_f32(ptr %68, ptr %70, ptr %72, i64 1, i64 1, i64 8)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %66
}

define void @_mlir_ciface_lm_head_main(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 1
  %12 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 0
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 1
  %20 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @lm_head_main(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
