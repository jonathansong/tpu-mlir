; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @memrefCopy(i64, ptr, ptr)

define private i32 @rxops_bridge_add_f32(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = call i32 @_mlir_ciface_rxops_bridge_add_f32(ptr %0, ptr %1, ptr %2, i64 %3)
  ret i32 %5
}

declare i32 @_mlir_ciface_rxops_bridge_add_f32(ptr, ptr, ptr, i64)

define private i32 @rxops_bridge_matmul_f32(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5) {
  %7 = call i32 @_mlir_ciface_rxops_bridge_matmul_f32(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5)
  ret i32 %7
}

declare i32 @_mlir_ciface_rxops_bridge_matmul_f32(ptr, ptr, ptr, i64, i64, i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @block7_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
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
  %29 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64), i64 64))
  %30 = ptrtoint ptr %29 to i64
  %31 = add i64 %30, 63
  %32 = urem i64 %31, 64
  %33 = sub i64 %31, %32
  %34 = inttoptr i64 %33 to ptr
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %29, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, ptr %34, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 0, 2
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 1, 3, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 8, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 8, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 1, 4, 1
  %42 = call ptr @llvm.stacksave()
  %43 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %43, align 8
  %44 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %43, 1
  %45 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, ptr %45, align 8
  %46 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %45, 1
  %47 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %44, ptr %47, align 8
  %48 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %46, ptr %48, align 8
  call void @memrefCopy(i64 4, ptr %47, ptr %48)
  call void @llvm.stackrestore(ptr %42)
  %49 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 64) to i64))
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %49, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, ptr %49, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 0, 2
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 8, 3, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 8, 3, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 8, 4, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 1, 4, 1
  %57 = call ptr @llvm.stacksave()
  %58 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, ptr %58, align 8
  %59 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %58, 1
  %60 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, ptr %60, align 8
  %61 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %60, 1
  %62 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %59, ptr %62, align 8
  %63 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %61, ptr %63, align 8
  call void @memrefCopy(i64 4, ptr %62, ptr %63)
  call void @llvm.stackrestore(ptr %57)
  %64 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %64, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, ptr %64, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 0, 2
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 1, 3, 0
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 8, 3, 1
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, i64 8, 4, 0
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 1, 4, 1
  %72 = ptrtoint ptr %64 to i64
  %73 = inttoptr i64 %72 to ptr
  %74 = ptrtoint ptr %34 to i64
  %75 = inttoptr i64 %74 to ptr
  %76 = ptrtoint ptr %49 to i64
  %77 = inttoptr i64 %76 to ptr
  %78 = call i32 @rxops_bridge_matmul_f32(ptr %73, ptr %75, ptr %77, i64 1, i64 8, i64 8)
  %79 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %79, 0
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, ptr %79, 1
  %82 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %81, i64 0, 2
  %83 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %82, i64 1, 3, 0
  %84 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %83, i64 8, 3, 1
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, i64 8, 4, 0
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, i64 1, 4, 1
  %87 = call ptr @llvm.stacksave()
  %88 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %88, align 8
  %89 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %88, 1
  %90 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %86, ptr %90, align 8
  %91 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %90, 1
  %92 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %89, ptr %92, align 8
  %93 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %91, ptr %93, align 8
  call void @memrefCopy(i64 4, ptr %92, ptr %93)
  call void @llvm.stackrestore(ptr %87)
  %94 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %94, 0
  %96 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %95, ptr %94, 1
  %97 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %96, i64 0, 2
  %98 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %97, i64 1, 3, 0
  %99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %98, i64 8, 3, 1
  %100 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %99, i64 8, 4, 0
  %101 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %100, i64 1, 4, 1
  %102 = ptrtoint ptr %94 to i64
  %103 = inttoptr i64 %102 to ptr
  %104 = ptrtoint ptr %64 to i64
  %105 = inttoptr i64 %104 to ptr
  %106 = ptrtoint ptr %79 to i64
  %107 = inttoptr i64 %106 to ptr
  %108 = call i32 @rxops_bridge_add_f32(ptr %103, ptr %105, ptr %107, i64 8)
  call void @free(ptr %29)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %101
}

define void @_mlir_ciface_block7_main(ptr %0, ptr %1, ptr %2) {
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
  %20 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @block7_main(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19)
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
