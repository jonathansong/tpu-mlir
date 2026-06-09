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

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @embedding_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %9 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %1, 1
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %9, i64 %2, 2
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, i64 %3, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 %5, 4, 0
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 %4, 3, 1
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 %6, 4, 1
  %15 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64), i64 64))
  %16 = ptrtoint ptr %15 to i64
  %17 = add i64 %16, 63
  %18 = urem i64 %17, 64
  %19 = sub i64 %17, %18
  %20 = inttoptr i64 %19 to ptr
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %15, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, ptr %20, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 0, 2
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 1, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 8, 3, 1
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 8, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 1, 4, 1
  %28 = call ptr @llvm.stacksave()
  %29 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, ptr %29, align 8
  %30 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %29, 1
  %31 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, ptr %31, align 8
  %32 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %31, 1
  %33 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %30, ptr %33, align 8
  %34 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %32, ptr %34, align 8
  call void @memrefCopy(i64 4, ptr %33, ptr %34)
  call void @llvm.stackrestore(ptr %28)
  %35 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %35, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %35, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 0, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 1, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 8, 3, 1
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 8, 4, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 1, 4, 1
  %43 = call ptr @llvm.stacksave()
  %44 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, ptr %44, align 8
  %45 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %44, 1
  %46 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, ptr %46, align 8
  %47 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %46, 1
  %48 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %45, ptr %48, align 8
  %49 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %47, ptr %49, align 8
  call void @memrefCopy(i64 4, ptr %48, ptr %49)
  call void @llvm.stackrestore(ptr %43)
  %50 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64))
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %50, 0
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 1, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 8, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 8, 4, 0
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 1, 4, 1
  %58 = ptrtoint ptr %50 to i64
  %59 = inttoptr i64 %58 to ptr
  %60 = ptrtoint ptr %20 to i64
  %61 = inttoptr i64 %60 to ptr
  %62 = ptrtoint ptr %35 to i64
  %63 = inttoptr i64 %62 to ptr
  %64 = call i32 @rxops_bridge_add_f32(ptr %59, ptr %61, ptr %63, i64 8)
  call void @free(ptr %15)
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %57
}

define void @_mlir_ciface_embedding_main(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 1
  %11 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @embedding_main(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)
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
