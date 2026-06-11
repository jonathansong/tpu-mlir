; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define private i32 @rxops_bridge_gather_nd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22) {
  %24 = call i32 @_mlir_ciface_rxops_bridge_gather_nd(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22)
  ret i32 %24
}

declare i32 @_mlir_ciface_rxops_bridge_gather_nd(ptr, ptr, ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @embedding_cache_main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
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
  %28 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)))
  %29 = ptrtoint ptr %28 to i64
  %30 = add i64 %29, sub (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 1)
  %31 = urem i64 %30, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %32 = sub i64 %30, %31
  %33 = inttoptr i64 %32 to ptr
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %28, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %33, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 0, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 1, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 1, 3, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 1, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 1, 4, 1
  %41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, 3, 0
  %42 = mul i64 %41, 1
  %43 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, 3, 1
  %44 = mul i64 %42, %43
  %45 = mul i64 %44, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, 1
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, 2
  %48 = getelementptr i32, ptr %46, i64 %47
  %49 = getelementptr i32, ptr %33, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr %49, ptr %48, i64 %45, i1 false)
  %50 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (half, ptr null, i32 2048) to i64))
  %51 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %50, 0
  %52 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %51, ptr %50, 1
  %53 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %52, i64 0, 2
  %54 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %53, i64 1, 3, 0
  %55 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %54, i64 1, 3, 1
  %56 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %55, i64 2048, 3, 2
  %57 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %56, i64 2048, 4, 0
  %58 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %57, i64 2048, 4, 1
  %59 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %58, i64 1, 4, 2
  %60 = ptrtoint ptr %50 to i64
  %61 = inttoptr i64 %60 to ptr
  %62 = ptrtoint ptr %20 to i64
  %63 = inttoptr i64 %62 to ptr
  %64 = ptrtoint ptr %33 to i64
  %65 = inttoptr i64 %64 to ptr
  %66 = call i32 @rxops_bridge_gather_nd(ptr %61, ptr %63, ptr %65, i64 0, i64 2, i64 248320, i64 2048, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 2048, i64 1, i64 1, i64 1)
  call void @free(ptr %15)
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %59
}

define void @_mlir_ciface_embedding_cache_main(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %4 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, 4, 1
  %11 = call { ptr, ptr, i64, [3 x i64], [3 x i64] } @embedding_cache_main(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10)
  store { ptr, ptr, i64, [3 x i64], [3 x i64] } %11, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
