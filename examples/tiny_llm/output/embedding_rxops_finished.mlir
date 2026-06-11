module @embedding attributes {ada300.rxops_codegen = "full", ada300.rxops_placeholder_ops = 1 : i64, ada300.rxops_supported_ops = 6 : i64, llvm.data_layout = "", module.FLOPs = 0 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 1017118720 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 12599296 : i64, module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "embedding_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @rxops_bridge_gather_nd(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_gather_nd(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_gather_nd(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(1024 : i64) : i64
    %9 = llvm.mlir.constant(1 : i64) : i64
    %10 = llvm.mlir.constant(2048 : i64) : i64
    %11 = llvm.mlir.constant(248320 : i64) : i64
    %12 = llvm.mlir.constant(2 : i64) : i64
    %13 = llvm.mlir.constant(0 : i64) : i64
    %14 = llvm.mlir.constant(248320 : index) : i64
    %15 = llvm.mlir.constant(2048 : index) : i64
    %16 = llvm.mlir.constant(1 : index) : i64
    %17 = llvm.mlir.constant(508559360 : index) : i64
    %18 = llvm.mlir.null : !llvm.ptr
    %19 = llvm.getelementptr %18[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %20 = llvm.ptrtoint %19 : !llvm.ptr to i64
    %21 = llvm.mlir.constant(64 : index) : i64
    %22 = llvm.add %20, %21  : i64
    %23 = llvm.call @malloc(%22) : (i64) -> !llvm.ptr
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.sub %21, %25  : i64
    %27 = llvm.add %24, %26  : i64
    %28 = llvm.urem %27, %21  : i64
    %29 = llvm.sub %27, %28  : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    %31 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %23, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %14, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %15, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %16, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(1 : index) : i64
    %41 = llvm.mlir.constant(1024 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.mlir.constant(1024 : index) : i64
    %44 = llvm.mlir.null : !llvm.ptr
    %45 = llvm.getelementptr %44[1024] : (!llvm.ptr) -> !llvm.ptr, i32
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.mlir.null : !llvm.ptr
    %48 = llvm.getelementptr %47[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.add %46, %49  : i64
    %51 = llvm.call @malloc(%50) : (i64) -> !llvm.ptr
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.sub %49, %53  : i64
    %55 = llvm.add %52, %54  : i64
    %56 = llvm.urem %55, %49  : i64
    %57 = llvm.sub %55, %56  : i64
    %58 = llvm.inttoptr %57 : i64 to !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.insertvalue %51, %59[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %40, %63[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %41, %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %41, %65[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %42, %66[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.intr.stacksave : !llvm.ptr
    %69 = llvm.mlir.constant(2 : i64) : i64
    %70 = llvm.mlir.constant(1 : index) : i64
    %71 = llvm.alloca %70 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %72 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %73 = llvm.insertvalue %69, %72[0] : !llvm.struct<(i64, ptr)> 
    %74 = llvm.insertvalue %71, %73[1] : !llvm.struct<(i64, ptr)> 
    %75 = llvm.mlir.constant(2 : i64) : i64
    %76 = llvm.mlir.constant(1 : index) : i64
    %77 = llvm.alloca %76 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %67, %77 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %78 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %79 = llvm.insertvalue %75, %78[0] : !llvm.struct<(i64, ptr)> 
    %80 = llvm.insertvalue %77, %79[1] : !llvm.struct<(i64, ptr)> 
    %81 = llvm.mlir.constant(1 : index) : i64
    %82 = llvm.alloca %81 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %74, %82 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %83 = llvm.alloca %81 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %80, %83 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %84 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%84, %82, %83) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %68 : !llvm.ptr
    %85 = llvm.mlir.constant(1 : index) : i64
    %86 = llvm.mlir.constant(1024 : index) : i64
    %87 = llvm.mlir.constant(2048 : index) : i64
    %88 = llvm.mlir.constant(1 : index) : i64
    %89 = llvm.mlir.constant(2097152 : index) : i64
    %90 = llvm.mlir.constant(2097152 : index) : i64
    %91 = llvm.mlir.null : !llvm.ptr
    %92 = llvm.getelementptr %91[2097152] : (!llvm.ptr) -> !llvm.ptr, f16
    %93 = llvm.ptrtoint %92 : !llvm.ptr to i64
    %94 = llvm.call @malloc(%93) : (i64) -> !llvm.ptr
    %95 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %96 = llvm.insertvalue %94, %95[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %97 = llvm.insertvalue %94, %96[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %98 = llvm.mlir.constant(0 : index) : i64
    %99 = llvm.insertvalue %98, %97[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %100 = llvm.insertvalue %85, %99[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %101 = llvm.insertvalue %86, %100[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %102 = llvm.insertvalue %87, %101[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.insertvalue %89, %102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %104 = llvm.insertvalue %87, %103[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %105 = llvm.insertvalue %88, %104[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %106 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %107 = llvm.inttoptr %106 : i64 to !llvm.ptr
    %108 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %109 = llvm.inttoptr %108 : i64 to !llvm.ptr
    %110 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %111 = llvm.inttoptr %110 : i64 to !llvm.ptr
    %112 = llvm.call @rxops_bridge_gather_nd(%107, %109, %111, %13, %12, %11, %10, %9, %9, %9, %9, %9, %8, %9, %9, %9, %9, %9, %8, %10, %9, %9, %9) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
    llvm.call @free(%23) : (!llvm.ptr) -> ()
    llvm.return %105 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    llvm.return
  }
}

