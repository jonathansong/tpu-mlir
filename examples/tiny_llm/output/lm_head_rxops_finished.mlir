module @tiny_lm_head attributes {ada300.rxops_codegen = "minimal", ada300.rxops_placeholder_ops = 0 : i64, ada300.rxops_supported_ops = 7 : i64, llvm.data_layout = "", module.FLOPs = 16 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 0 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 512 : i64, module.platform = "ONNX", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.weight_file = "tiny_lm_head_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @rxops_bridge_matmul_f32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_matmul_f32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_matmul_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(8 : i64) : i64
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.mlir.constant(8 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(8 : index) : i64
    %22 = llvm.mlir.null : !llvm.ptr
    %23 = llvm.getelementptr %22[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.call @malloc(%24) : (i64) -> !llvm.ptr
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %25, %26[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %18, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %19, %31[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %19, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %20, %33[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.intr.stacksave : !llvm.ptr
    %36 = llvm.mlir.constant(2 : i64) : i64
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.alloca %37 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %38 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %39 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %40 = llvm.insertvalue %36, %39[0] : !llvm.struct<(i64, ptr)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(i64, ptr)> 
    %42 = llvm.mlir.constant(2 : i64) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.alloca %43 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %34, %44 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %45 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %46 = llvm.insertvalue %42, %45[0] : !llvm.struct<(i64, ptr)> 
    %47 = llvm.insertvalue %44, %46[1] : !llvm.struct<(i64, ptr)> 
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.alloca %48 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %41, %49 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %50 = llvm.alloca %48 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %47, %50 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %51 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%51, %49, %50) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %35 : !llvm.ptr
    %52 = llvm.mlir.constant(8 : index) : i64
    %53 = llvm.mlir.constant(1 : index) : i64
    %54 = llvm.mlir.constant(1 : index) : i64
    %55 = llvm.mlir.null : !llvm.ptr
    %56 = llvm.getelementptr %55[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %57 = llvm.ptrtoint %56 : !llvm.ptr to i64
    %58 = llvm.call @malloc(%57) : (i64) -> !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.insertvalue %58, %59[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mlir.constant(0 : index) : i64
    %63 = llvm.insertvalue %62, %61[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %52, %63[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %53, %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %53, %65[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %54, %66[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.intr.stacksave : !llvm.ptr
    %69 = llvm.mlir.constant(2 : i64) : i64
    %70 = llvm.mlir.constant(1 : index) : i64
    %71 = llvm.alloca %70 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %15, %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
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
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.mlir.constant(1 : index) : i64
    %88 = llvm.mlir.null : !llvm.ptr
    %89 = llvm.getelementptr %88[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %90 = llvm.ptrtoint %89 : !llvm.ptr to i64
    %91 = llvm.call @malloc(%90) : (i64) -> !llvm.ptr
    %92 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.insertvalue %91, %92[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %91, %93[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.insertvalue %95, %94[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.insertvalue %85, %96[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.insertvalue %86, %97[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %86, %98[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.insertvalue %87, %99[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %102 = llvm.inttoptr %101 : i64 to !llvm.ptr
    %103 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %104 = llvm.inttoptr %103 : i64 to !llvm.ptr
    %105 = llvm.ptrtoint %58 : !llvm.ptr to i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.call @rxops_bridge_matmul_f32(%102, %104, %106, %17, %17, %16) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    llvm.return %100 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  llvm.func @_mlir_ciface_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %16, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

