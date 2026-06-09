module @tiny_embedding attributes {ada300.rxops_codegen = "minimal", ada300.rxops_placeholder_ops = 0 : i64, ada300.rxops_supported_ops = 4 : i64, llvm.data_layout = "", module.FLOPs = 8 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 0 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 320 : i64, module.platform = "ONNX", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.weight_file = "tiny_embedding_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @rxops_bridge_add_f32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_add_f32(%arg0, %arg1, %arg2, %arg3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_add_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(8 : i64) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(8 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(8 : index) : i64
    %13 = llvm.mlir.null : !llvm.ptr
    %14 = llvm.getelementptr %13[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.mlir.constant(64 : index) : i64
    %17 = llvm.add %15, %16  : i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.sub %16, %20  : i64
    %22 = llvm.add %19, %21  : i64
    %23 = llvm.urem %22, %16  : i64
    %24 = llvm.sub %22, %23  : i64
    %25 = llvm.inttoptr %24 : i64 to !llvm.ptr
    %26 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %18, %26[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %25, %27[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %9, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %10, %31[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %10, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %11, %33[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
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
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.mlir.constant(8 : index) : i64
    %54 = llvm.mlir.constant(1 : index) : i64
    %55 = llvm.mlir.constant(8 : index) : i64
    %56 = llvm.mlir.null : !llvm.ptr
    %57 = llvm.getelementptr %56[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    %60 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.insertvalue %59, %60[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %59, %61[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.mlir.constant(0 : index) : i64
    %64 = llvm.insertvalue %63, %62[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %52, %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %53, %65[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %53, %66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %54, %67[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.intr.stacksave : !llvm.ptr
    %70 = llvm.mlir.constant(2 : i64) : i64
    %71 = llvm.mlir.constant(1 : index) : i64
    %72 = llvm.alloca %71 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %72 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %73 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %74 = llvm.insertvalue %70, %73[0] : !llvm.struct<(i64, ptr)> 
    %75 = llvm.insertvalue %72, %74[1] : !llvm.struct<(i64, ptr)> 
    %76 = llvm.mlir.constant(2 : i64) : i64
    %77 = llvm.mlir.constant(1 : index) : i64
    %78 = llvm.alloca %77 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %68, %78 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %79 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %80 = llvm.insertvalue %76, %79[0] : !llvm.struct<(i64, ptr)> 
    %81 = llvm.insertvalue %78, %80[1] : !llvm.struct<(i64, ptr)> 
    %82 = llvm.mlir.constant(1 : index) : i64
    %83 = llvm.alloca %82 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %75, %83 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %84 = llvm.alloca %82 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %81, %84 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %85 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%85, %83, %84) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %69 : !llvm.ptr
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.mlir.constant(8 : index) : i64
    %88 = llvm.mlir.constant(1 : index) : i64
    %89 = llvm.mlir.constant(8 : index) : i64
    %90 = llvm.mlir.null : !llvm.ptr
    %91 = llvm.getelementptr %90[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %92 = llvm.ptrtoint %91 : !llvm.ptr to i64
    %93 = llvm.call @malloc(%92) : (i64) -> !llvm.ptr
    %94 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %95 = llvm.insertvalue %93, %94[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.insertvalue %93, %95[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.mlir.constant(0 : index) : i64
    %98 = llvm.insertvalue %97, %96[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %86, %98[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.insertvalue %87, %99[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.insertvalue %87, %100[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %88, %101[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.ptrtoint %93 : !llvm.ptr to i64
    %104 = llvm.inttoptr %103 : i64 to !llvm.ptr
    %105 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %108 = llvm.inttoptr %107 : i64 to !llvm.ptr
    %109 = llvm.call @rxops_bridge_add_f32(%104, %106, %108, %8) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    llvm.call @free(%18) : (!llvm.ptr) -> ()
    llvm.return %102 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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
    %8 = llvm.call @main(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

