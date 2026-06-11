module @lm_head attributes {ada300.rxops_codegen = "full", ada300.rxops_placeholder_ops = 1 : i64, ada300.rxops_supported_ops = 7 : i64, llvm.data_layout = "", module.FLOPs = 1017118720 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 1017118720 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 2007168 : i64, module.platform = "LLM_QUANTIZED", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.top_run_mode = "STATIC", module.weight_file = "lm_head_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @rxops_bridge_topk_f16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_topk_f16(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_topk_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @rxops_bridge_reshape_bytes(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_reshape_bytes(%arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_reshape_bytes(!llvm.ptr, !llvm.ptr, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func private @rxops_bridge_matmul_f16(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64, %arg4: i64, %arg5: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_matmul_f16(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_matmul_f16(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.mlir.constant(496640 : i64) : i64
    %10 = llvm.mlir.constant(2048 : i64) : i64
    %11 = llvm.mlir.constant(248320 : i64) : i64
    %12 = llvm.mlir.constant(248320 : index) : i64
    %13 = llvm.mlir.constant(2048 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(508559360 : index) : i64
    %16 = llvm.mlir.null : !llvm.ptr
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.add %18, %19  : i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.sub %19, %23  : i64
    %25 = llvm.add %22, %24  : i64
    %26 = llvm.urem %25, %19  : i64
    %27 = llvm.sub %25, %26  : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    %29 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %21, %29[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.constant(0 : index) : i64
    %33 = llvm.insertvalue %32, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %12, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %13, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.mlir.constant(2048 : index) : i64
    %40 = llvm.mlir.constant(1 : index) : i64
    %41 = llvm.mlir.constant(2048 : index) : i64
    %42 = llvm.mlir.null : !llvm.ptr
    %43 = llvm.getelementptr %42[2048] : (!llvm.ptr) -> !llvm.ptr, f32
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.call @malloc(%44) : (i64) -> !llvm.ptr
    %46 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.insertvalue %45, %46[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.insertvalue %45, %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.mlir.constant(0 : index) : i64
    %50 = llvm.insertvalue %49, %48[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %38, %50[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %39, %51[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %39, %52[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %40, %53[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.intr.stacksave : !llvm.ptr
    %56 = llvm.mlir.constant(2 : i64) : i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.alloca %57 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %58 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %59 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %60 = llvm.insertvalue %56, %59[0] : !llvm.struct<(i64, ptr)> 
    %61 = llvm.insertvalue %58, %60[1] : !llvm.struct<(i64, ptr)> 
    %62 = llvm.mlir.constant(2 : i64) : i64
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.alloca %63 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %54, %64 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %65 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %66 = llvm.insertvalue %62, %65[0] : !llvm.struct<(i64, ptr)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(i64, ptr)> 
    %68 = llvm.mlir.constant(1 : index) : i64
    %69 = llvm.alloca %68 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %61, %69 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %70 = llvm.alloca %68 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %67, %70 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %71 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%71, %69, %70) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %55 : !llvm.ptr
    %72 = llvm.mlir.constant(248320 : index) : i64
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.mlir.null : !llvm.ptr
    %76 = llvm.getelementptr %75[248320] : (!llvm.ptr) -> !llvm.ptr, f16
    %77 = llvm.ptrtoint %76 : !llvm.ptr to i64
    %78 = llvm.call @malloc(%77) : (i64) -> !llvm.ptr
    %79 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %80 = llvm.insertvalue %78, %79[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.insertvalue %78, %80[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(0 : index) : i64
    %83 = llvm.insertvalue %82, %81[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %84 = llvm.insertvalue %72, %83[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.insertvalue %73, %84[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.insertvalue %73, %85[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %74, %86[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    %90 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %91 = llvm.inttoptr %90 : i64 to !llvm.ptr
    %92 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %93 = llvm.inttoptr %92 : i64 to !llvm.ptr
    %94 = llvm.call @rxops_bridge_matmul_f16(%89, %91, %93, %11, %10, %10) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %95 = llvm.mlir.constant(1 : index) : i64
    %96 = llvm.mlir.constant(248320 : index) : i64
    %97 = llvm.mlir.constant(1 : index) : i64
    %98 = llvm.mlir.constant(248320 : index) : i64
    %99 = llvm.mlir.null : !llvm.ptr
    %100 = llvm.getelementptr %99[248320] : (!llvm.ptr) -> !llvm.ptr, f16
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.call @malloc(%101) : (i64) -> !llvm.ptr
    %103 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %104 = llvm.insertvalue %102, %103[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %102, %104[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mlir.constant(0 : index) : i64
    %107 = llvm.insertvalue %106, %105[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %95, %107[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %96, %108[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %96, %109[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %97, %110[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %113 = llvm.inttoptr %112 : i64 to !llvm.ptr
    %114 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %115 = llvm.inttoptr %114 : i64 to !llvm.ptr
    %116 = llvm.call @rxops_bridge_reshape_bytes(%113, %115, %9) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, i64) -> i32
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.mlir.constant(1 : index) : i64
    %119 = llvm.mlir.constant(1 : index) : i64
    %120 = llvm.mlir.null : !llvm.ptr
    %121 = llvm.getelementptr %120[1] : (!llvm.ptr) -> !llvm.ptr, f16
    %122 = llvm.ptrtoint %121 : !llvm.ptr to i64
    %123 = llvm.call @malloc(%122) : (i64) -> !llvm.ptr
    %124 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %125 = llvm.insertvalue %123, %124[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %123, %125[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.mlir.constant(0 : index) : i64
    %128 = llvm.insertvalue %127, %126[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %117, %128[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %118, %129[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %118, %130[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %119, %131[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.mlir.constant(1 : index) : i64
    %134 = llvm.mlir.constant(1 : index) : i64
    %135 = llvm.mlir.constant(1 : index) : i64
    %136 = llvm.mlir.null : !llvm.ptr
    %137 = llvm.getelementptr %136[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %138 = llvm.ptrtoint %137 : !llvm.ptr to i64
    %139 = llvm.call @malloc(%138) : (i64) -> !llvm.ptr
    %140 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %141 = llvm.insertvalue %139, %140[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.insertvalue %139, %141[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.mlir.constant(0 : index) : i64
    %144 = llvm.insertvalue %143, %142[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %133, %144[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.insertvalue %134, %145[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.insertvalue %134, %146[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.insertvalue %135, %147[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.ptrtoint %123 : !llvm.ptr to i64
    %150 = llvm.inttoptr %149 : i64 to !llvm.ptr
    %151 = llvm.ptrtoint %139 : !llvm.ptr to i64
    %152 = llvm.inttoptr %151 : i64 to !llvm.ptr
    %153 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %154 = llvm.inttoptr %153 : i64 to !llvm.ptr
    %155 = llvm.call @rxops_bridge_topk_f16(%150, %152, %154, %8, %8, %11, %8) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64) -> i32
    llvm.call @free(%21) : (!llvm.ptr) -> ()
    llvm.return %148 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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

