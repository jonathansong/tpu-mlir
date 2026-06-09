module @tiny_block7 attributes {ada300.rxops_codegen = "minimal", ada300.rxops_placeholder_ops = 0 : i64, ada300.rxops_supported_ops = 8 : i64, llvm.data_layout = "", module.FLOPs = 136 : i64, module.addr_mode = "basic", module.asymmetric = false, module.chip = "ada300", module.coeff_addr = 0 : i64, module.coeff_size = 0 : i64, module.cores = 1 : i64, module.devices = 1 : i64, module.high_precision = false, module.mode = "F16", module.neuron_addr = 0 : i64, module.neuron_size = 1024 : i64, module.platform = "ONNX", module.q_group_size = 0 : i64, module.q_symmetric = false, module.state = "TPU_ADDRESSED", module.weight_file = "tiny_block7_tpu_addressed_ada300_f16_weight.npz", tpu.gmem_bytes = 536870912 : i64, tpu.sram_bank_bytes = 262144 : i64, tpu.sram_bank_count = 5 : i64, tpu.target = "ada300", tpu.weight_memory_bytes = 268435456 : i64} {
  llvm.func @free(!llvm.ptr)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func private @rxops_bridge_add_f32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_rxops_bridge_add_f32(%arg0, %arg1, %arg2, %arg3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    llvm.return %0 : i32
  }
  llvm.func @_mlir_ciface_rxops_bridge_add_f32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "private"}
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
    %25 = llvm.mlir.constant(64 : index) : i64
    %26 = llvm.add %24, %25  : i64
    %27 = llvm.call @malloc(%26) : (i64) -> !llvm.ptr
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.sub %25, %29  : i64
    %31 = llvm.add %28, %30  : i64
    %32 = llvm.urem %31, %25  : i64
    %33 = llvm.sub %31, %32  : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %35 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.insertvalue %27, %35[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %34, %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.insertvalue %38, %37[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %18, %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %19, %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %19, %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %20, %42[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.intr.stacksave : !llvm.ptr
    %45 = llvm.mlir.constant(2 : i64) : i64
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.alloca %46 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %47 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %48 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %49 = llvm.insertvalue %45, %48[0] : !llvm.struct<(i64, ptr)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(i64, ptr)> 
    %51 = llvm.mlir.constant(2 : i64) : i64
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.alloca %52 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %43, %53 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %54 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %55 = llvm.insertvalue %51, %54[0] : !llvm.struct<(i64, ptr)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(i64, ptr)> 
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.alloca %57 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %50, %58 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %59 = llvm.alloca %57 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %56, %59 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %60 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%60, %58, %59) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %44 : !llvm.ptr
    %61 = llvm.mlir.constant(8 : index) : i64
    %62 = llvm.mlir.constant(8 : index) : i64
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.mlir.constant(64 : index) : i64
    %65 = llvm.mlir.null : !llvm.ptr
    %66 = llvm.getelementptr %65[64] : (!llvm.ptr) -> !llvm.ptr, f32
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.call @malloc(%67) : (i64) -> !llvm.ptr
    %69 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.insertvalue %68, %69[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %68, %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.insertvalue %61, %73[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %62, %74[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %62, %75[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %63, %76[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.intr.stacksave : !llvm.ptr
    %79 = llvm.mlir.constant(2 : i64) : i64
    %80 = llvm.mlir.constant(1 : index) : i64
    %81 = llvm.alloca %80 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %15, %81 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %82 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %83 = llvm.insertvalue %79, %82[0] : !llvm.struct<(i64, ptr)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(i64, ptr)> 
    %85 = llvm.mlir.constant(2 : i64) : i64
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.alloca %86 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %77, %87 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %88 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %89 = llvm.insertvalue %85, %88[0] : !llvm.struct<(i64, ptr)> 
    %90 = llvm.insertvalue %87, %89[1] : !llvm.struct<(i64, ptr)> 
    %91 = llvm.mlir.constant(1 : index) : i64
    %92 = llvm.alloca %91 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %84, %92 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %93 = llvm.alloca %91 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %90, %93 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %94 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%94, %92, %93) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %78 : !llvm.ptr
    %95 = llvm.mlir.constant(1 : index) : i64
    %96 = llvm.mlir.constant(8 : index) : i64
    %97 = llvm.mlir.constant(1 : index) : i64
    %98 = llvm.mlir.constant(8 : index) : i64
    %99 = llvm.mlir.null : !llvm.ptr
    %100 = llvm.getelementptr %99[8] : (!llvm.ptr) -> !llvm.ptr, f32
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
    %114 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %115 = llvm.inttoptr %114 : i64 to !llvm.ptr
    %116 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %117 = llvm.inttoptr %116 : i64 to !llvm.ptr
    %118 = llvm.call @rxops_bridge_matmul_f32(%113, %115, %117, %17, %16, %16) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
    %119 = llvm.mlir.constant(1 : index) : i64
    %120 = llvm.mlir.constant(8 : index) : i64
    %121 = llvm.mlir.constant(1 : index) : i64
    %122 = llvm.mlir.constant(8 : index) : i64
    %123 = llvm.mlir.null : !llvm.ptr
    %124 = llvm.getelementptr %123[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    %126 = llvm.call @malloc(%125) : (i64) -> !llvm.ptr
    %127 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.insertvalue %126, %127[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %126, %128[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.mlir.constant(0 : index) : i64
    %131 = llvm.insertvalue %130, %129[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %119, %131[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %120, %132[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %120, %133[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %121, %134[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.intr.stacksave : !llvm.ptr
    %137 = llvm.mlir.constant(2 : i64) : i64
    %138 = llvm.mlir.constant(1 : index) : i64
    %139 = llvm.alloca %138 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %7, %139 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %140 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %141 = llvm.insertvalue %137, %140[0] : !llvm.struct<(i64, ptr)> 
    %142 = llvm.insertvalue %139, %141[1] : !llvm.struct<(i64, ptr)> 
    %143 = llvm.mlir.constant(2 : i64) : i64
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.alloca %144 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %135, %145 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %146 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %147 = llvm.insertvalue %143, %146[0] : !llvm.struct<(i64, ptr)> 
    %148 = llvm.insertvalue %145, %147[1] : !llvm.struct<(i64, ptr)> 
    %149 = llvm.mlir.constant(1 : index) : i64
    %150 = llvm.alloca %149 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %142, %150 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %151 = llvm.alloca %149 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %148, %151 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %152 = llvm.mlir.constant(4 : index) : i64
    llvm.call @memrefCopy(%152, %150, %151) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %136 : !llvm.ptr
    %153 = llvm.mlir.constant(1 : index) : i64
    %154 = llvm.mlir.constant(8 : index) : i64
    %155 = llvm.mlir.constant(1 : index) : i64
    %156 = llvm.mlir.constant(8 : index) : i64
    %157 = llvm.mlir.null : !llvm.ptr
    %158 = llvm.getelementptr %157[8] : (!llvm.ptr) -> !llvm.ptr, f32
    %159 = llvm.ptrtoint %158 : !llvm.ptr to i64
    %160 = llvm.call @malloc(%159) : (i64) -> !llvm.ptr
    %161 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %162 = llvm.insertvalue %160, %161[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %160, %162[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.mlir.constant(0 : index) : i64
    %165 = llvm.insertvalue %164, %163[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %153, %165[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.insertvalue %154, %166[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.insertvalue %154, %167[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.insertvalue %155, %168[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.ptrtoint %160 : !llvm.ptr to i64
    %171 = llvm.inttoptr %170 : i64 to !llvm.ptr
    %172 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %173 = llvm.inttoptr %172 : i64 to !llvm.ptr
    %174 = llvm.ptrtoint %126 : !llvm.ptr to i64
    %175 = llvm.inttoptr %174 : i64 to !llvm.ptr
    %176 = llvm.call @rxops_bridge_add_f32(%171, %173, %175, %16) {ada300.rxops.keep = true} : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
    llvm.call @free(%27) : (!llvm.ptr) -> ()
    llvm.return %169 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
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

