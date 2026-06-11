# Tiny LLM Ada300 rx-ops demo.
#
# This intentionally mirrors examples/Ada300, but builds three independent
# MLIR subgraphs and links them into one baremetal ELF:
#
#   embedding.mlir -> embedding_tpu.mlir -> embedding_tpu_addr.mlir -> embedding_rxops.ll -> embedding_rv64.o
#   block_7.mlir   -> block_7_tpu.mlir   -> block_7_tpu_addr.mlir   -> block_7_rxops.ll   -> block_7_rv64.o
#   lm_head.mlir   -> lm_head_tpu.mlir   -> lm_head_tpu_addr.mlir   -> lm_head_rxops.ll   -> lm_head_rv64.o
#   tiny_llm_baremetal_main.c + rx_ops_bridge_ada300.c + rx-ops Ada300 backend
#     -> tiny_llm.elf

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(TinyLlmAda300 C CXX)
  find_package(MLIR REQUIRED CONFIG)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  get_filename_component(_TINY_LLM_REPO_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

  find_program(TPUC_OPT tpuc-opt
    HINTS "${_TINY_LLM_REPO_ROOT}/install/bin"
          "${_TINY_LLM_REPO_ROOT}/build/bin"
    DOC "Path to tpuc-opt")
  if(NOT TPUC_OPT)
    message(FATAL_ERROR "tiny_llm: tpuc-opt not found; pass -DTPUC_OPT=/path/to/tpuc-opt")
  endif()

  set(RXOPS_SRC_DIR "${_TINY_LLM_REPO_ROOT}/third_party/rx-ops")
  set(ADA300_EXAMPLE_DIR "${_TINY_LLM_REPO_ROOT}/examples/Ada300")
else()
  set(TPUC_OPT ${CMAKE_BINARY_DIR}/bin/tpuc-opt)
  set(RXOPS_SRC_DIR ${CMAKE_SOURCE_DIR}/third_party/rx-ops)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(ADA300_EXAMPLE_DIR ${CMAKE_SOURCE_DIR}/examples/Ada300)
endif()

set(TINY_LLM_MLIR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mlir)
set(TINY_LLM_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/output)
set(TINY_LLM_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${TINY_LLM_OUT_DIR})
file(MAKE_DIRECTORY ${TINY_LLM_BUILD_DIR})

# Helper script: patches tpu.weight_memory_bytes to 0 so that the
# address-assign pass skips the hardware capacity check.  This lets the
# demo run in QEMU (8 GiB RAM) even when a subgraph weight (e.g. the
# full 248320-entry lm_head) exceeds the real Ada300 256 MiB limit.
file(GENERATE
  OUTPUT "${TINY_LLM_BUILD_DIR}/relax_weight_limit.py"
  CONTENT "import sys, re
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
t = re.sub(r'tpu\\.weight_memory_bytes = \\d+ : i64',
           'tpu.weight_memory_bytes = 0 : i64', t)
open(dst, 'w').write(t)
")

option(TINY_LLM_BAREMETAL
  "Cross-compile tiny LLM demo for Ada300 RISC-V and QEMU" ON)

set(_TINY_BUFFERIZE_OPT
    "-one-shot-bufferize=bufferize-function-boundaries allow-return-allocs")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_TINY_TPUC_OPT_DEP "${TPUC_OPT}")
else()
  set(_TINY_TPUC_OPT_DEP tpuc-opt)
endif()

function(add_tiny_llm_subgraph name source_mlir entry_name)
  # ── output files, one per pipeline step ──────────────────────────────
  # Step 1 : source copy                   → ${name}_top.mlir
  # Step 2 : processor-assign + init       → ${name}_top_assigned.mlir
  # Step 3 : convert-top-to-tpu            → ${name}_tpu.mlir
  # Step 4 : address-assign                → ${name}_tpu_addr.mlir
  # Step 5 : convert-tpu-to-rxops          → ${name}_rxops.mlir
  # Step 6 : bufferize (tensor → memref)   → ${name}_rxops_bufferize.mlir
  # Step 7 : LLVM dialect lowering         → ${name}_rxops_finished.mlir
  # Step 8 : mlir-translate                → ${name}_rxops.ll
  set(_top_mlir       ${TINY_LLM_OUT_DIR}/${name}_top.mlir)
  set(_assigned_mlir  ${TINY_LLM_OUT_DIR}/${name}_top_assigned.mlir)
  set(_tpu_mlir       ${TINY_LLM_OUT_DIR}/${name}_tpu.mlir)
  set(_addr_mlir      ${TINY_LLM_OUT_DIR}/${name}_tpu_addr.mlir)
  set(_rxops_mlir     ${TINY_LLM_OUT_DIR}/${name}_rxops.mlir)
  set(_buf_mlir       ${TINY_LLM_OUT_DIR}/${name}_rxops_bufferize.mlir)
  set(_finished_mlir  ${TINY_LLM_OUT_DIR}/${name}_rxops_finished.mlir)
  set(_ll             ${TINY_LLM_OUT_DIR}/${name}_rxops.ll)
  set(_obj            ${TINY_LLM_BUILD_DIR}/${name}_rxops_rv64.o)

  # Step 1 – copy source MLIR verbatim
  add_custom_command(
    OUTPUT ${_top_mlir}
    COMMAND ${CMAKE_COMMAND} -E copy ${source_mlir} ${_top_mlir}
    DEPENDS ${source_mlir}
    COMMENT "tiny_llm [${name}] step 1/8: copy source MLIR"
    VERBATIM)

  # Step 2 – assign chip target and run init preprocessing passes
  add_custom_command(
    OUTPUT ${_assigned_mlir}
    COMMAND ${TPUC_OPT}
              "--processor-assign=chip=ada300 mode=F16"
              "--init=freq=0 level=0"
              ${_top_mlir}
              -o ${_assigned_mlir}
    DEPENDS ${_top_mlir} ${_TINY_TPUC_OPT_DEP}
    COMMENT "tiny_llm [${name}] step 2/8: processor-assign + init"
    VERBATIM)

  # Step 3 – lower Top dialect → TPU dialect
  add_custom_command(
    OUTPUT ${_tpu_mlir}
    COMMAND ${TPUC_OPT}
              --convert-top-to-tpu
              ${_assigned_mlir}
              -o ${_tpu_mlir}
    DEPENDS ${_assigned_mlir} ${_TINY_TPUC_OPT_DEP}
    COMMENT "tiny_llm [${name}] step 3/8: convert-top-to-tpu"
    VERBATIM)

  # Step 4 – assign buffer addresses.
  # skip_weight_check=true bypasses the 256 MiB hardware weight capacity
  # check so large subgraphs (e.g. full 248320-entry lm_head) compile for QEMU.
  add_custom_command(
    OUTPUT ${_addr_mlir}
    COMMAND ${TPUC_OPT}
              "--address-assign=reuse_addr=false skip_weight_check=true"
              ${_tpu_mlir}
              -o ${_addr_mlir}
    DEPENDS ${_tpu_mlir} ${_TINY_TPUC_OPT_DEP}
    COMMENT "tiny_llm [${name}] step 4/8: address-assign"
    VERBATIM)

  # Step 5 – lower TPU dialect → rx-ops calls
  add_custom_command(
    OUTPUT ${_rxops_mlir}
    COMMAND ${TPUC_OPT}
              "--convert-tpu-to-rxops=allow-stubs=true"
              ${_addr_mlir}
              -o ${_rxops_mlir}
    DEPENDS ${_addr_mlir} ${_TINY_TPUC_OPT_DEP}
    COMMENT "tiny_llm [${name}] step 5/8: convert-tpu-to-rxops"
    VERBATIM)

  # Step 6 – bufferize: tensor values → memref allocations
  add_custom_command(
    OUTPUT ${_buf_mlir}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              ${_rxops_mlir}
              -empty-tensor-to-alloc-tensor
              "${_TINY_BUFFERIZE_OPT}"
              -o ${_buf_mlir}
    DEPENDS ${_rxops_mlir}
    COMMENT "tiny_llm [${name}] step 6/8: bufferize (tensor -> memref)"
    VERBATIM)

  # Step 7 – lower memref + func to LLVM dialect
  add_custom_command(
    OUTPUT ${_finished_mlir}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt
              ${_buf_mlir}
              -finalize-memref-to-llvm
              -convert-arith-to-llvm
              -llvm-request-c-wrappers
              -convert-func-to-llvm
              -reconcile-unrealized-casts
              -o ${_finished_mlir}
    DEPENDS ${_buf_mlir}
    COMMENT "tiny_llm [${name}] step 7/8: lower to LLVM dialect"
    VERBATIM)

  # Step 8 – translate LLVM dialect → LLVM IR text; rename @main symbols
  add_custom_command(
    OUTPUT ${_ll}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate
              --mlir-to-llvmir
              ${_finished_mlir}
              -o ${_ll}
    COMMAND perl -0pi -e "s/([@])main[(]/$1${entry_name}(/g; s/([@])_mlir_ciface_main[(]/$1_mlir_ciface_${entry_name}(/g"
            ${_ll}
    DEPENDS ${_finished_mlir}
    COMMENT "tiny_llm [${name}] step 8/8: translate to LLVM IR"
    VERBATIM)

  set(TINY_LLM_${name}_TOP          ${_top_mlir}      PARENT_SCOPE)
  set(TINY_LLM_${name}_TOP_ASSIGNED ${_assigned_mlir} PARENT_SCOPE)
  set(TINY_LLM_${name}_TPU          ${_tpu_mlir}      PARENT_SCOPE)
  set(TINY_LLM_${name}_TPU_ADDR     ${_addr_mlir}     PARENT_SCOPE)
  set(TINY_LLM_${name}_RXOPS        ${_rxops_mlir}    PARENT_SCOPE)
  set(TINY_LLM_${name}_RXOPS_BUF    ${_buf_mlir}      PARENT_SCOPE)
  set(TINY_LLM_${name}_RXOPS_FIN    ${_finished_mlir} PARENT_SCOPE)
  set(TINY_LLM_${name}_LL           ${_ll}            PARENT_SCOPE)
  set(TINY_LLM_${name}_OBJ          ${_obj}           PARENT_SCOPE)
endfunction()


add_tiny_llm_subgraph(
  embedding
  ${TINY_LLM_MLIR_DIR}/embedding.mlir
  embedding_main)

add_tiny_llm_subgraph(
  embedding_cache
  ${TINY_LLM_MLIR_DIR}/embedding_cache.mlir
  embedding_cache_main)

add_tiny_llm_subgraph(
  block_7
  ${TINY_LLM_MLIR_DIR}/block_7.mlir
  block7_main)

add_tiny_llm_subgraph(
  block_cache_7
  ${TINY_LLM_MLIR_DIR}/block_cache_7.mlir
  block_cache7_main)

add_tiny_llm_subgraph(
  lm_head
  ${TINY_LLM_MLIR_DIR}/lm_head.mlir
  lm_head_main)

add_custom_target(tiny-llm-rxops-ll
  DEPENDS
    ${TINY_LLM_embedding_LL}
    ${TINY_LLM_embedding_cache_LL}
    ${TINY_LLM_block_7_LL}
    ${TINY_LLM_block_cache_7_LL}
    ${TINY_LLM_lm_head_LL}
  COMMENT "tiny_llm: all rx-ops LLVM IR files are ready")

if(TINY_LLM_BAREMETAL)
  set(ADA300_LLVM_ROOT "/workspace/rx-llvm/build"
      CACHE PATH "Ada300 LLVM toolchain root")

  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_filename_component(_sdk_default
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../embed-system/ada300_snpu_sdk"
      ABSOLUTE)
  else()
    get_filename_component(_sdk_default
      "${CMAKE_SOURCE_DIR}/../embed-system/ada300_snpu_sdk"
      ABSOLUTE)
  endif()

  set(ADA300_SDK_ROOT "${_sdk_default}"
      CACHE PATH "Path to ada300_snpu_sdk")
  set(ADA300_QEMU "/workspace/rx-qemu/build/qemu-system-riscv64"
      CACHE FILEPATH "QEMU binary for ada300s_evk")
  set(ADA300_NEWLIB_SYSROOT "/usr/lib/picolibc/riscv64-unknown-elf"
      CACHE PATH "Picolibc/newlib sysroot for riscv64-unknown-elf")

  set(_ADA300_CC "${ADA300_LLVM_ROOT}/bin/clang")
  set(_ADA300_OBJDUMP "${ADA300_LLVM_ROOT}/bin/llvm-objdump")
  set(_MARCH "rv64gcv_zfh_zvfh_zfbfmin0p8_zvfbfmin0p8_xadatmm_xadacv")
  set(_ADA300_ARCH_FLAGS
      --target=riscv64-unknown-elf
      -march=${_MARCH}
      -mabi=lp64d
      -mcmodel=medany
      -menable-experimental-extensions
      -ffreestanding
      -fno-builtin
      -fno-strict-aliasing)
  set(_LD_SCRIPT "${ADA300_SDK_ROOT}/bsp/ld/ada300s_evk.ld")

  macro(tiny_ada300_compile _out _src)
    set(_extra_incs ${ARGN})
    add_custom_command(
      OUTPUT "${_out}"
      COMMAND "${_ADA300_CC}"
                ${_ADA300_ARCH_FLAGS}
                -O2 -g
                -ffunction-sections
                -fdata-sections
                -DRXNN_BUILD_RTOS=1
                -I${ADA300_SDK_ROOT}/bsp/include
                $<$<NOT:$<BOOL:${ADA300_NEWLIB_SYSROOT}>>:-I${CMAKE_CURRENT_SOURCE_DIR}/freestanding_include>
                -I${ADA300_EXAMPLE_DIR}
                -I${RXOPS_SRC_DIR}/include
                -I${RXOPS_SRC_DIR}/include/interface
                -I${RXOPS_SRC_DIR}/include/backend
                -I${RXOPS_SRC_DIR}/include/internal
                -I${RXOPS_SRC_DIR}/third_party/rx-memory/include
                $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-isystem${ADA300_NEWLIB_SYSROOT}/include>
                ${_extra_incs}
                -c "${_src}"
                -o "${_out}"
      DEPENDS "${_src}"
      COMMENT "tiny_llm: Ada300 compile ${_src}"
      VERBATIM)
  endmacro()

  foreach(_sg embedding embedding_cache block_7 block_cache_7 lm_head)
    add_custom_command(
      OUTPUT "${TINY_LLM_BUILD_DIR}/${_sg}_rxops_rv64.o"
      COMMAND "${_ADA300_CC}"
                ${_ADA300_ARCH_FLAGS}
                -O3
                -c "${TINY_LLM_OUT_DIR}/${_sg}_rxops.ll"
                -o "${TINY_LLM_BUILD_DIR}/${_sg}_rxops_rv64.o"
      DEPENDS "${TINY_LLM_OUT_DIR}/${_sg}_rxops.ll"
      COMMENT "tiny_llm: clang ${_sg}_rxops.ll -> RISC-V object"
      VERBATIM)
  endforeach()

  tiny_ada300_compile(
    "${TINY_LLM_BUILD_DIR}/tiny_llm_baremetal_main.o"
    "${CMAKE_CURRENT_SOURCE_DIR}/tiny_llm_baremetal_main.c")

  tiny_ada300_compile(
    "${TINY_LLM_BUILD_DIR}/rx_ops_bridge_ada300.o"
    "${CMAKE_CURRENT_SOURCE_DIR}/rx_ops_bridge_ada300_compat.c")

  tiny_ada300_compile(
    "${TINY_LLM_BUILD_DIR}/rxops_c_init_stub.o"
    "${CMAKE_CURRENT_SOURCE_DIR}/rxops_c_init_stub.c")


  tiny_ada300_compile(
    "${TINY_LLM_BUILD_DIR}/rxops_tiny_ada300_setup.o"
    "${CMAKE_CURRENT_SOURCE_DIR}/rxops_tiny_ada300_setup.c"
    -I${RXOPS_SRC_DIR}/include/ada300
    -I${RXOPS_SRC_DIR}/src/backend/ada300)

  tiny_ada300_compile(
    "${TINY_LLM_BUILD_DIR}/bm_startup.o"
    "${ADA300_SDK_ROOT}/bsp/src/startup.S")

  set(_BSP_OBJS "${TINY_LLM_BUILD_DIR}/bm_startup.o")
  foreach(_bsp_src bsp_main.c uart.c mini_printf.c mr_heap.c)
    get_filename_component(_bsp_base "${_bsp_src}" NAME_WE)
    tiny_ada300_compile(
      "${TINY_LLM_BUILD_DIR}/bm_${_bsp_base}.o"
      "${ADA300_SDK_ROOT}/bsp/src/${_bsp_src}")
    list(APPEND _BSP_OBJS "${TINY_LLM_BUILD_DIR}/bm_${_bsp_base}.o")
  endforeach()

  # rx-ops Ada300 backend sources for all ops used by the Qwen LLM subgraphs.
  # Avoids upstream setup.c and full backend globs; the local setup file
  # registers only the ops that convert-tpu-to-rxops actually emits.
  set(_RXOPS_MINIMAL_SRCS
      # infrastructure
      "${RXOPS_SRC_DIR}/src/rx_ops_register.c"
      "${RXOPS_SRC_DIR}/src/rx_ops_utils.c"
      "${RXOPS_SRC_DIR}/src/log.c"
      # ops dispatch layer
      "${RXOPS_SRC_DIR}/src/ops/add.c"
      "${RXOPS_SRC_DIR}/src/ops/mul.c"
      "${RXOPS_SRC_DIR}/src/ops/matmul.c"
      "${RXOPS_SRC_DIR}/src/ops/fullyconnected.c"
      "${RXOPS_SRC_DIR}/src/ops/rms_norm.c"
      "${RXOPS_SRC_DIR}/src/ops/rope.c"
      "${RXOPS_SRC_DIR}/src/ops/full_attention.c"
      "${RXOPS_SRC_DIR}/src/ops/gather.c"
      "${RXOPS_SRC_DIR}/src/ops/transpose.c"
      "${RXOPS_SRC_DIR}/src/ops/concat.c"
      "${RXOPS_SRC_DIR}/src/ops/reshape.c"
      "${RXOPS_SRC_DIR}/src/ops/silu.c"
      "${RXOPS_SRC_DIR}/src/ops/sigmoid.c"
      "${RXOPS_SRC_DIR}/src/ops/topk.c"
      "${RXOPS_SRC_DIR}/src/ops/argmax.c"
      # Ada300 fp16/ and kernel C wrappers
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp16/fullyconnected.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp16/transpose.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp16/gather.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp16/topk.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp16/argmax.c"
      # rope and full_attention fp16 live at the ada300/ top level
      "${RXOPS_SRC_DIR}/src/backend/ada300/rope.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/full_attention.c"
      # prepare_tensor_as / release_tensor_view stub (avoids quant.c's fp32 asm deps)
      "${CMAKE_CURRENT_SOURCE_DIR}/rxops_ada300_tensor_utils_stub.c"
      # fp32 shape/copy primitives (reshape_copy, concat_copy used by fp16 reshape/concat)
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp32/copy_reorder_ops.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/fp32/sigmoid.c"
      # Ada300 kernel C wrappers
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/add.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/mul.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/matmul.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/rms_norm.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/gather.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/transpose.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/concat.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/silu.c"
      "${RXOPS_SRC_DIR}/src/backend/ada300/kernels/sigmoid.c")
  set(_RXOPS_MINIMAL_ASM_SRCS
      # standard scalar asm (define *_scalar_* symbols)
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_add_f32.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_add_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_mul_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_matmul_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_rms_norm_bf16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_rope_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_full_attention.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_full_attention_bf16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_fullyconnected_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_gather.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_transpose.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_concat_copy.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_copy_bytes.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_split_copy.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_broadcast_to_copy.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_sigmoid_f32.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_silu_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_sigmoid_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_topk_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_argmax_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_mul_f32.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_compound_f32.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_compound_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm/ada300_rope_f32.S"
      # vector-length (vlen) asm – defines *_vlen_* symbols, no conflict with scalar
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_add_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_mul_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_rms_norm_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_gather_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_concat_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_transpose_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_silu_f16.S"
      "${RXOPS_SRC_DIR}/src/backend/ada300/asm_vlen/ada300_sigmoid_f16.S")

  set(_RXOPS_OBJS)
  foreach(_rsrc ${_RXOPS_MINIMAL_SRCS})
    get_filename_component(_rbase "${_rsrc}" NAME_WE)
    string(MD5 _rhash "${_rsrc}")
    string(SUBSTRING "${_rhash}" 0 6 _rhash)
    set(_robj "${TINY_LLM_BUILD_DIR}/rxops_${_rbase}_${_rhash}.o")
    tiny_ada300_compile("${_robj}" "${_rsrc}"
      -I${RXOPS_SRC_DIR}/include/ada300
      -I${RXOPS_SRC_DIR}/src/backend/ada300)
    list(APPEND _RXOPS_OBJS "${_robj}")
  endforeach()

  foreach(_rsrc ${_RXOPS_MINIMAL_ASM_SRCS})
    get_filename_component(_rbase "${_rsrc}" NAME_WE)
    string(MD5 _rhash "${_rsrc}")
    string(SUBSTRING "${_rhash}" 0 6 _rhash)
    set(_robj "${TINY_LLM_BUILD_DIR}/rxops_asm_${_rbase}_${_rhash}.o")
    tiny_ada300_compile("${_robj}" "${_rsrc}"
      -I${RXOPS_SRC_DIR}/include/ada300)
    list(APPEND _RXOPS_OBJS "${_robj}")
  endforeach()

  set(_TINY_ELF "${TINY_LLM_BUILD_DIR}/tiny_llm.elf")

  add_custom_command(
    OUTPUT "${_TINY_ELF}"
    COMMAND "${_ADA300_CC}"
              ${_ADA300_ARCH_FLAGS}
              -nostdlib
              -fuse-ld=lld
              -Wl,--gc-sections
              -Wl,-Map=${TINY_LLM_BUILD_DIR}/tiny_llm.map
              "-T${_LD_SCRIPT}"
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-L${ADA300_NEWLIB_SYSROOT}/lib>
              ${_BSP_OBJS}
              "${TINY_LLM_BUILD_DIR}/tiny_llm_baremetal_main.o"
              "${TINY_LLM_BUILD_DIR}/rx_ops_bridge_ada300.o"
              "${TINY_LLM_BUILD_DIR}/rxops_c_init_stub.o"
              "${TINY_LLM_BUILD_DIR}/rxops_tiny_ada300_setup.o"
              "${TINY_LLM_BUILD_DIR}/embedding_rxops_rv64.o"
              "${TINY_LLM_BUILD_DIR}/embedding_cache_rxops_rv64.o"
              "${TINY_LLM_BUILD_DIR}/block_7_rxops_rv64.o"
              "${TINY_LLM_BUILD_DIR}/block_cache_7_rxops_rv64.o"
              "${TINY_LLM_BUILD_DIR}/lm_head_rxops_rv64.o"
              ${_RXOPS_OBJS}
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-Wl,--start-group>
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-lm>
              $<$<BOOL:${ADA300_NEWLIB_SYSROOT}>:-Wl,--end-group>
              -o "${_TINY_ELF}"
    DEPENDS
      ${_BSP_OBJS}
      "${TINY_LLM_BUILD_DIR}/tiny_llm_baremetal_main.o"
      "${TINY_LLM_BUILD_DIR}/rx_ops_bridge_ada300.o"
      "${TINY_LLM_BUILD_DIR}/rxops_c_init_stub.o"
      "${TINY_LLM_BUILD_DIR}/rxops_tiny_ada300_setup.o"
      "${TINY_LLM_BUILD_DIR}/embedding_rxops_rv64.o"
      "${TINY_LLM_BUILD_DIR}/embedding_cache_rxops_rv64.o"
      "${TINY_LLM_BUILD_DIR}/block_7_rxops_rv64.o"
      "${TINY_LLM_BUILD_DIR}/block_cache_7_rxops_rv64.o"
      "${TINY_LLM_BUILD_DIR}/lm_head_rxops_rv64.o"
      ${_RXOPS_OBJS}
    COMMENT "tiny_llm: link tiny_llm.elf"
    VERBATIM)

  add_custom_command(
    OUTPUT "${TINY_LLM_BUILD_DIR}/tiny_llm.asm"
    COMMAND "${_ADA300_OBJDUMP}" -d "${_TINY_ELF}"
            > "${TINY_LLM_BUILD_DIR}/tiny_llm.asm"
    DEPENDS "${_TINY_ELF}"
    COMMENT "tiny_llm: generate disassembly"
    VERBATIM)

  add_custom_target(tiny-llm-baremetal
    DEPENDS "${_TINY_ELF}" "${TINY_LLM_BUILD_DIR}/tiny_llm.asm"
    COMMENT "tiny_llm: Ada300 baremetal ELF ready")

  # Generate a wrapper script so make never sees || or other shell operators.
  file(GENERATE
    OUTPUT "${TINY_LLM_BUILD_DIR}/run_qemu.sh"
    CONTENT "#!/bin/sh
${ADA300_QEMU} -machine ada300s_evk -cpu ada300s -nographic -semihosting -serial mon:stdio -kernel ${_TINY_ELF}
exit 0
")

  add_custom_target(run-tiny-llm-baremetal
    COMMAND sh "${TINY_LLM_BUILD_DIR}/run_qemu.sh"
    DEPENDS "${_TINY_ELF}"
    USES_TERMINAL
    COMMENT "tiny_llm: boot tiny_llm.elf on QEMU")
endif()

