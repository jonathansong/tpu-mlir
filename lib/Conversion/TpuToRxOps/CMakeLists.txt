add_mlir_conversion_library(TPUMLIRTpuToRxOps
  TpuToRxOpsPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/tpu_mlir/Dialect/Tpu

  DEPENDS
  TpuMlirConversionPassIncGen

  LINK_LIBS PUBLIC
  TPUMLIRTop
  TPUMLIRTpu
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  )
