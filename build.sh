#!/bin/bash
# ./build.sh [DEBUG|RELEASE] [CUDA|CPU]
set -e
RED='\033[0;31m'
NC='\033[0m'

if [[ -z "$INSTALL_PATH" ]]; then
  echo "${RED}ERROR${NC}: Please source envsetup.sh firstly."
  exit 1
fi

if [[ -z "$ENVSETUP_LAST_UPDATED" || "$ENVSETUP_LAST_UPDATED" != "2025-05-22" ]];
then
  echo -e "${RED}ERROR${NC}: envsetup.sh has updated. Please source envsetup.sh again."
  exit 1
fi

# Function to show usage information
usage() {
  echo "Usage: $0 [RELEASE|DEBUG] [CPU|CUDA]"
}

echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"
#echo "BUILD_FLAG: $BUILD_FLAG"

BUILD_TYPE=""
CXX_FLAGS="-O2"
USE_CUDA="OFF"
ENABLE_COVERAGE_FLAG="OFF"

if [ -n "$1" ]; then
    if [ "$1" = "DEBUG" ]; then
        BUILD_TYPE="Debug"
        CXX_FLAGS="-ggdb"
    elif [ "$1" != "RELEASE" ]; then
        echo "Invalid build mode: $1"
        usage
        exit 1
    fi
fi

# Check for CUDA support
if [ -n "$2" ]; then
    if [ "$2" = "CUDA" ]; then
        USE_CUDA="ON"
    elif [ "$2" != "CPU" ]; then
        echo "Invalid CUDA option: $2"
        usage
        exit 1
    fi
fi

# Check coverage environment variables
if [ "${ENABLE_COVERAGE}" = "True" ]; then
    ENABLE_COVERAGE_FLAG="ON"
    echo "Building with code coverage enabled"
fi

# prepare install/build dir
# Only re-run cmake configure when the build directory does not exist or when
# a relevant cmake option has changed. This enables incremental builds when
# the build directory is persisted (e.g. via a Docker volume mount).
CMAKE_CACHE="${BUILD_PATH}/CMakeCache.txt"
NEEDS_CONFIGURE=0
if [[ ! -f "${CMAKE_CACHE}" ]]; then
  NEEDS_CONFIGURE=1
else
  # Re-configure if key options differ from the cached values.
  cached_type=$(grep -m1 "^CMAKE_BUILD_TYPE:" "${CMAKE_CACHE}" | cut -d= -f2)
  cached_cuda=$(grep -m1 "^TPUMLIR_USE_CUDA:" "${CMAKE_CACHE}" | cut -d= -f2)
  if [[ "${cached_type}" != "${BUILD_TYPE}" || "${cached_cuda}" != "${USE_CUDA}" ]]; then
    NEEDS_CONFIGURE=1
  fi
fi

if [[ "${NEEDS_CONFIGURE}" == "1" ]]; then
  echo "Running CMake configure..."
  cmake -G Ninja \
    -B "${BUILD_PATH}" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
    -DTPUMLIR_USE_LLD=ON \
    -DTPUMLIR_INCLUDE_TESTS=ON \
    -DTPUMLIR_USE_CUDA="${USE_CUDA}" \
    -DTPUMLIR_ENABLE_COVERAGE="${ENABLE_COVERAGE_FLAG}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}" \
    "${PROJECT_ROOT}"
else
  echo "CMake cache up-to-date, skipping configure."
fi

cpu_num=$(nproc)
cmake --build "$BUILD_PATH" --target install -j${cpu_num}

cmake --build "$BUILD_PATH" --target passes_json_files builder_python install_passes_files

# build ppl code
bash lib/PplBackend/build.sh "$1"

# Clean up some files for release build
if [ "$1" != "DEBUG" ]; then
  # build doc
  echo "check document ..."
  ./release_doc.sh >doc.log 2>&1
  if grep -i 'error' doc.log; then
    exit 1
  fi
  rm doc.log
  echo "document check passed."

  # strip mlir tools
  pushd $INSTALL_PATH
  find ./ -name "*.so" ! -name "*_kernel_module.so" ! -name "*_atomic_kernel.so" | xargs strip
  ls bin/* | xargs strip
  find ./ -name "*.a" ! -name "*_kernel_module.a" | xargs rm
  popd
fi
