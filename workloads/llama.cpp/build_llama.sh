#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
LLAMA_DIR="${SCRIPT_DIR}/llama.cpp"

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --cuda      Build with CUDA support"
    echo "  -m, --metal     Build with Metal support (macOS)"
    echo "  -v, --vulkan    Build with Vulkan support"
    echo "  -j, --jobs N    Number of parallel jobs (default: nproc)"
    echo "  -d, --debug     Build in debug mode"
    echo "  -u, --update    Update submodule before building"
    echo "  -h, --help      Show this help message"
}

BUILD_TYPE="Release"
CUDA_SUPPORT="OFF"
METAL_SUPPORT="OFF"
VULKAN_SUPPORT="OFF"
NUM_JOBS=$(nproc 2>/dev/null || echo 4)
UPDATE_SUBMODULE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cuda)
            CUDA_SUPPORT="ON"
            shift
            ;;
        -m|--metal)
            METAL_SUPPORT="ON"
            shift
            ;;
        -v|--vulkan)
            VULKAN_SUPPORT="ON"
            shift
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -u|--update)
            UPDATE_SUBMODULE=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [ ! -d "${LLAMA_DIR}" ]; then
    echo "Error: llama.cpp directory not found. Initializing submodule..."
    git submodule update --init --recursive
fi

if [ ${UPDATE_SUBMODULE} -eq 1 ]; then
    echo "Updating llama.cpp submodule..."
    cd "${LLAMA_DIR}"
    git pull origin master
    cd "${SCRIPT_DIR}"
fi

echo "Building llama.cpp..."
echo "  Build type: ${BUILD_TYPE}"
echo "  CUDA support: ${CUDA_SUPPORT}"
echo "  Metal support: ${METAL_SUPPORT}"
echo "  Vulkan support: ${VULKAN_SUPPORT}"
echo "  Parallel jobs: ${NUM_JOBS}"

mkdir -p "${BUILD_DIR}"

cd "${LLAMA_DIR}"
cmake -B "../${BUILD_DIR}" . \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_CUDA="${CUDA_SUPPORT}" \
    -DGGML_METAL="${METAL_SUPPORT}" \
    -DGGML_VULKAN="${VULKAN_SUPPORT}" \
    -DGGML_SYCL=OFF

cd "${BUILD_DIR}"
make -j${NUM_JOBS}

echo "Build complete!"
echo "Binaries available in: ${BUILD_DIR}"
echo "Main binary: ${BUILD_DIR}/llama-cli"

if [ -f "${BUILD_DIR}/llama-cli" ]; then
    echo "Testing binary..."
    "${BUILD_DIR}/llama-cli" --version
fi