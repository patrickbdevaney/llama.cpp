#!/bin/bash
# Build this branch for Jetson AGX Thor (sm_110). Compiles only; runs nothing on the GPU.
# Other NVIDIA GPUs: CUDA_ARCH=<compute capability without the dot, e.g. 89 or "86;89"> thor/build.sh
# (the build dir keeps the name build-thor either way, so thor/start.sh finds it).
set -euo pipefail
cd "$(dirname "$0")/.."
J=${J:-4}
echo "build.sh: CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-110}"
cmake -S . -B build-thor -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-110}" \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=ON
nice -n 19 cmake --build build-thor -j "$J" --target llama-server llama-cli llama-mtmd-cli llama-bench llama-gguf-split
ls -la build-thor/bin/llama-server build-thor/bin/llama-mtmd-cli
