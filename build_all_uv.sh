#!/usr/bin/env bash
# Build native extensions for a uv (or plain venv) environment.
# Native C++ deps (Eigen3, Boost, compiler) must come from the system (apt).
# BundleSDF CUDA ops (bundlesdf/mycuda) are omitted; model-free NeRF path needs them separately.
set -euo pipefail

PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

: "${VIRTUAL_ENV:?Activate your uv/venv environment first (source .venv/bin/activate).}"

PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
PYBIND11_CMAKE_DIR="$("${PYTHON_EXECUTABLE}" -m pybind11 --cmakedir)"

# Let CMake find pybind11 (from venv) plus system Eigen3/Boost.
export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}:${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH:-}"

cd "${PROJ_ROOT}/mycpp"
rm -rf build
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DPython3_ROOT_DIR="${VIRTUAL_ENV}" \
  -DPython3_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -DPYBIND11_PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
cmake --build . -j"$(nproc)"

cd "${PROJ_ROOT}"
