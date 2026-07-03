#!/usr/bin/env bash
# Configure and build cpp/ (requires ONNXRUNTIME_ROOT).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${ONNXRUNTIME_ROOT:-}" ]] && [[ -d /opt/homebrew/opt/onnxruntime ]]; then
  export ONNXRUNTIME_ROOT="/opt/homebrew/opt/onnxruntime"
fi
if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
  echo "error: set ONNXRUNTIME_ROOT to your onnxruntime prebuilt tree (include/ + lib/)." >&2
  echo "  See cpp/README.md" >&2
  exit 1
fi

cmake -S . -B build "-DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
cmake --build build --parallel
