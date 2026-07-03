#!/usr/bin/env bash
# Export community-1 segmentation and embedding to ONNX + JSON metadata.
# Requires HF access: export HF_TOKEN=hf_... (or HUGGING_FACE_HUB_TOKEN)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv pip install -e ".[export]"

mkdir -p cpp/artifacts

python cpp/scripts/export_segmentation_onnx.py -o cpp/artifacts/community1-segmentation.onnx
python cpp/scripts/export_embedding_onnx.py -o cpp/artifacts/community1-embedding.onnx

python cpp/scripts/verify_onnx_parity.py

echo "Done. Outputs under cpp/artifacts/"
