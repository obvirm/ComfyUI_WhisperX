#!/usr/bin/env bash
# Run SpeakerDiarization and write NPZ/JSON golden artifacts under OUTPUT_DIR.
# Usage: ./scripts/dump_golden.sh OUTPUT_DIR FILE.wav [FILE2.wav ...]
# Requires HF_TOKEN for gated checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR AUDIO.wav [AUDIO2.wav ...]" >&2
  exit 1
fi

OUT="$1"
shift

uv pip install -e ".[export]" >/dev/null

python cpp/scripts/dump_diarization_golden.py -o "$OUT" "$@"

echo "Golden bundle: $OUT"
