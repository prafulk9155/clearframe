#!/usr/bin/env bash
# Downloads the LaMa ONNX model (~208 MB, Apache-2.0) used by the "AI" engine.
set -euo pipefail
DEST="${1:-models/lama_fp32.onnx}"
URL="${LAMA_MODEL_URL:-https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx}"
mkdir -p "$(dirname "$DEST")"
if [ -s "$DEST" ]; then echo "Model already at $DEST"; exit 0; fi
echo "Downloading LaMa model -> $DEST"
curl -fL --retry 3 -o "$DEST.part" "$URL"
mv "$DEST.part" "$DEST"
ls -lh "$DEST"
