#!/usr/bin/env bash
# Готовит tar.gz для релиза jarvis-models (GitHub Releases, тег models-v1).
#
#   build/models/silero.tar.gz        -> silero/v5_cis_base.safetensors, silero/alphabet.json
#   build/models/qwen3.5-0.8b.tar.gz  -> qwen3.5-0.8b/{config.json,tokenizer.json,model.safetensors}
#
# Загрузите их в релиз:
#   gh release create models-v1 build/models/*.tar.gz \
#     --repo pavel807/jarvis-models --title "Models v1"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/build/models"
mkdir -p "$OUT"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# --- silero (TTS) ---
mkdir -p "$STAGE/silero"
cp "$ROOT/models/silero/v5_cis_base.safetensors" "$ROOT/models/silero/alphabet.json" "$STAGE/silero/"
tar -czf "$OUT/silero.tar.gz" -C "$STAGE" silero
echo "==> $OUT/silero.tar.gz ($(du -h "$OUT/silero.tar.gz" | cut -f1))"

# --- Qwen3.5-0.8B (для бэкенда candle) ---
mkdir -p "$STAGE/qwen3.5-0.8b"
cp "$ROOT/models/qwen3.5-0.8b/config.json" "$ROOT/models/qwen3.5-0.8b/tokenizer.json" "$STAGE/qwen3.5-0.8b/"
# Веса у HF называются model.safetensors-00001-of-00001.safetensors — кладём
# как model.safetensors (загрузчик принимает любое имя model.safetensors*).
W="$(ls "$ROOT"/models/qwen3.5-0.8b/model.safetensors*.safetensors | head -1)"
cp "$W" "$STAGE/qwen3.5-0.8b/model.safetensors"
tar -czf "$OUT/qwen3.5-0.8b.tar.gz" -C "$STAGE" qwen3.5-0.8b
echo "==> $OUT/qwen3.5-0.8b.tar.gz ($(du -h "$OUT/qwen3.5-0.8b.tar.gz" | cut -f1))"

echo "==> Готово. Загрузите в релиз models-v1 репозитория pavel807/jarvis-models."
