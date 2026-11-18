#!/usr/bin/env bash
# Скачивает веса ZONOS3-Mini в папку моделей Jarvis.
#
# Обычный путь — приложение качает само при первом запуске (hf-hub). Скрипт
# нужен для ручной загрузки/докачки, в том числе с зеркалом, когда
# huggingface.co недоступен напрямую (DPI/прокси): HF_ENDPOINT=
# https://hf-mirror.com ./scripts/fetch_zonos3.sh
#
# Файлы (~0.9 ГБ суммарно): бэкбон 0.66 ГБ + компактный ZAC-декодер ~0.25 ГБ
# + спикер-энкодер ~45 МБ. Полный zac.gguf не нужен — звук тот же.
set -euo pipefail

REPO="Zyphra/ZONOS3-MINI"
# Имя репозитория у Zyphra — ZONOS3-Mini (реестр регистронезависим, но
# URL принято писать как в HF).
REPO="Zyphra/ZONOS3-Mini"
ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODELS_DIR="${JARVIS_MODELS_DIR:-$HOME/Library/Application Support/Jarvis/models}"
DEST="$MODELS_DIR/zonos3/gguf"

mkdir -p "$DEST"

FILES=(
  "backbone-ptq.gguf"
  "zac-keep2.gguf"
  "spk-encoder.gguf"
)

for f in "${FILES[@]}"; do
  out="$DEST/$f"
  if [[ -s "$out" ]]; then
    echo "✅ $f уже на месте ($(du -h "$out" | cut -f1))"
    continue
  fi
  url="$ENDPOINT/$REPO/resolve/main/gguf/$f"
  echo "📥 $f ← $url"
  curl -L --fail --retry 3 --retry-delay 2 -C - -o "$out" "$url"
done

echo
echo "Готово. Эмбеддинг голоса «jarvis» (если ещё нет):"
echo "  cargo run -p jarvis-core --example voice_clone -- \\"
echo "    \"$HOME/Library/Application Support/Jarvis/voice_refs/jarvis_ref.wav\" jarvis"
