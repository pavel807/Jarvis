#!/bin/zsh
# Упаковка FastConformer для релиза jarvis-models.
#
# models/fastconformer (веса из .nemo NVIDIA + config + токенизатор) ->
# fastconformer.tar.gz (~460 МБ) для загрузки на GitHub Releases
# (pavel807/jarvis-models, тег models-v1). Пути внутри архива — относительно
# models/, как ожидает model_fetch.rs.
#
# Запуск: zsh tools/pack_fastconformer.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="build/fastconformer.tar.gz"
mkdir -p build

echo "📦 Упаковка models/fastconformer -> $OUT"
tar -czf "$OUT" \
    -C models \
    fastconformer/model.safetensors \
    fastconformer/config.json \
    fastconformer/tokenizer.vocab

echo "✅ Готово: $(du -h "$OUT" | cut -f1)"
echo "Загрузка на релиз:"
echo "  gh release upload models-v1 $OUT --repo pavel807/jarvis-models --clobber"
