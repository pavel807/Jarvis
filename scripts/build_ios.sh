#!/usr/bin/env bash
# Собирает iOS/iPadOS-приложение Jarvis (Tauri 2 mobile).
#
# Бандл НЕ несёт моделей: приложение маленькое, голосовые модели при первом
# запуске качаются в отдельную папку (Documents/Jarvis/models в сэндбоксе,
# видна в приложении «Файлы»). На iOS работает настоящий голосовой движок
# (jarvis_core::mobile): tap-to-talk — микрофон → FastConformer STT → LLM
# (облако или локальный Qwen) → озвучка Silero; гейты фактов/шуток/«почитай»
# те же, что на десктопе.
#
# Режимы:
#   ./scripts/build_ios.sh                 — сборка для симулятора (отладка UI)
#   ./scripts/build_ios.sh device          — сборка на устройство (нужна
#                                            подпись: бесплатная учётка Apple
#                                            в Xcode → Signing & Capabilities)
#
# Требования: Xcode с iOS SDK, Rust-таргеты:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/app"
MODE="${1:-simulator}"

CONF="$APP_DIR/src-tauri/tauri.conf.json"
VERSION="$(python3 -c "import json;print(json.load(open('$CONF'))['version'])")"
echo "==> Jarvis (iOS) $VERSION — режим: $MODE"

# 0. Rust-таргеты iOS.
for t in aarch64-apple-ios aarch64-apple-ios-sim; do
  rustup target list --installed | grep -q "$t" || rustup target add "$t"
done

# 1. Проект Xcode (gen/apple). Создаётся один раз, повторные запуски его не трогают.
if [ ! -d "$APP_DIR/src-tauri/gen/apple" ]; then
  echo "==> Генерирую Xcode-проект (tauri ios init)…"
  (cd "$APP_DIR" && bun tauri ios init)
fi

# 1б. Разрешение микрофона: ключ NSMicrophoneUsageDescription приедает в
#     сгенерированный plist из app/src-tauri/Info.plist (см. infoPlist в
#     tauri.conf.json). Без него iOS молча запрещает запись — проверяем
#     честно и падаем с понятной причиной, а не ловим тишину в записи.
#     Формулировку синхронизируем с исходником: gen/ перегенерируется
#     только при tauri ios init, а текст видит пользователь в диалоге.
PLIST="$(find "$APP_DIR/src-tauri/gen/apple" -name Info.plist -not -path "*ExportOptions*" | head -1)"
if [ -n "$PLIST" ]; then
  if ! /usr/libexec/PlistBuddy -c "Print :NSMicrophoneUsageDescription" "$PLIST" >/dev/null 2>&1; then
    echo "❌ В $PLIST нет NSMicrophoneUsageDescription."
    echo "   Добавь ключ в app/src-tauri/Info.plist и перегенерируй проект (tauri ios init)."
    exit 1
  fi
  TEXT="$(plutil -extract NSMicrophoneUsageDescription raw -o - "$APP_DIR/src-tauri/Info.plist" 2>/dev/null || true)"
  if [ -n "$TEXT" ]; then
    /usr/libexec/PlistBuddy -c "Set :NSMicrophoneUsageDescription $TEXT" "$PLIST"
  fi
fi

# 2. Сборка. Таргеты tauri-cli: aarch64 | aarch64-sim | x86_64 (не Rust-триплеты).
case "$MODE" in
  device)
    echo "==> Собираю для устройства (device)…"
    (cd "$APP_DIR" && bun tauri ios build)
    echo "==> Готово: открой Xcode-проект app/src-tauri/gen/apple/jarvis-app.xcodeproj"
    echo "    и запусти на устройстве (подпись — в Signing & Capabilities)."
    ;;
  simulator|*)
    echo "==> Собираю для симулятора…"
    (cd "$APP_DIR" && bun tauri ios build --target aarch64-sim)
    echo "==> Готово: приложение для симулятора — установи через Xcode"
    echo "    (xcodebuild -project gen/apple/jarvis-app.xcodeproj -scheme jarvis-app_iOS -destination 'platform=iOS Simulator')."
    ;;
esac

echo
echo "Что работает на iOS:"
echo "  • tap-to-talk: кнопка микрофона — говоришь, Джарвис распознаёт"
echo "    (FastConformer RU) и отвечает голосом Silero,"
echo "  • текстовый чат с облачной LLM или локальным Qwen (модели — в папке моделей),"
echo "  • «почитай статью про …» и «читай дальше» — чтение вслух,"
echo "  • «расскажи интересный факт» — мгновенно, из локальной базы,"
echo "  • интерфейс под iPhone/iPad, модели качаются в Documents/Jarvis/models."
