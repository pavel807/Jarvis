#!/usr/bin/env bash
# Собирает готовый macOS-дистрибутив Jarvis:
#   build/Jarvis-<версия>.dmg
#
# По умолчанию тяжёлые модели в .app не встраиваются — приложение скачивает их
# при первом запуске в ~/Library/Application Support/Jarvis/models (STT/stress/uzu
# из HF/Mirai, silero и Qwen3.5 — из релиза jarvis-models). Внутри архива —
# звуки, конфиги и крошечные wake word-модели. Офлайн-сборка с моделями:
# JARVIS_EMBED_MODELS=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/app"
TAURI_DIR="$APP_DIR/src-tauri"
BUILD_DIR="$ROOT/build"
CONF="$TAURI_DIR/tauri.conf.json"

VERSION="$(python3 -c "import json;print(json.load(open('$CONF'))['version'])")"
DMG_NAME="Jarvis-$VERSION"
echo "==> Jarvis $VERSION"

# 0. Иконки (icns нужен бандлеру macOS).
if [ ! -f "$TAURI_DIR/icons/icon.icns" ]; then
  echo "==> Генерирую иконки из icon.png"
  (cd "$APP_DIR" && bun tauri icon src-tauri/icons/icon.png)
fi

# 1. Сборка .app (release). Frontend собирается через beforeBuildCommand.
echo "==> Собираю .app (release)…"
(cd "$APP_DIR" && bun tauri build --bundles app)

APP="$(find "$ROOT/target" "$TAURI_DIR/target" -maxdepth 7 -type d -name 'Jarvis.app' 2>/dev/null | head -1 || true)"
if [ -z "$APP" ] || [ ! -d "$APP" ]; then
  echo "!! Не нашёл Jarvis.app после сборки" >&2
  exit 1
fi
echo "==> .app: $APP"

# 2. Архив данных внутрь приложения.
echo "==> Готовлю архив данных (звуки + конфиги)…"

# По умолчанию тяжёлые модели (LLM/STT/TTS, ~3 ГБ) НЕ встраиваются: приложение
# качает их при первом запуске в ~/Library/Application Support/Jarvis/models.
# Крошечные wake word-модели идут в архиве — своего источника у них нет.
# Офлайн-сборка со всем внутри: JARVIS_EMBED_MODELS=1.
EMBED_MODELS="${JARVIS_EMBED_MODELS:-0}"

mkdir -p "$APP/Contents/Resources/payload"
PAYLOAD="$APP/Contents/Resources/payload/jarvis-data.tar.gz"
rm -f "$PAYLOAD"

TAR_ITEMS=(resources/audio commands.toml settings.toml)
if [ "$EMBED_MODELS" = "1" ]; then
  echo "    JARVIS_EMBED_MODELS=1 — встраиваю models/ ($(du -sh "$ROOT/models" 2>/dev/null | cut -f1))"
  TAR_ITEMS=(models "${TAR_ITEMS[@]}")
else
  echo "    модели не встраиваются — скачаются при первом запуске"
  for f in models/jarvis_model.rpw models/jarvis_model_2.rpw; do
    [ -f "$ROOT/$f" ] && TAR_ITEMS+=("$f")
  done
fi

tar --exclude='.DS_Store' -czf "$PAYLOAD" -C "$ROOT" "${TAR_ITEMS[@]}"
echo "    архив: $(du -h "$PAYLOAD" | cut -f1)"

# Приблизительный объём распакованных данных — для Read Me.
DATA_GB="$(du -sm "$ROOT/models" "$ROOT/resources/audio" 2>/dev/null | awk '{s+=$1} END {printf "%d", s/1024 + 0.5}')"

# 3. Ad-hoc подпись (изменили содержимое бандла после сборки).
echo "==> Подписываю (ad-hoc)…"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || echo "   (codesign недоступен, пропускаю)"

# 4. Содержимое DMG.
echo "==> Собираю образ…"
rm -rf "$BUILD_DIR/dmg"
mkdir -p "$BUILD_DIR/dmg/.background"
cp -R "$APP" "$BUILD_DIR/dmg/"
ln -s /Applications "$BUILD_DIR/dmg/Applications"
python3 "$ROOT/scripts/dmg_background.py" "$BUILD_DIR/dmg/.background/bg.png"

if [ -f "$TAURI_DIR/icons/icon.icns" ]; then
  cp "$TAURI_DIR/icons/icon.icns" "$BUILD_DIR/dmg/.VolumeIcon.icns"
fi

if [ "$EMBED_MODELS" = "1" ]; then
  STEP2="2. Запустите Jarvis. При первом старте он распакует модели (~${DATA_GB:-3} ГБ) в ~/Library/Application Support/Jarvis."
  FIRST="Первый запуск занимает около минуты — идёт распаковка моделей."
else
  STEP2="2. Запустите Jarvis. При первом старте он скачает модели (~${DATA_GB:-3} ГБ) в ~/Library/Application Support/Jarvis — нужен интернет."
  FIRST="Первый запуск занимает несколько минут — идёт загрузка моделей."
fi

cat > "$BUILD_DIR/dmg/Read Me.txt" <<EOF
Jarvis — локальный голосовой ассистент
======================================

1. Перетащите Jarvis в папку «Программы».
$STEP2
3. Разрешите доступ к микрофону, когда система спросит.

$FIRST
Скажите «Джарвис», затем команду или вопрос.

Приложение без подписи Apple: при первом запуске нажмите
правой кнопкой по Jarvis → «Открыть» → «Открыть».
EOF

# 5. Записываемый образ, настройка окна Finder, сжатие в UDZO.
TMP_DMG="$BUILD_DIR/$DMG_NAME.tmp.dmg"
FINAL_DMG="$BUILD_DIR/$DMG_NAME.dmg"
rm -f "$TMP_DMG" "$FINAL_DMG"
hdiutil create -volname "Jarvis" -srcfolder "$BUILD_DIR/dmg" -ov -format UDRW "$TMP_DMG" >/dev/null
ATTACH="$(hdiutil attach "$TMP_DMG" -nobrowse -readwrite)"
VOL="$(printf '%s\n' "$ATTACH" | grep -o '/Volumes/.*' | tail -1)"
if [ -z "$VOL" ]; then
  echo "!! Не удалось смонтировать образ" >&2
  exit 1
fi
echo "    том: $VOL"

osascript >/dev/null 2>&1 <<OSA || true
tell application "Finder"
  tell disk "Jarvis"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {400, 120, 1000, 520}
    set vopt to the icon view options of container window
    set arrangement of vopt to not arranged
    set icon size of vopt to 120
    set background picture of vopt to file ".background:bg.png"
    set position of item "Jarvis.app" of container window to {150, 205}
    set position of item "Applications" of container window to {450, 205}
    try
      set position of item "Read Me.txt" of container window to {300, 330}
    end try
    update without registering applications
    delay 2
    close
  end tell
end tell
OSA

# Finder иногда убирает иконку тома — возвращаем и включаем её.
rm -rf "$VOL/.fseventsd"
cp "$TAURI_DIR/icons/icon.icns" "$VOL/.VolumeIcon.icns" 2>/dev/null || true
SetFile -a C "$VOL" 2>/dev/null || true
sync
hdiutil detach "$VOL" >/dev/null 2>&1 || hdiutil detach "$VOL" -force >/dev/null 2>&1 || true
hdiutil convert "$TMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$FINAL_DMG" >/dev/null
rm -f "$TMP_DMG"

echo "==> Готово: $FINAL_DMG ($(du -h "$FINAL_DMG" | cut -f1))"
