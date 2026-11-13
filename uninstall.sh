#!/usr/bin/env bash
# Деинсталлятор Jarvis: убирает бинарь, ярлык и (опционально) данные.
# Модели и плагины не трогает без явного --purge: там голос пользователя.
set -euo pipefail

PREFIX="${JARVIS_PREFIX:-$HOME/.local}"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/Jarvis"
PURGE=0
[ "${1:-}" = "--purge" ] && PURGE=1

say() { printf '%s\n' "$1"; }

say "▸ Убираю бинарь и ярлык"
rm -fv "$PREFIX/bin/jarvis" "$PREFIX/bin/jarvis-cli"
rm -fv "$PREFIX/share/applications/jarvis.desktop"

if [ "$PURGE" = 1 ]; then
  say "▸ --purge: удаляю данные, модели и плагины целиком"
  rm -rfv "$DATA_DIR"
else
  say "▸ Данные сохранены ($DATA_DIR): модели, плагины, настройки."
  say "  Удалить и их: $0 --purge"
fi

say "✔ Jarvis удалён"
