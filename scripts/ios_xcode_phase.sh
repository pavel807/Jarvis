#!/bin/sh
# Build-фаза Xcode «Build Rust Code» для Jarvis (Tauri 2 mobile).
#
# Зачем отдельный скрипт, а не строка в сгенерированном project.yml:
#
# 1. PATH. Xcode из GUI запускает build-фазы с минимальным PATH
#    (/usr/bin:/bin:/usr/sbin:/sbin). Ни bun, ни cargo там не лежат, и
#    сборка падает с «Command PhaseScriptExecution failed with a nonzero
#    exit code» (в логе Xcode: «line 2: bun: command not found»).
#    Здесь PATH дополняется явно.
#
# 2. Живой Tauri CLI. Фаза спрашивает у CLI адрес локального сервера опций
#    (gen/apple/.tauri/cli-options-server.json). Файл живёт только пока CLI
#    работает, поэтому сборка из Xcode требует запущенного `tauri ios dev`
#    (или `tauri ios build`). Без него фаза падала с невнятной ошибкой —
#    здесь мы говорим прямо, что запустить.
#
# Остальные аргументы (--platform, --sdk-root, --configuration, ARCHS и
# прочие переменные Xcode) пробрасываются в CLI как есть.
set -e

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.cargo/bin:$HOME/.bun/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_ROOT/app"
CLI_OPTIONS="$APP_DIR/src-tauri/gen/apple/.tauri/cli-options-server.json"

if [ ! -f "$CLI_OPTIONS" ]; then
  echo "❌ Tauri CLI не запущен: сборка Rust в Xcode невозможна." >&2
  echo "" >&2
  echo "   Что делать:" >&2
  echo "   1) cd app && bun tauri ios dev --open   # CLI остаётся живым, дальше Run в Xcode" >&2
  echo "   2) ./scripts/build_ios.sh               # сборка + установка + запуск без Xcode" >&2
  exit 1
fi

cd "$APP_DIR"
exec bun tauri ios xcode-script "$@"
