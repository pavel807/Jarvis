#!/usr/bin/env bash
# Установщик Jarvis: красивый, честный, нативный путь на Arch/Omarchy.
# Контейнерная сборка — через compose.yml (docker compose run --rm build).
set -euo pipefail

# ── Оформление ────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  B=$'\e[1m'; D=$'\e[2m'; R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'
  C=$'\e[36m'; M=$'\e[35m'; X=$'\e[0m'
else
  B=""; D=""; R=""; G=""; Y=""; C=""; M=""; X=""
fi

step()  { printf '\n%s▸ %s%s\n' "$C$B" "$1" "$X"; }
ok()    { printf '  %s✔%s %s\n' "$G" "$X" "$1"; }
warn()  { printf '  %s●%s %s\n' "$Y" "$X" "$1"; }
fail()  { printf '  %s✘%s %s\n' "$R" "$X" "$1"; }
die()   { fail "$1"; exit 1; }

banner() {
  printf '%s' "$M$B"
  cat <<'ART'
     ▄█████████████▄
   ▄█▀             ▀█▄      J A R V I S
  ██   ▄▄     ▄▄    ██      голосовой ассистент
  ██   ▀▀  █  ▀▀    ██      для вашего десктопа
  ██▄    ▄███▄    ▄██▀
   ▀██▄▄       ▄▄██▀
     ▀█████████████▀
ART
  printf '%s\n' "$X"
}

# ── Параметры ──────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${JARVIS_PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/Jarvis"
SKIP_BUILD=0
DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --skip-build) SKIP_BUILD=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --help|-h) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done

banner
printf '%sРепозиторий:%s %s\n' "$D" "$X" "$REPO_DIR"
printf '%sУстановка в:%s   %s (бинарь %s)\n' "$D" "$X" "$DATA_DIR" "$BIN_DIR"
[ "$DRY_RUN" = 1 ] && warn "режим просмотра (--dry-run): изменений не будет"

# ── Шаг 1. Зависимости ─────────────────────────────────────────────────────
step "Проверяю зависимости"

have() { command -v "$1" >/dev/null 2>&1; }

MISSING=()
check() {
  if have "$1"; then ok "$1 — есть"; else MISSING+=("$2"); warn "$1 — нет"; fi
}

check bun  "bun"
check cargo "rust (cargo)"
if [ "$(ldconfig -p 2>/dev/null | grep -c libwebkit2gtk-4.1)" -gt 0 ] || pkg-config --exists webkit2gtk-4.1 2>/dev/null; then
  ok "webkit2gtk-4.1 — есть"
else
  MISSING+=("webkit2gtk-4.1"); warn "webkit2gtk-4.1 — нет"
fi
if [ "$(ldconfig -p 2>/dev/null | grep -c libpulse)" -gt 0 ] || pkg-config --exists libpulse 2>/dev/null; then
  ok "libpulse — есть"
else
  MISSING+=("libpulse"); warn "libpulse — нет"
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
  printf '\n  Не хватает: %s%s%s\n' "$B" "${MISSING[*]}" "$X"
  if have pacman && [ "$DRY_RUN" = 0 ]; then
    printf '  %sСтавлю через pacman (попросит пароль)…%s\n' "$D" "$X"
    sudo pacman -S --needed "${MISSING[@]}" || die "pacman не смог установить зависимости"
    ok "зависимости установлены"
  else
    die "Поставь их вручную и запусти установщик снова${DRY_RUN:+ (или без --dry-run, чтобы я поставил сам через pacman)}"
  fi
fi

# ── Шаг 2. Сборка ──────────────────────────────────────────────────────────
step "Собираю релиз"
if [ "$SKIP_BUILD" = 1 ] && [ -x "$REPO_DIR/target/release/jarvis-app" ]; then
  warn "--skip-build: использую уже собранный бинарь"
elif [ "$DRY_RUN" = 1 ]; then
  warn "dry-run: сборка пропущена (будет: bun install + bun tauri build)"
else
  (cd "$REPO_DIR/app" && bun install --frozen-lockfile)
  (cd "$REPO_DIR/app" && bun tauri build) || die "сборка не удалась"
  ok "сборка готова: target/release/jarvis-app"
fi

# ── Шаг 3. Установка ───────────────────────────────────────────────────────
step "Устанавливаю"
BIN_SRC="$REPO_DIR/target/release/jarvis-app"

install_file() { # src dst mode
  if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$(dirname "$2")"
    install -m "$3" "$1" "$2"
  fi
  ok "$2"
}

install_file "$BIN_SRC" "$BIN_DIR/jarvis" 755

# Каталоги данных: модели качаются при первом запуске приложения.
for d in "$DATA_DIR" "$DATA_DIR/plugins" "$DATA_DIR/resources"; do
  if [ "$DRY_RUN" = 0 ]; then mkdir -p "$d"; fi
  ok "$d/"
done
if [ "$DRY_RUN" = 0 ]; then
  cp -n "$REPO_DIR"/plugins/*.sk "$DATA_DIR/plugins/" 2>/dev/null || true
fi

# Ядро движка в PATH тоже: CLI для тестов и скриптов.
if [ -x "$REPO_DIR/target/release/jarvis-cli" ]; then
  install_file "$REPO_DIR/target/release/jarvis-cli" "$BIN_DIR/jarvis-cli" 755
fi

# ── Шаг 4. Ярлык в меню ────────────────────────────────────────────────────
step "Ярлык в меню приложений"
DESKTOP="$PREFIX/share/applications/jarvis.desktop"
if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$PREFIX/share/applications"
  cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Jarvis
GenericName=Голосовой ассистент
Comment=Слушает активационную фразу и выполняет команды
Exec=$BIN_DIR/jarvis
Icon=$REPO_DIR/app/src-tauri/icons/128x128.png
Terminal=false
Categories=Utility;AudioVideo;
StartupNotify=true
EOF
fi
ok "$DESKTOP"
if have update-desktop-database && [ "$DRY_RUN" = 0 ]; then
  update-desktop-database "$PREFIX/share/applications" 2>/dev/null || true
fi

# ── Готово ─────────────────────────────────────────────────────────────────
printf '\n%s%s Готово!%s\n\n' "$G$B" "✔" "$X"
printf '  Запуск:            %sbun tauri dev%s (разработка) или %sjarvis%s из меню\n' "$B" "$X" "$B" "$X"
printf '  Фраза-активация:   настраивается в приложении, раздел «Wake word»\n'
printf '  Плагины:           кладите .sk в %s%s\n' "$B" "$DATA_DIR/plugins$X"
printf '  Модели:            скачаются при первом запуске (STT ~300 МБ, TTS ~40 МБ)\n'
printf '  Контейнерная сборка: %sdocker compose run --rm build%s → ./dist-out\n\n' "$B" "$X"
