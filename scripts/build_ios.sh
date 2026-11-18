#!/usr/bin/env bash
# Собирает iOS/iPadOS-приложение Jarvis (Tauri 2 mobile) и ставит его туда,
# куда выберешь в меню.
#
#   ./scripts/build_ios.sh            — меню: выбери симулятор или устройство,
#                                       Джарвис соберётся, поставится и запустится
#   ./scripts/build_ios.sh list       — показать симуляторы и подключённые устройства
#   ./scripts/build_ios.sh sim [имя]  — то же, но без вопросов (для скриптов):
#                                       имя — подстрока («iPhone 17»)
#   ./scripts/build_ios.sh device [имя] — сборка+установка на устройство
#   ./scripts/build_ios.sh dev        — открыть Xcode и держать Tauri CLI живым
#                                       (нужно, чтобы Run в Xcode собрал Rust)
#
# Про сборку прямо из Xcode: build-фаза «Build Rust Code» сама Rust не
# собирает — она спрашивает у Tauri CLI адрес локального сервера опций, а тот
# живёт только пока CLI работает. Поэтому либо запускай сборку этим скриптом
# (он зовёт CLI сам), либо держи `bun tauri ios dev --open` живым — тогда
# Run в Xcode работает. Подробности в scripts/ios_xcode_phase.sh.
#
# Бандл НЕ несёт моделей: голосовые модели при первом запуске качаются в
# отдельную папку Documents/Jarvis/models (видна в приложении «Файлы»).
# На iOS работает настоящий голосовой движок (jarvis_core::mobile):
# tap-to-talk — микрофон → FastConformer STT → LLM → озвучка Silero.
#
# Подпись для устройства: учётка Apple в Xcode → Signing & Capabilities,
# либо DEVELOPMENT_TEAM=<ID команды> — сборка сама пропишет её в проект.
#
# Требования: Xcode с iOS SDK, Rust-таргеты:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
#
# Внимание: macOS несёт bash 3.2 — не ставим переменные вплотную к юникоду
# в расширениях, парсер от этого ломается (найдено на «DEST: unbound»).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/app"
GEN_DIR="$APP_DIR/src-tauri/gen/apple"
# Идентификатор берём из собранного приложения (build_sim/build_device
# возвращают путь) — смена bundle id в project.yml не ломает запуск.
BUNDLE_ID=""
MODE="${1:-menu}"
DEST="${2:-}"

CONF="$APP_DIR/src-tauri/tauri.conf.json"
VERSION="$(python3 -c "import json;print(json.load(open('$CONF'))['version'])")"

# ---------------------------------------------------------------------
# Списки устройств.
# ---------------------------------------------------------------------
# Симуляторы iPhone/iPad из simctl: имя и UDID из строки вида
# «iPhone 17 (UUID) (Shutdown)» (имя может содержать скобки — «iPad mini
# (A20 Pro)»), поэтому UUID вырезаем с конца, имя — до него.
SIM_NAMES=(); SIM_UDIDS=()
while IFS= read -r line; do
  udid="$(printf '%s' "$line" | sed -E 's/.*\(([0-9A-Fa-f-]{25,36})\) \(.*/\1/')"
  [ "$udid" = "$line" ] && continue
  name="$(printf '%s' "$line" | sed -E "s/^ *//; s/ *\(${udid}\) .*$//")"
  case "$name" in *Watch*) continue ;; esac
  SIM_NAMES+=("$name")
  SIM_UDIDS+=("$udid")
done < <(xcrun simctl list devices available | grep -E '[A-Za-z].*\([0-9A-Fa-f-]{25,36}\)')

# Физические iPhone/iPad из devicectl: JSON → name \t identifier \t marketing.
DEV_NAMES=(); DEV_UDIDS=(); DEV_MODELS=""
DEV_TMP="$(mktemp)"
if xcrun devicectl list devices --json-output "$DEV_TMP" >/dev/null 2>&1; then
  while IFS="$(printf '\t')" read -r name ident model; do
    [ -z "$name" ] && continue
    DEV_NAMES+=("$name")
    DEV_UDIDS+=("$ident")
    DEV_MODELS="${DEV_MODELS}${model}"$'\n'
  done < <(python3 - "$DEV_TMP" <<'EOF'
import json, sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for d in data.get("result", {}).get("devices", []):
    p = d.get("properties", {})
    st = p.get("state", {})
    hw = p.get("hardware", {})
    if st.get("visibilityClass") == "simulators":
        continue
    if hw.get("deviceType") not in ("iPhone", "iPad"):
        continue  # Watch и прочее не подходит: бандл для телефона
    name = st.get("name") or hw.get("marketingName") or ""
    model = hw.get("marketingName") or ""
    print(f"{name}\t{d.get('identifier', '')}\t{model}")
EOF
)
fi
rm -f "$DEV_TMP"

sim_udid_for() {
  local want idx lname
  # Подчёркивание → пробел: в справке имена симуляторов записаны как
  # «iPhone_17», а в simctl они с пробелом («iPhone 17»).
  want="$(printf '%s' "$1" | tr '_' ' ' | tr '[:upper:]' '[:lower:]')"
  # Точное имя важнее подстроки: «iPhone 17» не должен попадать в «iPhone 17e».
  for ((idx = 0; idx < ${#SIM_UDIDS[@]}; idx++)); do
    lname="$(printf '%s' "${SIM_NAMES[$idx]}" | tr '[:upper:]' '[:lower:]')"
    if [ "$lname" = "$want" ]; then printf '%s\n' "${SIM_UDIDS[$idx]}"; return 0; fi
  done
  for ((idx = 0; idx < ${#SIM_UDIDS[@]}; idx++)); do
    lname="$(printf '%s' "${SIM_NAMES[$idx]}" | tr '[:upper:]' '[:lower:]')"
    case "$lname" in
      *"$want"*) printf '%s\n' "${SIM_UDIDS[$idx]}"; return 0 ;;
    esac
  done
  return 1
}

device_udid_for() {
  local want idx lname
  want="$(printf '%s' "$1" | tr '_' ' ' | tr '[:upper:]' '[:lower:]')"
  for ((idx = 0; idx < ${#DEV_UDIDS[@]}; idx++)); do
    lname="$(printf '%s' "${DEV_NAMES[$idx]}" | tr '[:upper:]' '[:lower:]')"
    if [ "$lname" = "$want" ] || [ "${DEV_UDIDS[$idx]}" = "$1" ]; then
      printf '%s\n' "${DEV_UDIDS[$idx]}"; return 0
    fi
  done
  for ((idx = 0; idx < ${#DEV_UDIDS[@]}; idx++)); do
    lname="$(printf '%s' "${DEV_NAMES[$idx]}" | tr '[:upper:]' '[:lower:]')"
    case "$lname" in
      *"$want"*) printf '%s\n' "${DEV_UDIDS[$idx]}"; return 0 ;;
    esac
  done
  return 1
}

# ---------------------------------------------------------------------
# Сборка. Таргеты tauri-cli: aarch64 | aarch64-sim | x86_64 (не Rust-триплеты).
# ---------------------------------------------------------------------
ensure_targets() {
  # Весь вывод — в stderr: stdout скрипта занят путём к собранному .app,
  # и любая лишняя строка там ломает установку.
  for t in aarch64-apple-ios aarch64-apple-ios-sim; do
    rustup target list --installed | grep -q "$t" || rustup target add "$t" >&2
  done
}

ensure_project() {
  if [ ! -d "$GEN_DIR" ]; then
    echo "==> Генерирую Xcode-проект (tauri ios init)…" >&2
    (cd "$APP_DIR" && bun tauri ios init)
  fi

  # Правки project.yml (генерируется заново при tauri ios init, поэтому
  # накатываем их сами, каждый раз):
  #  1. PATH в build-фазе «Build Rust Code». Xcode из GUI запускает скрипт
  #     с минимальным PATH (/usr/bin:/bin:/usr/sbin:/sbin) — bun и cargo там
  #     не лежат, и сборка падает с «Command PhaseScriptExecution failed with
  #     a nonzero exit code» (в логе: «line 2: bun: command not found»).
  #  2. Саму фазу зовём нашим скриптом scripts/ios_xcode_phase.sh: там же
  #     внятная проверка, что Tauri CLI запущен, иначе Xcode падает с
  #     невнятным «must be running while Xcode builds the app».
  #  3. Команда подписи: DEVELOPMENT_TEAM из окружения — иначе xcodebuild
  #     требует её до всякой сборки Rust.
  local changed
  changed="$(python3 - "$GEN_DIR/project.yml" "${DEVELOPMENT_TEAM:-}" <<'EOF'
import pathlib, re, sys

path, team = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
changed = False

# Явный PATH: Xcode из GUI не видит /opt/homebrew/bin и ~/.cargo/bin.
PATH_LINE = 'export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.cargo/bin:$HOME/.bun/bin:$PATH"'
SCRIPT_MARKER = "      - script: |"
OLD_CMD = "bun tauri ios xcode-script"
# Наш скрипт вместо прямого вызова CLI: путь считается от project.yml
# (gen/apple) — четыре уровня вверх до корня репозитория.
NEW_CMD = '"$SRCROOT/../../../../scripts/ios_xcode_phase.sh"'

if PATH_LINE not in text:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.rstrip("\n") == SCRIPT_MARKER:
            # Отступ блока берём у следующей строки (содержимое `|`),
            # иначе YAML сломается.
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            indent = " " * (len(nxt) - len(nxt.lstrip(" ")))
            lines.insert(i + 1, f"{indent}{PATH_LINE}\n")
            text = "".join(lines)
            changed = True
            break
    else:
        sys.exit("В project.yml нет build-фазы «- script: |» — патч PATH не применился")

if OLD_CMD in text:
    text = text.replace(OLD_CMD, NEW_CMD, 1)
    changed = True

if team:
    if "DEVELOPMENT_TEAM:" in text:
        new = re.sub(r"DEVELOPMENT_TEAM:.*", f"DEVELOPMENT_TEAM: {team}", text)
    else:
        new = text.replace(
            "        ENABLE_BITCODE: false",
            f"        ENABLE_BITCODE: false\n        DEVELOPMENT_TEAM: {team}",
            1,
        )
    if new != text:
        text = new
        changed = True

path.write_text(text)
print("1" if changed else "0")
EOF
)"
  if [ "$changed" = "1" ]; then
    echo "==> Правлю Xcode-проект (PATH в build-фазе) и перегенерирую…" >&2
    (cd "$GEN_DIR" && xcodegen generate) >&2
  fi

  # Разрешение микрофона: без NSMicrophoneUsageDescription iOS молча
  # запрещает запись (cpal откроется, но подаст тишину). Ключ живёт в
  # project.yml (info.properties) — здесь честная проверка.
  #
  # Info.plist берём ИМЕННО приложения (info.path из project.yml): в
  # gen/apple/build/ лежит свой, без этого ключа, и проверка цепляла его.
  local plist
  plist="$(python3 - "$GEN_DIR/project.yml" <<'EOF'
import pathlib, re, sys

text = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r"^\s*path:\s*(\S*/Info\.plist)\s*$", text, re.M)
print(m.group(1) if m else "")
EOF
)"
  if [ -n "$plist" ]; then
    plist="$GEN_DIR/$plist"
  else
    plist="$(ls "$GEN_DIR"/*_iOS/Info.plist 2>/dev/null | head -1)"
  fi
  if [ -z "$plist" ] || ! /usr/libexec/PlistBuddy -c "Print :NSMicrophoneUsageDescription" "$plist" >/dev/null 2>&1; then
    echo "❌ В сгенерированном Info.plist нет NSMicrophoneUsageDescription." >&2
    echo "   Файл: ${plist:-не найден}" >&2
    echo "   Проверь gen/apple/project.yml (info.properties) и перегенерируй:" >&2
    echo "   cd app/src-tauri/gen/apple && xcodegen generate" >&2
    exit 1
  fi

  # Иконка: каталог Assets.xcassets в gen/apple пересоздаётся при
  # tauri ios init (там снова дефолтный значок Tauri) — ставим реактор
  # Mark V: квадратный и непрозрачный, как iOS требует.
  if ! python3 "$ROOT/scripts/ios_icon.py" >&2; then
    echo "⚠️ Иконку iOS обновить не удалось — в приложении будет старая." >&2
  fi
}

# Команда подписи: DEVELOPMENT_TEAM из окружения или первый действительный
# сертификат из keychain (…(TEAMID)). Нужна только для установки на устройство.
detect_team() {
  [ "${DEVELOPMENT_TEAM:-}" != "" ] && return 0
  local found
  found="$(security find-identity -v -p codesigning 2>/dev/null \
    | sed -n -E 's/.*"Apple Development.*\(([0-9A-Z]{10})\)".*/\1/p' | head -1)"
  [ -n "$found" ] && DEVELOPMENT_TEAM="$found"
  return 0
}

build_sim() {
  ensure_targets
  ensure_project
  echo "==> Собираю для симулятора…" >&2
  # Прогресс — только в stderr: функция возвращает в stdout ровно путь,
  # иначе длинный лог попадает в APP и его ставят на устройство как путь.
  if ! (cd "$APP_DIR" && bun tauri ios build --target aarch64-sim) >&2; then
    echo "❌ Сборка симулятора не удалась — смотри лог выше." >&2
    return 1
  fi
  # tauri CLI собирает с -derivedDataPath gen/apple/build — продукт там.
  local app="$GEN_DIR/build/arm64-sim/Jarvis.app"
  [ -d "$app" ] || { echo "❌ После сборки нет $app" >&2; return 1; }
  echo "$app"
}

build_device() {
  detect_team
  if [ "${DEVELOPMENT_TEAM:-}" = "" ]; then
    echo "❌ Нет команды подписи для установки на устройство." >&2
    echo "   Открой учётку Apple в Xcode (Settings → Accounts) или задай:" >&2
    echo "   DEVELOPMENT_TEAM=<ID> ./scripts/build_ios.sh" >&2
    return 1
  fi
  ensure_targets
  ensure_project
  echo "==> Собираю для устройства (команда $DEVELOPMENT_TEAM)…" >&2
  if ! (cd "$APP_DIR" && bun tauri ios build --target aarch64) >&2; then
    echo "❌ Сборка для устройства не удалась — смотри лог выше." >&2
    echo "   «No Account for Team» — залогинься в Xcode один раз:" >&2
    echo "   Xcode → Settings → Accounts → «+» → Apple ID этой команды ($DEVELOPMENT_TEAM)," >&2
    echo "   после этого ./scripts/build_ios.sh сам создаст профиль и поставит приложение." >&2
    return 1
  fi
  local app="$GEN_DIR/build/arm64/Jarvis.ipa"
  [ -f "$app" ] || { echo "❌ После сборки нет $app" >&2; return 1; }
  # На устройство ставится .app, а не .ipa: берём его из xcarchive, который
  # CLI оставляет рядом (Payload/ из самого .ipa — запасной путь).
  local bundle="$GEN_DIR/build/jarvis-app_iOS.xcarchive/Products/Applications/Jarvis.app"
  if [ ! -d "$bundle" ]; then
    local out="$GEN_DIR/build/arm64/app"
    rm -rf "$out"
    mkdir -p "$out"
    unzip -q "$app" -d "$out" 2>/dev/null || true
    bundle="$(find "$out" -maxdepth 2 -name '*.app' -type d 2>/dev/null | head -1)"
  fi
  [ -n "$bundle" ] && [ -d "$bundle" ] || {
    echo "❌ В .ipa нет приложения .app — ставить нечего." >&2
    return 1
  }
  echo "$bundle"
}

# ---------------------------------------------------------------------
# Установка и запуск.
# ---------------------------------------------------------------------
bundle_id_of() {
  plutil -extract CFBundleIdentifier raw -o - "$1/Info.plist"
}

install_sim() {
  local udid="$1" app="$2"
  echo "==> Ставлю в симулятор ($udid)…"
  if ! xcrun simctl list devices booted | grep -q "$udid"; then
    echo "==> Запускаю симулятор…"
    xcrun simctl boot "$udid" || true
    # GUI симулятора опциональна (сборка ставится и без неё), путь ищем
    # рядом с активным Xcode — app name может не находиться через open -a.
    local simui
    simui="$(xcode-select -p)/Applications/Simulator.app"
    [ -d "$simui" ] && open "$simui" || true
    xcrun simctl bootstatus "$udid" -b >/dev/null 2>&1 || true
  fi
  xcrun simctl install "$udid" "$app"
  xcrun simctl launch "$udid" "$(bundle_id_of "$app")"
  echo "==> Джарвис запущен в симуляторе."
}

install_device() {
  local udid="$1" app="$2"
  echo "==> Ставлю на устройство ($udid)…"
  xcrun devicectl device install app --device "$udid" "$app"
  if xcrun devicectl device process launch --device "$udid" "$(bundle_id_of "$app")"; then
    echo "==> Джарвис запущен на устройстве."
  else
    echo "⚠️ Приложение установлено, но не запустилось — открой его вручную."
  fi
}

# ---------------------------------------------------------------------
# Меню выбора: куда ставить.
# ---------------------------------------------------------------------
menu_pick() {
  echo "Куда поставить Jarvis $VERSION?"
  local i=1 idx model n pick kind
  for ((idx = 0; idx < ${#SIM_NAMES[@]}; idx++)); do
    echo "  $i) ${SIM_NAMES[$idx]}  (симулятор)"
    i=$((i + 1))
  done
  for ((idx = 0; idx < ${#DEV_UDIDS[@]}; idx++)); do
    model="$(printf '%s\n' "$DEV_MODELS" | sed -n "$((idx + 1))p")"
    if [ -n "$model" ]; then
      echo "  $i) ${DEV_NAMES[$idx]}  (устройство, $model)"
    else
      echo "  $i) ${DEV_NAMES[$idx]}  (устройство)"
    fi
    i=$((i + 1))
  done
  echo "  b) Только собрать, не ставить (сборка для симулятора)"
  echo "  d) Открыть Xcode (Tauri CLI остаётся живым — горячая перезагрузка)"
  printf 'Выбор (номер, Enter — b): '
  IFS= read -r pick
  case "$pick" in
    ""|b|B)
      MODE="sim"; DEST=""
      ;;
    d|D)
      MODE="dev"; DEST=""
      ;;
    *)
      if ! [ "$pick" -ge 1 ] 2>/dev/null || ! [ "$pick" -le "$((i - 1))" ]; then
        echo "❌ Нет такого номера: $pick"
        exit 1
      fi
      n=1; kind=""
      for ((idx = 0; idx < ${#SIM_UDIDS[@]}; idx++)); do
        if [ "$n" = "$pick" ]; then MODE="sim"; DEST="${SIM_UDIDS[$idx]}"; kind="sim"; fi
        n=$((n + 1))
      done
      for ((idx = 0; idx < ${#DEV_UDIDS[@]}; idx++)); do
        if [ "$n" = "$pick" ]; then MODE="device"; DEST="${DEV_UDIDS[$idx]}"; kind="dev"; fi
        n=$((n + 1))
      done
      [ -n "$kind" ] || { echo "❌ Нет такого номера: $pick"; exit 1; }
      ;;
  esac
}

# ---------------------------------------------------------------------
# Режимы.
# ---------------------------------------------------------------------
case "$MODE" in
  list)
    echo "==> Jarvis (iOS) $VERSION — доступные назначения:"
    echo
    echo "Симуляторы:"
    i=1
    for ((idx = 0; idx < ${#SIM_NAMES[@]}; idx++)); do
      echo "  $i) ${SIM_NAMES[$idx]}  (${SIM_UDIDS[$idx]})"
      i=$((i + 1))
    done
    echo
    echo "Устройства (подключённые):"
    for ((idx = 0; idx < ${#DEV_UDIDS[@]}; idx++)); do
      echo "  ${DEV_NAMES[$idx]}  (${DEV_UDIDS[$idx]})"
    done
    echo
    echo "Установка: ./scripts/build_ios.sh  (меню)  |  sim iPhone_17  |  device iPad_Pavel"
    echo "Xcode:     ./scripts/build_ios.sh dev   (Tauri CLI живой → Run в Xcode собирает Rust)"
    exit 0
    ;;
  menu)
    menu_pick
    ;;
esac

APP=""
case "$MODE" in
  dev)
    # Xcode сам Rust не соберёт: build-фаза спрашивает у Tauri CLI адрес
    # локального сервера опций, и файл живёт только пока CLI работает.
    # Поэтому здесь CLI остаётся живым, а Xcode открывается для Run.
    ensure_targets
    ensure_project
    echo "==> Открываю Xcode: Tauri CLI живёт, пока эта команда работает." >&2
    echo "    Run в Xcode соберёт Rust и запустит приложение. Ctrl-C здесь — остановить всё." >&2
    cd "$APP_DIR" && exec bun tauri ios dev --open
    ;;
  sim)
    if [ -n "$DEST" ]; then
      # DEST может быть UDID из меню или подстрокой имени из аргумента.
      if printf '%s' "$DEST" | grep -qE '^[0-9A-Fa-f-]{25,36}$' \
         && printf '%s\n' "${SIM_UDIDS[@]}" | grep -q "$DEST"; then
        UDID="$DEST"
      else
        UDID="$(sim_udid_for "$DEST")" || {
          echo "❌ Симулятор «${DEST}» не найден. Список: ./scripts/build_ios.sh list"
          exit 1
        }
      fi
    else
      UDID=""
    fi
    APP="$(build_sim)" || exit 1
    if [ -n "$UDID" ]; then
      install_sim "$UDID" "$APP"
    else
      echo "==> Готово (без установки): $APP"
      echo "    Запуск: ./scripts/build_ios.sh  (меню)  или  ./scripts/build_ios.sh sim iPhone_17"
    fi
    ;;
  device)
    if [ -n "$DEST" ]; then
      if printf '%s\n' "${DEV_UDIDS[@]}" | grep -q "$DEST"; then
        UDID="$DEST"
      else
        UDID="$(device_udid_for "$DEST")" || {
          echo "❌ Устройство «${DEST}» не найдено. Список: ./scripts/build_ios.sh list"
          echo "   (устройство должно быть подключено и разблокировано)"
          exit 1
        }
      fi
    else
      UDID=""
    fi
    APP="$(build_device)" || exit 1
    if [ -n "$UDID" ]; then
      install_device "$UDID" "$APP"
    else
      echo "==> Готово (без установки): $APP"
      echo "    Запуск: ./scripts/build_ios.sh device iPad_Pavel"
      if [ "${DEVELOPMENT_TEAM:-}" = "" ]; then
        echo "    Для установки нужна подпись: DEVELOPMENT_TEAM=<ID> ./scripts/build_ios.sh device …"
      fi
    fi
    ;;
  *)
    echo "❌ Неизвестный режим «${MODE}». Режимы: (без аргументов — меню) | list | sim [имя] | device [имя] | dev"
    exit 1
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
