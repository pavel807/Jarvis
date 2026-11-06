#!/usr/bin/env bash
# jarvis-fixes-2.sh — дополнение к jarvis-fixes.sh: разбор действий Holo и
# английский вопрос для screen_look.
#
# Запуск из корня репозитория (где лежит crates/jarvis-core):
#     bash jarvis-fixes-2.sh
# Необязательно:  SKIP_CHECK=1  — не гонять cargo check;  RUN_TESTS=1 — после
# проверки сборки запустить юнит-тесты jarvis-core.
#
# Что делает: сохраняет копии затрагиваемых файлов, применяет патч, собирает
# `cargo check -p jarvis-core --lib --tests` и при ошибке сборки откатывает всё.
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"
if [ ! -f Cargo.toml ] || [ ! -d crates/jarvis-core/src ]; then
  echo "❌ Запустите скрипт из корня репозитория (там, где crates/jarvis-core)." >&2
  exit 1
fi
if [ ! -f crates/jarvis-core/src/tool_guard.rs ]; then
  echo "❌ Это дополнение к jarvis-fixes.sh: сначала примените его." >&2
  exit 1
fi
if grep -q 'fn parse_tool_call_json' crates/jarvis-core/src/holo/action.rs; then
  echo "ℹ️  Дополнение уже применено. Ничего не делаю."
  exit 0
fi
command -v patch >/dev/null 2>&1 || { echo "❌ Нужна утилита patch." >&2; exit 1; }
[ -f "$0" ] || { echo "❌ Запускайте как файл: bash jarvis-fixes-2.sh (не через pipe)." >&2; exit 1; }

PATCH_FILE="$(mktemp "${TMPDIR:-/tmp}/jarvis-fixes.XXXXXX")"
trap 'rm -f "$PATCH_FILE"' EXIT
sed -n '/^__PATCH_BELOW__$/,$p' "$0" | tail -n +2 > "$PATCH_FILE"

echo "🔎 Проверяю, что патч ложится на ваши файлы…"
if ! patch -p1 --dry-run --silent < "$PATCH_FILE" >/dev/null 2>&1; then
  echo "❌ Патч не применяется: ваши файлы отличаются от версии, под которую он сделан." >&2
  echo "   Ничего не изменено. Покажите вывод: patch -p1 --dry-run < <(sed -n '/^__PATCH_BELOW__\$/,\$p' jarvis-fixes-2.sh | tail -n +2)" >&2
  exit 1
fi

BACKUP=".jarvis-fixes-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP"
touch "$BACKUP/.stamp"
FILES="$(grep '^+++ ' "$PATCH_FILE" | awk '{print $2}' | sed 's#^[^/]*/##')"
for f in $FILES; do
  if [ -f "$f" ]; then mkdir -p "$BACKUP/$(dirname "$f")"; cp -p "$f" "$BACKUP/$f"; fi
done
echo "💾 Копии изменяемых файлов: $BACKUP/"

rollback() {
  echo "↩️  Откатываю изменения…" >&2
  for f in $FILES; do
    if [ -f "$BACKUP/$f" ]; then cp -p "$BACKUP/$f" "$f"; else rm -f "$f"; fi
  done
  echo "✅ Откат выполнен, репозиторий как был." >&2
}

patch -p1 --silent < "$PATCH_FILE"
find crates -name '*.orig' -newer "$BACKUP/.stamp" -delete 2>/dev/null || true
echo "🩹 Патч применён."

if [ "${SKIP_CHECK:-0}" != "1" ]; then
  echo "🔨 cargo check -p jarvis-core --lib --tests (на Metal-сборке это может занять несколько минут)…"
  if ! cargo check -p jarvis-core --lib --tests; then
    echo "❌ Сборка не прошла после патча." >&2
    rollback
    exit 1
  fi
  # Примеры собираются отдельно и не блокируют патч: часть из них может быть
  # сломана и без него (устаревшие вызовы API).
  if ! cargo check -p jarvis-core --examples; then
    echo "⚠️  Примеры (examples/) не собираются — это не блокирует патч. Если ошибка не в llm_honesty.rs и не в ui_agent_e2e.rs, она была и до патча." >&2
  fi
fi

if [ "${RUN_TESTS:-0}" = "1" ]; then
  echo "🧪 cargo test -p jarvis-core --lib"
  cargo test -p jarvis-core --lib || {
    echo "⚠️  Часть тестов упала — патч оставлен, копии в $BACKUP/. Откат: скопируйте файлы из копии." >&2
    exit 1
  }
fi

cat <<'DONE'

✅ Готово. Пересоберите и запустите ассистента (cargo run …) — старый бинарник
   этих исправлений не содержит.

Быстрая проверка чистой логики без моделей:
   cargo test -p jarvis-core --lib tool_guard
DONE
exit 0

__PATCH_BELOW__
diff -ruN a/crates/jarvis-core/src/holo/action.rs b/crates/jarvis-core/src/holo/action.rs
--- a/crates/jarvis-core/src/holo/action.rs
+++ b/crates/jarvis-core/src/holo/action.rs
@@ -101,14 +101,14 @@
             // Стоп-строка генерации обрезает закрывающий тег — сам JSON уже
             // полон; парсим хвост как есть.
             let json = rest.trim();
-            if let Ok(call) = serde_json::from_str::<ModelToolCall>(json) {
+            if let Ok(call) = parse_tool_call_json(json) {
                 actions.push(tool_call_to_action(call)?);
             }
             break;
         };
         let json = rest[..end].trim();
         rest = &rest[end + "</tool_call>".len()..];
-        match serde_json::from_str::<ModelToolCall>(json) {
+        match parse_tool_call_json(json) {
             Ok(call) => actions.push(tool_call_to_action(call)?),
             Err(e) => {
                 // Не JSON: возможно, heredoc-формат Qwen3.5/Holo — уступаем.
@@ -128,7 +128,7 @@
             bail!("Некорректный JSON внутри <tool_call>: {e}");
         }
         // Some model-serving paths return the tool-call JSON without XML tags.
-        if let Ok(call) = serde_json::from_str::<ModelToolCall>(trimmed) {
+        if let Ok(call) = parse_tool_call_json(trimmed) {
             actions.push(tool_call_to_action(call)?);
         } else {
             bail!("В ответе модели не найдено допустимых действий");
@@ -207,6 +207,56 @@
     Ok(calls)
 }
 
+/// Разбор JSON одного вызова инструмента с поправкой на то, что реально
+/// выдаёт модель: `{"name":"click","x":[500,54]}` — без обёртки `arguments`
+/// и с координатами парой. Из-за этого `ui_task` трижды подряд отказывал.
+fn parse_tool_call_json(raw: &str) -> Result<ModelToolCall, serde_json::Error> {
+    let mut value: Value = serde_json::from_str(raw)?;
+    if let Some(object) = value.as_object_mut() {
+        if !object.contains_key("arguments") {
+            if let Some(args) = object.remove("args").or_else(|| object.remove("parameters")) {
+                object.insert("arguments".into(), args);
+            } else if object.contains_key("name") {
+                // Плоский вариант: всё, кроме имени, — аргументы.
+                let mut args = serde_json::Map::new();
+                for key in object.keys().cloned().collect::<Vec<_>>() {
+                    if key != "name" {
+                        if let Some(v) = object.remove(&key) {
+                            args.insert(key, v);
+                        }
+                    }
+                }
+                object.insert("arguments".into(), Value::Object(args));
+            }
+        }
+        if let Some(Value::Object(args)) = object.get_mut("arguments") {
+            normalize_coordinates(args);
+        }
+    }
+    serde_json::from_value(value)
+}
+
+/// `x: [500, 54]` или `coordinate: [500, 54]` → `x: 500, y: 54`.
+fn normalize_coordinates(args: &mut serde_json::Map<String, Value>) {
+    if args.contains_key("y") {
+        return;
+    }
+    for key in ["x", "coordinate", "coordinates", "point", "position", "xy"] {
+        let pair = match args.get(key) {
+            Some(Value::Array(p)) if p.len() >= 2 && p[0].is_number() && p[1].is_number() => {
+                Some((p[0].clone(), p[1].clone()))
+            }
+            _ => None,
+        };
+        if let Some((x, y)) = pair {
+            args.remove(key);
+            args.insert("x".into(), x);
+            args.insert("y".into(), y);
+            return;
+        }
+    }
+}
+
 fn tool_call_to_action(call: ModelToolCall) -> Result<GuiAction> {
     let action_name = match call.name.as_str() {
         "click" => "click",
@@ -366,6 +416,32 @@
         assert_eq!(actions, vec![DesktopAction::Click { x: 960, y: 270, button: MouseButton::Left, count: 1 }]);
     }
     #[test]
+    fn accepts_flat_tool_call_with_coordinate_pair_and_no_closing_tag() {
+        // Реальный ответ Holo: нет `arguments`, координаты парой, тег не закрыт
+        // (стоп-строка обрезает его).
+        let raw = "<tool_call>\n{\"name\": \"click\", \"x\": [500, 54]}\n";
+        let batch = parse_model_output(raw).unwrap();
+        assert_eq!(
+            batch.actions,
+            vec![GuiAction::Click { x: 500.0, y: 54.0, button: None, clicks: None }]
+        );
+    }
+
+    #[test]
+    fn accepts_args_alias_and_point_arrays() {
+        let raw = r#"<tool_call>{"name":"click","args":{"coordinate":[10,20]}}</tool_call>"#;
+        assert_eq!(
+            parse_model_output(raw).unwrap().actions,
+            vec![GuiAction::Click { x: 10.0, y: 20.0, button: None, clicks: None }]
+        );
+        let flat_type = r#"<tool_call>{"name":"type_text","text":"aviasales"}</tool_call>"#;
+        assert_eq!(
+            parse_model_output(flat_type).unwrap().actions,
+            vec![GuiAction::Type { text: "aviasales".into() }]
+        );
+    }
+
+    #[test]
     fn parses_official_chat_template_tool_call_envelope() {
         let batch = parse_model_output(r#"<tool_call>{"name":"click","arguments":{"x":0.2,"y":0.8}}</tool_call>"#).unwrap();
         assert_eq!(batch.actions.len(), 1);
diff -ruN a/crates/jarvis-core/src/tools.rs b/crates/jarvis-core/src/tools.rs
--- a/crates/jarvis-core/src/tools.rs
+++ b/crates/jarvis-core/src/tools.rs
@@ -1794,6 +1794,16 @@
         .is_some_and(|at| at.elapsed() < BLIND_INPUT_BAN)
 }
 
+/// Holo заметно хуже пишет по-русски (вставляет китайские токены и
+/// придумывает детали), поэтому смотрим по-английски, а пересказывает
+/// пользователю основная LLM.
+fn vision_question(question: &str) -> String {
+    format!(
+        "{question}\n\nAnswer in English in at most three short plain sentences. \
+         Mention only what is clearly visible; do not guess."
+    )
+}
+
 /// Готов ли агент зрения. Идёт загрузка — ждём её (один раз сказав об этом
 /// вслух), иначе возвращаем причину отказа.
 fn holo_gate(audio_tx: &Sender<AudioMessage>) -> Result<(), String> {
@@ -2291,12 +2301,12 @@
                     format!("{why}. Экран увидеть нельзя: не вводи текст и не кликай вслепую, сообщи пользователю причину."),
                 )
             } else {
-                match crate::holo::service::with_service(|h| h.look(&question.clone())) {
+                match crate::holo::service::with_service(|h| h.look(&vision_question(&question))) {
                 Ok(description) => {
                     let bounded: String = description.chars().take(900).collect();
                         ToolOutcome::with_detail(
                             "Экран разобран агентом зрения".to_string(),
-                            format!("{question}\nОтвет агента зрения: {bounded}"),
+                            format!("{question}\nОтвет агента зрения (по-английски, перескажи пользователю по-русски): {bounded}"),
                         )
                     }
                     Err(e) => ToolOutcome::fail(format!("Разбор экрана недоступен: {e:#}")),
