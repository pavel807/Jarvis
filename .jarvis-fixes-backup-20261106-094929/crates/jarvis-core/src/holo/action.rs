use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;
use crate::platform::desktop::{DesktopAction, MouseButton, ScrollDirection};

/// Jarvis-side action envelope: координаты в конвенции grounding Holo-3.1 —
/// целые 0..1000 относительно скриншота. Это граница безопасности, а не
/// утверждение о сыром выводе модели.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum GuiAction {
    Click { x: f32, y: f32, button: Option<String>, clicks: Option<u8> },
    Type { text: String },
    Key { key: String },
    Hotkey { keys: Vec<String> },
    Scroll { direction: String, amount: Option<u8> },
    Move { x: f32, y: f32 },
    /// Внутренний адаптер запуска (пп. 25): приложение или сайт — надёжнее
    /// кликов по доку. Исполняется через системный `open`.
    Launch { app: Option<String>, url: Option<String> },
    Wait { milliseconds: Option<u64> },
    Done { message: Option<String> },
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GuiActionBatch { pub actions: Vec<GuiAction> }

/// Parse only a complete JSON object or JSON fenced block. Never extract a
/// guessed coordinate from arbitrary prose.
pub fn parse_actions(raw: &str) -> Result<GuiActionBatch> {
    let body = raw.trim();
    let body = body.strip_prefix("```json").unwrap_or(body).trim_start();
    let body = body.strip_suffix("```").unwrap_or(body).trim_end();
    let batch: GuiActionBatch = serde_json::from_str(body).context("Ответ GUI-модели не соответствует action JSON")?;
    if batch.actions.is_empty() || batch.actions.len() > 4 { bail!("В пакете должно быть от 1 до 4 действий: после значимых изменений нужен свежий скриншот"); }
    Ok(batch)
}

/// Convert 0..1000 grounding coordinates to desktop points. On retina the
/// screenshot is in physical pixels while cliclick expects logical points —
/// `logical` (ширина/высота рабочего стола в точках) задаёт масштаб.
pub fn to_desktop_actions(
    batch: &GuiActionBatch,
    width: u32,
    height: u32,
    logical: Option<(u32, u32)>,
) -> Result<Vec<DesktopAction>> {
    if width == 0 || height == 0 { bail!("Размер скриншота не может быть нулевым"); }
    // Масштаб пиксель → точка (пп. 35): без него клики на retina уходят
    // за экран (y=2561 при высоте 982).
    let (scale_x, scale_y) = match logical {
        Some((lw, lh)) if lw > 0 && lh > 0 && (lw != width || lh != height) => {
            (lw as f32 / width as f32, lh as f32 / height as f32)
        }
        _ => (1.0, 1.0),
    };
    let mut out = Vec::new();
    for action in &batch.actions {
        match action {
            GuiAction::Click { x, y, button, clicks } => out.push(DesktopAction::Click {
                x: pixel(*x, width, scale_x)?, y: pixel(*y, height, scale_y)?,
                button: parse_button(button.as_deref())?, count: clicks.unwrap_or(1),
            }),
            GuiAction::Type { text } => out.push(DesktopAction::TypeText { text: text.clone() }),
            GuiAction::Key { key } => out.push(DesktopAction::PressKey { key: key.clone() }),
            GuiAction::Hotkey { keys } => out.push(DesktopAction::Hotkey { keys: keys.clone() }),
            GuiAction::Scroll { direction, amount } => out.push(DesktopAction::Scroll {
                direction: parse_direction(direction)?, amount: amount.unwrap_or(3),
            }),
            GuiAction::Move { x, y } => out.push(DesktopAction::Move { x: pixel(*x, width, scale_x)?, y: pixel(*y, height, scale_y)? }),
            // Launch исполняется отдельно (execute_batch), в desktop-команды
            // не превращается.
            GuiAction::Launch { .. } | GuiAction::Wait { .. } | GuiAction::Done { .. } => {},
        }
    }
    Ok(out)
}


#[derive(Debug, Deserialize)]
struct ModelToolCall {
    name: String,
    arguments: Value,
}

/// Parse either Jarvis's strict action envelope, the Qwen JSON `<tool_call>`
/// envelope, or the Qwen3.5/Holo heredoc form. Only explicitly allow-listed
/// GUI tools are accepted.
pub fn parse_model_output(raw: &str) -> Result<GuiActionBatch> {
    let trimmed = raw.trim();
    if trimmed.starts_with('{') {
        if let Ok(batch) = parse_actions(trimmed) { return Ok(batch); }
    }
    let mut actions = Vec::new();
    let mut rest = trimmed;
    let mut json_error: Option<serde_json::Error> = None;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let Some(end) = rest.find("</tool_call>") else {
            // Стоп-строка генерации обрезает закрывающий тег — сам JSON уже
            // полон; парсим хвост как есть.
            let json = rest.trim();
            if let Ok(call) = serde_json::from_str::<ModelToolCall>(json) {
                actions.push(tool_call_to_action(call)?);
            }
            break;
        };
        let json = rest[..end].trim();
        rest = &rest[end + "</tool_call>".len()..];
        match serde_json::from_str::<ModelToolCall>(json) {
            Ok(call) => actions.push(tool_call_to_action(call)?),
            Err(e) => {
                // Не JSON: возможно, heredoc-формат Qwen3.5/Holo — уступаем.
                json_error = Some(e);
                actions.clear();
                break;
            }
        }
        if actions.len() > 4 { bail!("Модель вернула более 4 действий за один ответ — нужен свежий скриншот"); }
    }
    if actions.is_empty() {
        // Qwen3.5/Holo: heredoc-формат `<function=имя><parameter=ключ>…`.
        if let Ok(batch) = parse_heredoc_tool_calls(trimmed) {
            return Ok(batch);
        }
        if let Some(e) = json_error {
            bail!("Некорректный JSON внутри <tool_call>: {e}");
        }
        // Some model-serving paths return the tool-call JSON without XML tags.
        if let Ok(call) = serde_json::from_str::<ModelToolCall>(trimmed) {
            actions.push(tool_call_to_action(call)?);
        } else {
            bail!("В ответе модели не найдено допустимых действий");
        }
    }
    Ok(GuiActionBatch { actions })
}

/// Один heredoc-вызов: `<function=имя>` + `<parameter=ключ>значение</parameter>`.
#[derive(Debug, PartialEq)]
struct HeredocCall {
    name: String,
    params: Vec<(String, String)>,
}

/// Разбирает блоки `<tool_call><function=имя>…</function></tool_call>` —
/// официальный формат вызовов инструментов Qwen3.5/Holo. Значения параметров
/// могут быть многострочными; числа и булевы приводятся к JSON-типам.
pub fn parse_heredoc_tool_calls(raw: &str) -> Result<GuiActionBatch> {
    let calls = heredoc_calls(raw)?;
    if calls.is_empty() {
        bail!("heredoc-вызовов не найдено");
    }
    if calls.len() > 12 {
        bail!("Модель вернула более 4 действий за один ответ — нужен свежий скриншот");
    }
    let mut actions = Vec::with_capacity(calls.len());
    for call in calls {
        let mut object = serde_json::Map::new();
        for (key, value) in call.params {
            let json_value = if value == "true" {
                serde_json::Value::Bool(true)
            } else if value == "false" {
                serde_json::Value::Bool(false)
            } else if let Ok(n) = value.parse::<f64>()
                && value.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-')
                && value.chars().filter(|c| *c == '.').count() <= 1
            {
                serde_json::Value::Number(serde_json::Number::from_f64(n)
                    .unwrap_or_else(|| serde_json::Number::from(0)))
            } else {
                serde_json::Value::String(value)
            };
            object.insert(key, json_value);
        }
        actions.push(tool_call_to_action(ModelToolCall { name: call.name, arguments: Value::Object(object) })?);
    }
    Ok(GuiActionBatch { actions })
}

fn heredoc_calls(raw: &str) -> Result<Vec<HeredocCall>> {
    let mut calls = Vec::new();
    let mut rest = raw;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let Some(end) = rest.find("</tool_call>") else { break; };
        let body = rest[..end].trim();
        rest = &rest[end + "</tool_call>".len()..];
        let Some(fn_part) = body.strip_prefix("<function=") else { continue };
        let Some(fn_end) = fn_part.find('>') else { continue; };
        let name = fn_part[..fn_end].trim().to_string();
        let mut params = Vec::new();
        let mut cursor = &fn_part[fn_end + 1..];
        while let Some(p_start) = cursor.find("<parameter=") {
            cursor = &cursor[p_start + "<parameter=".len()..];
            let Some(gt) = cursor.find('>') else { break; };
            let key = cursor[..gt].trim().to_string();
            cursor = &cursor[gt + 1..];
            let Some(p_end) = cursor.find("</parameter>") else { break; };
            let value = cursor[..p_end].trim_end_matches('\n').trim().to_string();
            cursor = &cursor[p_end + "</parameter>".len()..];
            params.push((key, value));
        }
        calls.push(HeredocCall { name, params });
    }
    Ok(calls)
}

fn tool_call_to_action(call: ModelToolCall) -> Result<GuiAction> {
    let action_name = match call.name.as_str() {
        "click" => "click",
        "type_text" | "type" => "type",
        "press_key" | "key" => "key",
        "hotkey" => "hotkey",
        "scroll" => "scroll",
        "move_mouse" | "move" => "move",
        "launch" | "open" | "visit" => "launch",
        "wait" => "wait",
        "finish" | "done" => "done",
        other => bail!("Неизвестный GUI-инструмент: {other}"),
    };
    let mut args = call.arguments;
    // Модели иногда отдают аргументы строкой с JSON внутри — разворачиваем.
    if let Value::String(inner) = &args
        && let Ok(parsed) = serde_json::from_str::<Value>(inner)
    {
        args = parsed;
    }
    let object = match args.as_object_mut() {
        Some(object) => object,
        None => bail!("Аргументы {} не объект: {args}", call.name),
    };
    object.insert("action".into(), Value::String(action_name.into()));
    serde_json::from_value(args).with_context(|| {
        format!("Аргументы GUI-инструмента {} не соответствуют схеме", call.name)
    })
}

/// Tool definitions are passed through the chat-template `tools` variable.
/// The model can only request these allow-listed desktop actions; coordinates
/// follow the Holo-3.1 grounding convention (integers 0..1000).
pub fn tool_definitions() -> Value {
    serde_json::json!([
        {"type":"function","function":{"name":"launch","description":"Open an application by name or a URL in the default browser — the reliable way to start a task (do not click the Dock).","parameters":{"type":"object","properties":{"app":{"type":"string","description":"Application name, e.g. Safari"},"url":{"type":"string","description":"URL to open, e.g. https://www.aviasales.ru"}}}}},
        {"type":"function","function":{"name":"click","description":"Click a point in the current screenshot. x and y are integers from 0 to 1000 relative to the screenshot (0,0 top-left, 1000,1000 bottom-right).","parameters":{"type":"object","properties":{"x":{"type":"integer","minimum":0,"maximum":1000},"y":{"type":"integer","minimum":0,"maximum":1000},"button":{"type":"string","enum":["left","right","middle"]},"clicks":{"type":"integer","minimum":1,"maximum":3}},"required":["x","y"]}}},
        {"type":"function","function":{"name":"type_text","description":"Type text into the focused control.","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}},
        {"type":"function","function":{"name":"press_key","description":"Press one non-modifier keyboard key.","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}},
        {"type":"function","function":{"name":"hotkey","description":"Press a keyboard shortcut with modifiers and one main key.","parameters":{"type":"object","properties":{"keys":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":6}},"required":["keys"]}}},
        {"type":"function","function":{"name":"scroll","description":"Scroll the current view.","parameters":{"type":"object","properties":{"direction":{"type":"string","enum":["up","down","left","right"]},"amount":{"type":"integer","minimum":1,"maximum":50}},"required":["direction"]}}},
        {"type":"function","function":{"name":"move_mouse","description":"Move the pointer to a screenshot coordinate (0..1000).","parameters":{"type":"object","properties":{"x":{"type":"integer","minimum":0,"maximum":1000},"y":{"type":"integer","minimum":0,"maximum":1000}},"required":["x","y"]}}},
        {"type":"function","function":{"name":"wait","description":"Wait for the UI to settle, at most 30 seconds.","parameters":{"type":"object","properties":{"milliseconds":{"type":"integer","minimum":0,"maximum":30000}}}}},
        {"type":"function","function":{"name":"finish","description":"Finish the task and report the result.","parameters":{"type":"object","properties":{"message":{"type":"string"}}}}}
    ])
}

/// Stable system instructions for desktop tasks: Holo-3.1 вызывает
/// инструменты JSON-вызовами `<tool_call>`, координаты 0..1000.
pub fn system_prompt() -> &'static str {
    r#"You are the Jarvis desktop GUI agent. You receive a screenshot and one user task. Inspect the screenshot before acting and use only the provided GUI tools. Coordinates x and y are integers from 0 to 1000 relative to the screenshot: top-left is (0,0), bottom-right is (1000,1000). Never invent coordinates for controls you cannot see. Prefer one or two actions per reply, then let the system take a fresh screenshot before continuing. After typing a URL or search query, press Enter (press_key) — typing alone does not navigate. When a later screenshot shows the goal is reached, call finish immediately with a concise factual result; do not keep repeating the same action. Do not claim an action succeeded until a later screenshot confirms the result. Never use shell, scripts, terminal commands, or hidden APIs. If the target is ambiguous or an action could destroy data, call finish and say what needs clarification."#
}

/// Executes a validated batch in order. A failed operation stops the batch;
/// wait is capped and INTERRUPTIBLE by `cancel` (пп. 50-51) — длинные паузы
/// не блокируют отмену задачи. `logical` — размер рабочего стола в точках
/// для retina-масштаба (пп. 35).
pub fn execute_batch(
    batch: &GuiActionBatch,
    width: u32,
    height: u32,
    logical: Option<(u32, u32)>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<String>> {
    use std::sync::atomic::Ordering;
    let cancelled = |c: Option<&std::sync::atomic::AtomicBool>| {
        c.is_some_and(|f| f.load(Ordering::Relaxed))
    };
    let mut log = Vec::new();
    for action in &batch.actions {
        if cancelled(cancel) {
            bail!("Остановлено пользователем");
        }
        match action {
            GuiAction::Wait { milliseconds } => {
                let ms = milliseconds.unwrap_or(500).min(30_000);
                // Прерываемое ожидание: срезы по 200 мс с проверкой отмены.
                let mut waited = 0u64;
                while waited < ms {
                    if cancelled(cancel) {
                        bail!("Остановлено пользователем");
                    }
                    let slice = 200u64.min(ms - waited);
                    std::thread::sleep(Duration::from_millis(slice));
                    waited += slice;
                }
                log.push(format!("Ожидание {ms} мс"));
            }
            GuiAction::Launch { app, url } => {
                let opened = launch(app.as_deref(), url.as_deref())
                    .map_err(|e| anyhow::anyhow!("Запуск не выполнен: {e}"))?;
                log.push(opened);
            }
            GuiAction::Done { message } => {
                log.push(message.clone().unwrap_or_else(|| "Задача завершена по сообщению модели".into()));
                break;
            }
            _ => {
                let one = GuiActionBatch { actions: vec![action.clone()] };
                let mut desktop = to_desktop_actions(&one, width, height, logical)?;
                let desktop_action = desktop.pop().context("Действие не преобразовалось в desktop-команду")?;
                let result = crate::platform::desktop::execute(&desktop_action)
                    .map_err(|e| anyhow::anyhow!("Действие {:?} не выполнено: {e}", action))?;
                log.push(result);
            }
        }
    }
    Ok(log)
}

/// Внутренний адаптер запуска (пп. 25): системный `open -a app` / `open url`.
fn launch(app: Option<&str>, url: Option<&str>) -> Result<String> {
    let mut command = std::process::Command::new("open");
    match (app, url) {
        (Some(app), _) => {
            command.args(["-a", app]);
            Ok::<String, anyhow::Error>(format!("Запущено приложение {app}"))
        }
        (None, Some(url)) => {
            command.arg(url);
            Ok(format!("Открыт адрес {url}"))
        }
        _ => bail!("launch требует app или url"),
    };
    let status = command
        .status()
        .map_err(|e| anyhow::anyhow!("не удалось запустить open: {e}"))?;
    if !status.success() {
        bail!("open завершился с кодом {status}");
    }
    Ok(String::new()) // текст лога уже сформирован выше
}

/// Координата 0..1000 (конвенция grounding Holo-3.1) → точка экрана:
/// пиксель скриншота × retina-масштаб (пп. 35).
fn pixel(v: f32, extent: u32, scale: f32) -> Result<i32> {
    if !v.is_finite() || !(0.0..=1000.0).contains(&v) {
        bail!("Координата должна быть в диапазоне 0..1000, получено {v}");
    }
    let px = (v / 1000.0 * extent.saturating_sub(1) as f32).round();
    Ok((px * scale).round() as i32)
}
fn parse_button(v: Option<&str>) -> Result<MouseButton> {
    match v.unwrap_or("left") { "left" => Ok(MouseButton::Left), "right" => Ok(MouseButton::Right), "middle" => Ok(MouseButton::Middle), x => bail!("Неизвестная кнопка мыши: {x}") }
}
fn parse_direction(v: &str) -> Result<ScrollDirection> {
    match v { "up" => Ok(ScrollDirection::Up), "down" => Ok(ScrollDirection::Down), "left" => Ok(ScrollDirection::Left), "right" => Ok(ScrollDirection::Right), x => bail!("Неизвестное направление прокрутки: {x}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grounding_click_maps_to_screen_pixels() {
        let batch = parse_actions(r#"{"actions":[{"action":"click","x":500,"y":250}]}"#).unwrap();
        let actions = to_desktop_actions(&batch, 1920, 1080, None).unwrap();
        assert_eq!(actions, vec![DesktopAction::Click { x: 960, y: 270, button: MouseButton::Left, count: 1 }]);
    }
    #[test]
    fn parses_official_chat_template_tool_call_envelope() {
        let batch = parse_model_output(r#"<tool_call>{"name":"click","arguments":{"x":0.2,"y":0.8}}</tool_call>"#).unwrap();
        assert_eq!(batch.actions.len(), 1);
        assert!(matches!(batch.actions[0], GuiAction::Click { .. }));
    }

    #[test]
    fn rejects_unlisted_model_tool() {
        assert!(parse_model_output(r#"<tool_call>{"name":"shell","arguments":{"cmd":"rm -rf /"}}</tool_call>"#).is_err());
    }

    #[test]
    fn parses_heredoc_tool_calls_with_numeric_args() {
        let raw = "<tool_call>\n<function=click>\n<parameter=x>\n500\n</parameter>\n<parameter=y>\n250\n</parameter>\n</function>\n</tool_call>";
        let batch = parse_model_output(raw).unwrap();
        assert!(matches!(batch.actions[0], GuiAction::Click { x, y, .. } if (x - 500.0).abs() < 1e-6 && (y - 250.0).abs() < 1e-6));
    }

    #[test]
    fn heredoc_multiline_text_and_rejects_unknown_tool() {
        let raw = "<tool_call>\n<function=type_text>\n<parameter=text>\nпервая строка\nвторая\n</parameter>\n</function>\n</tool_call>";
        let batch = parse_model_output(raw).unwrap();
        assert!(matches!(&batch.actions[0], GuiAction::Type { text } if text.contains("вторая")));
        assert!(parse_model_output("<tool_call>\n<function=shell>\n<parameter=cmd>\nrm -rf /\n</parameter>\n</function>\n</tool_call>").is_err());
    }

    #[test]
    fn out_of_range_coordinates_are_rejected() {
        let batch = parse_actions(r#"{"actions":[{"action":"click","x":1200,"y":500}]}"#).unwrap();
        assert!(to_desktop_actions(&batch, 100, 100, None).is_err());
    }
}
