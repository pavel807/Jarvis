//! Accessibility-запросы к UI-элементам macOS через System Events.
//!
//! Все сценарии — AppleScript-строки, исполняемые существующим движком
//! `crate::applescript` (in-process NSAppleScript с внешним fallback).
//! Элементы ищутся по имени/роли в окнах активного приложения — так же, как
//! это делают Automator и скриптовые бриджи с Mac OS X 10.3.

use anyhow::{anyhow, Result};

// Словарь элементов и канонизация ролей — общие для всех ОС: перечисляет
// ui_read и принимает ui_click/ui_type на macOS и Linux.
use super::{normalize_role, ELEMENT_ROLES};

/// Снимок экрана: активное приложение и элементы управления.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScreenDump {
    /// Имя активного процесса (может быть пустым).
    pub app: String,
    /// Текстовый отчёт, собранный AppleScript'ом (уже человекочитаемый).
    pub report: String,
}

impl ScreenDump {
    /// Текст для пользователя и для модели.
    pub fn to_text(&self) -> String {
        if self.report.trim().is_empty() {
            format!(
                "Активное приложение: {}. Элементов не видно (окно может быть пустым или защищённым).",
                self.app
            )
        } else {
            self.report.trim().to_string()
        }
    }
}

/// Экранирование строки для AppleScript-литерала в двойных кавычках.
pub fn esc(s: &str) -> String {
    crate::platform::macos::applescript_escape(s)
}

/// Выполнить System Events-скрипт с человекопонятной ошибкой доступа.
pub fn run_ax(script: &str) -> Result<String> {
    crate::platform::macos::osascript(script)
        .map_err(|e| anyhow!(crate::platform::macos::friendly_access_error(&e)))
}

// =========================================================================
// Чистые построители (покрыты тестами без обращения к системе)
// =========================================================================

/// Предложение окна для AppleScript: число — индекс, иначе поиск по имени.
pub fn window_clause(window: Option<&str>) -> String {
    match window.map(str::trim).filter(|s| !s.is_empty()) {
        None => "window 1".to_string(),
        Some(w) if w.chars().all(|c| c.is_ascii_digit()) => format!("window {w}"),
        Some(w) => format!("window 1 whose name contains \"{}\"", esc(w)),
    }
}

/// Цепочка меню для AppleScript: `["Файл", "Сохранить"]` →
/// `menu item "Сохранить" of menu 1 of menu bar item "Файл" of menu bar 1`.
pub fn menu_chain(path: &[&str]) -> Result<String> {
    if path.is_empty() || path.len() > 3 {
        return Err(anyhow!(
            "путь меню: 1–3 пункта, получено {}",
            path.len()
        ));
    }
    let mut chain = format!("menu bar item \"{}\" of menu bar 1", esc(path[0]));
    for part in path.iter().skip(1) {
        chain = format!("menu item \"{}\" of menu 1 of {}", esc(part), chain);
    }
    Ok(chain)
}

/// Скрипт клика по элементу окна по имени (name → description → title).
pub fn click_script(target: &str, role: &str, window: &str, name: &str) -> String {
    let name_ax = esc(name);
    format!(
        r#"tell application "System Events"
  if not (exists application process "{t}") then error "приложение не запущено"
  tell application process "{t}"
    set frontmost to true
    delay 0.2
    set matches to every {role} of {window} whose name is "{n}"
    if (count of matches) = 0 then
      set matches to every {role} of {window} whose description is "{n}"
    end if
    if (count of matches) = 0 then
      set matches to every {role} of {window} whose title is "{n}"
    end if
    if (count of matches) = 0 then
      error "{role} «{n}» не найден в {window} приложения {t}"
    end if
    click item 1 of matches
    return "ok"
  end tell
end tell"#,
        t = esc(target),
        role = role,
        window = window,
        n = name_ax,
    )
}

/// Скрипт записи в текстовое поле + readback фактического значения.
pub fn type_script(target: &str, role: &str, window: &str, name: &str, text: &str, submit: bool) -> String {
    let name_ax = esc(name);
    let text_ax = esc(text);
    let submit_part = if submit { "delay 0.1\n    key code 36" } else { "" };
    format!(
        r#"tell application "System Events"
  if not (exists application process "{t}") then error "приложение не запущено"
  tell application process "{t}"
    set frontmost to true
    delay 0.2
    set matches to every {role} of {window} whose name is "{n}"
    if (count of matches) = 0 then
      set matches to every {role} of {window} whose placeholder text is "{n}"
    end if
    if (count of matches) = 0 then
      set matches to every {role} of {window} whose title is "{n}"
    end if
    if (count of matches) = 0 then
      error "{role} «{n}» не найден в {window} приложения {t}"
    end if
    set theField to item 1 of matches
    set focused of theField to true
    set value of theField to "{v}"
    {submit_part}
    return value of theField
  end tell
end tell"#,
        t = esc(target),
        role = role,
        window = window,
        n = name_ax,
        v = text_ax,
        submit_part = submit_part,
    )
}

/// Сценарий снимка экрана: окна и элементы активного приложения.
/// Каждый блок в `try` — AX-деревья приложений неровные, отсутствие узла
/// не должно ронять весь снимок.
pub const READ_SCREEN_SCRIPT: &str = r#"set out to ""
tell application "System Events"
  set procNames to name of every application process whose frontmost is true and background only is false
  if (count of procNames) = 0 then return "активного приложения нет"
  set pName to item 1 of procNames
  tell application process pName
    set wCount to count of windows
    if wCount = 0 then return "Активное приложение: " & pName & " (окон нет)"
    set maxW to wCount
    if maxW > 3 then set maxW to 3
    repeat with wIdx from 1 to maxW
      set wTitle to ""
      try
        set wTitle to name of window wIdx
      end try
      set out to out & "Окно " & wIdx & ": «" & wTitle & "»" & linefeed
      try
        set out to out & "Кнопки: " & joinList(name of every button of window wIdx) & linefeed
      end try
      try
        set out to out & "Флажки: " & joinList(name of every checkbox of window wIdx) & linefeed
      end try
      try
        set out to out & "Списки: " & joinList(name of every pop up button of window wIdx) & linefeed
      end try
      try
        set tfCount to count of text fields of window wIdx
        if tfCount > 0 then
          set fieldLine to ""
          set maxF to tfCount
          if maxF > 8 then set maxF to 8
          repeat with fIdx from 1 to maxF
            set fName to ""
            set fValue to ""
            try
              set fName to name of text field fIdx of window wIdx
            end try
            try
              set fValue to value of text field fIdx of window wIdx
            end try
            if fName is missing value then set fName to ""
            if fValue is missing value then set fValue to ""
            set fieldLine to fieldLine & "поле «" & fName & "» = «" & clip(fValue, 80) & "»"
            if fIdx < maxF then set fieldLine to fieldLine & ", "
          end repeat
          set out to out & "Поля: " & fieldLine & linefeed
        end if
      end try
      try
        set out to out & "Надписи: " & joinList(value of every static text of window wIdx) & linefeed
      end try
    end repeat
    try
      set out to out & "Меню: " & joinList(name of every menu bar item of menu bar 1) & linefeed
    end try
  end tell
end tell
return out

on joinList(itemsList)
  if (count of itemsList) = 0 then return "—"
  set clean to {}
  repeat with anItem in itemsList
    if anItem is not missing value then
      set end of clean to clip(anItem as text, 60)
    end if
  end repeat
  if (count of clean) = 0 then return "—"
  set AppleScript's text item delimiters to ", "
  set joined to (clean as text)
  set AppleScript's text item delimiters to ""
  return joined
end joinList

on clip(txt, maxLen)
  set t to txt as text
  if (length of t) > maxLen then return (text 1 thru maxLen of t) & "…"
  return t
end clip
"#;

// =========================================================================
// Исполнимые примитивы
// =========================================================================

/// Имя активного приложения (frontmost, без фоновых процессов).
pub fn front_app() -> Result<String> {
    let script = r#"tell application "System Events"
  set names to name of every application process whose frontmost is true and background only is false
  if (count of names) = 0 then return ""
  return item 1 of names
end tell"#;
    Ok(run_ax(script)?.trim().to_string())
}

/// Перечислить активное приложение, окна и элементы управления.
pub fn read_screen() -> Result<ScreenDump> {
    let report = run_ax(READ_SCREEN_SCRIPT)?;
    Ok(ScreenDump {
        app: front_app().unwrap_or_default(),
        report,
    })
}

/// Разрешить имя приложения в имя процесса (русские алиасы, транслит).
fn resolve_app(app: Option<&str>) -> Result<String> {
    match app.map(str::trim).filter(|s| !s.is_empty()) {
        Some(name) => Ok(crate::platform::macos::app_resolve(name)),
        None => Ok(front_app()
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .next()
            .unwrap_or_else(|| "Finder".into())),
    }
}

fn ensure_not_cancelled() -> Result<()> {
    if crate::auto::is_cancelled() {
        return Err(anyhow!("автоматизация остановлена голосом"));
    }
    Ok(())
}

/// Клик по элементу окна по имени и роли. Проверка результата (перечитывание
/// состояния) — на вызывающей стороне через [`read_screen`]/[`element_value`].
pub fn click_element(
    app: Option<&str>,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let role = normalize_role(role);
    // Меню-пункты живут не в окне, а в menu bar — отдельный путь.
    if role == "menu bar item" {
        return menu_path(app, &[name]);
    }
    if !ELEMENT_ROLES.contains(&role.as_str()) {
        return Err(anyhow!(
            "роль «{role}» не поддерживается: используй одну из {}",
            ELEMENT_ROLES.join(", ")
        ));
    }
    let target = resolve_app(app)?;
    let win = window_clause(window);
    run_ax(&click_script(&target, &role, &win, name))?;
    Ok(format!("нажато: {role} «{name}» в приложении {target}"))
}

/// Вписать значение в текстовое поле, вернуть фактическое значение поля
/// после записи (честный readback для верификации).
pub fn set_field_value(
    app: Option<&str>,
    role: &str,
    name: &str,
    text: &str,
    submit: bool,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let role = normalize_role(role);
    if role != "text field" && role != "secure text field" {
        return Err(anyhow!(
            "вписывать текст можно только в поле, получена роль «{role}»"
        ));
    }
    let target = resolve_app(app)?;
    let win = window_clause(window);
    let actual = run_ax(&type_script(&target, &role, &win, name, text, submit))?;
    Ok(actual.trim().to_string())
}

/// Прочитать значение элемента после действия (readback для верификации).
pub fn element_value(
    app: Option<&str>,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<String> {
    let role = normalize_role(role);
    let target = resolve_app(app)?;
    let win = window_clause(window);
    let name_ax = esc(name);
    let script = format!(
        r#"tell application "System Events"
  tell application process "{t}"
    set matches to every {role} of {window} whose name is "{n}"
    if (count of matches) = 0 then error "{role} «{n}» не найден"
    return value of item 1 of matches
  end tell
end tell"#,
        t = esc(&target),
        role = role,
        window = win,
        n = name_ax,
    );
    Ok(run_ax(&script)?.trim().to_string())
}

/// Выбрать пункт меню приложения: путь из 1–3 имён
/// (например `["Файл", "Экспортировать", "PDF"]`).
pub fn menu_path(app: Option<&str>, path: &[&str]) -> Result<String> {
    ensure_not_cancelled()?;
    let chain = menu_chain(path)?;
    let target = resolve_app(app)?;
    let script = format!(
        r#"tell application "System Events"
  if not (exists application process "{t}") then error "приложение не запущено"
  tell application process "{t}"
    set frontmost to true
    delay 0.25
    click {chain}
    return "ok"
  end tell
end tell"#,
        t = esc(&target),
        chain = chain,
    );
    run_ax(&script)?;
    Ok(format!(
        "выбран пункт меню «{}» в приложении {target}",
        path.join(" → ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_quotes_and_backslashes() {
        assert_eq!(esc("привет\"мир"), "привет\\\"мир");
        assert_eq!(esc("a\\b"), "a\\\\b");
        assert_eq!(esc("обычный текст"), "обычный текст");
    }

    #[test]
    fn role_aliases_normalize() {
        assert_eq!(normalize_role("Кнопка"), "button");
        assert_eq!(normalize_role("чекбокс"), "checkbox");
        assert_eq!(normalize_role("поле ввода"), "text field");
        assert_eq!(normalize_role(""), "button");
        assert_eq!(normalize_role("button"), "button");
        assert_eq!(normalize_role("пункт меню"), "menu bar item");
    }

    #[test]
    fn window_clause_supports_index_and_name() {
        assert_eq!(window_clause(Some("2")), "window 2");
        assert_eq!(window_clause(None), "window 1");
        assert_eq!(
            window_clause(Some("Настройки")),
            "window 1 whose name contains \"Настройки\""
        );
    }

    #[test]
    fn menu_chain_builds_nested_path() {
        assert_eq!(
            menu_chain(&["Файл"]).unwrap(),
            "menu bar item \"Файл\" of menu bar 1"
        );
        assert_eq!(
            menu_chain(&["Файл", "Сохранить"]).unwrap(),
            "menu item \"Сохранить\" of menu 1 of menu bar item \"Файл\" of menu bar 1"
        );
        assert_eq!(
            menu_chain(&["A", "B", "C"]).unwrap(),
            "menu item \"C\" of menu 1 of menu item \"B\" of menu 1 of menu bar item \"A\" of menu bar 1"
        );
        assert!(menu_chain(&[]).is_err());
        assert!(menu_chain(&["a", "b", "c", "d"]).is_err());
    }

    #[test]
    fn click_rejects_unknown_role() {
        let err = click_element(None, "неведомость", "кнопка", None).unwrap_err();
        assert!(err.to_string().contains("не поддерживается"));
    }

    #[test]
    fn type_rejects_non_field_role() {
        let err = set_field_value(None, "button", "поле", "текст", false, None).unwrap_err();
        assert!(err.to_string().contains("только в поле"));
    }

    #[test]
    fn scripts_embed_escaped_names() {
        let s = click_script("Zed", "button", "window 1", "Ок\"Х");
        assert!(s.contains("\"Ок\\\"Х\""));
        assert!(s.contains("\"Zed\""));
        let t = type_script("Zed", "text field", "window 1", "Поиск", "кавы\"чка", true);
        assert!(t.contains("\"кавы\\\"чка\""));
        assert!(t.contains("key code 36"));
    }
}
