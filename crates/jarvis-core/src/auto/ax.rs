//! Accessibility-запросы к UI-элементам macOS через System Events.
//!
//! Дерево читается **глубоко**: обход рекурсивный, а не по прямым детям
//! окна. Это главное отличие от прежней реализации: у Electron-приложений,
//! браузеров и Office кнопки лежат на глубине 3–6, и плоский запрос
//! `every button of window 1` возвращал пустоту — агент видел только
//! заголовок окна и не мог ни на что нажать.
//!
//! Чтение и действие разведены:
//! - **чтение** (`collect`) обходит дерево и отдаёт плоский список элементов;
//!   решение «какой элемент имеется в виду» принимает Rust (`elements.rs`);
//! - **действие** (`act_on`) идёт по адресу элемента (роль ОС, имя,
//!   описание, номер повтора) и повторяет тот же обход, поэтому находит
//!   именно тот элемент, который был показан.
//!
//! Так модель не обязана угадывать точное имя: она говорит «кликни 12» или
//! «кликни кнопку “Настройки”», а нечёткое совпадение делает остальное.

use anyhow::{anyhow, Result};

use super::elements::{Element, ScreenSnapshot};
use super::{normalize_role, ELEMENT_ROLES};

/// Снимок экрана: активное приложение и элементы управления.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScreenDump {
    /// Имя активного процесса (может быть пустым).
    pub app: String,
    /// Текстовый отчёт, собранный обходом дерева (уже человекочитаемый).
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
    // Отладочный дамп: полезно, когда скрипт не компилируется или ведёт себя
    // не так, как ожидалось. Каталог задаётся переменной окружения.
    if let Ok(dir) = std::env::var("JARVIS_DUMP_AX_SCRIPT")
        && !dir.trim().is_empty()
    {
        let path = std::path::Path::new(&dir).join("jarvis_ax.applescript");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&path, script);
        eprintln!("💾 скрипт AX сохранён: {}", path.display());
    }
    crate::platform::macos::osascript(script)
        .map_err(|e| anyhow!(crate::platform::macos::friendly_access_error(&e)))
}

// =========================================================================
// Общие куски AppleScript
// =========================================================================

/// Скрипт сбора дерева элементов.
///
/// Каждая строка результата — один элемент, поля разделены табуляцией:
/// `E <глубина> <роль ОС> <имя> <описание> <значение> <номер окна>`.
/// Строка окна: `W <номер окна> <заголовок>`.
///
/// Правила чтения (обязаны совпадать в действии, иначе элемент найдётся не с
/// тем описанием):
/// - имя читается всегда, кроме контейнеров — у них имени нет вовсе, а
///   запрос стоит времени (контейнеров в дереве большинство);
/// - описание — только если имя пустое;
/// - значение — только у текстовых элементов.
///
/// `using terms from` обязателен: обработчики теряют терминологию System
/// Events, и без неё `UI elements` не распознаётся — скрипт не компилируется.
const COLLECT_BODY: &str = r#"
on safeStr(v)
	if v is missing value then return ""
	try
		set t to (v as text)
	on error
		return ""
	end try
	set t to my repAll(t, tab, " ")
	set t to my repAll(t, return, " ")
	set t to my repAll(t, linefeed, " ")
	return t
end safeStr

on repAll(t, f, r)
	set AppleScript's text item delimiters to f
	set parts to (t as text)
	set AppleScript's text item delimiters to r
	set out to (parts as text)
	set AppleScript's text item delimiters to ""
	return out
end repAll

on isContainer(rn)
	return rn is "AXGroup" or rn is "AXSplitGroup" or rn is "AXSplitter" or rn is "AXScrollArea" or rn is "AXToolbar" or rn is "AXWindow" or rn is "AXTabGroup" or rn is "AXApplication" or rn is "AXRow" or rn is "AXCell" or rn is "AXTable" or rn is "AXOutline" or rn is "AXLayoutArea" or rn is "AXCollectionView" or rn is "AXBrowser" or rn is "AXSheet"
end isContainer

-- Добавление строки в результат. Вызывается через `my`: внутри tell-блока
-- присваивание `gOut` обращалось бы к свойству приложения, а не к переменной.
on appendOut(t)
	global gOut
	set gOut to gOut & t & linefeed
end appendOut

on walkNode(el, depth, winIdx)
	global gOut
	global gCount
	using terms from application "System Events"
		if depth > gMaxDepth then return
		if gCount >= gMaxNodes then return
		set gCount to gCount + 1
		set rn to ""
		try
			set rn to (role of el) as text
		end try
		if rn is not "" then
			set nm to ""
			set ds to ""
			set vl to ""
			if not (my isContainer(rn)) then
				try
					set nm to my safeStr(name of el)
				end try
				if nm is "" then
					try
						set ds to my safeStr(description of el)
					end try
				end if
				if rn contains "TextField" or rn contains "TextArea" or rn contains "ComboBox" or rn contains "PopUpButton" then
					try
						set vl to my safeStr(value of el)
					end try
				end if
			end if
			set gOut to gOut & "E" & tab & depth & tab & rn & tab & nm & tab & ds & tab & vl & tab & winIdx & linefeed
		end if
		try
			set kids to UI elements of el
			repeat with k in kids
				if gCount >= gMaxNodes then exit repeat
				my walkNode(k, depth + 1, winIdx)
			end repeat
		end try
	end using terms from
end walkNode
"#;

/// Скрипт действия: тот же обход, но по найденному адресу — действие.
const ACTION_BODY: &str = r#"
on safeStr(v)
	if v is missing value then return ""
	try
		set t to (v as text)
	on error
		return ""
	end try
	set t to my repAll(t, tab, " ")
	set t to my repAll(t, return, " ")
	set t to my repAll(t, linefeed, " ")
	return t
end safeStr

on repAll(t, f, r)
	set AppleScript's text item delimiters to f
	set parts to (t as text)
	set AppleScript's text item delimiters to r
	set out to (parts as text)
	set AppleScript's text item delimiters to ""
	return out
end repAll

on performOn(el, mode, tText)
	using terms from application "System Events"
		if mode is "press" then
			try
				perform action "AXPress" of el
				return "OK"
			on error
				try
					click el
					return "OK"
				on error errMsg
					return "ERR " & errMsg
				end try
			end try
		else
			try
				set focused of el to true
			end try
			try
				set value of el to tText
				return "VALUE " & (my safeStr(value of el))
			on error errMsg
				return "ERR " & errMsg
			end try
		end if
	end using terms from
end performOn

on findAndAct(el, depth)
	global gCount
	global gFound
	using terms from application "System Events"
		if depth > gMaxDepth then return "NOTFOUND"
		if gCount >= gMaxNodes then return "NOTFOUND"
		set gCount to gCount + 1
		set rn to ""
		try
			set rn to (role of el) as text
		end try
		if rn is "{RAW_ROLE}" then
			set nm to ""
			set ds to ""
			try
				set nm to my safeStr(name of el)
			end try
			if nm is "" then
				try
					set ds to my safeStr(description of el)
				end try
			end if
			if nm is "{NAME}" and ds is "{DESC}" then
				if gFound = {OCCURRENCE} then
					return my performOn(el, "{MODE}", "{TEXT}")
				end if
				set gFound to gFound + 1
			end if
		end if
		try
			set kids to UI elements of el
			repeat with k in kids
				if gCount >= gMaxNodes then exit repeat
				set r to my findAndAct(k, depth + 1)
				if r is not "NOTFOUND" then return r
			end repeat
		end try
	end using terms from
	return "NOTFOUND"
end findAndAct
"#;

/// Сколько узлов дерева обходить и на какую глубину.
const MAX_NODES: usize = 400;
const MAX_DEPTH: u32 = 8;
/// Сколько окон читать.
const MAX_WINDOWS: usize = 2;

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

/// Перевод роли AX в общий словарь ролей инструментов.
pub fn ax_role(raw: &str) -> &'static str {
    match raw {
        "AXButton" => "button",
        "AXTextField" | "AXTextArea" => "text field",
        "AXSecureTextField" => "secure text field",
        "AXCheckBox" => "checkbox",
        "AXRadioButton" => "radio button",
        "AXPopUpButton" | "AXComboBox" => "pop up button",
        "AXMenuBarItem" | "AXMenuItem" => "menu bar item",
        "AXLink" => "link",
        "AXSlider" | "AXIncrementor" => "slider",
        "AXStaticText" | "AXText" => "static text",
        "AXCell" | "AXRow" | "AXOutlineRow" => "list item",
        "AXTabButton" => "tab",
        "AXGroup" | "AXScrollArea" | "AXTable" | "AXOutline" | "AXToolbar" | "AXWindow" | "AXSheet"
        | "AXScrollBar" | "AXSplitGroup" | "AXBrowser" | "AXCollectionView" | "AXLayoutArea"
        | "AXApplication" => "",
        _ => "",
    }
}

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

/// Собрать дерево элементов активного (или указанного) приложения.
pub fn collect(app: Option<&str>) -> Result<ScreenSnapshot> {
    let target = resolve_app(app)?;
    let script = format!(
        r#"set gOut to ""
set gCount to 0
set gMaxNodes to {MAX_NODES}
set gMaxDepth to {MAX_DEPTH}
{COLLECT_BODY}
tell application "System Events"
  if not (exists application process "{APP}") then error "приложение не запущено"
  tell application process "{APP}"
    set wCount to count of windows
    if wCount = 0 then return "NOWINDOWS"
    set maxW to wCount
    if maxW > {MAX_WINDOWS} then set maxW to {MAX_WINDOWS}
    repeat with wIdx from 1 to maxW
      set wTitle to ""
      try
        set wTitle to my safeStr(name of window wIdx)
      end try
      my appendOut("W" & tab & wIdx & tab & wTitle)
      try
        my walkNode(window wIdx, 1, wIdx)
      end try
    end repeat
  end tell
end tell
return gOut"#,
        APP = esc(&target),
    );
    let raw = run_ax(&script)?;
    Ok(parse_snapshot(&raw, &target))
}

/// Разбор ответа обхода в список элементов.
pub fn parse_snapshot(raw: &str, app: &str) -> ScreenSnapshot {
    let mut elements: Vec<Element> = Vec::new();
    let mut window = String::new();
    for line in raw.lines() {
        let mut parts = line.split('\t');
        match parts.next() {
            Some("W") => {
                let idx = parts.next().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
                let title = parts.next().unwrap_or("").trim().to_string();
                if idx == 1 {
                    window = title;
                }
            }
            Some("E") => {
                let depth = parts
                    .next()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(1);
                let raw_role = parts.next().unwrap_or("").trim().to_string();
                let name = parts.next().unwrap_or("").trim().to_string();
                let description = parts.next().unwrap_or("").trim().to_string();
                let value = parts.next().unwrap_or("").trim().to_string();
                let win = parts.next().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(1);
                let role = ax_role(&raw_role);
                if role.is_empty() {
                    continue;
                }
                elements.push(Element {
                    id: elements.len(),
                    role: role.to_string(),
                    raw_role,
                    name,
                    description,
                    value,
                    depth,
                    window: win,
                    app: app.to_string(),
                });
            }
            _ => {}
        }
    }
    ScreenSnapshot::new(app, window, elements)
}

/// Снимок экрана: активное приложение и элементы управления.
pub fn read_screen() -> Result<ScreenDump> {
    let snapshot = collect(None)?;
    Ok(ScreenDump {
        app: snapshot.app.clone(),
        report: snapshot.render(),
    })
}

/// Действие по адресу элемента: `press` (клик) или `value` (ввод текста).
///
/// Повтор считается по списку элементов того же снимка: у дерева нет
/// устойчивых идентификаторов, но есть порядок обхода — он и есть адрес.
fn act_on(elements: &[Element], el: &Element, mode: &str, text: &str) -> Result<String> {
    let occurrence = crate::auto::elements::occurrence_of(elements, el);
    // Подстановки — внутрь тела скрипта: там, где живут плейсхолдеры.
    let body = ACTION_BODY
        .replace("{RAW_ROLE}", &esc(&el.raw_role))
        .replace("{NAME}", &esc(&el.name))
        .replace("{DESC}", &esc(&el.description))
        .replace("{OCCURRENCE}", &occurrence.to_string())
        .replace("{MODE}", &esc(mode))
        .replace("{TEXT}", &esc(text));
    let script = format!(
        r#"set gCount to 0
set gFound to 0
set gMaxNodes to {MAX_NODES}
set gMaxDepth to {MAX_DEPTH}
{body}
tell application "System Events"
  if not (exists application process "{APP}") then error "приложение не запущено"
  tell application process "{APP}"
    set frontmost to true
    delay 0.25
    set wCount to count of windows
    if wCount = 0 then return "ERR нет окон"
    repeat with wIdx from 1 to wCount
      if wIdx = {WIN} then
        return my findAndAct(window wIdx, 1)
      end if
    end repeat
  end tell
end tell
return "NOTFOUND""#,
        APP = esc(&el.app),
        WIN = el.window.max(1),
    );
    let out = run_ax(&script)?;
    match out.trim() {
        "OK" => Ok("ok".into()),
        v if v.starts_with("VALUE ") => Ok(v["VALUE ".len()..].trim().to_string()),
        v if v.starts_with("ERR ") => Err(anyhow!("{}", v["ERR ".len()..].trim())),
        "NOTFOUND" => Err(anyhow!("элемент не найден на экране")),
        other => Err(anyhow!("непонятный ответ системы: {other}")),
    }
}

/// Найти элемент по роли и имени (нечётко) и нажать его.
pub fn click_element(
    app: Option<&str>,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let role = normalize_role(role);
    if !ELEMENT_ROLES.contains(&role.as_str()) {
        return Err(anyhow!(
            "роль «{role}» не поддерживается: используй одну из {}",
            ELEMENT_ROLES.join(", ")
        ));
    }
    let snapshot = collect(app)?;
    let target = pick(&snapshot, &role, name, window)?;
    act_on(&snapshot.elements, &target, "press", "")?;
    Ok(format!(
        "нажато: {} «{}»",
        target.role,
        target.label()
    ))
}

/// Выбрать элемент из снимка: сначала точное имя, потом нечёткое.
fn pick(
    snapshot: &ScreenSnapshot,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<Element> {
    let mut candidates: Vec<&Element> = snapshot
        .elements
        .iter()
        .filter(|el| el.role == role)
        .filter(|el| match window {
            Some(w) if !w.trim().is_empty() => el.window.to_string() == w.trim(),
            _ => true,
        })
        .collect();
    // Роль может прийти словом на русском — тогда роль элемента не совпадёт.
    if candidates.is_empty() {
        candidates = snapshot
            .elements
            .iter()
            .filter(|el| match window {
                Some(w) if !w.trim().is_empty() => el.window.to_string() == w.trim(),
                _ => true,
            })
            .collect();
    }
    let want = super::elements::strip_role_words(name);
    let want = if want.trim().is_empty() { name } else { want.as_str() };
    let mut scored: Vec<(u32, &Element)> = candidates
        .iter()
        .map(|el| (super::elements::score_element(el, want), *el))
        .filter(|(score, _)| *score > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
    match scored.first() {
        Some((score, el)) if *score >= 50 => Ok((*el).clone()),
        Some(_) | None => {
            // Не нашли — показываем, что есть, чтобы модель выбрала другое имя.
            let names: Vec<String> = candidates
                .iter()
                .take(8)
                .map(|el| format!("{} «{}»", el.role, el.label()))
                .collect();
            Err(anyhow!(
                "элемент «{name}» не найден. Похожее на экране: {}",
                if names.is_empty() {
                    "ничего похожего нет".to_string()
                } else {
                    names.join(", ")
                }
            ))
        }
    }
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
    let snapshot = collect(app)?;
    let target = pick(&snapshot, &role, name, window)?;
    let actual = act_on(&snapshot.elements, &target, "value", text)?;
    if submit {
        let script = r#"tell application "System Events" to key code 36"#;
        let _ = run_ax(script);
    }
    Ok(actual)
}

/// Прочитать значение элемента после действия (readback для верификации).
pub fn element_value(
    app: Option<&str>,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<String> {
    let role = normalize_role(role);
    let snapshot = collect(app)?;
    let target = pick(&snapshot, &role, name, window)?;
    Ok(target.value.trim().to_string())
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
        assert!(menu_chain(&[]).is_err());
        assert!(menu_chain(&["a", "b", "c", "d"]).is_err());
    }

    #[test]
    fn ax_roles_map_to_common_dictionary() {
        assert_eq!(ax_role("AXButton"), "button");
        assert_eq!(ax_role("AXTextField"), "text field");
        assert_eq!(ax_role("AXStaticText"), "static text");
        assert_eq!(ax_role("AXMenuItem"), "menu bar item");
        assert_eq!(ax_role("AXGroup"), "");
    }

    #[test]
    fn parses_collected_lines_into_elements() {
        let raw = "W\t1\tSafari\n\
                   E\t2\tAXButton\tНастройки\t\t\t1\n\
                   E\t3\tAXTextField\tПоиск\t\tпусто\t1\n\
                   E\t4\tAXGroup\t\t\t\t1\n";
        let snapshot = parse_snapshot(raw, "Safari");
        assert_eq!(snapshot.window, "Safari");
        assert_eq!(snapshot.elements.len(), 2);
        assert_eq!(snapshot.elements[0].role, "button");
        assert_eq!(snapshot.elements[0].name, "Настройки");
        assert_eq!(snapshot.elements[1].role, "text field");
        assert_eq!(snapshot.elements[1].value, "пусто");
        assert_eq!(snapshot.elements[0].id, 0);
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
}
