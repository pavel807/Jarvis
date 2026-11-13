//! Прочие ОС (BSD и т.п.): честное «не поддерживается» вместо паники или
//! тихой лжи. Всё, что не завязано на систему, продолжает работать: wake
//! word, STT, LLM, файлы, веб — они платформенно-нейтральны.

pub fn alias(_word: &str) -> Option<&'static str> {
    None
}

pub fn app_running(_name: &str) -> bool {
    false
}

pub fn app_resolve(name: &str) -> String {
    name.to_string()
}

pub fn app_lookup(_name: &str) -> Option<String> {
    None
}

pub fn app_open(_name: &str) -> Result<String, String> {
    Err("Управление приложениями поддерживается на macOS, Linux и Windows".to_string())
}

pub fn app_quit(_name: &str) -> bool {
    false
}

pub fn window_control(_action: &str, _app: Option<&str>) -> String {
    "Управление окнами поддерживается на macOS, Linux и Windows".to_string()
}

pub fn window_control_checked(
    action: &str,
    app: Option<&str>,
) -> crate::platform::WindowControlOutcome {
    crate::platform::WindowControlOutcome::Done(window_control(action, app))
}

pub fn window_control_target(action: &str, app: &str, _title: &str) -> String {
    window_control(action, Some(app))
}

pub fn music_control(_action: &str) -> Result<String, String> {
    Err("Управление музыкой поддерживается на macOS, Linux и Windows".to_string())
}

pub fn set_volume(_percent: u8) -> bool {
    false
}

pub fn set_brightness(_percent: u8) -> bool {
    false
}

pub fn screenshot() -> Option<String> {
    None
}

pub fn lock_screen() -> bool {
    false
}

pub fn sleep_now() -> bool {
    false
}

pub fn notify(_title: &str, _body: &str) -> bool {
    false
}

pub fn clipboard_read() -> Option<String> {
    None
}

pub fn clipboard_write(_text: &str) -> bool {
    false
}

pub fn system_info() -> String {
    "Информация недоступна".to_string()
}

pub fn send_message(_to: &str, _text: &str) -> Result<String, String> {
    Err("Отправка сообщений поддерживается только на macOS".to_string())
}

pub fn add_note(_title: &str, _body: &str) -> Result<String, String> {
    Err("Заметки поддерживаются только на macOS".to_string())
}

pub fn create_event(_title: &str, _date: &str, _time: &str) -> Result<String, String> {
    Err("Календарь поддерживается только на macOS".to_string())
}

pub fn active_window() -> Option<(String, String)> {
    None
}

pub fn music_status() -> Option<String> {
    None
}

/// Записи базы приложений: на этой платформе пока каталог не собирается.
pub fn app_db_entries() -> Vec<(String, String, String)> {
    Vec::new()
}

pub fn app_catalog() -> Vec<String> {
    Vec::new()
}
