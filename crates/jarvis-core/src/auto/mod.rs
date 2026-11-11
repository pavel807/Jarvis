//! Автоматизация экрана без модели зрения.
//!
//! Один контракт — системные примитивы каждой ОС:
//! - macOS: **Accessibility (System Events) + AppleScript** (`ax.rs`);
//! - Linux: **AT-SPI2** (`atspi_linux.rs`) — дерево доступности GTK/Qt
//!   через шину at-spi2-core, работает и на X11, и на Wayland.
//!
//! Вместо скриншота и нейросети-зрения основная LLM читает экран текстом
//! (приложение, окна, кнопки, поля) и действует по именам элементов: клик
//! по кнопке «Продолжить», ввод в поле «Поиск», пункт меню «Файл →
//! Сохранить». Результат каждого действия перечитывается из системы —
//! честная самопроверка вместо обещаний.
//!
//! Инструменты модели: [`ui_read`], [`ui_click`], [`ui_type`], [`ui_menu`]
//! (реализация в `tools.rs`, здесь — системные примитивы).
//!
//! Отмена: голосовое «стоп» вызывает [`request_cancel`]; каждый инструмент
//! проверяет флаг между шагами и останавливается, не трогая экран дальше.

use anyhow::Result;

/// Роли, которые перечисляет `ui_read` и принимает `ui_click`/`ui_type`.
/// Словарь один на все ОС: Linux переводит классы в свои роли AT-SPI
/// (`atspi_linux::atspi_roles`).
pub const ELEMENT_ROLES: &[&str] = &[
    "button",
    "checkbox",
    "radio button",
    "pop up button",
    "text field",
    "secure text field",
    "static text",
    "link",
    "slider",
];

/// Канонизация роли: русские и частые синонимы → класс словаря.
pub fn normalize_role(role: &str) -> String {
    match role.trim().to_lowercase().as_str() {
        "кнопка" | "кнопку" | "" => "button".into(),
        "флажок" | "чекбокс" | "галочка" => "checkbox".into(),
        "переключатель" => "radio button".into(),
        "список" | "выпадающий список" | "выпадашка" => "pop up button".into(),
        "поле" | "поле ввода" | "текстовое поле" => "text field".into(),
        "пароль" | "поле пароля" => "secure text field".into(),
        "надпись" | "текст" => "static text".into(),
        "ссылка" => "link".into(),
        "ползунок" => "slider".into(),
        "меню" | "пункт меню" | "menu bar item" => "menu bar item".into(),
        other => other.to_string(),
    }
}

#[cfg(target_os = "macos")]
pub mod ax;

#[cfg(target_os = "macos")]
pub use ax::{click_element, menu_path, read_screen, set_field_value, ScreenDump};

// Linux: полная реализация через AT-SPI2 (read/click/type/menu).
#[cfg(target_os = "linux")]
pub mod atspi_linux;

#[cfg(target_os = "linux")]
pub use atspi_linux::{
    click_element, describe_app, menu_path, read_screen, set_field_value, ScreenDump,
};

// Остальные ОС: честная причина невозможности — реализация не написана.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn ui_unavailable() -> anyhow::Error {
    anyhow::anyhow!(
        "Работа с элементами интерфейса (Accessibility) поддерживается на macOS и Linux"
    )
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScreenDump {
    pub app: String,
    pub report: String,
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
impl ScreenDump {
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn read_screen() -> Result<ScreenDump> {
    Err(ui_unavailable())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn click_element(
    _app: Option<&str>,
    _role: &str,
    _name: &str,
    _window: Option<&str>,
) -> Result<String> {
    Err(ui_unavailable())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn set_field_value(
    _app: Option<&str>,
    _role: &str,
    _name: &str,
    _text: &str,
    _submit: bool,
    _window: Option<&str>,
) -> Result<String> {
    Err(ui_unavailable())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn menu_path(_app: Option<&str>, _path: &[&str]) -> Result<String> {
    Err(ui_unavailable())
}

/// Флаг отмены автоматизации: выставляется голосовым «стоп».
static CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Запросить остановку автоматизации (голосовая команда «стоп»).
pub fn request_cancel() {
    CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Снять флаг отмены: новый запрос пользователя начинает новую задачу.
pub fn clear_cancel() {
    CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
}

/// Автоматизация остановлена голосом?
pub fn is_cancelled() -> bool {
    CANCEL.load(std::sync::atomic::Ordering::SeqCst)
}

/// Человекочитаемое описание экрана для ответа «что на экране» и для
/// модели перед действием. Никаких скриншотов и нейросетей: только
/// дерево доступности активного приложения.
pub fn describe_screen() -> Result<String> {
    let dump = read_screen()?;
    Ok(dump.to_text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_flag_roundtrip() {
        clear_cancel();
        assert!(!is_cancelled());
        request_cancel();
        assert!(is_cancelled());
        clear_cancel();
        assert!(!is_cancelled());
    }

    #[test]
    fn normalize_role_maps_russian_and_synonyms() {
        assert_eq!(normalize_role("Кнопка"), "button");
        assert_eq!(normalize_role("чекбокс"), "checkbox");
        assert_eq!(normalize_role("поле ввода"), "text field");
        assert_eq!(normalize_role(""), "button");
        assert_eq!(normalize_role("button"), "button");
        assert_eq!(normalize_role("пункт меню"), "menu bar item");
    }
}
