//! Автоматизация экрана без модели зрения.
//!
//! Классический стек macOS, работающий десятилетиями: **Accessibility
//! (System Events) + AppleScript**. Вместо скриншота и нейросети-зрения
//! основная LLM читает экран текстом (приложение, окна, кнопки, поля) и
//! действует по именам элементов: клик по кнопке «Продолжить», ввод в поле
//! «Поиск», пункт меню «Файл → Сохранить». Результат каждого действия
//! перечитывается из системы — честная самопроверка вместо обещаний.
//!
//! Инструменты модели: [`ui_read`], [`ui_click`], [`ui_type`], [`ui_menu`]
//! (реализация в `tools.rs`, здесь — системные примитивы).
//!
//! Отмена: голосовое «стоп» вызывает [`request_cancel`]; каждый инструмент
//! проверяет флаг между шагами и останавливается, не трогая экран дальше.

use anyhow::{anyhow, Result};

// Accessibility-автоматизация через System Events — стек macOS; на Linux и
// Windows код не компилируется (зависит от platform::macos), поэтому здесь
// он под гейтом, а вызывающий код получает честную причину невозможности.
#[cfg(target_os = "macos")]
pub mod ax;

#[cfg(target_os = "macos")]
pub use ax::{click_element, menu_path, read_screen, set_field_value, ScreenDump};

#[cfg(not(target_os = "macos"))]
fn ui_unavailable() -> anyhow::Error {
    anyhow!(
        "Работа с элементами интерфейса (Accessibility) поддерживается только на macOS"
    )
}

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScreenDump {
    pub app: String,
    pub report: String,
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
pub fn read_screen() -> Result<ScreenDump> {
    Err(ui_unavailable())
}

#[cfg(not(target_os = "macos"))]
pub fn click_element(
    _app: Option<&str>,
    _role: &str,
    _name: &str,
    _window: Option<&str>,
) -> Result<String> {
    Err(ui_unavailable())
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
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
/// Accessibility-дерево активного приложения.
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
}
