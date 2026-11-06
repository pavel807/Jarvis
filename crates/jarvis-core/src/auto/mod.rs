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

use anyhow::Result;

pub mod ax;

pub use ax::{click_element, menu_path, read_screen, set_field_value, ScreenDump};

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
