//! Диагностика AT-SPI2 на Linux: печатает активное приложение и дерево
//! элементов через `crate::auto` (тот же путь, что ui_read/ui_click/ui_type).
//! С аргументом — отчёт по названному приложению, даже неактивному:
//! cargo run -p jarvis-core --example atspi_probe [-- имя-приложения]

fn main() {
    #[cfg(target_os = "linux")]
    match std::env::args().nth(1) {
        Some(app) => match jarvis_core::auto::describe_app(&app) {
            Ok(text) => println!("✅ Элементы «{app}»:\n{text}"),
            Err(e) => println!("❌ «{app}» недоступно: {e:#}"),
        },
        None => match jarvis_core::auto::describe_screen() {
            Ok(text) => println!("✅ Снимок экрана:\n{text}"),
            Err(e) => {
                println!("❌ Экран прочитать не удалось: {e:#}");
                println!("Подсказки: установлен ли at-spi2-core? Запущена ли шина org.a11y.Bus?");
            }
        },
    }
    #[cfg(not(target_os = "linux"))]
    println!("ℹ️ Пример только для Linux (AT-SPI2)");
}
