//! Журнал: dev — терминал, prod — только интерфейс.
//!
//! Голосовой и модельный стек пишет через [`info`]/[`warn`]/[`error`] вместо
//! прямых `println!`/`eprintln!`. Правила:
//!
//! * приложение (`app`) ставит приёмник ([`set_sink`]) — сообщения уходят в
//!   GUI как события `Log`; в терминал они дублируются только в dev-сборке
//!   (debug-сборка или `JARVIS_DEV=1`);
//! * CLI и примеры — терминальные инструменты: приёмник не ставят, печать
//!   идёт всегда, независимо от сборки.

use std::fmt::Display;
use std::sync::OnceLock;

/// Важность сообщения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

type Sink = Box<dyn Fn(Level, &str) + Send + Sync>;

static SINK: OnceLock<Sink> = OnceLock::new();

/// Поставить приёмник журнала (интерфейс). Срабатывает один раз за процесс.
pub fn set_sink(sink: Sink) {
    let _ = SINK.set(sink);
}

/// Терминальный вывод включён только в dev: debug-сборка или `JARVIS_DEV=1`.
pub fn dev_terminal() -> bool {
    cfg!(debug_assertions) || std::env::var_os("JARVIS_DEV").is_some_and(|v| v == "1")
}

/// Довести сообщение до приёмника (GUI) и, в dev, до терминала.
pub fn log(level: Level, message: impl Display) {
    let text = message.to_string();
    if let Some(sink) = SINK.get() {
        sink(level, &text);
    }
    // Без приёмника (CLI/примеры) печатаем всегда — это терминальные
    // инструменты. С приёмником (приложение) — только в dev.
    if SINK.get().is_none() || dev_terminal() {
        match level {
            Level::Info => println!("{text}"),
            Level::Warn => eprintln!("{text}"),
            Level::Error => eprintln!("{text}"),
        }
    }
}

/// Информационное сообщение журнала.
pub fn info(message: impl Display) {
    log(Level::Info, message);
}

/// Предупреждение журнала.
pub fn warn(message: impl Display) {
    log(Level::Warn, message);
}

/// Ошибка журнала.
pub fn error(message: impl Display) {
    log(Level::Error, message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dev_flag() {
        // Флаг не задан в тестовой среде — решает тип сборки.
        let dev = dev_terminal();
        assert_eq!(dev, cfg!(debug_assertions));
    }
}
