//! Контракты агентного исполнения: статусы, верификация, реестр инструментов.
//!
//! Это единственный источник метаданных об инструментах: канонические имена
//! ([`known_names`]), уровень риска, таймаут и положенный тип постусловия
//! ([`ToolSpec`]). Модель не может вызвать инструмент вне реестра; аргументы
//! проверяются до выполнения (см. `tools::build_call`); результат проверки
//! ([`VerificationResult`]) — основа честного ответа пользователю: успех
//! не объявляется без подтверждённого факта.

use std::time::Duration;

/// Уровень риска операции — определяет необходимость подтверждения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Чтение и локальные безопасные операции.
    Low,
    /// Изменение пользовательских данных, запуск приложений, shell.
    Medium,
    /// Удаление данных, массовые и системные операции.
    High,
}

/// Какой тип постусловия положен инструменту — чем результат подтверждается.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Postcondition {
    /// Проверка процесса/приложения (open_app, quit_app).
    ProcessState,
    /// Код возврата и stderr (shell, applescript).
    ExitCode,
    /// Чтение файла обратно после записи (file_write, file_append, file_edit).
    FileContent,
    /// Проверка существования артефакта (screenshot → файл).
    EvidenceFile,
    /// Чтение фактического состояния через системный API (volume, brightness,
    /// music_control) — эталон честности.
    StateReadback,
    /// Постусловие невозможно: статус остаётся неподтверждённым, и ответ
    /// обязан это отражать (open_url — «браузер запущен», а не «сайт открыт»).
    None,
}

/// Спецификация инструмента: метаданные реестра.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    /// Каноническое имя (совпадает с `ToolCall`-вариантом и событиями).
    pub name: String,
    pub risk: RiskLevel,
    /// Разумный лимит исполнения; зависшие операции прерываются.
    pub timeout: Duration,
    pub postcondition: Postcondition,
    /// Платформы, где инструмент реализован; на остальных — честный
    /// `Unsupported`, а не выдуманный результат.
    pub platforms: &'static [&'static str],
}

mod platforms {
    pub const ALL: &[&str] = &["macos", "linux", "windows"];
    pub const MACOS_ONLY: &[&str] = &["macos"];
}

/// Канонические имена инструментов — единственный список, по которому
/// парсится и валидируется вызов модели (см. `tools::parse`).
pub const KNOWN_NAMES: &[&str] = &[
    "shell",
    "open_app",
    "quit_app",
    "open_url",
    "open_path",
    "set_timer",
    "set_alarm",
    "timer_stop",
    "alarm_stop",
    "open_utils",
    "close_utils",
    "play_reactor_intro",
    "set_volume",
    "set_brightness",
    "screenshot",
    "lock_screen",
    "sleep",
    "notify",
    "clipboard_read",
    "clipboard_write",
    "applescript",
    "system_info",
    "file_read",
    "file_write",
    "file_append",
    "file_edit",
    "file_list",
    "file_search",
    "file_glob",
    "task_add",
    "task_list",
    "task_done",
    "task_remove",
    "task_run",
    "task_open",
    "web_search",
    "window",
    "run_command",
    "send_message",
    "add_note",
    "create_event",
    "music_control",
];

const FIVE_SECONDS: Duration = Duration::from_secs(5);
const THIRTY_SECONDS: Duration = Duration::from_secs(30);

/// Спецификация по каноническому имени. `None` — имя вне реестра: вызов
/// отклоняется до выполнения.
pub fn spec_for(name: &str) -> Option<ToolSpec> {
    // Спецификация выдаётся только для канонических имён реестра: parse
    // нормализует алиасы до вызова, так что сюда приходят именно они.
    if !KNOWN_NAMES.contains(&name) {
        return None;
    }
    // Группы инструментов делят риск/таймаут/постусловие; имя — всегда
    // каноническое имя запроса (тест every_known_name_has_spec ловит
    // расхождение группового arm'а с реестром).
    let (risk, timeout, postcondition, platforms) = match name {
        "shell" | "applescript" => (
            RiskLevel::Medium,
            THIRTY_SECONDS,
            Postcondition::ExitCode,
            if name == "applescript" {
                platforms::MACOS_ONLY
            } else {
                platforms::ALL
            },
        ),
        "open_app" | "quit_app" => (
            RiskLevel::Medium,
            Duration::from_secs(10),
            Postcondition::ProcessState,
            platforms::ALL,
        ),
        "open_url" | "open_path" => (
            RiskLevel::Medium,
            FIVE_SECONDS,
            Postcondition::None,
            platforms::ALL,
        ),
        "set_timer" | "set_alarm" | "timer_stop" | "alarm_stop" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::StateReadback,
            platforms::ALL,
        ),
        "open_utils" | "close_utils" | "play_reactor_intro" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::None,
            platforms::ALL,
        ),
        "set_volume" | "set_brightness" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::StateReadback,
            platforms::ALL,
        ),
        "screenshot" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::EvidenceFile,
            platforms::ALL,
        ),
        "lock_screen" | "sleep" | "notify" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::ExitCode,
            platforms::ALL,
        ),
        "clipboard_read" | "clipboard_write" | "system_info" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::None,
            platforms::ALL,
        ),
        "file_read" | "file_list" | "file_search" | "file_glob" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::None,
            platforms::ALL,
        ),
        "file_write" | "file_append" | "file_edit" => (
            RiskLevel::Medium,
            FIVE_SECONDS,
            Postcondition::FileContent,
            platforms::ALL,
        ),
        "task_add" | "task_done" | "task_remove" | "task_open" | "task_list" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::None,
            platforms::ALL,
        ),
        "task_run" => (
            RiskLevel::Medium,
            Duration::from_secs(300),
            Postcondition::ExitCode,
            platforms::ALL,
        ),
        "web_search" => (
            RiskLevel::Low,
            Duration::from_secs(20),
            Postcondition::None,
            platforms::ALL,
        ),
        "window" => (
            RiskLevel::Medium,
            FIVE_SECONDS,
            Postcondition::StateReadback,
            platforms::ALL,
        ),
        "run_command" => (
            RiskLevel::Medium,
            Duration::from_secs(15),
            Postcondition::ProcessState,
            platforms::ALL,
        ),
        "send_message" | "add_note" | "create_event" => (
            RiskLevel::Medium,
            FIVE_SECONDS,
            Postcondition::None,
            platforms::MACOS_ONLY,
        ),
        "music_control" => (
            RiskLevel::Low,
            FIVE_SECONDS,
            Postcondition::StateReadback,
            platforms::ALL,
        ),
        _ => return None,
    };
    Some(ToolSpec {
        name: name.to_string(),
        risk,
        timeout,
        postcondition,
        platforms,
    })
}

/// Итоговый статус исполнения. Успех ставится только после подтверждения
/// постусловия; без возможности проверки — `Unknown`/`PartiallySucceeded`,
/// но никогда не `Succeeded` ради красивого ответа.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Succeeded,
    Failed,
    NeedsConfirmation,
    Unsupported,
    Cancelled,
    TimedOut,
    PartiallySucceeded,
    Unknown,
}

impl ExecutionStatus {
    /// Мост от текущего `ToolOutcome` (ok/detail) — статусов в нём ещё нет,
    /// поэтому ok=true даёт `Succeeded` только для инструментов, чьё
    /// постусловие фактически проверено движком (см. verification).
    pub fn from_verified(verified: bool, had_error: bool) -> Self {
        if verified {
            Self::Succeeded
        } else if had_error {
            Self::Failed
        } else {
            Self::Unknown
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::NeedsConfirmation => "needs_confirmation",
            Self::Unsupported => "unsupported",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
            Self::PartiallySucceeded => "partially_succeeded",
            Self::Unknown => "unknown",
        }
    }
}

/// Результат проверки постусловия: что именно подтверждено и чем.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub verified: bool,
    pub message: String,
    /// Доказательство для диагностики: имя процесса, код возврата,
    /// размер файла, фактическое значение настройки.
    pub evidence: Option<String>,
}

impl VerificationResult {
    pub fn verified(message: impl Into<String>, evidence: impl Into<String>) -> Self {
        Self {
            verified: true,
            message: message.into(),
            evidence: Some(evidence.into()),
        }
    }

    /// Выполнено без возможности проверить — не успех.
    pub fn unverified(message: impl Into<String>) -> Self {
        Self {
            verified: false,
            message: message.into(),
            evidence: None,
        }
    }

    pub fn failed(message: impl Into<String>, evidence: impl Into<String>) -> Self {
        Self {
            verified: false,
            message: message.into(),
            evidence: Some(evidence.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_known_name_has_spec() {
        for name in KNOWN_NAMES {
            let spec = spec_for(name).unwrap_or_else(|| panic!("нет спецификации для {name}"));
            assert_eq!(spec.name, *name);
        }
    }

    #[test]
    fn unknown_name_has_no_spec() {
        assert!(spec_for("delete_everything").is_none());
        assert!(spec_for("").is_none());
    }

    #[test]
    fn shell_and_applescript_are_medium_risk() {
        assert_eq!(spec_for("shell").unwrap().risk, RiskLevel::Medium);
        assert_eq!(spec_for("applescript").unwrap().risk, RiskLevel::Medium);
    }

    #[test]
    fn music_control_requires_state_readback() {
        assert_eq!(
            spec_for("music_control").unwrap().postcondition,
            Postcondition::StateReadback
        );
    }

    #[test]
    fn file_writes_verify_content() {
        for name in ["file_write", "file_append", "file_edit"] {
            assert_eq!(
                spec_for(name).unwrap().postcondition,
                Postcondition::FileContent
            );
        }
    }

    #[test]
    fn status_from_verified() {
        assert_eq!(
            ExecutionStatus::from_verified(true, false),
            ExecutionStatus::Succeeded
        );
        assert_eq!(
            ExecutionStatus::from_verified(false, true),
            ExecutionStatus::Failed
        );
        // Без проверки и без ошибки — не успех.
        assert_eq!(
            ExecutionStatus::from_verified(false, false),
            ExecutionStatus::Unknown
        );
    }
}
