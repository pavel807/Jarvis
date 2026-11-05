//! Единый объект GUI-задачи и контракт результата.
//!
//! Основная модель (Uzu) формирует цель и критерии успеха — но не управляет
//! мышью. GUI-агент исполняет задачу замкнутым циклом
//! Observe → Decide → Act → Verify и возвращает [`TaskResult`] с честным
//! состоянием: успешная отправка события ≠ выполненная задача.

use serde::Serialize;

/// Единая GUI-задача: создаётся один раз из пользовательской фразы,
/// живёт до терминального состояния.
#[derive(Debug, Clone)]
pub struct GuiTask {
    /// Уникальный идентификатор в пределах сессии (для событий и логов).
    pub id: u32,
    /// Исходная фраза пользователя как была сказана.
    pub utterance: String,
    /// Нормализованная цель (по-английски — рабочий язык Holo).
    pub goal: String,
    /// Критерии успешного завершения (по-английски, проверяемые по экрану).
    pub success_criteria: String,
    /// Ограничения: не подтверждать платежи, не удалять данные и т.п.
    pub constraints: &'static str,
}

/// Ограничения по умолчанию: безопасность важнее послушания.
pub const DEFAULT_CONSTRAINTS: &str =
    "Never confirm payments, purchases, sending messages, deleting data, or changing security settings without explicit user approval in the task. Do not enter passwords or personal data.";

/// Жизненный цикл задачи (пп. 30, 52): терминальные — Succeeded,
/// PartiallySucceeded, Failed, TimedOut, Cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// Создана, ещё не запущена.
    Pending,
    /// Цикл агента работает.
    Running,
    /// Идёт проверка результата по свежему скриншоту.
    Verifying,
    /// Цель подтверждена свежим наблюдением.
    Succeeded,
    /// Действия выполнялись, но цель не подтверждена (лимит/ошибка в конце).
    PartiallySucceeded,
    /// Цель не достигнута, действий не было или они не удались.
    Failed,
    /// Цель достигнута по сообщению модели, но подтвердить по экрану нельзя.
    Unverified,
    /// Временной бюджет исчерпан.
    TimedOut,
    /// Остановлено пользователем.
    Cancelled,
}

impl TaskState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskState::Succeeded
                | TaskState::PartiallySucceeded
                | TaskState::Failed
                | TaskState::Unverified
                | TaskState::TimedOut
                | TaskState::Cancelled
        )
    }

    /// Человеческая формулировка для голоса и журнала.
    pub fn speech(self) -> &'static str {
        match self {
            TaskState::Pending => "задача в очереди",
            TaskState::Running => "задача выполняется",
            TaskState::Verifying => "проверяю результат",
            TaskState::Succeeded => "задача выполнена",
            TaskState::PartiallySucceeded => "задача выполнена частично",
            TaskState::Failed => "задачу не удалось выполнить",
            TaskState::Unverified => "выполнено, но результат не удалось подтвердить",
            TaskState::TimedOut => "не хватило отведённого времени",
            TaskState::Cancelled => "остановлено",
        }
    }
}

/// Итог задачи: состояние + факты для голоса и основной модели.
#[derive(Debug, Clone, Serialize)]
pub struct TaskResult {
    pub task_id: u32,
    pub state: TaskState,
    /// Краткий итог по-русски (что сделано / что не вышло и почему).
    pub summary: String,
    /// Последние действия с результатами — факты, не выдумка модели.
    pub evidence: Vec<String>,
    pub steps: usize,
    pub total_actions: usize,
    pub elapsed_ms: u128,
}

impl TaskResult {
    /// Честный ok для ToolOutcome: только Succeeded считается успехом.
    pub fn succeeded(&self) -> bool {
        self.state == TaskState::Succeeded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_states_are_correct() {
        assert!(TaskState::Succeeded.is_terminal());
        assert!(TaskState::Cancelled.is_terminal());
        assert!(!TaskState::Running.is_terminal());
        assert!(!TaskState::Verifying.is_terminal());
        assert!(!TaskState::Pending.is_terminal());
    }

    #[test]
    fn only_succeeded_counts_as_success() {
        let mut result = TaskResult {
            task_id: 1,
            state: TaskState::Succeeded,
            summary: "готово".into(),
            evidence: vec![],
            steps: 2,
            total_actions: 2,
            elapsed_ms: 100,
        };
        assert!(result.succeeded());
        result.state = TaskState::Unverified;
        assert!(!result.succeeded(), "неподтверждённый успех — не успех");
        result.state = TaskState::PartiallySucceeded;
        assert!(!result.succeeded());
    }
}
