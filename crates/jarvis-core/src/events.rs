//! Шина событий движка: движок публикует, UI (Tauri) и CLI подписываются.

use serde::Serialize;
use std::sync::mpsc::Sender;

/// Вариант выбора для интерфейса: номер (с нуля) и подпись.
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceOptionView {
    pub index: usize,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JarvisEvent {
    StateChanged {
        state: String,
    },
    WakeWordDetected {
        name: String,
        score: f32,
    },
    Transcript {
        text: String,
    },
    CommandMatched {
        name: String,
    },
    CommandNotFound {
        text: String,
    },
    /// Джарвис начал выполнять действие (tool-вызов или команду).
    ActionStarted {
        tool: String,
        summary: String,
    },
    /// Итог действия: честный статус и фраза, которую слышит пользователь.
    ActionResult {
        ok: bool,
        summary: String,
    },
    /// Джарвис вернулся к незакрытой теме.
    TopicResumed {
        topic: String,
    },
    /// Вердикт роутера: почему запрос поехал действием или разговором.
    /// Диагностика выбора инструмента из журнала.
    RouterVerdict {
        intent: String,
        confidence: f32,
        force_tool: bool,
    },
    /// У команды несколько равноправных целей: UI показывает варианты,
    /// голосовой ответ («второе», «2», название) разрешает тот же запрос.
    ChoiceRequested {
        choice_id: u64,
        prompt: String,
        options: Vec<ChoiceOptionView>,
    },
    /// Выбор закрыт. `index = None` — отмена, истечение или новая команда.
    ChoiceResolved {
        choice_id: u64,
        index: Option<usize>,
        label: Option<String>,
    },
    /// Tool-вызов модели не разобран (битый JSON, неизвестный инструмент).
    ToolParseFailed {
        reason: String,
        raw: String,
    },
    Reply {
        text: String,
    },
    LlmChunk {
        text: String,
    },
    LlmDone {
        text: String,
    },
    LlmStatus {
        status: String,
    },
    /// Доступ к микрофону: false — показать меню разрешения,
    /// true — доступ получен (меню закрыть, микрофон подключается).
    MicPermission {
        granted: bool,
    },
    /// Таймер поставлен: окно «Утилиты» открывается и тикает.
    TimerStarted {
        id: u64,
        label: String,
        seconds: u64,
        kind: crate::timers::TimerKind,
    },
    /// Таймер сработал: голос уже озвучил, окно мигает, рингтон звенит.
    TimerFired {
        id: u64,
        label: String,
        seconds: u64,
        kind: crate::timers::TimerKind,
    },
    /// Таймер отменён («останови таймер» или кнопкой).
    TimerStopped {
        id: u64,
        label: String,
    },
    /// Открыть окно утилит на вкладке: "timer" | "stopwatch" | "calendar".
    OpenUtils {
        tab: String,
    },
    /// Закрыть окно утилит («закрой утилиты» или крестиком).
    CloseUtils,
    /// Проиграть сборку реактора заново — эстетика интерфейса, поэтому
    /// событие идёт в UI, а не в систему.
    ReactorIntro,
    TrainProgress {
        epoch: usize,
        total: usize,
        loss: f64,
    },
    TrainDone {
        path: String,
    },
    RecordSaved {
        path: String,
    },
    /// Итог диктанта для профиля голоса: совпадение слов с фразой (0..1)
    /// и расшифровка. `ok` — диктант принят и сохранён в профиль.
    VoiceCheck {
        ok: bool,
        overlap: f32,
        transcript: String,
    },
    /// Ход загрузки модели: что качается и сколько готово (0..=100).
    /// Событие только для UI (прогресс-бар), движок его не слушает.
    DownloadProgress {
        what: String,
        percent: u8,
    },
    AudioLevel {
        level: f32,
        source: String,
    },
    Error {
        message: String,
    },
    Log {
        message: String,
    },
}

pub type EventTx = Sender<JarvisEvent>;