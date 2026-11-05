//! Holo-3.1-4B (Qwen3.5-гибрид) — глаза и руки Jarvis.
//!
//! Один выделенный воркер владеет моделью (~9,7 ГБ BF16): чат по своей
//! системной роли, чтение экрана (screen_look) и замкнутый GUI-цикл
//! (ui_task: скриншот → действия → самопроверка). Текстовый мозг Jarvis —
//! uzu/mirai-l; он поручает агенту задачи инструментами `ui_task` /
//! `screen_look`. Модуль отделяет загрузку/валидацию весов от инференса.

pub mod action;
pub mod agent;
pub mod capture;
pub mod checkpoint;
pub mod hybrid;
pub mod mrope;
pub mod processor;
pub mod repository;
pub mod runtime;
pub mod service;
pub mod text_engine;
pub mod vision;

pub use action::{
    GuiAction, GuiActionBatch, execute_batch, parse_actions, parse_heredoc_tool_calls,
    parse_model_output, system_prompt, tool_definitions, to_desktop_actions,
};
pub use agent::{
    AgentConfig, AgentReport, Brain, DesktopScreen, Screen, StepRecord, Termination, VlBrain,
    run_loop, run_loop_with_observer,
};
pub use capture::{CapturedScreenshot, capture_desktop};
pub use checkpoint::{
    VerifyReport, expected_weights_hybrid, parse_header_json, read_safetensors_header,
    verify_checkpoint_files_hybrid, weight_files,
};
pub use hybrid::{HybridConfig, HybridLayerType, HybridModel};
pub use processor::{PreparedPrompt, ProcessedImage, UiVenusProcessor};
pub use repository::{ModelStatus, ensure_model, model_dir, model_status, remove_model};
pub use runtime::{
    GenerationOutput, SamplingParams, StepModel, StopReason, StreamEvent, generate,
};
pub use service::{
    AgentSettings, ServiceHandle, ServiceQuery, ServiceStatus, is_loading, set_service,
    spawn_service, update_config, with_service,
};
pub use text_engine::{ChatMessage, TextEngine35, holo_tools_block, split_think};

pub const REPOSITORY: &str = "Hcompany/Holo-3.1-4B";
pub const MODEL_TYPE: &str = "qwen3_5";
pub const EXPECTED_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";

/// Runtime state must not conflate downloaded weights with loaded inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeStage {
    NotDownloaded,
    DownloadedNotLoaded,
    Loading,
    Ready,
    Failed,
}
