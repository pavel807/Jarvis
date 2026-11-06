//! Воркер Holo-3.1: один выделенный поток владеет моделью (8.3 ГБ BF16 +
//! draft 1.4 ГБ), все запросы идут через канал — как у speaker-воркера.
//! Загрузка ленивая: при первом обращении. Отмена (голос «стоп», кнопка UI)
//! — общим флагом, он опрашивается и в генерации, и в GUI-цикле.

use super::agent::{AgentConfig, AgentReport, DesktopScreen, VlBrain, run_loop_with_observer};
use super::capture::CapturedScreenshot;
use super::repository;
use super::runtime::SamplingParams;
use super::text_engine::{ChatMessage, TextEngine35};
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Sender, SyncSender, Receiver, channel, sync_channel};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub enum ServiceQuery {
    Status { reply: Sender<ServiceStatus> },
    /// Скачать чекпойнт (~9,7 ГБ); прогресс уходит в JarvisEvent::Log.
    Download { reply: Sender<Result<()>> },
    /// Горячая загрузка модели при старте приложения: воркер сам готовит
    /// веса, интерфейс остаётся живым; вопросы не про экран отвечаются,
    /// про экран — фраза ожидания, пока `is_loading()`.
    Preload,
    /// Единственный исполнительный контур GUI-задач.
    RunTask { task: super::task::GuiTask, config: AgentConfig, reply: Sender<Result<super::task::TaskResult>> },
    /// Чат: текстовый ответ (обычная или спекулятивная генерация).
    Chat {
        messages: Vec<ChatMessage>,
        tools: Option<Vec<String>>,
        thinking: bool,
        speculative: bool,
        params: SamplingParams,
        reply: Sender<Result<String>>,
    },
    /// Что на экране: текстовое описание по вопросу.
    Look { question: String, reply: Sender<Result<String>> },
    /// Выгрузить модель из памяти.
    Unload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceStatus {
    pub stage: super::RuntimeStage,
    pub detail: String,
    pub downloaded: bool,
    pub weights_bytes: u64,
}

#[derive(Clone)]
pub struct ServiceHandle {
    tx: SyncSender<ServiceQuery>,
    /// Общий флаг отмены: генерация и GUI-цикл опрашивают его между шагами.
    pub cancel: Arc<AtomicBool>,
}

impl ServiceHandle {
    /// Снапшот статуса БЕЗ запроса к воркеру: воркер бывает занят задачей
    /// на минуты, и блокирующий запрос отсюда вешал интерфейс.
    pub fn status(&self) -> ServiceStatus {
        let repo = repository::model_status();
        let code = STAGE.load(Ordering::Relaxed);
        let stage = stage_from_code(code);
        let (detail, downloaded) = if code == 3 {
            ("модель в памяти".to_string(), true)
        } else if repo.downloaded {
            ("веса на диске, память свободна".to_string(), true)
        } else if let Some(err) = repo.config_error.clone() {
            (format!("конфиг отклонён: {err}"), false)
        } else if !repo.missing_files.is_empty() {
            (format!("не скачана: отсутствуют {:?}", repo.missing_files), false)
        } else {
            ("шарды недокачаны".to_string(), false)
        };
        ServiceStatus { stage, detail, downloaded, weights_bytes: repo.weights_bytes }
    }

    pub fn request(&self, query: ServiceQuery) -> Result<()> {
        self.tx
            .send(query)
            .map_err(|_| anyhow::anyhow!("канал Holo-3.1 закрыт"))
    }

    /// GUI-задача: единый контракт — `GuiTask` на входе, `TaskResult` на
    /// выходе. Конфиг — из пользовательских настроек (`update_config`).
    pub fn run_task_full(&self, goal: &str, success: Option<&str>) -> Result<super::task::TaskResult> {
        let (settings, config) = config_snapshot();
        let (tx, rx) = channel();
        let task_id = next_task_id();
        let task = super::task::GuiTask {
            id: task_id,
            utterance: goal.to_string(),
            goal: goal.to_string(),
            success_criteria: success
                .map(|s| s.to_string())
                .unwrap_or_else(|| "The goal is visibly reached on the current screen.".into()),
            constraints: super::task::DEFAULT_CONSTRAINTS,
        };
        let _ = settings;
        self.request(ServiceQuery::RunTask { task, config, reply: tx })?;
        rx.recv().map_err(|_| anyhow::anyhow!("воркер Holo-3.1 не ответил"))?
    }

    /// Текстовое описание текущего экрана по вопросу.
    pub fn look(&self, question: &str) -> Result<String> {
        let (tx, rx) = channel();
        self.request(ServiceQuery::Look { question: question.to_string(), reply: tx })?;
        rx.recv().map_err(|_| anyhow::anyhow!("воркер Holo-3.1 не ответил"))?
    }
}

/// Пользовательские настройки GUI-агента (settings.toml → apply_hot).
static AGENT_CONFIG: OnceLock<RwLock<AgentSettings>> = OnceLock::new();

/// Счётчик задач для идентификаторов в событиях и журналах.
static TASK_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn next_task_id() -> u32 {
    TASK_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Честный маппинг Termination → TaskState (пп. 30-31, 63-65): успешная
/// отправка события ≠ выполненная задача; действия без подтверждённой цели —
/// PartiallySucceeded, не Succeeded.
fn task_result(
    task_id: u32,
    report: &AgentReport,
    _criteria: &str,
) -> super::task::TaskResult {
    use super::task::{TaskResult, TaskState};
    let evidence: Vec<String> = report
        .steps
        .iter()
        .rev()
        .take(3)
        .flat_map(|s| s.log.iter().cloned())
        .collect();
    let did_work = report.total_actions > 0;
    let (state, summary) = match &report.termination {
        crate::holo::Termination::Done(message) => {
            (TaskState::Succeeded, message.clone())
        }
        crate::holo::Termination::MaxSteps => (
            if did_work { TaskState::PartiallySucceeded } else { TaskState::Failed },
            format!("Лимит шагов ({}) исчерпан, цель не подтверждена", report.steps.len()),
        ),
        crate::holo::Termination::Timeout => (
            TaskState::TimedOut,
            "Время задачи истекло, цель не подтверждена".into(),
        ),
        crate::holo::Termination::Cancelled => {
            (TaskState::Cancelled, "Остановлено пользователем".into())
        }
        crate::holo::Termination::ActionError(e) => (
            if did_work { TaskState::PartiallySucceeded } else { TaskState::Failed },
            format!("Ошибка действия: {e}"),
        ),
        crate::holo::Termination::Stuck => (
            TaskState::Failed,
            "Агент повторял одно и то же действие без изменения экрана — остановлен".into(),
        ),
    };
    TaskResult {
        task_id,
        state,
        summary,
        evidence,
        steps: report.steps.len(),
        total_actions: report.total_actions,
        elapsed_ms: report.elapsed_ms,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AgentSettings {
    pub enabled: bool,
    pub max_steps: usize,
    pub timeout_secs: u64,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self { enabled: true, max_steps: 12, timeout_secs: 240 }
    }
}

/// Обновляет настройки агента на живую (hot-reload settings.toml).
pub fn update_config(settings: AgentSettings) {
    let slot = AGENT_CONFIG.get_or_init(|| RwLock::new(AgentSettings::default()));
    *slot.write().unwrap_or_else(|e| e.into_inner()) = settings;
}

fn config_snapshot() -> (AgentSettings, AgentConfig) {
    let slot = AGENT_CONFIG.get_or_init(|| RwLock::new(AgentSettings::default()));
    let s = *slot.read().unwrap_or_else(|e| e.into_inner());
    let cfg = AgentConfig {
        max_steps: s.max_steps,
        timeout: Some(std::time::Duration::from_secs(s.timeout_secs)),
        verify: true,
    };
    (s, cfg)
}

/// Стадия воркера для UI: атомарная, чтобы `status` никогда не ждал
/// занятый задачей воркер — иначе интерфейс зависает на минуту.
static STAGE: AtomicU8 = AtomicU8::new(0);

fn stage_code(stage: super::RuntimeStage) -> u8 {
    match stage {
        super::RuntimeStage::NotDownloaded => 0,
        super::RuntimeStage::DownloadedNotLoaded => 1,
        super::RuntimeStage::Loading => 2,
        super::RuntimeStage::Ready => 3,
        super::RuntimeStage::Failed => 4,
    }
}

fn stage_from_code(code: u8) -> super::RuntimeStage {
    match code {
        1 => super::RuntimeStage::DownloadedNotLoaded,
        2 => super::RuntimeStage::Loading,
        3 => super::RuntimeStage::Ready,
        4 => super::RuntimeStage::Failed,
        _ => super::RuntimeStage::NotDownloaded,
    }
}

/// Модель сейчас грузится в память? Пока да — вопросы про экран получают
/// вежливую фразу ожидания вместо минутного подвешивания движка.
pub fn is_loading() -> bool {
    stage_from_code(STAGE.load(Ordering::Relaxed)) == super::RuntimeStage::Loading
}

/// Доступность агента зрения для LLM-инструментов (`ui_task`, `screen_look`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoloAvailability {
    /// Воркер не запущен (тесты, примеры): инструменты остаются как были.
    NotStarted,
    /// Модель в памяти.
    Ready,
    /// Идёт загрузка: запрос стоит подождать.
    Loading,
    /// Веса на диске, в память ещё не загружены: первый запрос загрузит их сам.
    Idle,
    /// Не скачана или загрузка провалилась: пользоваться нельзя.
    Unavailable,
}

pub fn availability() -> HoloAvailability {
    let started = SERVICE
        .get()
        .is_some_and(|slot| slot.lock().unwrap_or_else(|e| e.into_inner()).is_some());
    if !started {
        return HoloAvailability::NotStarted;
    }
    match stage_from_code(STAGE.load(Ordering::Relaxed)) {
        super::RuntimeStage::Ready => HoloAvailability::Ready,
        super::RuntimeStage::Loading => HoloAvailability::Loading,
        super::RuntimeStage::DownloadedNotLoaded => HoloAvailability::Idle,
        super::RuntimeStage::Failed => HoloAvailability::Unavailable,
        // Начальное значение стадии — NotDownloaded, пока воркер не взялся за
        // Preload: смотрим на диск, а не считаем модель отсутствующей.
        super::RuntimeStage::NotDownloaded => {
            if repository::model_status().downloaded {
                HoloAvailability::Idle
            } else {
                HoloAvailability::Unavailable
            }
        }
    }
}

/// Ждёт, пока модель станет доступной для запроса. `false` — таймаут или
/// модель недоступна вовсе.
pub fn wait_until_ready(timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match availability() {
            HoloAvailability::Ready | HoloAvailability::Idle => return true,
            HoloAvailability::NotStarted | HoloAvailability::Unavailable => return false,
            HoloAvailability::Loading => {}
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

/// Глобальный хэндл: инструменты (`tools::execute`) не имеют доступа к
/// состоянию движка, поэтому сервис доступен через слот.
static SERVICE: OnceLock<Mutex<Option<ServiceHandle>>> = OnceLock::new();

pub fn set_service(handle: ServiceHandle) {
    let slot = SERVICE.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
}

pub fn with_service<T>(f: impl FnOnce(&ServiceHandle) -> Result<T>) -> Result<T> {
    let slot = SERVICE.get_or_init(|| Mutex::new(None));
    let guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    let Some(handle) = guard.as_ref() else {
        anyhow::bail!("Holo-3.1 сервис не запущен");
    };
    f(handle)
}

/// Запускает воркер. Возвращает хэндл для запросов.
pub fn spawn_service(events: crate::events::EventTx) -> ServiceHandle {
    let (tx, rx) = sync_channel::<ServiceQuery>(4);
    let cancel = Arc::new(AtomicBool::new(false));
    let handle =
        ServiceHandle { tx, cancel: Arc::clone(&cancel) };
    std::thread::Builder::new()
        .name("qwen3vl-service".into())
        .spawn(move || worker_loop(rx, events, cancel))
        .expect("поток Holo-3.1 должен стартовать");
    handle
}

struct WorkerState {
    engine: Option<TextEngine35>,
}

impl WorkerState {
    fn unload(&mut self) {
        self.engine = None;
        let repo = repository::model_status();
        STAGE.store(
            stage_code(if repo.downloaded {
                super::RuntimeStage::DownloadedNotLoaded
            } else {
                super::RuntimeStage::NotDownloaded
            }),
            Ordering::Relaxed,
        );
    }
}

fn worker_loop(rx: Receiver<ServiceQuery>, events: crate::events::EventTx, cancel: Arc<AtomicBool>) {
    let log = |msg: String| {
        let _ = events.send(crate::events::JarvisEvent::Log { message: msg });
    };
    let mut state = WorkerState { engine: None };

    while let Ok(query) = rx.recv() {
        match query {
            ServiceQuery::Status { reply } => {
                let repo = repository::model_status();
                let stage = if state.engine.is_some() {
                    super::RuntimeStage::Ready
                } else if repo.downloaded {
                    super::RuntimeStage::DownloadedNotLoaded
                } else {
                    super::RuntimeStage::NotDownloaded
                };
                STAGE.store(stage_code(stage), Ordering::Relaxed);
                let _ = reply.send(ServiceStatus {
                    stage,
                    detail: if state.engine.is_some() {
                        "модель в памяти".into()
                    } else if repo.downloaded {
                        "веса на диске, память свободна".into()
                    } else if let Some(err) = repo.config_error.clone() {
                        format!("конфиг отклонён: {err}")
                    } else if !repo.missing_files.is_empty() {
                        format!("не скачана: отсутствуют {:?}", repo.missing_files)
                    } else {
                        "шарды недокачаны".into()
                    },
                    downloaded: repo.downloaded,
                    weights_bytes: repo.weights_bytes,
                });
            }
            ServiceQuery::Download { reply } => {
                cancel.store(false, Ordering::Relaxed);
                log("⬇️ Загрузка Holo-3.1-4B (~9,7 ГБ)…".into());
                let flag = Arc::clone(&cancel);
                let mut progress = |percent: u8| {
                    log(format!("⬇️ Holo-3.1: {percent}%"));
                    if flag.load(Ordering::Relaxed) {
                        // ensure_model не принимает флаг; отмена загрузки
                        // выполняется удалением .part вручную — не критично.
                    }
                };
                let result = repository::ensure_model(&mut progress)
                    .map(|dir| log(format!("✅ Holo-3.1 скачана: {}", dir.display())));
                let _ = reply.send(result);
            }
            ServiceQuery::Unload => {
                state.unload();
                STAGE.store(stage_code(super::RuntimeStage::DownloadedNotLoaded), Ordering::Relaxed);
                log("🧹 Holo-3.1-4B выгружена из памяти".into());
            }
            ServiceQuery::Preload => {
                // Горячая загрузка: готовим модель в фоне. Пока грузится,
                // is_loading() отвечает за вежливую фразу ожидания.
                let WorkerState { engine, .. } = &mut state;
                if engine.is_some() {
                    continue;
                }
                if let Err(e) = ensure_engine(engine, &events) {
                    log(format!("⚠️ Горячая загрузка не удалась: {e:#}"));
                }
            }
            ServiceQuery::Chat { messages, tools, thinking, speculative: _, params, reply } => {
                // Спекулятивное декодирование отключено (решение владельца:
                // «без draft, главное качество») — обычная генерация.
                cancel.store(false, Ordering::Relaxed);
                let WorkerState { engine, .. } = &mut state;
                let result = (|| -> Result<String> {
                    let engine = ensure_engine(engine, &events)?;
                    engine.reply(&messages, tools.as_deref(), thinking, &params, Some(&cancel), |_| true)
                })();
                let _ = reply.send(result);
            }
            ServiceQuery::RunTask { task, config, reply } => {
                // Новый запрос сбрасывает отмену: «стоп» действует на текущую
                // задачу, а не на все будущие.
                cancel.store(false, Ordering::Relaxed);
                if !config_snapshot().0.enabled {
                    let _ = reply.send(Err(anyhow::anyhow!(
                        "GUI-агент отключён в настройках (ui_agent_enabled = false)"
                    )));
                    continue;
                }
                let _ = events.send(crate::events::JarvisEvent::UiAgentStatus {
                    stage: "running".into(),
                    detail: format!("#{}: {}", task.id, task.goal.chars().take(110).collect::<String>()),
                });
                let WorkerState { engine, .. } = &mut state;
                let events_for_steps = crate::events::EventTx::clone(&events);
                let task_id = task.id;
                let goal = task.goal.clone();
                let criteria = task.success_criteria.clone();
                let result = (|| -> Result<super::task::TaskResult> {
                    let engine = ensure_engine(engine, &events)?;
                    let params = SamplingParams {
                        // Thinking-блок + tool-call на шаг: 512 обрезал ответы
                        // посреди рассуждения — парсер терял вызов.
                        max_new_tokens: 768,
                        temperature: 0.0,
                        // Стоп ровно после завершённого вызова: длинный
                        // thinking больше не съедает хвост с действиями.
                        stop_strings: vec!["</tool_call>".into()],
                        ..SamplingParams::default()
                    };
                    let mut brain = VlBrain::new(engine.model_mut(), params);
                    let report = run_loop_with_observer(
                        &mut brain,
                        &DesktopScreen,
                        &goal,
                        &criteria,
                        &config,
                        Some(&cancel),
                        &|step, _actions, batch| {
                            let names: Vec<String> = batch
                                .actions
                                .iter()
                                .map(|a| format!("{a:?}"))
                                .collect();
                            let _ = events_for_steps.send(crate::events::JarvisEvent::UiAgentStep {
                                step,
                                max_steps: config.max_steps,
                                actions: names,
                            });
                        },
                    )?;
                    Ok(task_result(task_id, &report, &criteria))
                })();
                match &result {
                    Ok(task_result) => {
                        let _ = events.send(crate::events::JarvisEvent::UiAgentDone {
                            ok: task_result.succeeded(),
                            message: format!("#{}: {}", task_result.task_id, task_result.summary),
                            steps: task_result.steps,
                            elapsed_ms: task_result.elapsed_ms,
                        });
                    }
                    Err(e) => {
                        let _ = events.send(crate::events::JarvisEvent::UiAgentDone {
                            ok: false,
                            message: format!("#{task_id}: {e:#}"),
                            steps: 0,
                            elapsed_ms: 0,
                        });
                    }
                }
                let _ = reply.send(result);
            }
            ServiceQuery::Look { question, reply } => {
                let WorkerState { engine, .. } = &mut state;
                let result = (|| -> Result<String> {
                    let engine = ensure_engine(engine, &events)?;
                    let shot: CapturedScreenshot = super::capture::capture_desktop()
                        .context("Не удалось сделать скриншот для разбора экрана")?;
                    let params = SamplingParams {
                        max_new_tokens: 512,
                        temperature: 0.0,
                        ..SamplingParams::default()
                    };
                    let mut brain = VlBrain::new(engine.model_mut(), params);
                    brain.describe(&shot, &question)
                })();
                let _ = reply.send(result);
            }
        }
    }
}

/// Гарантирует загруженный движок: ленивая загрузка при первом обращении.
fn ensure_engine<'a>(
    engine: &'a mut Option<TextEngine35>,
    events: &crate::events::EventTx,
) -> Result<&'a mut TextEngine35> {
    if engine.is_none() {
        let repo = repository::model_status();
        if !repo.downloaded {
            anyhow::bail!(
                "Holo-3.1-4B не скачана ({}); запустите загрузку в настройках",
                repo.config_error.clone().unwrap_or_else(|| format!("отсутствуют {:?}", repo.missing_files))
            );
        }
        let _ = events.send(crate::events::JarvisEvent::Log {
            message: "🧠 Загрузка Holo-3.1-4B в память…".into(),
        });
        STAGE.store(stage_code(super::RuntimeStage::Loading), Ordering::Relaxed);
        let device = pick_device();
        let loaded = match TextEngine35::load(&repository::model_dir(), device) {
            Ok(loaded) => loaded,
            Err(e) => {
                STAGE.store(stage_code(super::RuntimeStage::Failed), Ordering::Relaxed);
                return Err(e).context("Загрузка Holo-3.1-4B не удалась");
            }
        };
        *engine = Some(loaded);
        STAGE.store(stage_code(super::RuntimeStage::Ready), Ordering::Relaxed);
        let _ = events.send(crate::events::JarvisEvent::Log {
            message: "✅ Holo-3.1-4B готова".into(),
        });
    }
    Ok(engine.as_mut().expect("движок загружен"))
}

/// Metal на macOS, иначе CPU (как у остальных движков Jarvis).
fn pick_device() -> candle_core::Device {
    #[cfg(target_os = "macos")]
    {
        candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
    }
    #[cfg(not(target_os = "macos"))]
    {
        candle_core::Device::Cpu
    }
}
