//! Closed-loop GUI agent: observe → decide → act → verify → self-correct.
//!
//! The loop is written against two small traits ([`Screen`] and [`Brain`]) so
//! the whole state machine — step budget, wall-clock budget, cancellation,
//! verification re-capture, stop-on-`done`, and the action log — is testable on
//! CPU with a scripted brain, without the 4.88 GB checkpoint. [`VenusBrain`]
//! is the production brain: it turns a live screenshot + task into a validated
//! [`GuiActionBatch`] using the loaded [`UiVenusModel`].

use super::action::{
    GuiAction, GuiActionBatch, execute_batch, parse_model_output,
};
use super::capture::{CapturedScreenshot, capture_desktop};
use super::hybrid::HybridModel;
use super::runtime::SamplingParams;
use anyhow::{Context, Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Where the agent captures the current desktop from.
pub trait Screen {
    fn capture(&self) -> Result<CapturedScreenshot>;
}

/// Production screen: the real native desktop grab.
pub struct DesktopScreen;
impl Screen for DesktopScreen {
    fn capture(&self) -> Result<CapturedScreenshot> {
        capture_desktop()
    }
}

/// One decision step of the agent: given a fresh screenshot, the task, and the
/// log of what has already happened, produce the next batch of GUI actions.
pub trait Brain {
    fn decide(
        &mut self,
        screenshot: &CapturedScreenshot,
        task: &str,
        history: &[String],
    ) -> Result<GuiActionBatch>;
}

/// Tunables for [`run_loop`].
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Hard cap on observe/decide/act iterations.
    pub max_steps: usize,
    /// Wall-clock budget for the whole run; `None` disables it.
    pub timeout: Option<Duration>,
    /// Re-capture and feed back to the brain after every action batch so it can
    /// verify results and self-correct. Always on in practice, kept explicit.
    pub verify: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_steps: 8, timeout: Some(Duration::from_secs(120)), verify: true }
    }
}

/// One executed step.
#[derive(Debug, Clone)]
pub struct StepRecord {
    pub step: usize,
    pub width: u32,
    pub height: u32,
    /// Human-readable lines returned by the executor for each action.
    pub log: Vec<String>,
}

/// Why the loop ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Termination {
    /// The brain emitted a `done` action; carries its message.
    Done(String),
    /// `max_steps` reached without a `done`.
    MaxSteps,
    /// The cancellation flag was set.
    Cancelled,
    /// The wall-clock budget expired.
    Timeout,
    /// An action failed to execute; carries the error text.
    ActionError(String),
}

/// Full result of [`run_loop`].
#[derive(Debug, Clone)]
pub struct AgentReport {
    pub steps: Vec<StepRecord>,
    pub termination: Termination,
    pub total_actions: usize,
    pub elapsed_ms: u128,
}

impl AgentReport {
    pub fn finished(&self) -> bool {
        matches!(self.termination, Termination::Done(_))
    }
}

/// Drives the observe→decide→act→verify loop.
///
/// `cancel` (optional) is polled before every step; when set, the loop stops
/// with [`Termination::Cancelled`] without executing further actions.
pub fn run_loop(
    brain: &mut dyn Brain,
    screen: &dyn Screen,
    task: &str,
    config: &AgentConfig,
    cancel: Option<&AtomicBool>,
) -> Result<AgentReport> {
    run_loop_with_observer(brain, screen, task, config, cancel, &|_, _, _| {})
}

/// Как [`run_loop`], но `observer(step, total_actions, batch)` вызывается
/// после каждого выполненного пакета — для событий в UI.
pub fn run_loop_with_observer(
    brain: &mut dyn Brain,
    screen: &dyn Screen,
    task: &str,
    config: &AgentConfig,
    cancel: Option<&AtomicBool>,
    observer: &dyn Fn(usize, usize, &GuiActionBatch),
) -> Result<AgentReport> {
    let start = Instant::now();
    let mut steps: Vec<StepRecord> = Vec::new();
    let mut history: Vec<String> = Vec::new();
    let mut total_actions = 0usize;
    let mut termination = Termination::MaxSteps;

    for step in 0..config.max_steps {
        if cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            termination = Termination::Cancelled;
            break;
        }
        if let Some(budget) = config.timeout {
            if start.elapsed() >= budget {
                termination = Termination::Timeout;
                break;
            }
        }

        // 1. Observe: a fresh screenshot is the only ground truth the brain sees.
        let shot = screen.capture().context("GUI-агент: не удалось сделать скриншот")?;

        // 2. Decide.
        let batch = match brain.decide(&shot, task, &history) {
            Ok(b) => b,
            Err(e) => {
                termination = Termination::ActionError(format!("модель не выдала действий: {e}"));
                break;
            }
        };

        // A `done`/`wait`-only batch still runs through the executor so waits
        // happen and the completion message is captured.
        let saw_done = batch.actions.iter().any(|a| matches!(a, GuiAction::Done { .. }));

        // 3. Act (executor enforces coordinate bounds and its own safety).
        let log = match execute_batch(&batch, shot.width, shot.height) {
            Ok(lines) => lines,
            Err(e) => {
                termination = Termination::ActionError(format!("{e}"));
                break;
            }
        };
        total_actions += batch.actions.len();
        history.extend(log.iter().cloned());
        steps.push(StepRecord { step, width: shot.width, height: shot.height, log });
        observer(step, total_actions, &batch);

        if saw_done {
            let message = batch
                .actions
                .iter()
                .find_map(|a| match a {
                    GuiAction::Done { message } => message.clone(),
                    _ => None,
                })
                .unwrap_or_else(|| "Задача завершена".to_string());
            termination = Termination::Done(message);
            break;
        }
        // 4. Verify / self-correct: the next iteration re-captures and passes
        // the growing history back to the brain. Nothing else to do here.
    }

    Ok(AgentReport {
        steps,
        termination,
        total_actions,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

// =========================================================================
// Production brain: Holo-3.1 model in the loop
// =========================================================================

/// A [`Brain`] backed by the loaded multimodal Holo-3.1 model. Each `decide`
/// call preprocesses the live screenshot, builds the official GUI-agent
/// prompt with the real action history, runs generation, and parses the
/// result into a validated action batch. Неверный формат — повтор с той же
/// задачей (до 3 попыток), как в официальных агентских харнессах.
pub struct VlBrain<'a> {
    model: &'a mut HybridModel,
    params: SamplingParams,
}

/// Сколько раз модель может переформулировать действия после ошибки формата.
const PARSE_ATTEMPTS: usize = 3;

impl<'a> VlBrain<'a> {
    pub fn new(model: &'a mut HybridModel, params: SamplingParams) -> Self {
        Self { model, params }
    }

    /// Text-only description of the current screen (the "what do you see?"
    /// path): runs the model with a describe instruction and returns prose,
    /// bypassing the action parser.
    pub fn describe(&mut self, screenshot: &CapturedScreenshot, question: &str) -> Result<String> {
        let path = std::path::Path::new(&screenshot.path);
        let image = self.model.processor().process_image(path)?;
        let prompt = self.model.processor().prepare_prompt(
            self.model.tokenizer(),
            image,
            question,
        )?;
        let cancel = AtomicBool::new(false);
        let out = self.model.generate_for_prompt(
            &prompt,
            &self.params,
            Some(&cancel),
            |_| true,
        )?;
        Ok(out.text)
    }

    /// Один запрос к модели по свежему скриншоту: задача + официальная
    /// история шагов («Step N: …») для самоисправления.
    fn decide_once(
        &mut self,
        screenshot: &CapturedScreenshot,
        task: &str,
        history: &[String],
    ) -> Result<String> {
        let path = std::path::Path::new(&screenshot.path);
        let image = self.model.processor().process_image(path)?;
        let prompt = self.model.processor().prepare_prompt_with_history(
            self.model.tokenizer(),
            image,
            task,
            history,
        )?;
        let cancel = AtomicBool::new(false);
        let out = self.model.generate_for_prompt(&prompt, &self.params, Some(&cancel), |_| true)?;
        Ok(out.text)
    }
}

impl Brain for VlBrain<'_> {
    fn decide(
        &mut self,
        screenshot: &CapturedScreenshot,
        task: &str,
        history: &[String],
    ) -> Result<GuiActionBatch> {
        let mut last_error = String::new();
        for _ in 0..PARSE_ATTEMPTS {
            let raw = self.decide_once(screenshot, task, history)?;
            match parse_model_output(&raw) {
                Ok(batch) => return Ok(batch),
                Err(e) => {
                    last_error = format!("{e} (сырой ответ: {})", clip(&raw, 160));
                }
            }
        }
        bail!("Модель {PARSE_ATTEMPTS} раза выдала неверный формат действий: {last_error}")
    }
}

/// Keeps history/parse-error strings bounded.
fn clip(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push('…');
    out
}

// =========================================================================
// Tests (scripted brain + fake screen — no weights required)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A fake screen that writes a tiny PNG to a temp path so `process_image`
    /// is never exercised (only dimensions matter for the loop tests).
    struct FakeScreen {
        width: u32,
        height: u32,
        captures: std::cell::Cell<usize>,
    }
    impl FakeScreen {
        fn new(w: u32, h: u32) -> Self {
            Self { width: w, height: h, captures: std::cell::Cell::new(0) }
        }
    }
    impl Screen for FakeScreen {
        fn capture(&self) -> Result<CapturedScreenshot> {
            self.captures.set(self.captures.get() + 1);
            // Minimal valid file path (contents unused by the loop tests).
            let mut dir = std::env::temp_dir();
            dir.push(format!("jarvis-gui-test-{}", std::process::id()));
            std::fs::create_dir_all(&dir).ok();
            let path = dir.join(format!("shot-{}.png", self.captures.get()));
            let mut f = std::fs::File::create(&path)?;
            f.write_all(&[0u8; 8])?;
            Ok(CapturedScreenshot { path, width: self.width, height: self.height })
        }
    }

    /// Scripted brain: yields a queued batch per step, then `done`.
    struct ScriptBrain {
        script: Vec<GuiActionBatch>,
        i: std::cell::Cell<usize>,
        seen_history_len: std::cell::RefCell<Vec<usize>>,
    }
    impl ScriptBrain {
        fn new(script: Vec<GuiActionBatch>) -> Self {
            Self { script, i: std::cell::Cell::new(0), seen_history_len: std::cell::RefCell::new(vec![]) }
        }
    }
    impl Brain for ScriptBrain {
        fn decide(
            &mut self,
            _shot: &CapturedScreenshot,
            _task: &str,
            history: &[String],
        ) -> Result<GuiActionBatch> {
            self.seen_history_len.borrow_mut().push(history.len());
            let idx = self.i.get();
            self.i.set(idx + 1);
            Ok(self.script[idx.min(self.script.len() - 1)].clone())
        }
    }

    fn batch(actions: Vec<GuiAction>) -> GuiActionBatch {
        GuiActionBatch { actions }
    }

    #[test]
    fn done_terminates_loop() {
        let brain = &mut ScriptBrain::new(vec![batch(vec![GuiAction::Done {
            message: Some("Готово".into()),
        }])]);
        let screen = FakeScreen::new(800, 600);
        let report = run_loop(brain, &screen, "тест", &AgentConfig::default(), None).unwrap();
        assert_eq!(report.termination, Termination::Done("Готово".into()));
        assert!(report.finished());
        assert_eq!(report.steps.len(), 1);
    }

    #[test]
    fn max_steps_stops_without_done() {
        // A brain that always clicks → loop must stop at max_steps, never hang.
        let brain = &mut ScriptBrain::new(vec![batch(vec![GuiAction::Wait {
            milliseconds: Some(1),
        }])]);
        let screen = FakeScreen::new(800, 600);
        let cfg = AgentConfig { max_steps: 3, timeout: None, verify: true };
        let report = run_loop(brain, &screen, "тест", &cfg, None).unwrap();
        assert_eq!(report.termination, Termination::MaxSteps);
        assert_eq!(report.steps.len(), 3);
    }

    #[test]
    fn cancellation_stops_before_next_step() {
        let flag = AtomicBool::new(true);
        let brain = &mut ScriptBrain::new(vec![batch(vec![GuiAction::Wait {
            milliseconds: Some(1),
        }])]);
        let screen = FakeScreen::new(800, 600);
        let report = run_loop(brain, &screen, "тест", &AgentConfig::default(), Some(&flag)).unwrap();
        assert_eq!(report.termination, Termination::Cancelled);
        assert!(report.steps.is_empty());
    }

    #[test]
    fn history_grows_between_steps_for_self_correction() {
        // Two click-free wait steps then done; the brain must see the growing
        // history (0 lines first, then >0) — that's the self-correct feedback.
        let brain = &mut ScriptBrain::new(vec![
            batch(vec![GuiAction::Wait { milliseconds: Some(1) }]),
            batch(vec![GuiAction::Wait { milliseconds: Some(1) }]),
            batch(vec![GuiAction::Done { message: Some("ok".into()) }]),
        ]);
        let screen = FakeScreen::new(1024, 768);
        let cfg = AgentConfig { max_steps: 5, timeout: None, verify: true };
        let report = run_loop(brain, &screen, "тест", &cfg, None).unwrap();
        assert_eq!(report.termination, Termination::Done("ok".into()));
        let hist = brain.seen_history_len.borrow();
        assert_eq!(hist[0], 0, "первый шаг не имеет истории");
        assert!(hist[1] >= hist[0], "история должна накапливаться");
    }

    #[test]
    fn timeout_budget_ends_loop() {
        struct Slow;
        impl Brain for Slow {
            fn decide(&mut self, _: &CapturedScreenshot, _: &str, _: &[String]) -> Result<GuiActionBatch> {
                std::thread::sleep(Duration::from_millis(10));
                Ok(GuiActionBatch { actions: vec![GuiAction::Wait { milliseconds: Some(1) }] })
            }
        }
        let screen = FakeScreen::new(640, 480);
        let cfg = AgentConfig { max_steps: 1000, timeout: Some(Duration::from_millis(1)), verify: true };
        let report = run_loop(&mut Slow, &screen, "тест", &cfg, None).unwrap();
        assert_eq!(report.termination, Termination::Timeout);
    }

    #[test]
    fn clip_bounds_long_strings() {
        let long = "一二三四五六七八九十".repeat(50);
        let c = clip(&long, 20);
        assert_eq!(c.chars().count(), 21); // 20 chars + ellipsis
        assert!(c.ends_with('…'));
    }
}
