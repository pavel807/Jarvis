//! Multimodal runtime: vision→decoder fusion, incremental KV decode, and a
//! real generation engine (greedy / temperature / top-k / top-p / repetition
//! penalty / EOS / stop-strings / streaming / cancellation).
//!
//! The generation *engine* ([`generate`]) is generic over [`StepModel`] so it can
//! be exercised on tiny CPU fixtures without the 4.88 GB checkpoint. The
//! concrete [`UiVenusModel`] implements [`StepModel`] by wiring the verified
//! image processor, the vision tower, the text decoder and the HF tokenizer
//! together; it can only be constructed once the real weights are present —
//! there is no simulated fallback.

use super::vision::Qwen3VlVision;
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use rand::Rng;
use rand::SeedableRng;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokenizers::Tokenizer;

// =========================================================================
// Generation parameters
// =========================================================================

/// Everything a caller can tune about one generation run.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    /// 0.0 means greedy (argmax). Positive means sample with this temperature.
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    /// Multiplicative penalty on tokens already generated (1.0 disables it).
    pub repetition_penalty: f64,
    pub seed: Option<u64>,
    /// Token ids that stop generation in addition to the model's own EOS set.
    pub extra_eos_ids: Vec<u32>,
    /// Surface-level stop strings; generation halts at the first hit and the
    /// emitted text is trimmed before it.
    pub stop_strings: Vec<String>,
    /// Wall-clock budget. `None` disables the timeout.
    pub timeout: Option<std::time::Duration>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            seed: None,
            extra_eos_ids: Vec::new(),
            stop_strings: Vec::new(),
            timeout: None,
        }
    }
}

/// Why a generation run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Model emitted an end-of-sequence token.
    Eos,
    /// A configured stop string matched the decoded text.
    StopString,
    /// `max_new_tokens` reached.
    MaxTokens,
    /// A streaming callback or the cancel flag requested cancellation.
    Cancelled,
    /// The wall-clock budget expired.
    Timeout,
}

/// One streamed update. `delta` is the newly decoded text since the last
/// event; the raw token id is included for token-level consumers.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    pub token_id: u32,
    pub delta: String,
}

/// Final result of [`generate`].
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub stop_reason: StopReason,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub elapsed_ms: u128,
    /// Tokens per second over the decode window.
    pub decode_tps: f64,
}

// =========================================================================
// StepModel: what the engine needs from any model
// =========================================================================

/// The minimal interface the generation engine drives. `prefill` runs the whole
/// prompt at once and returns the logits for the next-token position; `step`
/// advances one token using the model's internal cache. `begin_sequence` must
/// fully reset per-session state so a stale cache never leaks across requests.
pub trait StepModel {
    /// Resets internal caches and records that a `prompt_len`-token sequence is
    /// about to be run. Called once before every [`StepModel::prefill`].
    fn begin_sequence(&mut self, prompt_len: usize);
    /// Runs the prompt, returns `(vocab,)` logits for the next-token position.
    fn prefill(&mut self, prompt_ids: &[u32]) -> Result<Tensor>;
    /// Feeds one newly generated token, returns `(vocab,)` next-token logits.
    fn step(&mut self, token: u32) -> Result<Tensor>;
    /// Decodes a token id buffer to text (for streaming deltas).
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    /// Model's end-of-sequence token ids.
    fn eos_ids(&self) -> Vec<u32>;
}

// =========================================================================
// Generation engine
// =========================================================================

/// Drives one generation run over a [`StepModel`].
///
/// `sink` receives a [`StreamEvent`] per accepted token and returns `false` to
/// request cancellation (the run stops with [`StopReason::Cancelled`]).
/// `cancel`, if provided, is polled between tokens so an out-of-band flag
/// (voice barge-in, UI stop button) also halts generation.
pub fn generate(
    model: &mut dyn StepModel,
    prompt_ids: &[u32],
    params: &SamplingParams,
    cancel: Option<&AtomicBool>,
    mut sink: impl FnMut(&StreamEvent) -> bool,
) -> Result<GenerationOutput> {
    if prompt_ids.is_empty() {
        bail!("Пустой промпт: нечего подавать модели");
    }
    if params.max_new_tokens == 0 {
        bail!("max_new_tokens должен быть больше нуля");
    }

    let start = Instant::now();
    let mut eos = model.eos_ids();
    eos.extend(params.extra_eos_ids.iter().copied());
    let mut rng = match params.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::from_entropy(),
    };

    model.begin_sequence(prompt_ids.len());
    let mut logits = model.prefill(prompt_ids)?;
    let mut generated: Vec<u32> = Vec::with_capacity(params.max_new_tokens);
    // Number of characters already emitted via streaming.
    let mut streamed_chars = 0usize;
    let mut stop_reason = StopReason::MaxTokens;
    let mut trimmed_text: Option<String> = None;

    for _step_index in 0..params.max_new_tokens {
        if cancel.is_some_and(|f| f.load(Ordering::Relaxed)) {
            stop_reason = StopReason::Cancelled;
            break;
        }
        if let Some(budget) = params.timeout {
            if start.elapsed() >= budget {
                stop_reason = StopReason::Timeout;
                break;
            }
        }

        let mut vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let token = sample_token(&mut vec, &generated, params, &mut rng)? as u32;

        if eos.contains(&token) {
            stop_reason = StopReason::Eos;
            break;
        }
        generated.push(token);

        // Streaming delta: re-decode the buffer and emit only the new tail.
        let decoded = model.decode(&generated)?;
        let delta = tail_delta(&decoded, streamed_chars);
        streamed_chars += delta.chars().count();
        let event = StreamEvent { token_id: token, delta };
        if !sink(&event) {
            stop_reason = StopReason::Cancelled;
            break;
        }

        // Stop-string detection on the decoded surface text.
        if !params.stop_strings.is_empty() {
            if let Some(cut) = earliest_stop(&decoded, &params.stop_strings) {
                trimmed_text = Some(decoded[..cut].to_string());
                stop_reason = StopReason::StopString;
                break;
            }
        }

        logits = if generated.len() < params.max_new_tokens {
            model.step(token)?
        } else {
            logits // budget exhausted; the loop will end without another step
        };
    }

    // Final text: use the trimmed version if a stop string fired, otherwise a
    // clean re-decode of the generated buffer.
    let text = match trimmed_text {
        Some(t) => t,
        None => model.decode(&generated)?,
    };
    let elapsed = start.elapsed();
    let decode_tps = if generated.is_empty() {
        0.0
    } else {
        generated.len() as f64 * 1000.0 / elapsed.as_secs_f64().max(1e-6)
    };

    Ok(GenerationOutput {
        text,
        generated_tokens: generated.len(),
        tokens: generated,
        stop_reason,
        prompt_tokens: prompt_ids.len(),
        elapsed_ms: elapsed.as_millis(),
        decode_tps,
    })
}

/// Returns the characters of `full` beyond the first `streamed` chars. Because
/// the buffer is re-decoded cumulatively each step, the emitted tail is always
/// valid UTF-8 (never splits a character).
fn tail_delta(full: &str, streamed: usize) -> String {
    let total = full.chars().count();
    if streamed >= total {
        return String::new();
    }
    full.chars().skip(streamed).collect()
}

/// Index into `text` of the earliest stop string, if any.
fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

// =========================================================================
// Sampler
// =========================================================================

/// Turns a raw logits row into the next token id, honouring repetition
/// penalty, temperature, top-k and top-p. Greedy when `temperature == 0`.
fn sample_token(
    logits: &mut [f32],
    generated: &[u32],
    params: &SamplingParams,
    rng: &mut rand::rngs::StdRng,
) -> Result<usize> {
    if logits.is_empty() {
        bail!("Пустая строка логитов");
    }
    if params.repetition_penalty != 1.0 {
        apply_repetition_penalty(logits, generated, params.repetition_penalty);
    }

    if params.temperature <= 0.0 {
        return Ok(argmax(logits));
    }

    // Temperature scaling.
    let inv_t = 1.0 / params.temperature as f32;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }

    // Top-k truncation: mask everything below the k-th largest to -inf.
    if let Some(k) = params.top_k {
        if k > 0 && k < logits.len() {
            let mut sorted = logits.to_vec();
            sorted.sort_unstable_by(|a, b| b.total_cmp(a));
            let threshold = sorted[k - 1];
            for v in logits.iter_mut() {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Top-p (nucleus) truncation.
    if let Some(p) = params.top_p {
        if p > 0.0 && p < 1.0 {
            apply_top_p(logits, p);
        }
    }

    // Softmax + multinomial draw over the (possibly masked) row.
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    let mut probs = Vec::with_capacity(logits.len());
    for &v in logits.iter() {
        if v == f32::NEG_INFINITY {
            probs.push(0.0f64);
            continue;
        }
        let e = ((v - max_v) as f64).exp();
        sum += e;
        probs.push(e);
    }
    if sum <= 0.0 {
        // Degenerate (everything masked) — fall back to argmax of unmasked vals.
        return Ok(argmax(logits));
    }
    let draw = rng.gen_range(0.0..sum);
    let mut acc = 0.0f64;
    for (i, &pr) in probs.iter().enumerate() {
        acc += pr;
        if draw < acc {
            return Ok(i);
        }
    }
    Ok(probs.len() - 1)
}

/// First index of the maximal value (matches torch.argmax tie-breaking).
fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v && v != f32::NEG_INFINITY {
            best_v = v;
            best = i;
        }
    }
    best
}

/// HF/generate-style repetition penalty: divide positive logits of seen tokens
/// by the penalty and multiply negative ones.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f64) {
    let p = penalty as f32;
    for &id in generated {
        let i = id as usize;
        if i < logits.len() {
            logits[i] = if logits[i] > 0.0 { logits[i] / p } else { logits[i] * p };
        }
    }
}

/// Nucleus sampling: drop the lowest-probability tokens whose cumulative mass
/// would exceed `top_p`, by masking them to -inf. Always keeps at least one.
fn apply_top_p(logits: &mut [f32], top_p: f64) {
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let e = if v == f32::NEG_INFINITY { 0.0 } else { ((v - max_v) as f64).exp() };
            (i, e)
        })
        .collect();
    let total: f64 = exps.iter().map(|(_, e)| e).sum();
    if total <= 0.0 {
        return;
    }
    exps.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let mut cumulative = 0.0f64;
    let mut cutoff_index = exps.len();
    for (rank, (_, e)) in exps.iter().enumerate() {
        cumulative += e / total;
        if cumulative > top_p {
            cutoff_index = rank;
            break;
        }
    }
    let keep: std::collections::HashSet<usize> =
        exps[..cutoff_index.max(1)].iter().map(|(i, _)| *i).collect();
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

// =========================================================================
// Concrete UI-Venus model (only constructible with real weights)
// =========================================================================

// =========================================================================
// Tests (engine + sampler, no weights required)
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A scripted [`StepModel`]: emits a fixed token sequence, then EOS. The
    /// logits always point at the next scripted token, so we exercise the whole
    /// engine (greedy, EOS, budget, cancellation, reset) without weights.
    struct ScriptedModel {
        script: Vec<u32>,
        cursor: usize,
        vocab: usize,
        eos: u32,
        prefill_calls: usize,
        step_calls:  usize,
        sequences:   usize,
    }

    impl ScriptedModel {
        fn new(script: Vec<u32>, vocab: usize, eos: u32) -> Self {
            Self { script, cursor: 0, vocab, eos, prefill_calls: 0, step_calls: 0, sequences: 0 }
        }
        fn one_hot(&self, token: u32) -> Tensor {
            let mut v = vec![0.0f32; self.vocab];
            v[token as usize] = 5.0;
            Tensor::from_vec(v, (self.vocab,), &Device::Cpu).unwrap()
        }
    }

    impl StepModel for ScriptedModel {
        fn begin_sequence(&mut self, _prompt_len: usize) {
            self.cursor = 0;
            self.sequences += 1;
        }
        fn prefill(&mut self, _ids: &[u32]) -> Result<Tensor> {
            self.prefill_calls += 1;
            let tok = *self.script.first().unwrap_or(&self.eos);
            Ok(self.one_hot(tok))
        }
        fn step(&mut self, _token: u32) -> Result<Tensor> {
            self.step_calls += 1;
            self.cursor += 1;
            let tok = *self.script.get(self.cursor).unwrap_or(&self.eos);
            Ok(self.one_hot(tok))
        }
        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens.iter().map(|t| (b'a' + (*t % 26) as u8) as char).collect())
        }
        fn eos_ids(&self) -> Vec<u32> {
            vec![self.eos]
        }
    }

    #[test]
    fn greedy_stops_at_max_tokens_and_resets_per_sequence() {
        let mut m = ScriptedModel::new(vec![1, 2, 3], 8, 7);
        let out = generate(
            &mut m,
            &[10, 11],
            &SamplingParams { max_new_tokens: 3, ..Default::default() },
            None,
            |_| true,
        )
        .unwrap();
        assert_eq!(out.tokens, vec![1, 2, 3]);
        assert_eq!(out.stop_reason, StopReason::MaxTokens);
        assert_eq!(m.prefill_calls, 1);
        assert_eq!(m.sequences, 1);
        assert_eq!(out.generated_tokens, 3);
    }

    #[test]
    fn eos_token_halts_generation() {
        let mut m = ScriptedModel::new(vec![4, 4, 7, 5], 8, 7);
        let out = generate(
            &mut m,
            &[1],
            &SamplingParams { max_new_tokens: 10, ..Default::default() },
            None,
            |_| true,
        )
        .unwrap();
        assert_eq!(out.tokens, vec![4, 4]); // third draw is EOS → not kept
        assert_eq!(out.stop_reason, StopReason::Eos);
    }

    #[test]
    fn cancellation_stops_stream() {
        let mut m = ScriptedModel::new(vec![1; 20], 8, 7);
        let mut seen = 0;
        let out = generate(
            &mut m,
            &[1],
            &SamplingParams { max_new_tokens: 20, ..Default::default() },
            None,
            |_| {
                seen += 1;
                seen < 5
            },
        )
        .unwrap();
        assert_eq!(out.stop_reason, StopReason::Cancelled);
        assert_eq!(seen, 5);
    }

    #[test]
    fn external_cancel_flag_halts_before_first_token() {
        let mut m = ScriptedModel::new(vec![1; 20], 8, 7);
        let flag = AtomicBool::new(true);
        let out = generate(
            &mut m,
            &[1],
            &SamplingParams { max_new_tokens: 20, ..Default::default() },
            Some(&flag),
            |_| true,
        )
        .unwrap();
        assert_eq!(out.stop_reason, StopReason::Cancelled);
        assert!(out.tokens.is_empty());
    }

    #[test]
    fn timeout_budget_stops_generation() {
        struct Slow;
        impl StepModel for Slow {
            fn begin_sequence(&mut self, _: usize) {}
            fn prefill(&mut self, _: &[u32]) -> Result<Tensor> {
                std::thread::sleep(std::time::Duration::from_millis(5));
                let mut v = vec![0.0f32; 4];
                v[0] = 5.0;
                Ok(Tensor::from_vec(v, (4,), &Device::Cpu).unwrap())
            }
            fn step(&mut self, _: u32) -> Result<Tensor> {
                std::thread::sleep(std::time::Duration::from_millis(5));
                let mut v = vec![0.0f32; 4];
                v[0] = 5.0;
                Ok(Tensor::from_vec(v, (4,), &Device::Cpu).unwrap())
            }
            fn decode(&self, _: &[u32]) -> Result<String> {
                Ok(String::new())
            }
            fn eos_ids(&self) -> Vec<u32> {
                vec![3]
            }
        }
        let mut m = Slow;
        let out = generate(
            &mut m,
            &[1],
            &SamplingParams {
                max_new_tokens: 1000,
                timeout: Some(std::time::Duration::from_millis(1)),
                ..Default::default()
            },
            None,
            |_| true,
        )
        .unwrap();
        assert_eq!(out.stop_reason, StopReason::Timeout);
    }

    #[test]
    fn sampler_greedy_picks_argmax() {
        let mut logits = vec![0.1f32, 3.0, -1.0, 2.0];
        let params = SamplingParams::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        assert_eq!(sample_token(&mut logits, &[], &params, &mut rng).unwrap(), 1);
    }

    #[test]
    fn sampler_repetition_penalty_suppresses_seen_tokens() {
        let mut logits = vec![0.0f32, 3.0, 2.5, 0.0];
        let params =
            SamplingParams { temperature: 0.0, repetition_penalty: 3.0, ..Default::default() };
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let chosen = sample_token(&mut logits, &[1], &params, &mut rng).unwrap();
        assert_eq!(chosen, 2, "повтор token 1 должен был подавиться");
    }

    #[test]
    fn sampler_top_k_restricts_candidates() {
        let logits: Vec<f32> =
            vec![10.0, 5.0, 4.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: Some(3),
            seed: Some(42),
            ..Default::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..50 {
            let c = sample_token(&mut logits.clone(), &[], &params, &mut rng).unwrap();
            assert!(c < 3, "top-k пропустил токен вне тройки: {c}");
        }
    }

    #[test]
    fn sampler_top_p_nucleus_keeps_only_the_peak() {
        let logits = vec![0.0f32, 0.0, 0.0, 8.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: Some(0.2),
            seed: Some(7),
            ..Default::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..200 {
            let c = sample_token(&mut logits.clone(), &[], &params, &mut rng).unwrap();
            assert_eq!(c, 3, "top-p=0.2 оставил только вероятностный пик token 3");
        }
    }

    #[test]
    fn apply_top_p_masks_low_mass() {
        let mut logits = vec![0.0f32, 0.0, 8.0, 0.0];
        apply_top_p(&mut logits, 0.1);
        assert_eq!(logits[2], 8.0);
        assert!(logits[0] == f32::NEG_INFINITY && logits[1] == f32::NEG_INFINITY);
    }

    #[test]
    fn prefill_runs_once_and_steps_after() {
        let mut m = ScriptedModel::new(vec![1, 2, 3], 8, 7);
        generate(
            &mut m,
            &[9, 9],
            &SamplingParams { max_new_tokens: 2, ..Default::default() },
            None,
            |_| true,
        )
        .unwrap();
        assert_eq!(m.prefill_calls, 1);
        assert_eq!(m.step_calls, 1); // only steps after the first sampled token
    }

    #[test]
    fn tail_delta_splits_on_char_boundary() {
        // Multi-byte text; streaming must never split a char.
        let full = "привет мир";
        let first = tail_delta(full, 0);
        assert_eq!(first, full); // whole thing first time
        let rest = tail_delta(full, full.chars().count());
        assert_eq!(rest, ""); // nothing new
        // Splitting at 3 chars yields the remaining chars, valid UTF-8.
        let tail = tail_delta(full, 3);
        assert_eq!(format!("при{tail}"), full);
    }
}
