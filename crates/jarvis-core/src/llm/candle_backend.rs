//! Candle-бэкенд LLM: MiniCPM5-1B-ru и Qwen3.5-0.8B, реализации архитектур
//! на чистом candle + Metal. Был единственным движком `LlmEngine`, теперь
//! стоит за диспетчером в `mod.rs` (второй бэкенд — uzu на macOS).

use super::config::{self, ModelConfig};
use super::streaming::{MAX_SENTENCE_CHARS, find_chunk_end, speakable, strip_think};
use super::template;
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Терминаторы ответа: добавляются к eos из конфига, если есть в словаре.
const EXTRA_STOP_TOKENS: &[&str] = &["<|eot_id|>", "<|im_end|>", "<|endoftext|>"];

fn pick_device() -> Device {
    // Metal есть только на macOS: на Linux и Windows считаем на CPU
    // (сначала пробуем его на всех ОС, чтобы не тащить cfg в горячий путь).
    #[cfg(target_os = "macos")]
    match Device::new_metal(0) {
        Ok(d) => {
            println!("✅ LLM: устройство Metal");
            return d;
        }
        Err(e) => {
            eprintln!("⚠️ Metal недоступен ({e}), LLM будет работать на CPU");
        }
    }
    Device::Cpu
}

/// Бюджет генерации: сколько токенов отведено ответу.
///
/// `max_tokens = 0` — **без искусственного лимита**: весь остаток контекста
/// модели; обрыв ответа по счётчику давал ложные «не могу» на длинные
/// команды, поэтому по умолчанию лимита нет. Ненулевое значение —
/// страховочный кап; контекст (rope) — жёсткая граница в любом случае.
fn generation_budget(rope_len: usize, prompt_len: usize, max_tokens: usize) -> usize {
    let headroom = rope_len.saturating_sub(prompt_len + 8);
    if max_tokens == 0 {
        headroom
    } else {
        max_tokens.min(headroom)
    }
}

/// Загруженная модель вместе с её (модельно-специфичным) кэшем.
enum CandleModel {
    /// MiniCPM5-1B-ru (Llama-архитектура, KV-кэш на каждый слой).
    Llama {
        model: super::model::LlamaModel,
        caches: Vec<super::model::KvCache>,
    },
    /// Qwen3.5-0.8B (гибрид Gated DeltaNet + gated attention).
    Qwen35 {
        model: super::qwen35::Qwen35Model,
        cache: super::qwen35::Qwen35Cache,
    },
}

impl CandleModel {
    fn forward(&mut self, ids: &[u32], start_pos: usize) -> Result<candle_core::Tensor> {
        match self {
            Self::Llama { model, caches } => model.forward(ids, start_pos, caches),
            Self::Qwen35 { model, cache } => model.forward(ids, start_pos, cache),
        }
    }

    fn rope_len(&self) -> usize {
        match self {
            Self::Llama { model, .. } => model.rope_len(),
            Self::Qwen35 { model, .. } => model.rope_len(),
        }
    }

    fn num_hidden_layers(&self) -> usize {
        match self {
            Self::Llama { model, .. } => model.config().num_hidden_layers,
            Self::Qwen35 { model, .. } => model.config.num_hidden_layers,
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Llama { model, .. } => model.config().vocab_size,
            Self::Qwen35 { model, .. } => model.config.vocab_size,
        }
    }

    fn reset_caches(&mut self) {
        match self {
            Self::Llama { caches, .. } => {
                for c in caches {
                    c.reset();
                }
            }
            Self::Qwen35 { model, cache } => {
                *cache = model.new_cache();
            }
        }
    }
}

/// Загружает модель из каталога по её config.json (qwen3_5_text → Qwen3.5,
/// иначе — Llama-семейство: MiniCPM5).
fn load_model_dir(
    models_dir: &std::path::Path,
) -> Result<(CandleModel, Tokenizer, Vec<u32>, bool, &'static str)> {
    let dir = models_dir.join(config::model_name());
    // Qwen3.5-0.8B не качается ни из HF, ни из Mirai — тянем из релиза
    // jarvis-models, если файлов ещё нет.
    if config::model_name() == super::qwen35::LOCAL_DIR_NAME {
        crate::model_fetch::ensure_qwen35(&dir)
            .with_context(|| format!("Не удалось подготовить Qwen3.5 в {}", dir.display()))?;
    }
    if super::qwen35::Qwen35Config::is_qwen35_config(&dir.join("config.json")) {
        let device = pick_device();
        let model = super::qwen35::Qwen35Model::load(&dir, &device)
            .with_context(|| format!("Не удалось загрузить Qwen3.5 из {}", dir.display()))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {e}"))?;
        let mut eos_ids = model.config.eos_ids.clone();
        for stop in EXTRA_STOP_TOKENS {
            if let Some(id) = tokenizer.token_to_id(stop)
                && !eos_ids.contains(&id)
            {
                eos_ids.push(id);
            }
        }
        println!(
            "   LLM: модель {}/qwen3_5, eos_ids={:?}",
            config::model_name(),
            eos_ids
        );
        let cache = model.new_cache();
        Ok((
            CandleModel::Qwen35 { model, cache },
            tokenizer,
            eos_ids,
            false,
            template::QWEN_ASSISTANT_PREFIX,
        ))
    } else {
        let started = Instant::now();
        let files = config::ensure_model_downloaded(models_dir)?;
        let cfg = ModelConfig::load(&files.config)?;
        let eos_ids = cfg.eos_ids();
        let bos_id = cfg.bos_id();
        let prompt_bos = cfg.use_bos_in_template();

        let device = pick_device();
        let model = super::model::LlamaModel::load(&files, cfg, &device)
            .context("Не удалось загрузить веса MiniCPM5-1B")?;

        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {e}"))?;
        let mut eos_ids = eos_ids;
        for stop in EXTRA_STOP_TOKENS {
            if let Some(id) = tokenizer.token_to_id(stop)
                && !eos_ids.contains(&id)
            {
                eos_ids.push(id);
            }
        }
        println!(
            "   LLM: bos_id={:?}, eos_ids={:?}, слоёв={}, vocab={}",
            bos_id,
            eos_ids,
            model.config().num_hidden_layers,
            model.config().vocab_size
        );
        println!("✅ MiniCPM5-1B загружена за {:.1?}", started.elapsed());
        let caches = vec![super::model::KvCache::new(); model.config().num_hidden_layers];
        Ok((
            CandleModel::Llama { model, caches },
            tokenizer,
            eos_ids,
            prompt_bos,
            "",
        ))
    }
}

pub struct CandleEngine {
    engine: CandleModel,
    tokenizer: Tokenizer,
    eos_ids: Vec<u32>,
    prompt_bos: bool,
    /// Дописка после `<|im_start|>assistant\n`: у Qwen3.5 — принудительно
    /// закрытый блок рассуждений, у MiniCPM5 — пустая строка.
    assistant_prefix: &'static str,
    /// Токены, уже лежащие в KV-кэше (префикс последнего промпта + ответа).
    /// Следующий вызов с тем же префиксом префиллится только с дельты.
    prefix_cache: Vec<u32>,
}

impl CandleEngine {
    /// Загрузка модели из `models/` (каталог задаёт `llm_model` в settings.toml).
    pub fn load(models_dir: &std::path::Path) -> Result<Self> {
        let started = Instant::now();
        let (engine, tokenizer, eos_ids, prompt_bos, assistant_prefix) =
            load_model_dir(models_dir)?;

        println!(
            "   LLM: слоёв={}, vocab={}, потолок контекста={}",
            engine.num_hidden_layers(),
            engine.vocab_size(),
            engine.rope_len()
        );
        println!("✅ LLM загружена за {:.1?}", started.elapsed());

        Ok(Self {
            engine,
            tokenizer,
            eos_ids,
            prompt_bos,
            assistant_prefix,
            prefix_cache: Vec::new(),
        })
    }

    fn reset(&mut self) {
        self.engine.reset_caches();
        self.prefix_cache.clear();
    }

    pub fn generate_streaming(
        &mut self,
        system: &str,
        history: &[template::ChatTurn],
        user: &str,
        cancel: &AtomicBool,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_streaming_with_params(
            system,
            history,
            user,
            None,
            cancel,
            on_sentence,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_params(
        &mut self,
        system: &str,
        history: &[template::ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_streaming_inner(
            system,
            history,
            user,
            params_override,
            cancel,
            None,
            on_sentence,
            on_progress,
        )
    }

    /// Генерация с «вставленным началом ответа ассистента»: `prefill`
    /// дописывается в промпт сразу после заголовка `assistant`, и модель
    /// продолжает только его. Так она не тратит токены на рассуждения
    /// и не может выдать кривой JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_prefill(
        &mut self,
        system: &str,
        history: &[template::ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        prefill: &str,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_streaming_inner(
            system,
            history,
            user,
            params_override,
            cancel,
            Some(prefill),
            on_sentence,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_streaming_inner(
        &mut self,
        system: &str,
        history: &[template::ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        prefill: Option<&str>,
        mut on_sentence: impl FnMut(&str),
        mut on_progress: impl FnMut(&str),
    ) -> Result<String> {
        let mut prompt = template::build_chat_prompt(
            system,
            history,
            user,
            if self.prompt_bos {
                Some(template::MINICPM_BOS)
            } else {
                None
            },
            self.assistant_prefix,
        );
        if let Some(prefix) = prefill {
            prompt.push_str(prefix);
        }
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Ошибка токенизации промпта: {e}"))?;
        let mut context: Vec<u32> = encoding.get_ids().to_vec();
        if context.is_empty() {
            anyhow::bail!("Промпт не дал токенов");
        }

        let params = params_override.unwrap_or_else(|| {
            config::generation_params()
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        });
        // Бюджет генерации: 0 — без искусственного лимита (весь остаток
        // контекста). Останов держат EOS, фильтр зацикливания, закрытый
        // tool-JSON и отмена — но не обрыв длинного ответа по счётчику.
        let token_budget =
            generation_budget(self.engine.rope_len(), context.len(), params.max_tokens);
        if token_budget == 0 {
            anyhow::bail!("Промпт слишком длинный: {} токенов", context.len());
        }

        // Префиксный кэш: если начало промпта уже посчитано в KV-кэше
        // (системный промпт + история + прошлые ходы агентного цикла),
        // префиллим только дельту.
        let cached_len = self.prefix_cache.len();
        let reusable = cached_len > 0
            && cached_len < context.len()
            && context[..cached_len] == self.prefix_cache[..];
        let prefill_from = if reusable {
            if context.len() - cached_len > 1 {
                println!(
                    "   LLM: префикс {} ток. в кэше, допрефилл {} ток.",
                    cached_len,
                    context.len() - cached_len
                );
            }
            cached_len
        } else {
            self.reset();
            0
        };
        let mut rng = rand::thread_rng();
        let prompt_len = context.len();
        let mut suppress_think = false;
        let mut loop_hits = 0u32;
        const LOOP_WINDOW: usize = 10;
        const LOOP_MAX_HITS: u32 = 3;

        let t_prefill = Instant::now();
        let mut logits = self
            .engine
            .forward(&context[prefill_from..], prefill_from)?;
        let mut next_pos = context.len();
        let t_prefill = t_prefill.elapsed();
        let t_decode = Instant::now();
        let mut text = String::new();
        let mut buf = String::new();

        // Флаг tool-call: если ответ начинается с <tool>, весь текст идёт
        // только в буфер, минуя TTS.
        let mut is_tool_call: Option<bool> = None;
        let mut tool_probe = String::new();
        if let Some(prefix) = prefill {
            text.push_str(prefix);
            tool_probe.push_str(prefix);
            is_tool_call = Some(prefix.trim_start().starts_with("<tool>"));
        }

        loop {
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            let last = logits.dim(0)? - 1;
            let mut row: Vec<f32> = logits
                .i((last, ..))?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            let penalty = params.repeat_penalty + 0.6 * loop_hits as f32;
            super::sampler::apply_repeat_penalty(&mut row, penalty, &context, params.repeat_window);

            let tok =
                super::sampler::sample_top_p(&row, params.temperature, params.top_p, &mut rng);
            if self.eos_ids.contains(&tok) {
                break;
            }

            context.push(tok);

            let generated = &context[prompt_len..];
            if generated.len() >= LOOP_WINDOW * 2
                && generated[generated.len() - LOOP_WINDOW..]
                    == generated[generated.len() - 2 * LOOP_WINDOW..generated.len() - LOOP_WINDOW]
            {
                loop_hits += 1;
                if loop_hits >= LOOP_MAX_HITS {
                    println!("🛑 LLM зациклилась — останавливаю генерацию");
                    break;
                }
            }

            let piece = self
                .tokenizer
                .decode(&[tok], true)
                .map_err(|e| anyhow::anyhow!("Ошибка декодирования токена: {e}"))?;

            let piece = strip_think(&piece, &mut suppress_think);
            if piece.is_empty() {
                if context.len() - prompt_len >= token_budget {
                    break;
                }
                logits = self.engine.forward(&[tok], next_pos)?;
                next_pos += 1;
                continue;
            }

            text.push_str(&piece);

            if is_tool_call.is_none() {
                tool_probe.push_str(&piece);
                if tool_probe.trim_start().len() >= 6 {
                    let trimmed = tool_probe.trim_start();
                    is_tool_call = Some(trimmed.starts_with("<tool>"));
                }
            }

            // JSON инструмента уже валиден — продолжать незачем.
            if is_tool_call == Some(true) && crate::tools::tool_json_complete(&text) {
                println!("   ⏹ tool-JSON закрыт — генерация остановлена");
                break;
            }

            if is_tool_call == Some(true) {
                if context.len() - prompt_len >= token_budget {
                    break;
                }
                logits = self.engine.forward(&[tok], next_pos)?;
                next_pos += 1;
                continue;
            }

            buf.push_str(&piece);

            while let Some(pos) = find_chunk_end(&buf) {
                let sentence: String = buf.drain(..pos).collect();
                if sentence.contains("<tool>") || sentence.contains("</tool>") {
                    continue;
                }
                if speakable(&sentence) {
                    on_sentence(&sentence);
                    on_progress(&text);
                }
            }

            if buf.chars().count() > MAX_SENTENCE_CHARS
                && let Some(pos) = buf.rfind(' ')
            {
                let piece_flush: String = buf.drain(..pos + 1).collect();
                if !piece_flush.contains("<tool>")
                    && !piece_flush.contains("</tool>")
                    && speakable(&piece_flush)
                {
                    on_sentence(&piece_flush);
                    on_progress(&text);
                }
            }

            if context.len() - prompt_len >= token_budget {
                break;
            }
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            logits = self.engine.forward(&[tok], next_pos)?;
            next_pos += 1;
        }

        if is_tool_call != Some(true) && !cancel.load(Ordering::SeqCst) {
            let tail = std::mem::take(&mut buf);
            if !tail.contains("<tool>") && !tail.contains("</tool>") && speakable(&tail) {
                on_sentence(&tail);
            }
        }

        let decoded = context.len() - prompt_len;
        let decode_ms = t_decode.elapsed().as_millis() as f64;
        println!(
            "   LLM: префилл {} ток. за {:.0?}, декод {} ток. за {:.0?} ({:.1} мс/ток.)",
            prompt_len,
            t_prefill,
            decoded,
            decode_ms,
            if decoded > 0 {
                decode_ms / decoded as f64
            } else {
                0.0
            }
        );

        self.prefix_cache = context[..next_pos.min(context.len())].to_vec();

        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::generation_budget;

    #[test]
    fn zero_max_tokens_means_no_artificial_cap() {
        // 0 — весь остаток контекста (минус 8 токенов запаса на закрытие).
        assert_eq!(generation_budget(8192, 100, 0), 8084);
        // Ненулевое значение — страховочный кап.
        assert_eq!(generation_budget(8192, 100, 256), 256);
        // Контекст — жёсткая граница: кап не выходит за остаток.
        assert_eq!(generation_budget(8192, 8000, 256), 184);
        assert_eq!(generation_budget(8192, 8100, 0), 84);
        // Промпт занял весь контекст — бюджета нет (вызывающий даёт ошибку).
        assert_eq!(generation_budget(8192, 8192, 0), 0);
    }
}
