//! Локальный LLM (Qwen/Qwen3-1.7B на candle + Metal): фоновая загрузка,
//! потоковая генерация с озвучкой по предложениям и отменой.
//!
//! Модули:
//! - [`config`] — конфиг модели и загрузка файлов с Hugging Face;
//! - [`template`] — ChatML-промпт (включая историю диалога);
//! - [`sampler`] — repeat penalty + top-p сэмплинг;
//! - [`model`] — реализация архитектуры (GQA, QK-norm, явный head_dim, KV-кэш).

pub mod config;
pub mod model;
pub mod sampler;
pub mod template;

use crate::events::{EventTx, JarvisEvent};
use crate::tts::AudioMessage;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp};
use config::ModelConfig;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

pub use template::ChatTurn;

/// Сколько последних реплик (вопрос + ответ = 2 записи) держим в контексте.
pub const MAX_HISTORY_ENTRIES: usize = 6;

/// Памятка: ответ будет озвучен TTS, поэтому — короткий устный русский текст.
pub const SYSTEM_PROMPT: &str = "Ты Джарвис — голосовой ассистент на устройстве пользователя. Отвечай только на русском языке, кратко и разговорно: 2-4 предложения, максимум 300 символов. Без markdown, без списков, без эмодзи и без служебных символов — твой ответ озвучивается синтезом речи. Если вопрос неясен, коротко переспроси.";

/// Промпт для ответов по результатам веб-поиска.
pub const WEB_SYSTEM_PROMPT: &str = "Ты Джарвис — голосовой ассистент. Пользователь задал вопрос, тебе даны свежие результаты поиска в интернете. Отвечай строго на русском языке, кратко и по существу: 1-3 предложения. Используй только факты из результатов. Без markdown, без ссылок, без эмодзи — ответ озвучивается синтезом речи.";

/// Первая загрузка может качать ~2.5 ГБ весов из Hugging Face.
const LLM_WAIT_TIMEOUT_SECS: u64 = 300;
/// Принудительный сброс длинных предложений без знаков препинания.
const MAX_SENTENCE_CHARS: usize = 240;

/// Готовый к работе LLM-движок: владение передаётся между потоками через
/// `Mutex` (все поля Send).
pub struct LlmEngine {
    model: model::LlamaModel,
    tokenizer: Tokenizer,
    caches: Vec<model::KvCache>,
    eos_ids: Vec<u32>,
    /// Добавлять ли bos-токен в шаблон (зависит от семейства модели).
    prompt_bos: bool,
}

impl LlmEngine {
    pub fn load(models_dir: &std::path::Path) -> Result<Self> {
        let started = Instant::now();
        let files = config::ensure_model_downloaded(models_dir)?;
        let cfg = ModelConfig::load(&files.config)?;
        let mut eos_ids = cfg.eos_ids();
        let bos_id = cfg.bos_id();
        let prompt_bos = cfg.use_bos_in_template();

        let device = match Device::new_metal(0) {
            Ok(d) => {
                println!("✅ LLM: устройство Metal");
                d
            }
            Err(e) => {
                eprintln!("⚠️ Metal недоступен ({e}), LLM будет работать на CPU");
                Device::Cpu
            }
        };

        let model = model::LlamaModel::load(&files, cfg, &device)
            .context("Не удалось загрузить веса LLM")?;

        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {e}"))?;

        // Стоп-токены: eos из config.json + закрытие реплики ассистента.
        if let Some(im_end) = tokenizer.token_to_id("<|im_end|>") {
            if !eos_ids.contains(&im_end) {
                eos_ids.push(im_end);
            }
        }
        println!(
            "   LLM: bos_id={:?}, eos_ids={:?}, слоёв={}, vocab={}",
            bos_id,
            eos_ids,
            model.config().num_hidden_layers,
            model.config().vocab_size
        );

        println!("✅ LLM загружена за {:.1?}", started.elapsed());

        let caches = vec![model::KvCache::new(); model.config().num_hidden_layers];

        Ok(Self {
            model,
            tokenizer,
            caches,
            eos_ids,
            prompt_bos,
        })
    }

    fn reset(&mut self) {
        for c in &mut self.caches {
            c.reset();
        }
    }

    /// Генерация ответа. Не блокирует вызывающего снаружи: каждое готовое
    /// предложение уходит в `on_sentence`, накопленный текст — в `on_progress`.
    /// Флаг `cancel` прерывает генерацию между токенами.
    pub fn generate_streaming(
        &mut self,
        system: &str,
        history: &[ChatTurn],
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

    /// То же с переопределением параметров генерации (для служебных задач).
    pub fn generate_streaming_with_params(
        &mut self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        mut on_sentence: impl FnMut(&str),
        mut on_progress: impl FnMut(&str),
    ) -> Result<String> {
        let prompt = template::build_chat_prompt(
            system,
            history,
            user,
            if self.prompt_bos {
                Some(template::MINICPM_BOS)
            } else {
                None
            },
        );
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Ошибка токенизации промпта: {e}"))?;
        let mut context: Vec<u32> = encoding.get_ids().to_vec();
        if context.is_empty() {
            bail!("Промпт не дал токенов");
        }
        // Параметры генерации берём из глобальных настроек на старте ответа.
        let params = params_override.unwrap_or_else(|| {
            config::generation_params()
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        });
        if context.len() + params.max_tokens >= self.model.rope_len() {
            bail!("Промпт слишком длинный: {} токенов", context.len());
        }

        self.reset();
        let mut rng = rand::thread_rng();
        let prompt_len = context.len();
        let mut suppress_think = false;
        // Анти-зацикливание: при обнаружении повторов усиливаем штраф,
        // после нескольких срабатываний останавливаем генерацию.
        let mut loop_hits = 0u32;
        const LOOP_WINDOW: usize = 10;
        const LOOP_MAX_HITS: u32 = 3;

        // Префилл всего промпта за один forward, декод — по одному токену.
        let t_prefill = Instant::now();
        let mut logits = self.model.forward(&context, 0, &mut self.caches)?;
        let mut next_pos = context.len();
        let t_prefill = t_prefill.elapsed();
        let t_decode = Instant::now();
        let mut text = String::new();
        let mut buf = String::new();

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
            sampler::apply_repeat_penalty(&mut row, penalty, &context, params.repeat_window);

            let tok =
                sampler::sample_top_p(&row, params.temperature, params.top_p, &mut rng);

            if self.eos_ids.contains(&tok) {
                break;
            }

            context.push(tok);

            // Детектор зацикливания: хвост из LOOP_WINDOW токенов повторил
            // предыдущий такой же блок — модель ходит по кругу.
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

            // Страховка: не озвучиваем и не показываем блок размышлений.
            let piece = strip_think(&piece, &mut suppress_think);
            if piece.is_empty() {
                if context.len() - prompt_len >= params.max_tokens {
                    break;
                }
                logits = self.model.forward(&[tok], next_pos, &mut self.caches)?;
                next_pos += 1;
                continue;
            }

            text.push_str(&piece);
            buf.push_str(&piece);

            // Выталкиваем готовые предложения в озвучку по мере генерации.
            loop {
                let Some(pos) = find_sentence_end(&buf) else {
                    break;
                };
                let sentence: String = buf.drain(..pos).collect();
                if speakable(&sentence) {
                    on_sentence(&sentence);
                    on_progress(&text);
                }
            }

            // Защита от «бесконечного» предложения без знаков препинания.
            if buf.chars().count() > MAX_SENTENCE_CHARS {
                if let Some(pos) = buf.rfind(' ') {
                    let piece_flush: String = buf.drain(..pos + 1).collect();
                    if speakable(&piece_flush) {
                        on_sentence(&piece_flush);
                        on_progress(&text);
                    }
                }
            }

            if context.len() - prompt_len >= params.max_tokens {
                break;
            }
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            logits = self.model.forward(&[tok], next_pos, &mut self.caches)?;
            next_pos += 1;
        }

        if !cancel.load(Ordering::SeqCst) {
            let tail = std::mem::take(&mut buf);
            if speakable(&tail) {
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
            if decoded > 0 { decode_ms / decoded as f64 } else { 0.0 }
        );

        Ok(text)
    }
}

/// Число сгенерированных токенов считается в generate_streaming как
/// `context.len() - prompt_len`.
fn find_sentence_end(buf: &str) -> Option<usize> {
    buf.find(|c| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(|i| i + 1)
}

fn speakable(sentence: &str) -> bool {
    let trimmed = sentence.trim();
    trimmed.chars().count() >= 8 && trimmed.chars().any(|c| c.is_alphanumeric())
}

/// Убирает блок размышлений `<think>...</think>` из потока токенов.
/// `suppress` — состояние между вызовами: true, пока ищем закрывающий тег.
fn strip_think(piece: &str, suppress: &mut bool) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";

    let mut out = String::new();
    let mut rest = piece;

    loop {
        if *suppress {
            match rest.find(CLOSE) {
                Some(i) => {
                    rest = &rest[i + CLOSE.len()..];
                    *suppress = false;
                }
                None => break,
            }
        } else {
            match rest.find(OPEN) {
                Some(i) => {
                    out.push_str(&rest[..i]);
                    rest = &rest[i + OPEN.len()..];
                    *suppress = true;
                }
                None => {
                    out.push_str(rest);
                    break;
                }
            }
        }
    }

    out
}

pub struct SharedLlmState {
    pub engine: Mutex<Option<LlmEngine>>,
    pub condvar: Condvar,
    pub failed: AtomicBool,
}

pub type SharedLlm = Arc<SharedLlmState>;

impl SharedLlmState {
    pub fn new() -> Self {
        Self {
            engine: Mutex::new(None),
            condvar: Condvar::new(),
            failed: AtomicBool::new(false),
        }
    }
}

/// Фоновая загрузка модели: по завершении уведомляет ожидающих.
pub fn spawn_llm_loader(shared: &SharedLlm, events: &EventTx) {
    let shared = Arc::clone(shared);
    let events = events.clone();

    std::thread::spawn(move || {
        let _ = events.send(JarvisEvent::LlmStatus {
            status: "loading".into(),
        });

        match LlmEngine::load(&crate::paths::models_dir()) {
            Ok(mut engine) => {
                // Прогрев Metal-шейдеров: candle специализирует кернелы по
                // форме тензоров, поэтому прогреваемся на реальной длине
                // промпта (SYSTEM_PROMPT) и с реальным декодом — иначе первый
                // вопрос пользователя платит секунды за компиляцию.
                let cancel = Arc::new(AtomicBool::new(false));
                let warm_params = config::GenerationParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    max_tokens: 16,
                    ..Default::default()
                };
                let started = Instant::now();
                let _ = engine.generate_streaming_with_params(
                    SYSTEM_PROMPT,
                    &[],
                    "Привет",
                    Some(warm_params),
                    &cancel,
                    |_| {},
                    |_| {},
                );
                println!("🔥 LLM прогрета за {:.1?}", started.elapsed());

                let mut guard = shared.engine.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(engine);
                drop(guard);
                shared.condvar.notify_all();
                let _ = events.send(JarvisEvent::LlmStatus {
                    status: "ready".into(),
                });
            }
            Err(e) => {
                eprintln!("❌ Ошибка загрузки LLM: {e}");
                shared.failed.store(true, Ordering::SeqCst);
                shared.condvar.notify_all();
                let _ = events.send(JarvisEvent::LlmStatus {
                    status: "failed".into(),
                });
            }
        }
    });
}

fn wait_for_engine(
    shared: &SharedLlm,
    timeout: Duration,
) -> Option<MutexGuard<'_, Option<LlmEngine>>> {
    let guard = shared.engine.lock().unwrap_or_else(|e| e.into_inner());
    let (guard, _) = shared
        .condvar
        .wait_timeout_while(
            guard,
            timeout,
            |engine| engine.is_none() && !shared.failed.load(Ordering::SeqCst),
        )
        .unwrap_or_else(|e| e.into_inner());

    if guard.is_none() {
        eprintln!("❌ Не дождался готовности LLM за {:?}", timeout);
        None
    } else {
        Some(guard)
    }
}

/// Задать вопрос LLM. Не блокирует вызывающего: генерация идёт в фоне,
/// готовые предложения отправляются в плеер.
///
/// `history` — общий буфер диалога: снапшот читается перед генерацией,
/// успешный обмен дописывается в конец (не более [`MAX_HISTORY_ENTRIES`]).
/// Отмена — через `cancel`; сбрасывать флаг должен тот, кто инициирует
/// запрос, до вызова `ask` (сброс внутри не делал бы смысла: отмена,
/// нажатая до старта потока, была бы затёрта).
pub fn ask(
    shared: SharedLlm,
    system: &'static str,
    prompt: String,
    history: Arc<Mutex<Vec<ChatTurn>>>,
    audio_tx: Sender<AudioMessage>,
    events: EventTx,
    cancel: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let history_snapshot: Vec<ChatTurn> = {
            let buf = history.lock().unwrap_or_else(|e| e.into_inner());
            buf.iter()
                .rev()
                .take(MAX_HISTORY_ENTRIES)
                .rev()
                .cloned()
                .collect()
        };

        let Some(mut guard) = wait_for_engine(&shared, Duration::from_secs(LLM_WAIT_TIMEOUT_SECS))
        else {
            let _ = events.send(JarvisEvent::Error {
                message: "LLM недоступна".into(),
            });
            return;
        };

        let Some(engine) = guard.as_mut() else {
            return;
        };

        let started = Instant::now();
        let result = engine.generate_streaming(
            system,
            &history_snapshot,
            &prompt,
            &cancel,
            |sentence| {
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(sentence.to_string()));
            },
            |full| {
                let _ = events.send(JarvisEvent::LlmChunk {
                    text: full.to_string(),
                });
            },
        );
        drop(guard);

        match result {
            Ok(text) => {
                println!(
                    "🤖 Ответ LLM за {:.1?} ({} симв.)",
                    started.elapsed(),
                    text.chars().count()
                );
                {
                    let mut buf = history.lock().unwrap_or_else(|e| e.into_inner());
                    buf.push((prompt, text.clone()));
                    let excess = buf.len().saturating_sub(MAX_HISTORY_ENTRIES);
                    buf.drain(..excess);
                }
                let _ = events.send(JarvisEvent::LlmDone { text });
            }
            Err(e) => {
                eprintln!("⚠️ Ошибка генерации LLM: {e}");
                let _ = events.send(JarvisEvent::Error {
                    message: format!("Ошибка LLM: {e}"),
                });
            }
        }
    });
}

/// Системный промпт нейро-нормализации текста для TTS.
pub const NORMALIZE_SYSTEM_PROMPT: &str = "Ты нормализатор текста для синтеза речи на русском. Перепиши текст, заменив все цифры, числа, единицы измерения, сокращения и символы на их произнесение русскими словами с правильными падежами. Примеры: «2 м/с» → «два метра в секунду»; «21°C» → «двадцать один градус Цельсия»; «1000» → «тысяча»; «12:30» → «двенадцать тридцать»; «iPhone 15» → «айфон пятнадцать». Ничего кроме этих замен не меняй: не переводи, не сокращай, не дополняй и не комментируй. Верни только готовый текст одной строкой без кавычек.";

/// Санитизация выхода нормализатора: защита от галлюцинаций и порчи текста.
pub fn sanitize_normalized(original: &str, output: &str) -> Option<String> {
    let out = output
        .trim()
        .trim_start_matches(['"', '\u{ab}', '\u{bb}'])
        .trim_end_matches(['"', '\u{ab}', '\u{bb}'])
        .trim()
        .to_string();

    if out.is_empty() {
        return None;
    }
    if !out.chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    if out.contains("пример") || out.contains("замени") {
        return None;
    }
    let orig_chars = original.chars().count();
    let out_chars = out.chars().count();
    if out_chars > orig_chars * 4 + 40 || out_chars < orig_chars / 4 {
        return None;
    }
    Some(out)
}

/// Нейро-нормализация текста для TTS без ожидания: если движок занят
/// генерацией ответа (а во время ответа он занят всегда) или ещё грузится,
/// сразу возвращает None — вызывающий откатывается на правила. Так нормализация
/// никогда не останавливает ни генерацию, ни воспроизведение.
pub fn try_normalize(shared: &SharedLlm, text: &str) -> Option<String> {
    let Ok(mut guard) = shared.engine.try_lock() else {
        return None;
    };
    let engine = guard.as_mut()?;
    let cancel = Arc::new(AtomicBool::new(false));
    let params = config::GenerationParams {
        temperature: 0.2,
        top_p: 0.8,
        repeat_penalty: 1.1,
        repeat_window: 48,
        max_tokens: 150,
    };

    let result = engine.generate_streaming_with_params(
        NORMALIZE_SYSTEM_PROMPT,
        &[],
        text,
        Some(params),
        &cancel,
        |_| {},
        |_| {},
    );

    let raw = result.ok()?;
    sanitize_normalized(text, &raw)
}

/// Синхронный вопрос-ответ (для спайка и тестов): ждёт модель и возвращает текст.
pub fn generate_blocking(
    shared: &SharedLlm,
    system: &str,
    history: &[ChatTurn],
    user: &str,
    cancel: &AtomicBool,
) -> Result<String> {
    let mut guard = wait_for_engine(shared, Duration::from_secs(LLM_WAIT_TIMEOUT_SECS))
        .context("LLM недоступна")?;
    let engine = guard.as_mut().context("LLM недоступна")?;
    engine.generate_streaming(system, history, user, cancel, |_| {}, |_| {})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_sanitizer_rejects_garbage() {
        assert_eq!(sanitize_normalized("2 м/с", "два метра в секунду").unwrap(), "два метра в секунду");
        // модель пересказала инструкцию — отбрасываем
        assert!(sanitize_normalized("2 м/с", "замени единицы на слова").is_none());
        // пустой вывод
        assert!(sanitize_normalized("2 м/с", "  ").is_none());
        // взрыв длины
        assert!(sanitize_normalized("2 м/с", &"два метра в секунду ".repeat(20)).is_none());
    }

    #[test]
    fn sentence_end_detection() {
        // Индексы байтовые: кириллица по 2 байта на символ.
        assert_eq!(find_sentence_end("Привет. Как дела"), Some(13));
        assert_eq!(find_sentence_end("Привет"), None);
        assert_eq!(find_sentence_end("Ваше\nВеличество"), Some(9));
    }

    #[test]
    fn speakable_filter() {
        assert!(!speakable("Стоп."));
        assert!(!speakable("..."));
        assert!(speakable("Это достаточно длинное предложение."));
    }

    #[test]
    fn think_block_stripping() {
        let mut suppress = false;
        assert_eq!(strip_think("Ответ.", &mut suppress), "Ответ.");
        assert_eq!(strip_think("<think>скрыто</think>Ответ.", &mut suppress), "Ответ.");
        assert!(!suppress);

        // Блок, разорванный между токенами.
        let mut suppress = false;
        assert_eq!(strip_think("<think>часть", &mut suppress), "");
        assert!(suppress);
        assert_eq!(strip_think("1</think>Готово.", &mut suppress), "Готово.");
        assert!(!suppress);
    }
}
