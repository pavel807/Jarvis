//! Запуск Qwen3.5-0.8B на чистом candle (Metal, с откатом на CPU).
//!
//! Модель берётся из `models/qwen3.5-0.8b/` (уже скачана с Hugging Face:
//! config.json, tokenizer.json и safetensors-веса).
//!
//! Режимы:
//! ```text
//! cargo run -p jarvis-core --example qwen35_chat -- chat   # диалог в терминале
//! cargo run -p jarvis-core --example qwen35_chat -- bench  # замер скорости
//! cargo run -p jarvis-core --example qwen35_chat -- bench "свой промпт"
//! ```

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp};
use jarvis_core::llm::qwen35::{self, Qwen35Model};
use jarvis_core::llm::sampler;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const SYSTEM_PROMPT: &str =
    "Ты Джарвис — голосовой ассистент. Отвечай кратко на русском, 1-3 предложения.";

/// Рекомендации Qwen для не-thinking режима.
const TEMPERATURE: f64 = 0.7;
const TOP_P: f64 = 0.8;
const REPEAT_PENALTY: f32 = 1.05;
const REPEAT_WINDOW: usize = 64;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn pick_device() -> Device {
    match Device::new_metal(0) {
        Ok(d) => {
            println!("✅ Устройство: Metal");
            d
        }
        Err(e) => {
            eprintln!("⚠️ Metal недоступен ({e}), работаем на CPU");
            Device::Cpu
        }
    }
}

struct Runner {
    model: Qwen35Model,
    tokenizer: Tokenizer,
    eos_ids: Vec<u32>,
}

impl Runner {
    fn load() -> Result<Self> {
        let dir = models_dir().join(qwen35::LOCAL_DIR_NAME);
        let device = pick_device();

        let model = Qwen35Model::load(&dir, &device)
            .with_context(|| format!("Не удалось загрузить модель из {}", dir.display()))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Не удалось загрузить токенизатор: {e}"))?;

        // EOS: из конфига (<|endoftext|>) плюс терминатор ChatML-шаблона.
        let mut eos_ids = model.config.eos_ids.clone();
        for special in ["<|endoftext|>", "<|im_end|>"] {
            if let Some(id) = tokenizer.token_to_id(special)
                && !eos_ids.contains(&id)
            {
                eos_ids.push(id);
            }
        }
        println!("   EOS-токены: {eos_ids:?}");

        Ok(Self {
            model,
            tokenizer,
            eos_ids,
        })
    }

    /// Генерация со стримингом. Возвращает текст и метрики (мс префилла,
    /// мс/токен декода).
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(&str),
    ) -> Result<(String, f64, f64)> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Ошибка токенизации: {e}"))?;
        let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            anyhow::bail!("Пустой промпт");
        }
        let mut context = prompt_tokens.clone();

        let mut cache = self.model.new_cache();
        let t0 = Instant::now();
        let mut logits = self.model.forward(&prompt_tokens, 0, &mut cache)?;
        let _ = self.model.device().synchronize();
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;

        let prompt_len = context.len();
        let t_decode = Instant::now();
        let mut text = String::new();
        let mut suppress_think = false;
        let mut rng = rand::thread_rng();

        let debug = std::env::var("QWEN35_DEBUG").is_ok();
        for step in 0..max_tokens {
            let last = logits.dim(0)? - 1;
            let mut row: Vec<f32> = logits
                .i((last, ..))?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;

            if debug && step < 8 {
                let mut idx: Vec<u32> = (0..row.len() as u32).collect();
                idx.select_nth_unstable_by(4, |&a, &b| row[b as usize].total_cmp(&row[a as usize]));
                idx.truncate(5);
                idx.sort_unstable_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
                for &i in &idx {
                    let s = self.tokenizer.decode(&[i], true).unwrap_or_default();
                    println!("   🔍 top{:?} {}: {:.4} {:?}", step, i, row[i as usize], s);
                }
            }

            sampler::apply_repeat_penalty(&mut row, REPEAT_PENALTY, &context, REPEAT_WINDOW);
            let tok = sampler::sample_top_p(&row, TEMPERATURE, TOP_P, &mut rng);
            if self.eos_ids.contains(&tok) {
                break;
            }
            context.push(tok);

            let piece = self
                .tokenizer
                .decode(&[tok], true)
                .map_err(|e| anyhow::anyhow!("Ошибка декодирования: {e}"))?;
            let piece = strip_think(&piece, &mut suppress_think);
            if !piece.is_empty() {
                text.push_str(&piece);
                on_token(&piece);
            }

            logits = self.model.forward(&[tok], prompt_len + step, &mut cache)?;
        }
        let _ = self.model.device().synchronize();
        let decoded = context.len() - prompt_len;
        let ms_per_token = t_decode.elapsed().as_secs_f64() * 1e3 / decoded.max(1) as f64;
        Ok((text, prefill_ms, ms_per_token))
    }
}

/// Простейшее отсечение рассуждений, если модель начала их писать
/// вопреки закрытому `<think>` в промпте.
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

fn bench(prompt: &str, max_tokens: usize) -> Result<()> {
    let runner = Runner::load()?;
    println!("\n▶ Промпт: {prompt:?}");
    let started = Instant::now();
    let (text, prefill_ms, ms_per_token) = runner.generate(prompt, max_tokens, |piece| {
        print!("{piece}");
        std::io::stdout().flush().ok();
    })?;
    println!(
        "\n\n📊 Префилл {prefill_ms:.0} мс; декод {ms_per_token:.1} мс/токен \
              ({:.0} ток/с); всего {:.1?}",
        1000.0 / ms_per_token,
        started.elapsed(),
    );
    println!("---\n{text}");
    Ok(())
}

fn chat() -> Result<()> {
    let runner = Runner::load()?;
    let mut history: Vec<(String, String)> = Vec::new();
    println!("Диалог с Qwen3.5-0.8B. Пустая строка или Ctrl-D — выход.\n");
    let stdin = std::io::stdin();
    loop {
        print!("Вы > ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let user = line.trim();
        if user.is_empty() {
            break;
        }

        let prompt = qwen35::build_chat_prompt(SYSTEM_PROMPT, &history, user);
        let started = Instant::now();
        let result = runner.generate(&prompt, 256, |piece| {
            print!("{piece}");
            std::io::stdout().flush().ok();
        });
        println!();
        match result {
            Ok((text, prefill_ms, ms_per_token)) => {
                println!(
                    "   [{:.1?}: префилл {:.0} мс, {:.1} мс/токен]",
                    started.elapsed(),
                    prefill_ms,
                    ms_per_token
                );
                history.push((user.to_string(), text));
                let excess = history.len().saturating_sub(6);
                history.drain(..excess);
            }
            Err(e) => eprintln!("⚠️ {e:#}"),
        }
    }
    Ok(())
}

/// Отладка: логиты префилла на фиксированном промпте → /tmp/qwen35_rust_logits.bin
/// (f32, все позиции) для сверки с python-референсом.
fn dump_logits() -> Result<()> {
    let runner = Runner::load()?;
    let prompt = qwen35::build_chat_prompt(SYSTEM_PROMPT, &[], "Привет");
    let encoding = runner
        .tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Ошибка токенизации: {e}"))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("ids ({}){:?}...", ids.len(), &ids[..ids.len().min(12)]);

    type LayerDumps = std::sync::Arc<std::sync::Mutex<Vec<(usize, Vec<f32>)>>>;

    let mut cache = runner.model.new_cache();
    let dumps: LayerDumps = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let d = std::sync::Arc::clone(&dumps);
    let logits = runner
        .model
        .forward_with_inspector(&ids, 0, &mut cache, &mut |i, t| {
            let v: Vec<f32> = t
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1())
                .unwrap_or_default();
            d.lock().unwrap().push((i, v));
        })?;
    {
        let dumps = dumps.lock().unwrap();
        for (i, v) in dumps.iter() {
            let name = if *i == usize::MAX {
                "embed".to_string()
            } else if *i == usize::MAX - 1 {
                "final".to_string()
            } else {
                format!("layer{i:02}")
            };
            let bytes: Vec<u8> = v.iter().flat_map(|f: &f32| f.to_le_bytes()).collect();
            std::fs::write(format!("/tmp/qwen35_rust_{name}.bin"), &bytes)?;
        }
        println!("слоёв сохранено: {}", dumps.len());
    }
    let _ = runner.model.device().synchronize();
    let rows: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let bytes: Vec<u8> = rows.iter().flat_map(|f: &f32| f.to_le_bytes()).collect();
    std::fs::write("/tmp/qwen35_rust_logits.bin", &bytes)?;
    println!(
        "сохранено /tmp/qwen35_rust_logits.bin: {} значений",
        rows.len()
    );

    let last = &rows[rows.len() - runner.model.config.vocab_size..];
    let mut idx: Vec<u32> = (0..last.len() as u32).collect();
    idx.select_nth_unstable_by(9, |&a, &b| last[b as usize].total_cmp(&last[a as usize]));
    idx.truncate(10);
    idx.sort_unstable_by(|&a, &b| last[b as usize].total_cmp(&last[a as usize]));
    for &i in &idx {
        println!(
            "   top {}: {:.4} {:?}",
            i,
            last[i as usize],
            runner.tokenizer.decode(&[i], true).unwrap_or_default()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "chat".into());
    match mode.as_str() {
        "chat" => chat(),
        "bench" => {
            let prompt = std::env::args().nth(2).unwrap_or_else(|| {
                "Привет! Расскажи коротко, чем знаменит Санкт-Петербург.".into()
            });
            bench(&prompt, 96)
        }
        "dump" => dump_logits(),
        other => {
            anyhow::bail!("Неизвестный режим {other:?}: используй chat, bench или dump");
        }
    }
}
