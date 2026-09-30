//! Бенчмарк TTS: время сессии (префилл), первого чанка (TTFB) и RTF
//! на фразах разной длины с разными max_length/chunk_frames.
//!
//! max_length определяет преаллокацию KV-кэша talker'а (28 слоёв,
//! ~115 КБ/токен): 2048 = 264 МБ нулей на каждую фразу.
//!
//! Запуск: cargo run --release --example tts_bench -p jarvis-core

use anyhow::Result;
use jarvis_tts::{auto_device, Language, Qwen3TTS, Speaker, SynthesisOptions};
use std::time::Instant;

const MODEL_DIR_NAME: &str = "qwen3-tts-0.6b-customvoice";
const SR: f64 = 24000.0;

fn bench(model: &Qwen3TTS, label: &str, text: &str, max_length: usize, chunk_frames: usize) -> Result<()> {
    let opts = SynthesisOptions {
        temperature: 0.7,
        max_length,
        chunk_frames,
        ..SynthesisOptions::default()
    };

    let t_new = Instant::now();
    let mut session =
        model.synthesize_streaming(text, Speaker::Aiden, Language::Russian, opts)?;
    let session_ms = t_new.elapsed().as_secs_f64() * 1000.0;

    let t_first = Instant::now();
    let mut first_ms = 0.0;
    let mut samples = 0usize;
    let mut sum_sq = 0f64;
    let mut chunks = 0usize;
    loop {
        match session.next_chunk()? {
            Some(c) => {
                if chunks == 0 {
                    first_ms = t_first.elapsed().as_secs_f64() * 1000.0;
                }
                for &s in &c.samples {
                    sum_sq += (s as f64) * (s as f64);
                }
                samples += c.samples.len();
                chunks += 1;
            }
            None => break,
        }
    }
    let total = t_new.elapsed().as_secs_f64();
    let audio = samples as f64 / SR;
    let rms = if samples > 0 { (sum_sq / samples as f64).sqrt() } else { 0.0 };
    println!(
        "{label:<44} сессия {session_ms:>6.1} мс | 1-й чанк {first_ms:>6.1} мс | TTFB {:>6.1} мс | аудио {audio:>6.2} с | RTF {:>5.2} | RMS {:.3}",
        session_ms + first_ms,
        total / audio,
        rms,
    );
    Ok(())
}

fn main() -> Result<()> {
    let dir = jarvis_core::paths::tts_models_dir().join(MODEL_DIR_NAME);
    println!("Модель: {:?}", dir);
    let device = auto_device()?;
    println!("Устройство: {device:?}");

    let t = Instant::now();
    let model = Qwen3TTS::from_pretrained(&dir.to_string_lossy(), device)?;
    println!("Загрузка модели: {:.1?}\n", t.elapsed());

    let short = "Да.";
    let medium = "Сейчас на улице двадцать один градус и небольшая облачность, ветра почти нет.";
    let long = "Сегодня в Москве ожидается переменная облачность, без осадков. Температура воздуха днём составит около двадцати трёх градусов тепла. Ветер северо-западный, четыре метра в секунду. Атмосферное давление в пределах нормы.";

    // Прогрев: компиляция Metal-шейдеров, чтобы мерить инференс, а не компиляцию.
    bench(&model, "ПРОГРЕВ", short, 2048, 10)?;

    println!("\n--- Влияние преаллокации KV (chunk 10) ---");
    bench(&model, "короткая,  max_length 2048", short, 2048, 10)?;
    bench(&model, "короткая,  max_length 128", short, 128, 10)?;
    bench(&model, "средняя,   max_length 2048", medium, 2048, 10)?;
    bench(&model, "средняя,   max_length 384", medium, 384, 10)?;

    println!("\n--- Влияние размера чанка (max_length 384) ---");
    bench(&model, "средняя,   chunk 10", medium, 384, 10)?;
    bench(&model, "средняя,   chunk 5", medium, 384, 5)?;
    bench(&model, "средняя,   chunk 3", medium, 384, 3)?;

    println!("\n--- Длинная фраза ---");
    bench(&model, "длинная,   max_length 2048, chunk 5", long, 2048, 5)?;
    bench(&model, "длинная,   max_length 512, chunk 5", long, 512, 5)?;

    Ok(())
}
