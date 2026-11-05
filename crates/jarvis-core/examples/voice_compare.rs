//! Сравнение мужских голосов Silero v5 для профиля «Джарвис»: одинаковые
//! фразы для каждого голоса, мужской тембр, спокойная подача.
//!
//! Кандидаты — все мужские голоса набора v5_cis_base; для финального выбора
//! прослушать сгенерированные WAV. Профиль Jarvis: скорость 0.9 (умеренно
//! медленно), ударения и омографы включены.
//!
//! Запуск: cargo run --release --example voice_compare -p jarvis-core
//! Результат: build/voice_compare/<voice>.wav

use jarvis_core::tts_silero::{SAMPLE_RATE, SileroEngine};

/// Фразы покрывают типовые роли голоса: приветствие, статус, числа,
/// технические названия (Premiere Pro), вопрос, спокойное завершение.
const PHRASES: &str = "Слушаю вас, Павел. Все системы в норме: процессор загружен на тридцать процентов, \
оперативной памяти свободно двенадцать гигабайт. Открываю премьер про. \
Похоже, сегодня будет дождь — взять с собой зонт? До связи.";

/// Мужские голоса v5.
const CANDIDATES: &[&str] = &[
    "ru_eduard",
    "ru_dmitriy",
    "ru_alexandr",
    "ru_bogdan",
    "ru_igor",
    "ru_marat",
    "ru_roman",
    "ru_gamat",
];

/// Скорость профиля Jarvis: умеренно медленная, уверенная подача.
const JARVIS_SPEED: f32 = 0.9;

fn write_wav(path: &str, samples: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("build/voice_compare")?;
    for voice in CANDIDATES {
        let engine = SileroEngine::new(voice)?;
        let started = std::time::Instant::now();
        let samples = engine.synthesize(PHRASES, JARVIS_SPEED)?;
        let path = format!("build/voice_compare/{voice}.wav");
        write_wav(&path, &samples)?;
        println!(
            "🎙 {voice}: {:.1} с речи за {:.1?} синтеза → {path}",
            samples.len() as f32 / SAMPLE_RATE as f32,
            started.elapsed()
        );
    }
    println!("Прослушай build/voice_compare/ и выбери голос в settings.toml (tts_voice).");
    Ok(())
}
