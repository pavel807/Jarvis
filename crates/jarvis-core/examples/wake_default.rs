//! Дефолтная wake-модель «Джарвис»: эталонные записи фразы синтезируются
//! Silero (несколько темпов — немного разнообразия для DTW), из них
//! строится reference-модель RustPotter. Годится как стартовая: после
//! записи своих сэмплов в приложении модель переучивается на голос
//! владельца.
//! Запуск: cargo run -p jarvis-core --example wake_default

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let (train_dir, _) = jarvis_core::wake_train::sample_dirs();
    std::fs::create_dir_all(&train_dir)?;

    let engine = jarvis_core::tts_silero::SileroEngine::new_no_stress("ru_eduard")?;
    // Разные дикторы Silero: reference-модель для «чужого» голоса должна
    // покрывать разнообразие тембров, поэтому эталоны мультиголосые.
    let variants: Vec<(&str, f32)> = vec![
        ("ru_eduard", 0.95),
        ("ru_dmitriy", 1.0),
        ("ru_alexandr", 1.0),
        ("ru_bogdan", 1.0),
        ("ru_igor", 1.0),
        ("ru_karina", 1.0),
        ("ru_eduard", 1.1),
        ("ru_dmitriy", 0.9),
    ];
    let mut paths = Vec::new();
    for (i, (voice, speed)) in variants.iter().enumerate() {
        let audio = engine.synthesize_with(voice, "Джарвис", *speed)?;
        let mono_16k = jarvis_core::audio::resample_audio(&audio, 48_000, 16_000);
        // Пауза вокруг фразы — как у живой записи: детектор видит фразу
        // в потоке, а не в изолированном буфере.
        let silence = vec![0_f32; 16_000 * 3 / 10];
        let mut padded = silence.clone();
        padded.extend(mono_16k);
        padded.extend(silence);
        let mono_16k = padded;
        let path = train_dir.join(format!("[jarvis]_tts_default_{voice}_{i}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec)
            .with_context(|| format!("не создать {}", path.display()))?;
        for s in &mono_16k {
            writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        println!("🎙️ Эталон {i} ({voice}, скорость {speed}): {}", path.display());
        paths.push(path);
    }

    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (etx, _erx) = std::sync::mpsc::channel::<jarvis_core::events::JarvisEvent>();
    let trained = jarvis_core::wake_train::build_reference_wakeword(&etx, &cancel)?;

    // Итог — jarvis_model.rpw: то имя, что качается из релиза по умолчанию.
    let out = trained.with_file_name("jarvis_model.rpw");
    std::fs::rename(&trained, &out)?;
    println!("✅ Дефолтная модель: {}", out.display());

    // Самопроверка: эталоны должны уверенно детектироваться, чужой звук — нет.
    for (i, path) in paths.iter().enumerate() {
        let score = jarvis_core::wake_train::test_wakeword(&out, path)?;
        println!("   эталон {i}: score {score:.3}");
    }
    let negative = std::path::Path::new("resources/audio/ok1.wav");
    if negative.exists() {
        let score = jarvis_core::wake_train::test_wakeword(&out, negative)?;
        println!("   чужой звук (ok1.wav): score {score:.3} (ожидаемо низкий)");
    }
    Ok(())
}
