//! Бенчмарк speakrs: CoreML (Metal/GPU) против CPU на реальной записи.
//!
//! Запуск: cargo run -p jarvis-core --release --example speaker_bench
//!
//! Замеряет прогон пайплайна (диаризация+эмбеддинги) на каноническом
//! 3-секундном окне (домен верификации фраз) и на 24-секундной записи
//! (домен пересборки профиля). Прогрев + серия замеров, медиана и среднее.

use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
use std::time::Instant;

fn read_wav_16k(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut samples: Vec<f32> = Vec::new();
    for frame in reader.samples::<i32>() {
        let raw = frame?;
        let max = 1i64 << (spec.bits_per_sample.saturating_sub(1));
        samples.push(raw as f32 / max as f32);
    }
    if channels > 1 {
        samples = samples
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect();
    }
    if spec.sample_rate != 16_000 {
        samples = jarvis_core::audio::resample_audio(&samples, spec.sample_rate as usize, 16_000);
    }
    Ok(samples)
}

fn bench(pipeline: &mut OwnedDiarizationPipeline, label: &str, audio: &[f32], runs: usize) {
    // Прогрев (компиляция шейдеров/кэш CoreML не должен попадать в замер).
    for _ in 0..5 {
        let _ = pipeline.run(audio);
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let _ = pipeline.run(audio).expect("прогон пайплайна");
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "  {label:>10}: медиана {median:6.1} мс, среднее {mean:6.1} мс ({runs} прогонов, {:.1} с аудио)",
        audio.len() as f32 / 16_000.0
    );
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new("train/voice");
    let wav = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "wav"))
        .ok_or_else(|| anyhow::anyhow!("в train/voice нет записей — нечего бенчмаркить"))?;
    let samples = read_wav_16k(&wav)?;
    println!("Запись: {:?} ({:.1} с)", wav.file_name().unwrap(), samples.len() as f32 / 16_000.0);

    // Каноническое окно верификации (3 с) и кусок пересборки профиля (24 с).
    let window_len: usize = 16_000 * 3;
    let long_len: usize = 16_000 * 24;
    let window: Vec<f32> = samples
        .iter()
        .take(window_len)
        .copied()
        .chain(std::iter::repeat_n(0.0, window_len.saturating_sub(samples.len())))
        .collect();
    let long: Vec<f32> = samples
        .iter()
        .take(long_len)
        .copied()
        .chain(std::iter::repeat_n(0.0, long_len.saturating_sub(samples.len())))
        .collect();

    #[cfg(target_os = "macos")]
    let modes: &[(&str, ExecutionMode)] = &[
        ("CoreML", ExecutionMode::CoreMl),
        ("CPU", ExecutionMode::Cpu),
    ];
    #[cfg(not(target_os = "macos"))]
    let modes: &[(&str, ExecutionMode)] = &[("CPU", ExecutionMode::Cpu)];

    for (label, mode) in modes {
        println!("\n=== {label} ===");
        let mut pipeline = match OwnedDiarizationPipeline::from_pretrained(*mode) {
            Ok(p) => p,
            Err(e) => {
                println!("  недоступен: {e}");
                continue;
            }
        };
        bench(&mut pipeline, "окно 3 с", &window, 7);
        bench(&mut pipeline, "запись 24 с", &long, 5);
    }
    Ok(())
}
