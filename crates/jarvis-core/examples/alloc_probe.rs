//! Подсчёт аллокаций при инференсе локальных моделей: FastConformer и speakrs.
//!
//! Запуск: cargo run -p jarvis-core --release --example alloc_probe
//!
//! Глобальный аллокатор считает каждое выделение/освобождение памяти, пока
//! включён флаг: замеряется один прогон инференса каждой модели (после
//! прогрева), отдельно — загрузка модели. Числа — Rust-аллокации; буферы,
//! которые уходят в Metal/ANE мимо аллокатора (objc2/CoreML), не считаются.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

struct TrackingAlloc;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static DEALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES_ALLOC: AtomicU64 = AtomicU64::new(0);
static BYTES_DEALLOC: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACKING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES_ALLOC.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        if TRACKING.load(Ordering::Relaxed) {
            DEALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES_DEALLOC.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() && TRACKING.load(Ordering::Relaxed) {
            // realloc = освободили старый блок, выделили новый.
            DEALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES_DEALLOC.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES_ALLOC.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        new
    }
}

#[global_allocator]
static GLOBAL: TrackingAlloc = TrackingAlloc;

pub mod counters {
    use super::*;

    pub struct Guard;

    impl Guard {
        pub fn start() -> Self {
            ALLOCS.store(0, Ordering::Relaxed);
            DEALLOCS.store(0, Ordering::Relaxed);
            BYTES_ALLOC.store(0, Ordering::Relaxed);
            BYTES_DEALLOC.store(0, Ordering::Relaxed);
            TRACKING.store(true, Ordering::Relaxed);
            Guard
        }
        pub fn stop(self) -> Report {
            TRACKING.store(false, Ordering::Relaxed);
            Report {
                allocs: ALLOCS.load(Ordering::Relaxed),
                deallocs: DEALLOCS.load(Ordering::Relaxed),
                bytes_alloc: BYTES_ALLOC.load(Ordering::Relaxed),
                bytes_dealloc: BYTES_DEALLOC.load(Ordering::Relaxed),
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct Report {
        pub allocs: u64,
        pub deallocs: u64,
        pub bytes_alloc: u64,
        pub bytes_dealloc: u64,
    }

    impl Report {
        pub fn print(&self, label: &str) {
            let net = self.bytes_alloc as i64 - self.bytes_dealloc as i64;
            println!(
                "    {label}: аллокаций {}, освобождений {}, выделено {}, нетто {:+}",
                self.allocs,
                self.deallocs,
                human(self.bytes_alloc),
                human(net.unsigned_abs()),
            );
        }
    }
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1_048_576.0 {
        format!("{:.1} МБ", b / 1_048_576.0)
    } else if b >= 1024.0 {
        format!("{:.1} КБ", b / 1024.0)
    } else {
        format!("{bytes} Б")
    }
}

use counters::{Guard, Report};

fn read_wav_16k(path: &std::path::Path, max_secs: usize) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let max_samples = spec.sample_rate as usize * max_secs;
    let mut samples: Vec<f32> = Vec::new();
    for frame in reader.samples::<i32>() {
        let raw = frame?;
        let max = 1i64 << (spec.bits_per_sample.saturating_sub(1));
        samples.push(raw as f32 / max as f32);
        if samples.len() >= max_samples * channels {
            break;
        }
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

fn first_dictation() -> anyhow::Result<std::path::PathBuf> {
    let dir = std::path::Path::new("train/voice");
    std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "wav"))
        .ok_or_else(|| anyhow::anyhow!("в train/voice нет записей — нечем мерить"))
}

fn main() -> anyhow::Result<()> {
    let wav = first_dictation()?;
    let fastconformer_audio = read_wav_16k(&wav, 3)?;
    let window_len = 16_000usize * 3;
    let speakrs_audio: Vec<f32> = fastconformer_audio
        .iter()
        .take(window_len)
        .copied()
        .chain(std::iter::repeat_n(
            0.0,
            window_len.saturating_sub(fastconformer_audio.len()),
        ))
        .collect();
    println!(
        "Аудио: {:?} ({:.1} с), счётчик аллокаций включён\n",
        wav.file_name().unwrap(),
        fastconformer_audio.len() as f32 / 16_000.0
    );

    // ---- FastConformer ----
    println!("FastConformer (CPU, candle):");
    let guard = Guard::start();
    let mut stt = jarvis_core::stt::Stt::load(&jarvis_core::paths::models_dir())?;
    let load: Report = guard.stop();
    load.print("загрузка модели");

    // Прогрев: JIT/кэши не попадают в замер.
    for _ in 0..2 {
        let _ = stt.transcribe(&fastconformer_audio)?;
    }
    let mut per_run: Vec<Report> = Vec::new();
    for _ in 0..5 {
        let guard = Guard::start();
        let _ = stt.transcribe(&fastconformer_audio)?;
        per_run.push(guard.stop());
    }
    let median = {
        let mut a: Vec<u64> = per_run.iter().map(|r| r.allocs).collect();
        a.sort_unstable();
        per_run.iter().find(|r| r.allocs == a[a.len() / 2]).copied().unwrap()
    };
    median.print("один прогон (медиана 5)");
    println!(
        "    детально 5 прогонов: {:?}",
        per_run.iter().map(|r| r.allocs).collect::<Vec<_>>()
    );
    println!();

    // ---- speakrs ----
    use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
    #[cfg(target_os = "macos")]
    let modes: &[(&str, ExecutionMode)] = &[
        ("CPU", ExecutionMode::Cpu),
        ("CoreML", ExecutionMode::CoreMl),
    ];
    #[cfg(not(target_os = "macos"))]
    let modes: &[(&str, ExecutionMode)] = &[("CPU", ExecutionMode::Cpu)];

    for (label, mode) in modes {
        println!("speakrs ({label}, окно 3 с):");
        let guard = Guard::start();
        let mut pipeline = OwnedDiarizationPipeline::from_pretrained(*mode)?;
        let load: Report = guard.stop();
        load.print("загрузка пайплайна");

        for _ in 0..2 {
            let _ = pipeline.run(&speakrs_audio)?;
        }
        let mut per_run: Vec<Report> = Vec::new();
        for _ in 0..5 {
            let guard = Guard::start();
            let _ = pipeline.run(&speakrs_audio)?;
            per_run.push(guard.stop());
        }
        let median = {
            let mut a: Vec<u64> = per_run.iter().map(|r| r.allocs).collect();
            a.sort_unstable();
            per_run.iter().find(|r| r.allocs == a[a.len() / 2]).copied().unwrap()
        };
        median.print("один прогон (медиана 5)");
        println!(
            "    детально 5 прогонов: {:?}",
            per_run.iter().map(|r| r.allocs).collect::<Vec<_>>()
        );
        println!();
    }

    println!("Примечание: считаются Rust-аллокации (куча). Буферы, уходящие");
    println!("в Metal/CoreML напрямую (objc2), аллокатор не видит — для CPU-режимов");
    println!("числа полные.");
    Ok(())
}
