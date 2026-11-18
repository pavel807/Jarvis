//! Транскрипция wav через встроенный FastConformer: cargo run --release -p jarvis-core --example fc_transcribe -- <wav>...

use anyhow::Result;

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let model = jarvis_core::asr::model::FastConformer::load(&jarvis_core::paths::models_dir())?;
    for path in std::env::args().skip(1) {
        let mut reader = hound::WavReader::open(&path)?;
        let rate = reader.spec().sample_rate;
        let raw: Vec<f32> = reader
            .samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / (1i64 << 15) as f32)
            .collect();
        // FastConformer обучен на 16 кГц — приводим к его частоте.
        let samples = if rate != 16000 {
            jarvis_core::tts_zonos::speaker::sinc_resample(&raw, rate, 16000)
        } else {
            raw
        };
        let t = model.transcribe(&samples)?;
        println!("{path} ({rate} Гц): {}", t.text.trim());
    }
    Ok(())
}
