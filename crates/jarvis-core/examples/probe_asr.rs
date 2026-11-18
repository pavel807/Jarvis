// Расшифровка WAV через FastConformer: что реально говорит синтез.
use jarvis_core::stt::Stt;

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let stt = Stt::load(&jarvis_core::paths::models_dir())?;
    for path in std::env::args().skip(1) {
        let mut reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        let raw: Vec<f32> = reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / (1 << (spec.bits_per_sample - 1)) as f32)
            .collect();
        let mono: Vec<f32> = if spec.channels == 1 {
            raw
        } else {
            raw.chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        };
        let mono = jarvis_core::audio::resample_audio(&mono, spec.sample_rate as usize, 16_000);
        let t = stt.model().transcribe(&mono)?;
        println!("{path}: «{}»", t.text);
    }
    Ok(())
}
