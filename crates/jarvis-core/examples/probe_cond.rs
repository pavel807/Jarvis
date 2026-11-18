// Проверка отклика модели на текст: два разных текста, greedy + min_frames=60.
use jarvis_core::tts_zonos::{Sampling, SpeechEngine};

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let mut engine = jarvis_core::tts::open_zonos_voice("jarvis")?;
    for text in ["Привет!", "Сегодня отличная погода, идём гулять в парк?"] {
        engine.set_sampling(Sampling {
            greedy: true,
            ..Sampling::default()
        });
        let audio = engine.speak(text, 1.0)?;
        let secs = audio.len() as f32 / 44100.0;
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len().max(1) as f32).sqrt();
        println!("{text:?}: {secs:.2} с, RMS {rms:.4}");
        let win = 4410;
        let env: Vec<f32> = audio
            .chunks(win)
            .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let short: Vec<String> = env.iter().map(|v| format!("{v:.2}")).collect();
        println!("   огибающая 100 мс: {}", short.join(" "));
        let name = if text.starts_with("При") { "probe_hi" } else { "probe_weather" };
        let path = format!("/tmp/{name}.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec)?;
        for s in &audio {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        w.finalize()?;
        println!("   → {path}");
    }
    Ok(())
}
