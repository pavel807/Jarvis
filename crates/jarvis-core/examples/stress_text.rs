//! Печатает текст с расставленными «+» (как его видит движок) и пишет WAV.
//! Запуск: cargo run --release --example stress_text -p jarvis-core -- "текст"
//!         [выходной wav]

use jarvis_core::tts_silero::{SAMPLE_RATE, SileroEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let text = args
        .next()
        .unwrap_or_else(|| "проверка ударений".to_string());
    let out = args.next();

    let engine = SileroEngine::new("ru_bogdan")?;
    let tokens = engine.debug_tokens(&text)?;

    // Декодируем токены обратно в строку через алфавит.
    let alphabet: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        jarvis_core::paths::models_dir().join("silero/alphabet.json"),
    )?)?;
    let id2sym: std::collections::HashMap<i64, char> = alphabet["id_to_symbol"]
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    let id: i64 = k.parse().ok()?;
                    let c = v.as_str()?.chars().next()?;
                    Some((id, c))
                })
                .collect()
        })
        .unwrap_or_default();

    let accented: String = tokens
        .iter()
        .filter_map(|t| id2sym.get(t))
        .collect::<String>()
        .trim_matches(|c| c == '|' || c == '~')
        .to_string();
    println!("ACCENTED:{accented}");

    if let Some(path) = out {
        let audio = engine.synthesize(&text, 1.0)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec)?;
        for &s in &audio {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        println!("WROTE:{path} {} samples", audio.len());
    }
    Ok(())
}
