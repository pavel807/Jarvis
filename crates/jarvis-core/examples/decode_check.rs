//! Что реально даёт decode_audio для mp3 vs wav.

use anyhow::Result;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("путь к аудио");
    let x = jarvis_core::tts::decode_audio(std::path::Path::new(&path))?;
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write("/tmp/decode_audio_out.raw", &bytes)?;
    println!("len {} ({:.1} c) min {} max {}", x.len(), x.len() as f64 / 44100.0,
        x.iter().fold(f32::INFINITY, |m, &v| m.min(v)),
        x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v)));
    Ok(())
}
