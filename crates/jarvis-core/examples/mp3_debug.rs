//! Отладка mp3-декода: частота/каналы/длина + корреляция с ffmpeg-эталоном.

use anyhow::Result;
use std::fs;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("путь к mp3");
    let file = fs::File::open(&path)?;
    let mut decoder = minimp3::Decoder::new(file);
    let mut raw: Vec<i16> = Vec::new();
    let mut rate = 0i32;
    let mut channels = 0usize;
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                rate = frame.sample_rate;
                channels = frame.channels;
                raw.extend_from_slice(&frame.data);
            }
            Err(minimp3::Error::Eof) => break,
            Err(minimp3::Error::SkippedData) => continue,
            Err(e) => anyhow::bail!("{e:?}"),
        }
    }
    println!("minimp3: {rate} Гц, {channels} кан., {} значений ({:.1} с)", raw.len(), raw.len() as f64 / (rate.max(1) as f64 * channels.max(1) as f64));
    // Сохраняем f32 mono в /tmp/sir_minimp3.raw для сравнения с ffmpeg.
    let mono: Vec<f32> = if channels > 1 {
        raw.chunks(channels).map(|c| c.iter().map(|&v| v as f32 / 32768.0).sum::<f32>() / channels as f32).collect()
    } else {
        raw.iter().map(|&v| v as f32 / 32768.0).collect()
    };
    let bytes: Vec<u8> = mono.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write("/tmp/sir_minimp3.raw", &bytes)?;
    println!("→ /tmp/sir_minimp3.raw");
    Ok(())
}
