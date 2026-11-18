//! Изоляция кодека: декодирование эталонных codes.npy и сравнение с исходным wav.
//!
//! cargo run -p jarvis-core --example probe_codec_ref -- /tmp/zonos_ref/jett.codes.npy /tmp/zonos_ref/jett.wav
use anyhow::Context;
use jarvis_core::tts_zonos::ZacDecoder;
use std::path::Path;

fn read_npy_codes(path: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(&data[0..6] == b"\x93NUMPY", "не npy");
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = String::from_utf8_lossy(&data[10..10 + hlen]).to_string();
    // shape: (T, 10), fortran_order: False, descr: '<i4'
    let shape_part = header.split("'shape':").nth(1).context("нет shape")?;
    let parens = shape_part.split('(').nth(1).context("нет (shape)")?;
    let nums: Vec<usize> = parens
        .split(|c| c == ',' || c == ')')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .collect();
    anyhow::ensure!(nums.len() >= 2, "shape {header:?}");
    let (t, books) = (nums[0], nums[1]);
    let off = 10 + hlen;
    let mut rows = Vec::with_capacity(t);
    for f in 0..t {
        let mut row = Vec::with_capacity(books);
        for b in 0..books {
            let i = off + (f * books + b) * 4;
            row.push(u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn read_wav(path: &Path) -> anyhow::Result<(u32, Vec<f32>)> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(&data[0..4] == b"RIFF", "не wav");
    let mut pos = 12usize;
    let mut rate = 0u32;
    let mut samples: Vec<f32> = Vec::new();
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let body = &data[pos + 8..pos + 8 + size];
        if id == b"fmt " {
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
        } else if id == b"data" {
            for c in body.chunks_exact(2) {
                samples.push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
            }
        }
        pos += 8 + size + (size & 1);
    }
    Ok((rate, samples))
}

/// RMS по окнам 250 мс, компактно.
fn rms_profile(x: &[f32], rate: u32, label: &str) -> Vec<f32> {
    let win = (rate / 4) as usize;
    let mut out = Vec::new();
    for chunk in x.chunks(win) {
        let s: f32 = chunk.iter().map(|v| v * v).sum();
        out.push((s / chunk.len().max(1) as f32).sqrt());
    }
    println!("{label}: {out:?}");
    out
}

/// Энергии полос через наивный Гёрцель по всему сигналу.
fn bands(x: &[f32], rate: u32) -> [(f64, f64); 6] {
    let freqs = [120.0f64, 350.0, 800.0, 1800.0, 4000.0, 9000.0];
    let mut out = [(0.0, 0.0); 6];
    for (fi, &f) in freqs.iter().enumerate() {
        let w = 2.0 * std::f64::consts::PI * f / rate as f64;
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &v in x {
            let s0 = v as f64 + 2.0 * w.cos() * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = s1 * s1 + s2 * s2 - 2.0 * w.cos() * s1 * s2;
        out[fi] = (f, (power / x.len() as f64).sqrt());
    }
    out
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (codes_path, wav_path) = match args.len() {
        3 => (args[1].clone(), args[2].clone()),
        _ => (
            "/tmp/zonos_ref/jett.codes.npy".into(),
            "/tmp/zonos_ref/jett.wav".into(),
        ),
    };
    let codes = read_npy_codes(Path::new(&codes_path))?;
    println!("кадры: {}, книг: {}", codes.len(), codes[0].len());

    let dir = jarvis_core::model_fetch::zonos3_dir();
    let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;

    let books: usize = std::env::var("BOOKS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let t0 = std::time::Instant::now();
    let audio = dec.decode_books(&codes, books)?;
    let (rate, samples) = (44100u32, audio);
    println!("книг: {books}");
    println!("декодировано {} сэмплов ({} c) за {:?}", samples.len(), samples.len() as f64 / rate as f64, t0.elapsed());

    // Эталон.
    let (ref_rate, ref_samples) = read_wav(Path::new(&wav_path))?;
    println!("эталон: {} Гц, {} сэмплов", ref_rate, ref_samples.len());

    // Сохраняем декодированный wav 16-bit.
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    let data_len: u32 = (samples.len() * 2) as u32;
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&rate.to_le_bytes());
    wav.extend_from_slice(&(rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for &s in &samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        wav.extend_from_slice(&v.to_le_bytes());
    }
    let out_path = Path::new(&codes_path).with_extension("decoded.wav");
    std::fs::write(&out_path, &wav)?;
    println!("записан {}", out_path.display());

    // Сравнение.
    let n = samples.len().min(ref_samples.len());
    let mine = &samples[..n];
    let theirs = &ref_samples[..n];
    println!("\nRMS-профиль (250 мс):");
    let a = rms_profile(mine, rate, "наш  ");
    let b = rms_profile(theirs, rate, "этал.");
    let ma: f64 = a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64;
    let mb: f64 = b.iter().map(|v| *v as f64).sum::<f64>() / b.len() as f64;
    let num: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64 - ma) * (*y as f64 - mb)).sum();
    let da: f64 = a.iter().map(|v| (*v as f64 - ma).powi(2)).sum();
    let db: f64 = b.iter().map(|v| (*v as f64 - mb).powi(2)).sum();
    println!("корреляция RMS-профилей: {:.3}", num / (da.sqrt() * db.sqrt() + 1e-12));

    println!("\nполосы (ампл. RMS):");
    for ((f1, a1), (_, b1)) in bands(mine, rate).iter().zip(bands(theirs, ref_rate).iter()) {
        println!("  {:>6.0} Гц: наш {:.5}  этал. {:.5}  отнош. {:.3}", f1, a1, b1, a1 / (b1 + 1e-9));
    }
    Ok(())
}
