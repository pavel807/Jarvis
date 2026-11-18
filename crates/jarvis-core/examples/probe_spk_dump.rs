//! Дамп стадий спикер-энкодера для python-пересчёта.
use jarvis_core::tts_zonos::speaker::SpeakerEncoder;
use std::sync::{Arc, Mutex};

fn read_wav(path: &str) -> anyhow::Result<(u32, Vec<f32>)> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(&data[0..4] == b"RIFF", "не wav");
    let mut pos = 12usize;
    let (mut rate, mut samples) = (0u32, Vec::new());
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]) as usize;
        let body = &data[pos+8..pos+8+size];
        if id == b"fmt " { rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]); }
        else if id == b"data" {
            for c in body.chunks_exact(2) { samples.push(i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0); }
        }
        pos += 8 + size + (size & 1);
    }
    Ok((rate, samples))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (wav_path, emb_path) = match args.len() {
        3 => (args[1].clone(), args[2].clone()),
        _ => ("/tmp/zonos_ref/jett.wav".into(), "/tmp/zonos_ref/jett.embedding.npy".into()),
    };
    let (rate, wav) = read_wav(&wav_path)?;
    // Ограничим до 3 секунд, чтобы python успел пересчитать.
    let wav = &wav[..wav.len().min(3 * rate as usize)];
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let enc = SpeakerEncoder::open(&dir.join("spk-encoder.gguf"))?;
    let dumps: Arc<Mutex<Vec<(String, Vec<f32>, usize, usize)>>> = Default::default();
    let d2 = dumps.clone();
    let emb = enc.embedding_debug(wav, rate, &mut |name, data, rows, cols| {
        println!("дамп: {name} [{rows}×{cols}]");
        d2.lock().unwrap().push((name.to_string(), data.to_vec(), rows, cols));
    })?;
    for (name, data, rows, cols) in dumps.lock().unwrap().iter() {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("/tmp/zcheck/spk_{name}.bin"), &bytes)?;
        std::fs::write(format!("/tmp/zcheck/spk_{name}.shape"), format!("({rows}, {cols})"))?;
    }
    let emb_bytes: Vec<u8> = emb.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write("/tmp/zcheck/spk_emb.bin", &emb_bytes)?;
    println!("эмбеддинг: {} изм. → /tmp/zcheck/", emb.len());
    // Эталонный эмбеддинг на том же отрезке — для сравнения косинуса.
    let full = read_wav(&wav_path)?.1;
    let full_emb = enc.embedding(&full[..], rate)?;
    let ref_emb_npy = std::fs::read(&emb_path)?;
    let hlen = u16::from_le_bytes([ref_emb_npy[8], ref_emb_npy[9]]) as usize;
    let ref_emb: Vec<f32> = ref_emb_npy[10 + hlen..].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        d / (na * nb)
    };
    println!("косинус(наш 3с, наш полн) = {:.4}", cos(&emb, &full_emb));
    println!("косинус(наш полн, эталон) = {:.4}", cos(&full_emb, &ref_emb));
    Ok(())
}
