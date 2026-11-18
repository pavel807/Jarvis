//! Проверка спикер-энкодера: наш эмбеддинг jett.wav vs эталонный .npy Zyphra.
use jarvis_core::tts_zonos::speaker::SpeakerEncoder;

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

fn read_npy(path: &str) -> anyhow::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let off = 10 + hlen;
    Ok(data[off..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> anyhow::Result<()> {
    let (rate, wav) = read_wav("/tmp/zonos_ref/jett.wav")?;
    println!("jett.wav: {} Гц, {} сэмплов", rate, wav.len());
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let enc = SpeakerEncoder::open(&dir.join("spk-encoder.gguf"))?;
    let mine = enc.embedding(&wav, rate)?;
    let theirs = read_npy("/tmp/zonos_ref/jett.embedding.npy")?;
    println!("наш: {} изм., эталон: {} изм.", mine.len(), theirs.len());
    let dot: f32 = mine.iter().zip(theirs.iter()).map(|(a, b)| a * b).sum();
    let nm = (mine.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let nt = (theirs.iter().map(|x| x * x).sum::<f32>()).sqrt();
    println!("косинус: {:.4}  |наш|={nm:.4} |этал|={nt:.4}", dot / (nm * nt));
    let max_diff = mine.iter().zip(theirs.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("max|Δ| = {max_diff:.5}");
    // Первые 8 значений.
    println!("наш    первые 8: {:?}", mine[..8].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    println!("эталон первые 8: {:?}", theirs[..8].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    Ok(())
}
