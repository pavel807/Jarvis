//! Эмбеддинги mp3-пути и wav-пути в одном процессе: decode → embedding → косинус.

use anyhow::Result;

fn read_npy(path: &std::path::Path) -> Result<Vec<f32>> {
    let d = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([d[8], d[9]]) as usize;
    let off = 10 + hlen;
    Ok(d[off..off + 192 * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let mp3 = jarvis_core::tts::decode_audio(std::path::Path::new(
        "/Users/pavellabushev/Downloads/[Jarvis]Сэр, ......ости..mp3",
    ))?;
    let wav = jarvis_core::tts::decode_audio(std::path::Path::new("/tmp/sir44.wav"))?;
    let cos_wf = {
        let (na, nb) = (mp3.len(), wav.len());
        let m = na.min(nb);
        let dot: f64 = mp3[..m].iter().zip(&wav[..m]).map(|(a, b)| (*a * *b) as f64).sum();
        let (sa, sb): (f64, f64) = (
            mp3[..m].iter().map(|v| (*v * *v) as f64).sum(),
            wav[..m].iter().map(|v| (*v * *v) as f64).sum(),
        );
        dot / (sa.sqrt() * sb.sqrt()).max(1e-30)
    };
    println!("waveform косинус mp3↔wav: {cos_wf:.4}");

    let enc = jarvis_core::tts_zonos::SpeakerEncoder::open(
        &jarvis_core::model_fetch::zonos3_dir().join("spk-encoder.gguf"),
    )?;
    let e_mp3 = enc.embedding(&mp3, 44100)?;
    let e_wav = enc.embedding(&wav, 44100)?;
    let dot: f32 = e_mp3.iter().zip(&e_wav).map(|(a, b)| a * b).sum();
    let cos = dot
        / (e_mp3.iter().map(|v| v * v).sum::<f32>().sqrt()
            * e_wav.iter().map(|v| v * v).sum::<f32>().sqrt());
    println!("эмбеддинг косинус mp3↔wav (один процесс): {cos:.4}");

    let saved = read_npy(std::path::Path::new(
        "/Users/pavellabushev/Library/Application Support/Jarvis/models/zonos3/voices/evgeny.embedding.npy",
    ))?;
    let dot: f32 = saved.iter().zip(&e_mp3).map(|(a, b)| a * b).sum();
    let cos = dot
        / (saved.iter().map(|v| v * v).sum::<f32>().sqrt()
            * e_mp3.iter().map(|v| v * v).sum::<f32>().sqrt());
    println!("evgeny.embedding ↔ свежий mp3-эмбеддинг: {cos:.4}");
    Ok(())
}
