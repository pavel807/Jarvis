//! Метрика похожести клонированного голоса: синтезируем фразу этим голосом,
//! кодируем синтез спикер-энкодером обратно и считаем косинус к целевому
//! эмбеддингу и к эталону (эмбеддингу исходной записи).
//!
//! USAGE: probe_voice_fit <голос.npy> [эталон.npy]

use anyhow::{Context, Result};

fn read_npy_f32(path: &std::path::Path) -> Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let off = 10 + hlen;
    let header = String::from_utf8_lossy(&data[10..10 + hlen]).to_string();
    let n: usize = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .unwrap_or("")
        .split(|c| c == ',' || c == ')')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .product();
    Ok(data[off..off + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-9)
}

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let mut args = std::env::args().skip(1);
    let voice_npy = args.next().context("Укажи npy голоса")?;
    let ref_npy = args.next();

    let dir = jarvis_core::model_fetch::zonos3_dir();
    let gguf = jarvis_core::tts_zonos::GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut engine = jarvis_core::tts_zonos::ZonosEngine::load(&gguf, candle_core::Device::Cpu)?;
    let dec = jarvis_core::tts_zonos::ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;
    let spk = jarvis_core::tts_zonos::SpeakerEncoder::open(&dir.join("spk-encoder.gguf"))?;

    let target = read_npy_f32(std::path::Path::new(&voice_npy))?;
    let target_ref = ref_npy.map(|p| read_npy_f32(std::path::Path::new(&p))).transpose()?;

    let cfg_scale: f32 = std::env::var("CFG").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let mut sampling = jarvis_core::tts_zonos::Sampling::default();
    if let Ok(v) = std::env::var("TEMP") {
        sampling.temperature = v.parse().unwrap_or(sampling.temperature);
    }
    if let Ok(v) = std::env::var("SEED") {
        sampling.seed = v.parse().unwrap_or(sampling.seed);
    }
    println!(
        "параметры: cfg {cfg_scale}, temp {:.2}, seed {}",
        sampling.temperature, sampling.seed
    );
    let phrases = [
        "Привет, сэр. Все системы работают в штатном режиме.",
        "Курс установлен, скорость максимальная.",
        "Обнаружено входящее сообщение, сэр.",
    ];
    let voice = read_npy_f32(std::path::Path::new(&voice_npy))?;
    let mut cos_t = 0.0f32;
    let mut cos_r = 0.0f32;
    for (i, text) in phrases.iter().enumerate() {
        let out = engine.synthesize(text, Some(&voice), [10, 10, 30], &sampling, cfg_scale, Some(&dec))?;
        let back = spk.embedding(&out.audio, 44100)?;
        let c1 = cosine(&back, &target);
        cos_t += c1;
        if let Some(r) = &target_ref {
            let c2 = cosine(&back, r);
            cos_r += c2;
            println!("  фраза {i}: синт↔цель {c1:.4}  синт↔эталон {c2:.4}");
        } else {
            println!("  фраза {i}: синт↔цель {c1:.4}");
        }
    }
    let n = phrases.len() as f32;
    println!("СРЕДНЕЕ: синт↔цель {:.4}  синт↔эталон {:.4}  (cfg {cfg_scale})", cos_t / n, cos_r / n);
    Ok(())
}
