//! Синтез текста LM'ом (с голосом или без) и декодирование в wav.
//!
//! USAGE: probe_synth_wav [текст] [имя_голоса | путь/npy]
//! GREEDY=1 — жадная выборка вместо реестровой.
use candle_core::IndexOp;
use jarvis_core::tts_zonos::{Sampling, ZonosEngine};

fn read_npy_f32(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = String::from_utf8_lossy(&data[10..10 + hlen]).to_string();
    let off = 10 + hlen;
    let nums: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .unwrap_or("()")
        .split('(')
        .nth(1)
        .unwrap_or("")
        .split(|c| c == ',' || c == ')')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .collect();
    let n: usize = nums.iter().product();
    Ok(data[off..off + n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let gguf = jarvis_core::tts_zonos::GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut engine = ZonosEngine::load(&gguf, candle_core::Device::Cpu)?;
    let greedy = std::env::var("TEMP").ok().and_then(|v| v.parse::<f32>().ok()) == Some(0.0);
    let _ = std::env::var("SEED_ENV_MARKER");
    let mut sp = Sampling { greedy, ..Sampling::default() };
    if let Ok(t) = std::env::var("TEMP") { if let Ok(v) = t.parse() { sp.temperature = v; } }
    if let Ok(sd) = std::env::var("SEED") { sp.seed = sd.parse().unwrap_or(0); }
    let dec = jarvis_core::tts_zonos::ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;

    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Привет! Я Джарвис, ваш голосовой помощник.".into());
    let speaker: Option<Vec<f32>> = match std::env::args().nth(2) {
        None => None,
        Some(name) => {
            let path = if name.ends_with(".npy") {
                std::path::PathBuf::from(&name)
            } else {
                dir.join("voices").join(format!("{name}.embedding.npy"))
            };
            let v = read_npy_f32(&path)?;
            println!("голос {}: {} измерений", path.display(), v.len());
            Some(v)
        }
    };

    let t0 = std::time::Instant::now();
    let buckets: [usize; 3] = std::env::var("BUCKETS")
        .ok()
        .map(|v| {
            let p: Vec<usize> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            [p.first().copied().unwrap_or(10), p.get(1).copied().unwrap_or(10), p.get(2).copied().unwrap_or(30)]
        })
        .unwrap_or([10, 10, 30]);
    println!("бакеты: {buckets:?}");
    let cfg_scale: f32 = std::env::var("CFG").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let out = engine.synthesize(&text, speaker.as_deref(), buckets, &sp, cfg_scale, Some(&dec))?;
    let codes = out.frames;
    let audio = out.audio;
    println!("кадров: {} за {:?}", codes.len(), t0.elapsed());
    // Дамп кодов [кадры, книги] для сравнения с эталоном.
    let nc = codes.first().map(|r| r.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(codes.len() * nc);
    for row in &codes {
        flat.extend_from_slice(row);
    }
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write("/tmp/synth_codes.bin", &bytes)?;
    std::fs::write("/tmp/synth_codes.shape", format!("{},{}\n", codes.len(), nc))?;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create("/tmp/synth.wav", spec)?;
    let mut rms = 0.0f64;
    for &s in &audio {
        rms += (s as f64).powi(2);
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    println!(
        "/tmp/synth.wav: {} сэмплов ({:.2} с), RMS {:.4}",
        audio.len(),
        audio.len() as f64 / 44100.0,
        (rms / audio.len().max(1) as f64).sqrt()
    );
    Ok(())
}
