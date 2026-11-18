//! Клонирование голоса для ZONOS3: опорная запись → 192-мерный эмбеддинг
//! → `voices/<имя>.embedding.npy` в каталоге весов.
//!
//! Запуск: cargo run -p jarvis-core --example voice_clone -- <запись.wav> <имя>
//! Вход — моно WAV (16/22.05/24/44.1 кГц): 44.1 кГц ресемплится точным
//! ядром из GGUF, остальное — оконным sinc. Достаточно 10–60 секунд
//! чистой речи одного диктора.

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let wav = args.next().context("Укажи путь к опорной записи (WAV)")?;
    let name = args.next().unwrap_or_else(|| "jarvis".to_string());
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        bail!("Имя голоса — простое слово без разделителей пути");
    }

    jarvis_core::init_tuning();
    jarvis_core::model_fetch::ensure_zonos3()
        .context("Не удалось подготовить веса ZONOS3 (spk-encoder.gguf)")?;

    // Читаем WAV (hound): моно f32, частота из заголовка.
    let mut reader = hound::WavReader::open(&wav)
        .with_context(|| format!("Не открыть {wav:?}"))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        bail!("Нужен моно-WAV, в файле {} каналов", spec.channels);
    }
    let bits_ok = matches!(spec.sample_format, hound::SampleFormat::Int)
        && matches!(spec.bits_per_sample, 8 | 16 | 24 | 32);
    if !bits_ok && spec.sample_format != hound::SampleFormat::Float {
        bail!("Неожиданный формат выборок в WAV");
    }
    let rate = spec.sample_rate;
    let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        reader.samples::<f32>().filter_map(|s| s.ok()).collect()
    } else {
        let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
        reader
            .samples::<i32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / max)
            .collect()
    };
    let secs = samples.len() as f64 / rate as f64;
    println!("🎙 {wav}: {secs:.1} с @ {rate} Гц");

    let encoder = jarvis_core::tts_zonos::SpeakerEncoder::open(
        &jarvis_core::model_fetch::zonos3_dir().join("spk-encoder.gguf"),
    )
    .context("Не удалось загрузить spk-encoder.gguf")?;
    let emb = encoder
        .embedding(&samples, rate)
        .context("Спикер-энкодер не справился с записью")?;
    if emb.iter().any(|v| !v.is_finite()) {
        bail!("В эмбеддинге не-числа — запись битая или тишина");
    }

    let dir = jarvis_core::model_fetch::zonos3_dir().join("voices");
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("{name}.embedding.npy"));
    write_npy_f32(&dest, &emb)?;
    println!("✅ Эмбеддинг «{name}» ({} измерений) → {}", emb.len(), dest.display());
    println!("   Голос появится в настройках после перезагрузки TTS (переключи движок или перезапусти).");
    Ok(())
}

/// Минимальная запись `.npy`: одномерный массив little-endian f32 — тот же
/// формат, что читает `load_voice_embedding`.
fn write_npy_f32(path: &std::path::Path, values: &[f32]) -> Result<()> {
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    // Заголовок выравнивается до 64 байт: magic(6) + версия(2) + длина(2).
    let header_len = 10 + dict.len();
    let pad = 64 - (header_len % 64);
    let dict = format!("{dict}{}", " ".repeat(pad - 1) + "\n");
    let mut bytes = Vec::with_capacity(10 + dict.len() + values.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("Не записать {path:?}"))
}
