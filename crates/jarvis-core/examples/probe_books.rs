// Проверка порядка кодовых книг: декодируем как есть и с обращёнными книгами.
use jarvis_core::tts_zonos::{Sampling, ZacDecoder};
use jarvis_core::model_fetch;

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    jarvis_core::model_fetch::ensure_zonos3()?;
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let gguf = jarvis_core::tts_zonos::GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut engine = jarvis_core::tts_zonos::ZonosEngine::load(&gguf, candle_core::Device::Cpu)?;
    let sp = Sampling { greedy: true, ..Sampling::default() };
    let codes = engine.synthesize_codes("Привет! Как дела?", None, [10, 10, 30], &sp)?;
    println!("кадров: {}", codes.len());

    let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;

    let ncb = 10usize;
    let write = |name: &str, frames: &[Vec<u32>]| -> anyhow::Result<()> {
        let audio = dec.decode(frames)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(name, spec)?;
        for s in &audio {
            w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        w.finalize()?;
        println!("{name}: {} сэмплов", audio.len());
        Ok(())
    };

    // (a) как есть (без 3 кадров префилла)
    let a: Vec<Vec<u32>> = codes.iter().skip(3).cloned().collect();
    write("/tmp/dec_as_is.wav", &a)?;
    // (b) книги в обратном порядке
    let b: Vec<Vec<u32>> = a.iter().map(|f| f.iter().rev().cloned().collect()).collect();
    write("/tmp/dec_reversed.wav", &b)?;
    Ok(())
}
