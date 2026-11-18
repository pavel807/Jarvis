// Спектры из head_out декодера для реальных кодов: формантная структура или шум?
use jarvis_core::tts_zonos::{Sampling, ZacDecoder};

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let dir = jarvis_core::model_fetch::zonos3_dir();
    // 1. Кодировщик недоступен — берём коды из LM (greedy).
    let gguf = jarvis_core::tts_zonos::GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut engine = jarvis_core::tts_zonos::ZonosEngine::load(&gguf, candle_core::Device::Cpu)?;
    let sp = Sampling { greedy: true, ..Sampling::default() };
    let frames = engine.synthesize_codes("Привет! Как дела?", None, [10, 10, 30], &sp)?;
    println!("кадров: {}", frames.len());

    // 2. Тишина-эталон: все книги = коды тишины.
    let silence = [1918u32, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337];
    let sil_frames: Vec<Vec<u32>> = (0..frames.len())
        .map(|_| silence.to_vec())
        .collect();

    let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;

    // 3. Прогоняем голову для: тишины (все кадры) и реальных кодов —
    //    печатаем межкадровую вариацию logmag по бинам.
    for (name, fr) in [("тишина", &sil_frames), ("речь-LM", &frames)] {
        let audio = dec.decode(fr)?;
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        println!("{name}: RMS {rms:.4}, len {}", audio.len());
    }
    Ok(())
}
