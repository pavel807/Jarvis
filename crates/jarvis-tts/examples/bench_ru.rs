//! Замер стадий синтеза: cargo run -p jarvis-tts --example bench_ru -- [путь] [голос]

use jarvis_tts::{auto_device, Language, Qwen3TTS, Speaker, SynthesisOptions};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "models/Qwen3-TTS/qwen3-tts-0.6b-customvoice".into());
    let speaker: Speaker = args.next().map(|s| s.parse()).transpose()?.unwrap_or(Speaker::Ryan);

    let device = auto_device()?;
    let model = Qwen3TTS::from_pretrained(&model_dir, device)?;
    let options = SynthesisOptions { temperature: 0.7, ..Default::default() };

    let text = "Привет! Я Джарвис, твой персональный голосовой ассистент.";
    let mut total_start = Instant::now();
    let mut total_dur = 0.0f64;
    for run in 1..=3 {
        let t = Instant::now();
        let (audio, timing) =
            model.synthesize_with_timing(text, speaker, Language::Russian, Some(options.clone()))?;
        let wall = t.elapsed().as_secs_f64();
        let dur = audio.samples.len() as f64 / audio.sample_rate as f64;
        total_dur += dur;
        println!(
            "run {run}: wall {wall:.2}с | prefill {:.0}мс | gen {:.0}мс ({} кадров) | decode {:.0}мс | аудио {dur:.2}с",
            timing.prefill_ms, timing.generation_ms, timing.generation_frames, timing.decode_ms
        );
        if run == 1 {
            total_start = Instant::now();
        }
    }
    let wall: f64 = total_start.elapsed().as_secs_f64() / 2.0;
    let rtf = wall / (total_dur / 2.0);
    println!("средний RTF (прогретый): {rtf:.2}");
    Ok(())
}
