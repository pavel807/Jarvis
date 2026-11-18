//! Клон голоса из файла (wav/mp3): cargo run --release -p jarvis-core --example voice_clone_file -- <файл> <имя>

use anyhow::Result;

fn main() -> Result<()> {
    jarvis_core::init_tuning();
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("путь к аудио");
    let name = args.next().unwrap_or_else(|| "custom".into());
    jarvis_core::tts::clone_voice_from_file(std::path::Path::new(&path), &name)?;
    println!("✅ голос «{name}» готов");
    Ok(())
}
