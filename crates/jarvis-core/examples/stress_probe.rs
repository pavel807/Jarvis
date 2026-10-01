//! Проба: поддерживает ли espeak-ng (Rust-порт) знаки ударения в тексте
//! (комбинируемый акут U+0301) и как меняет IPA.
//!
//! Запуск: cargo run --release --example stress_probe -p jarvis-core

use jarvis_core::tts_piper::PiperEngine;

fn main() -> anyhow::Result<()> {
    let piper = PiperEngine::new("ruslan")?;

    let pairs: Vec<(&str, String)> = vec![
        ("замок (без знака)", "замок".into()),
        ("ЗАмок (за́мок)", format!("за\u{301}мок")),
        ("замОк", format!("замо\u{301}к")),
        ("сейчас", "сейчас".into()),
        ("сейчАс", format!("сейча\u{301}с")),
        ("градуса", "градуса".into()),
        ("градУса", format!("гра\u{301}дуса")),
        ("открываю", "открываю".into()),
        ("открывАю", format!("открыва\u{301}ю")),
    ];

    for (label, text) in pairs {
        let ipa = piper.phonemize(&text)?;
        println!("{label:<22} -> {ipa}");
    }

    Ok(())
}
