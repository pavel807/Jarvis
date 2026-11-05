//! Диагностика резолва приложений: какие правила срабатывают на фразе.
//! Запуск: cargo run --release -p jarvis-core --example app_match_debug [фраза]

fn main() {
    let phrases: Vec<String> = std::env::args().skip(1).collect();
    let phrases = if phrases.is_empty() {
        vec![
            "открой электронный дневник".to_string(),
            "открой мэш".to_string(),
            "открой энкубик слайсер некст".to_string(),
            "открой сафари".to_string(),
        ]
    } else {
        phrases
    };
    for q in phrases {
        let r = jarvis_core::tools::try_open_app_from_prompt(&q);
        println!("{q} -> {r:?}");
    }
}
