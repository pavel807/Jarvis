use jarvis_core::web;

fn main() -> anyhow::Result<()> {
    for q in ["что такое дед икс фитнес", "дед икс фитнес", "что такое Дональд Трамп"] {
        println!("=== запрос: {q}");
        match web::search(q, 5) {
            Ok(hits) => {
                println!("  результатов: {}", hits.len());
                for h in &hits {
                    println!(
                        "  - [{}] {}",
                        h.title.chars().take(50).collect::<String>(),
                        h.snippet.chars().take(70).collect::<String>()
                    );
                }
                let prompt = web::build_summary_prompt(q, &hits);
                println!("  промпт построен: {}", prompt.is_some());
            }
            Err(e) => println!("  ошибка: {e:#}"),
        }
    }
    Ok(())
}
