use jarvis_core::plugins;

fn main() -> anyhow::Result<()> {
    if plugins::uv_path().is_none() {
        println!("== Установка uv ==");
        let path = plugins::install_uv(&mut |pct, text| println!("[{pct}%] {text}"))?;
        println!("uv: {}", path.display());
        plugins::ensure_sample_plugin()?;
    } else {
        println!("uv уже установлен: {:?}", plugins::uv_path());
    }
    println!("\n== Плагины ==");
    for p in plugins::scan() {
        println!("{} ({}): команды {:?}", p.id, p.title, p.commands.iter().map(|c| &c.name).collect::<Vec<_>>());
    }
    println!("\n== Запуск команды ==");
    let quotes = plugins::scan().into_iter().find(|p| p.id == "quotes").expect("пример-плагин");
    let outcome = plugins::run_command(&quotes, "quote", &serde_json::json!({"topic": "тест"}))?;
    println!("speech: {}", outcome.speech);
    Ok(())
}
