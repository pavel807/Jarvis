//! autocore — демон и CLI. Этап 1 спецификации: `doctor` — отчёт о
//! возможностях платформы, правах и дым-тест чтения без действий.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("doctor") => {
            let mut report = autocore::doctor::report();
            let smoke = args.iter().any(|a| a == "--smoke");
            if smoke {
                let note = report.smoke();
                println!("{note}");
            }
            print!("{}", report.render());
        }
        _ => {
            eprintln!("Использование: autocore doctor [--smoke]");
            std::process::exit(2);
        }
    }
}
