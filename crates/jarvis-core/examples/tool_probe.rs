//! Проверка одного инструмента на живой машине: разбирает tool-блок и
//! выполняет его, печатая итог так, как его услышит пользователь.
//!
//! Нужно, чтобы проверять действия без голоса и без моделей:
//!
//! ```text
//! JARVIS_HOME=. cargo run --release -p jarvis-core --example tool_probe -- \
//!   '<tool>{"name":"music_control","args":{"action":"pause"}}</tool>'
//! ```

use std::sync::mpsc::channel;

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    if arg.trim().is_empty() {
        eprintln!("Использование: tool_probe '<tool>{{...}}</tool>'");
        std::process::exit(1);
    }

    match jarvis_core::tools::parse_lenient(&arg) {
        jarvis_core::tools::ToolParse::Call(call) => {
            let (audio_tx, audio_rx) = channel();
            let (events, _events_rx) = channel();
            let outcome = jarvis_core::tools::execute(call, &audio_tx, &events);
            println!("ок:      {}", outcome.ok);
            println!("скажет:  {}", outcome.speech);
            if !outcome.detail.trim().is_empty() {
                println!("детали:  {}", outcome.detail.replace('\n', " ⏎ "));
            }
            // Аудио в этом режиме не проигрываем: пробуем инструмент, а не голос.
            let _ = audio_rx.try_recv();
        }
        jarvis_core::tools::ToolParse::Failed(reason) => {
            eprintln!("вызов не разобран: {reason}");
            std::process::exit(2);
        }
        jarvis_core::tools::ToolParse::None => {
            eprintln!("в аргументе нет tool-блока");
            std::process::exit(2);
        }
    }
}
