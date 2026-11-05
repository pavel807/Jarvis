//! Диагностика TCC-доступа к микрофону: печатает статус разрешения macOS.
//! Запуск: cargo run -p jarvis-core --example mic_check

fn main() {
    match jarvis_core::audio::mic_authorized() {
        Some(true) => println!("✅ Доступ к микрофону разрешён"),
        Some(false) => println!(
            "⏳ Доступа пока нет: запрещён или системный запрос не отвечен.\n\
             Системные настройки → Конфиденциальность и безопасность → Микрофон;\n\
             если пункта нет — tccutil reset Microphone и перезапуск приложения."
        ),
        None => println!("⚠️ Не удалось определить статус (класс AVCaptureDevice не найден)"),
    }
}
