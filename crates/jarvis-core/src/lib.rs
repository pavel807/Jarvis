//! jarvis-core: движок голосового ассистента (wake word, STT, TTS, LLM, команды).

/// Тюнинг рантайма до первой тензорной операции. candle на сотнях мелких
/// операций энкодера/фронтенда теряет до 20× на оверхеде rayon-пула из
/// многих потоков (замер: 12 потоков — 1956 мс на окно 1.2 с, 4 — 58 мс,
/// 1 — 99 мс). Cap задевает только элементные операции: matmul считает
/// через собственный трединг matrixmultiply и не меняется.
pub fn init_tuning() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let threads = cores.min(4).max(1).to_string();
        // RAYON_NUM_THREADS читается rayon'ом при создании глобального пула —
        // до первого par_iter. set_var небезопасен в edition 2024: других
        // записей окружения в процессе нет, гонок нет.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", threads) };
    });
}

pub mod agent;
pub mod applescript;
pub mod asr;
pub mod audio;
pub mod choice;
pub mod commands;
pub mod db;
pub mod engine;
pub mod events;
pub mod jokes;
pub mod learned;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod model_fetch;
pub mod nn;
pub mod observe;
pub mod omograph;
pub mod paths;
pub mod permissions;
pub mod platform;
pub mod plugins;

/// Текущее время в RFC3339 без внешних зависимостей: дата из `date +%F`,
/// время посекундно из системного вызова — точности снимка достаточно.
fn time_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86400;
    // Гражданская дата (алгоритм Ховарда Хиннанта): без chrono.
    let z = (days + 719_468) as i64;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let rem = secs % 86400;
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Стартовый прогрев «знания о системе»: сканирование установленных
/// приложений в базу (память + JSON на диске) и компиляция плагинов в
/// фоне. Вызывается один раз при запуске ассистента — модель получает
/// готовый каталог мгновенно, а не после первого обхода каталогов.
pub fn warm_startup() {
    std::thread::spawn(|| {
        // База приложений: список в памяти уже прогрет ленивым кэшем
        // платформенного слоя, на диск пишем снимок для наблюдаемости.
        let apps = platform::app_db_entries();
        let file = paths::app_data_dir().join("app_db.json");
        let db = serde_json::json!({
            "обновлено": time_now_rfc3339(),
            "приложений": apps.len(),
            "приложения": apps.iter().map(|(name, exec, id)| serde_json::json!({
                "имя": name, "бинарь": exec, "id": id,
            })).collect::<Vec<_>>(),
        });
        match serde_json::to_string_pretty(&db)
            .ok()
            .and_then(|s| std::fs::write(&file, s).ok())
        {
            Some(()) => println!("📚 База приложений: {} записей — {}", apps.len(), file.display()),
            None => eprintln!("⚠️ Не записал базу приложений в {}", file.display()),
        }
    });
    // Плагины на Spark: скан каталогов, компиляция с кэшем — в своём потоке.
    plugins::warm();
}
pub mod prompt_budget;
pub mod router;
pub mod schedule;
pub mod settings;
pub mod site_cache;
pub mod stress;
pub mod telegram;
pub mod speaker;
pub mod stt;
pub mod tasks;
pub mod text;
pub mod timers;
pub mod tool_guard;
pub mod tools;
pub mod topic;
pub mod tts;
pub mod tts_normalizer;
pub mod tts_silero;
pub mod auto;
pub mod wake;
pub mod wake_ctc;
pub mod wake_phrase;
pub mod wake_train;
pub mod weather;
pub mod web;
