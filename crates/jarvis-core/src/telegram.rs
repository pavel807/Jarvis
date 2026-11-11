//! Telegram-канал — по логике OpenJarvis: бот на Bot API long-polling,
//! whitelist chat_id, сообщение → LLM → ответ в тот же чат. Токен и
//! разрешённые чаты — в settings.toml (`telegram_bot_token`,
//! `telegram_chat_ids` — список через запятую). Пустой токен — канал
//! просто не запускается.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

/// Формат ответа getUpdates (нужные поля).
#[derive(Deserialize)]
struct Updates {
    result: Vec<Update>,
}

#[derive(Deserialize)]
struct Update {
    update_id: u64,
    message: Option<Message>,
}

#[derive(Deserialize)]
struct Message {
    chat: Chat,
    text: Option<String>,
}

#[derive(Deserialize)]
struct Chat {
    id: i64,
}

/// Смещение long-poll: следующий update_id, который ждём.
static OFFSET: AtomicU64 = AtomicU64::new(0);

/// Свой агент: long-poll живёт 25 секунд — глобальный таймаут с запасом.
fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(35)))
            .build()
            .new_agent()
    })
}

fn settings_token() -> String {
    crate::settings::Settings::load(&crate::paths::settings_file())
        .map(|s| s.telegram_bot_token)
        .unwrap_or_default()
}

fn allowed_chats() -> Vec<i64> {
    crate::settings::Settings::load(&crate::paths::settings_file())
        .map(|s| {
            s.telegram_chat_ids
                .split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

/// GET-вызов Bot API, ответ — сырой JSON.
fn api(token: &str, method: &str, query: &str) -> Result<String> {
    let url = format!("https://api.telegram.org/bot{token}/{method}?{query}");
    agent()
        .get(&url)
        .call()
        .context("Telegram Bot API недоступен")?
        .body_mut()
        .read_to_string()
        .context("пустой ответ Telegram Bot API")
}

/// Отправить текст в чат. Пишем в stderr, ошибки не поднимаем: канал не
/// должен ронять движок из-за сетевой секунды.
fn send_message(token: &str, chat_id: i64, text: &str) {
    let query = format!(
        "chat_id={}&text={}",
        chat_id,
        urlencoding::encode(text)
    );
    if let Err(e) = api(token, "sendMessage", &query) {
        eprintln!("⚠️ Telegram sendMessage: {e:#}");
    }
}

/// Один цикл long-poll: ждём апдейты до 25 с, обрабатываем сообщения.
/// true — были апдейты (сразу идём за следующими), false — пауза.
fn poll_once(token: &str, allowed: &[i64], shared: &crate::llm::SharedLlm) -> bool {
    let offset = OFFSET.load(Ordering::Relaxed);
    let query = format!("timeout=25&offset={offset}&allowed_updates=[\"message\"]");
    let body = match api(token, "getUpdates", &query) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️ Telegram getUpdates: {e:#} — повтор через 5 с");
            std::thread::sleep(Duration::from_secs(5));
            return false;
        }
    };
    let updates: Updates = match serde_json::from_str(&body) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("⚠️ Telegram: не разобрал ответ: {e}");
            std::thread::sleep(Duration::from_secs(5));
            return false;
        }
    };
    for update in &updates.result {
        OFFSET.store(update.update_id + 1, Ordering::Relaxed);
        let Some(message) = &update.message else {
            continue;
        };
        let chat_id = message.chat.id;
        // Whitelist: без разрешённых чатов бот молчит — чужие не обслуживаем.
        if !allowed.contains(&chat_id) {
            println!("🚫 Telegram: чат {chat_id} не в telegram_chat_ids");
            continue;
        }
        let Some(text) = message.text.clone() else {
            continue;
        };
        let reply = answer(shared, &text);
        send_message(token, chat_id, &reply);
    }
    !updates.result.is_empty()
}

/// Ответ на сообщение: разговорный промпт + память + дата. Изолированно,
/// без инструментов — бот не должен трогать рабочий стол по чужому чату.
fn answer(shared: &crate::llm::SharedLlm, text: &str) -> String {
    let system = format!(
        "{}{}Сегодняшняя дата: {}.\nТы отвечаешь в Telegram: тот же краткий голос, без markdown.",
        crate::llm::CHAT_SYSTEM_PROMPT,
        crate::memory::prompt_block(),
        crate::text::today_russian()
    );
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    match crate::llm::generate_blocking(shared, &system, &[], text, &cancel) {
        Ok(reply) if !reply.trim().is_empty() => reply,
        Ok(_) => "Ответ пустой — переформулируй вопрос.".into(),
        Err(e) => format!("Модель недоступна: {e}"),
    }
}

/// Запустить канал в фоне, если задан токен. Тихий выход — иначе это
/// не ошибка, а «канал не настроен».
pub fn spawn(shared: crate::llm::SharedLlm) {
    let token = settings_token();
    if token.trim().is_empty() {
        return;
    }
    let allowed = allowed_chats();
    if allowed.is_empty() {
        eprintln!(
            "⚠️ Telegram: токен есть, но telegram_chat_ids пуст — бот никому не отвечает"
        );
        return;
    }
    println!("📨 Telegram-канал запущен (чатов в whitelist: {})", allowed.len());
    std::thread::Builder::new()
        .name("telegram-channel".into())
        .spawn(move || {
            loop {
                let went_busy = poll_once(&token, &allowed, &shared);
                // Пустой poll = просто истёк long-poll — сразу за следующим.
                if !went_busy {
                    std::thread::sleep(Duration::from_millis(300));
                }
            }
        })
        .expect("поток Telegram-канала");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_parsing_accepts_message_and_skips_other_updates() {
        let body = r#"{"ok":true,"result":[
            {"update_id":10,"message":{"message_id":1,"chat":{"id":42},"text":"привет"}},
            {"update_id":11,"edited_message":{"chat":{"id":42},"text":"правка"}}
        ]}"#;
        let updates: Updates = serde_json::from_str(body).unwrap();
        assert_eq!(updates.result.len(), 2);
        assert_eq!(updates.result[0].message.as_ref().unwrap().chat.id, 42);
        // edited_message не считаем сообщением — None, канал его пропустит.
        assert!(updates.result[1].message.is_none());
    }

    #[test]
    fn answer_falls_back_on_empty_and_error() {
        // LLM недоступна — честный текст, а не паника.
        let shared = std::sync::Arc::new(crate::llm::SharedLlmState::new());
        shared.failed.store(true, Ordering::SeqCst);
        let reply = answer(&shared, "привет");
        assert!(!reply.trim().is_empty());
    }
}
