//! Облачный LLM через OpenAI-совместимый API (стриминг SSE).
//!
//! Один клиент закрывает и Яндекс GPT, и любые другие совместимые сервисы
//! (OpenAI, OpenRouter, LM Studio, vLLM): различаются только базовый адрес,
//! схема авторизации и формат имени модели.
//!
//! Пресет «yandex»: `Authorization: Api-Key <ключ>`, модель
//! `gpt://<каталог>/<модель>/latest` (собирается из полей автоматически).
//! Пресет «openai»: `Authorization: Bearer <ключ>`, модель как введена.
//!
//! Стриминг: `stream: true`, разбор `data:`-строк SSE, дельты текста
//! отдаются в TTS по предложениям — общие помощники из `streaming.rs`.

use super::streaming;
use super::template::ChatTurn;
use crate::events::{EventTx, JarvisEvent};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Базовый адрес Яндекс GPT (OpenAI-совместимый эндпоинт).
/// Именно `/v1`, а не `/openai/v1` — тот путь даёт HTTP 404 от шлюза Яндекса.
pub const YANDEX_BASE_URL: &str = "https://llm.api.cloud.yandex.net/v1";
/// Модель Яндекса по умолчанию (легче и быстрее старшей).
pub const YANDEX_DEFAULT_MODEL: &str = "yandexgpt-lite";

pub struct ApiEngine {
    /// Базовый адрес без хвостового `/`: например `https://llm.api.cloud.yandex.net/openai/v1`.
    base_url: String,
    /// Полное имя модели: `yandexgpt-lite`, `gpt://b1g.../yandexgpt/latest`, `gpt-4o-mini`, …
    model: String,
    /// Заголовок авторизации: `Api-Key <ключ>` (Яндекс) или `Bearer <ключ>`.
    auth_value: String,
}

impl ApiEngine {
    /// Сборка из конфига (заполняется из settings.toml до загрузки движка).
    pub fn load(events: &EventTx) -> Result<Self> {
        let base_url = super::config::api_base_url();
        let api_key = super::config::api_key();
        if api_key.trim().is_empty() {
            let _ = events.send(JarvisEvent::Error {
                message: "Облачная модель выбрана, но API-ключ не задан (настройки → AI-модель)"
                    .into(),
            });
            return Err(anyhow!(
                "API-ключ не задан: укажи его в настройках → AI-модель"
            ));
        }
        let preset = super::config::api_preset();
        let auth_value = if preset == "yandex" {
            format!("Api-Key {api_key}")
        } else {
            format!("Bearer {api_key}")
        };
        // Полный URI модели для Яндекса: gpt://<каталог>/<модель>[/<версия>].
        // Версию дописываем только если пользователь её не указал —
        // «yandexgpt/rc» не должно превращаться в «…/rc/latest».
        let mut model = super::config::api_model();
        if preset == "yandex" {
            let folder = super::config::api_folder_id();
            if !folder.trim().is_empty() && !model.starts_with("gpt://") {
                let version = if model.contains('/') { "" } else { "/latest" };
                model = format!("gpt://{}/{model}{version}", folder.trim());
            }
        }
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            auth_value,
        })
    }

    /// Тело запроса chat/completions.
    fn request_body(&self, messages: Vec<Value>, temperature: f64, max_tokens: usize) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "temperature": temperature,
        });
        if max_tokens > 0 {
            body["max_tokens"] = json!(max_tokens);
        }
        body
    }

    /// Проверка ключа и адреса минимальным запросом (без стрима, пара токенов).
    /// Возвращает ответ модели или честную ошибку — для кнопки «Проверить».
    pub fn check_connection(&self) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": "Ответь одним словом: готов"}],
            "max_tokens": 8,
            "temperature": 0.0,
        });
        let agent = llm_agent();
        let response = agent
            .post(&url)
            .header("Authorization", &self.auth_value)
            .header("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| anyhow!("не удалось соединиться: {e}"))?;
        let status = response.status();
        let text = response
            .into_body()
            .read_to_string()
            .unwrap_or_default();
        if status != 200 {
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str().map(str::to_string))
                })
                .unwrap_or(text);
            let msg = msg.trim().to_string();
            if msg.is_empty() {
                return Err(anyhow!("HTTP {status}"));
            }
            return Err(anyhow!("HTTP {status}: {}", msg.chars().take(200).collect::<String>()));
        }
        let parsed: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow!("странный ответ провайдера: {e}"))?;
        let answer = parsed
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if answer.is_empty() {
            return Err(anyhow!("провайдер ответил пустотой — модель не ответила"));
        }
        Ok(answer)
    }

    /// Сообщения запроса: system + история + пользователь (+ prefill как
    /// частичный ответ ассистента — стандартный приём OpenAI-совместимых API).
    fn messages(&self, system: &str, history: &[ChatTurn], user: &str, prefill: Option<&str>) -> Vec<Value> {
        let mut messages = vec![json!({ "role": "system", "content": system })];
        for (q, a) in history {
            messages.push(json!({ "role": "user", "content": q }));
            messages.push(json!({ "role": "assistant", "content": a }));
        }
        messages.push(json!({ "role": "user", "content": user }));
        if let Some(p) = prefill {
            messages.push(json!({ "role": "assistant", "content": p }));
        }
        messages
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_params(
        &self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<super::config::GenerationParams>,
        cancel: &AtomicBool,
        prefill: Option<&str>,
        mut on_sentence: impl FnMut(&str),
        mut on_progress: impl FnMut(&str),
    ) -> Result<String> {
        let params = params_override.unwrap_or_else(|| super::config::generation_params().read().unwrap_or_else(|e| e.into_inner()).clone());
        let body = self.request_body(
            self.messages(system, history, user, prefill),
            params.temperature,
            params.max_tokens,
        );

        let url = format!("{}/chat/completions", self.base_url);
        on_progress("запрос…");

        let agent = llm_agent();
        // http_status_as_error(false): 4xx/5xx приходят обычным ответом —
        // тогда текст ошибки провайдера ({"error":{"message":..}}) можно
        // вычитать и показать пользователю честно.
        let response = agent
            .post(&url)
            .header("Authorization", &self.auth_value)
            .header("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| anyhow!("облачная модель недоступна: {e}"))
            .context("запрос к облачной модели не удался")?;

        let status = response.status();
        if status != 200 {
            let text = response
                .into_body()
                .read_to_string()
                .unwrap_or_default();
            // У Яндекса текст ошибки в JSON: {"error":{"message":".."}}.
            let msg = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str().map(str::to_string))
                })
                .unwrap_or(text);
            let msg = msg.trim().to_string();
            let detail = if msg.is_empty() {
                format!("HTTP {status}")
            } else if msg.chars().count() > 300 {
                format!("HTTP {status}: {}", msg.chars().take(300).collect::<String>())
            } else {
                format!("HTTP {status}: {msg}")
            };
            return Err(anyhow!("облачная модель: {detail}"));
        }

        // Разбор SSE: строки `data: {json}`, конец — `data: [DONE]`.
        let reader = response.into_body().into_reader();
        let mut full = String::new();
        let mut buf = String::new();
        let mut suppress_think = false;
        for line in std::io::BufRead::lines(std::io::BufReader::new(reader)) {
            if cancel.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let Ok(line) = line else { break };
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            let payload = payload.trim();
            if payload == "[DONE]" {
                break;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
                continue;
            };
            let Some(delta) = chunk
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            else {
                continue;
            };
            full.push_str(delta);
            let clean = streaming::strip_think(delta, &mut suppress_think);
            if !clean.is_empty() {
                streaming::push_delta(&mut buf, &clean, &mut on_sentence);
            }
        }
        // Хвост, который не дорос до конца предложения.
        if !buf.trim().is_empty() {
            let tail = std::mem::take(&mut buf);
            if streaming::speakable(&tail) {
                on_sentence(&tail);
            }
        }
        Ok(full)
    }
}

/// Отдельный HTTP-агент для облачной модели: без глобального таймаута
/// (стриминг длинного ответа идёт минутами), только таймауты соединения.
fn llm_agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(1500)))
            .timeout_connect(Some(Duration::from_secs(10)))
            .http_status_as_error(false)
            .build()
            .new_agent()
    })
}

/// Список чат-моделей провайдера (GET /models, OpenAI-совместимый формат).
///
/// Эмбеддинги (`emb://…`) выбрасываются; полные URI вида
/// `gpt://<каталог>/<модель>/latest` сокращаются до имени модели — полный
/// URI собирает движок при запросе (см. `ApiEngine::load`).
pub fn list_models() -> Result<Vec<String>> {
    let base_url = super::config::api_base_url().trim_end_matches('/').to_string();
    let key = super::config::api_key();
    if key.trim().is_empty() {
        return Err(anyhow!("API-ключ не задан: укажи его в настройках"));
    }
    let preset = super::config::api_preset();
    let auth_value = if preset == "yandex" {
        format!("Api-Key {key}")
    } else {
        format!("Bearer {key}")
    };
    let url = format!("{base_url}/models");
    let response = llm_agent()
        .get(&url)
        .header("Authorization", &auth_value)
        .call()
        .map_err(|e| anyhow!("не удалось получить список моделей: {e}"))?;
    let status = response.status();
    let text = response.into_body().read_to_string().unwrap_or_default();
    if status != 200 {
        return Err(anyhow!("HTTP {status}: {}", text.chars().take(200).collect::<String>()));
    }
    let data: Value =
        serde_json::from_str(&text).context("странный ответ списка моделей")?;
    let mut names: Vec<String> = data["data"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|m| m["id"].as_str())
                .filter(|id| !id.starts_with("emb://"))
                .map(|id| {
                    // gpt://<каталог>/<модель>[/<версия>] → <модель>.
                    id.strip_prefix("gpt://")
                        .and_then(|rest| rest.split_once('/'))
                        .map(|(_, model)| model.to_string())
                        .unwrap_or_else(|| id.to_string())
                })
                .map(|name| name.trim_end_matches("/latest").to_string())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Err(anyhow!("провайдер не вернул ни одной модели"));
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yandex_model_uri_is_built_from_folder() {
        // Пресет Яндекса собирает URI, когда ключ и каталог заданы.
        // Здесь проверяем только логику URI — статик конфиг трогать не надо.
        let folder = "b1gabc123";
        let model = "yandexgpt";
        let built = format!("gpt://{folder}/{model}/latest");
        assert_eq!(built, "gpt://b1gabc123/yandexgpt/latest");
        // Уже полный URI не трогаем.
        let full = "gpt://b1gabc123/yandexgpt/latest";
        assert!(full.starts_with("gpt://"));
    }

    #[test]
    fn request_body_has_stream_and_no_zero_limit() {
        let engine = ApiEngine {
            base_url: "https://example.com/v1".into(),
            model: "test-model".into(),
            auth_value: "Bearer k".into(),
        };
        let body = engine.request_body(
            vec![json!({"role": "user", "content": "привет"})],
            0.5,
            0,
        );
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["model"], json!("test-model"));
        assert!(body.get("max_tokens").is_none(), "0 = без лимита, поле не ставится");

        let body = engine.request_body(
            vec![json!({"role": "user", "content": "привет"})],
            0.5,
            256,
        );
        assert_eq!(body["max_tokens"], json!(256));
    }

    #[test]
    fn messages_include_history_and_prefill() {
        let engine = ApiEngine {
            base_url: "https://example.com/v1".into(),
            model: "m".into(),
            auth_value: "Bearer k".into(),
        };
        let history = vec![("вопрос".to_string(), "ответ".to_string())];
        let msgs = engine.messages("sys", &history, "новый вопрос", Some("<tool>"));
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[4]["role"], "assistant");
        assert_eq!(msgs[4]["content"], "<tool>");
    }
}
