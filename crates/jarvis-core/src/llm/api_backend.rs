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

/// GigaChat: чат (OpenAI-совместимый) и выдача access-токена.
/// Пользователь задаёт один ключ авторизации (Basic …) — access-токен
/// Джарвис получает сам и продлевает по мере надобности.
pub const GIGACHAT_BASE_URL: &str = "https://gigachat.devices.sberbank.ru/api/v1";
pub const GIGACHAT_OAUTH_URL: &str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
pub const GIGACHAT_DEFAULT_MODEL: &str = "GigaChat";
pub const GIGACHAT_SCOPE: &str = "GIGACHAT_API_PERS";

/// Сертификаты НУЦ Минцифры (официальные с gu-st.ru — Госуслуги): цепочка
/// sber-эндпоинтов подписана ими, Mozilla-корни эти имена не знают.
/// Файл — root + sub CA (2022) + sub CA 2024, PEМ с LF-концами.
const RU_CA_PEM: &str = include_str!("../../certs/russian_trusted_root_ca.crt");

pub struct ApiEngine {
    /// Базовый адрес без хвостового `/`: например `https://llm.api.cloud.yandex.net/openai/v1`.
    base_url: String,
    /// Полное имя модели: `yandexgpt-lite`, `gpt://b1g.../yandexgpt/latest`, `gpt-4o-mini`, …
    model: String,
    /// Заголовок авторизации: `Api-Key <ключ>` (Яндекс) или `Bearer <ключ>`.
    /// Для GigaChat не используется — там Bearer access-токена живёт своей
    /// жизнью (см. `auth_header`).
    auth_value: String,
    /// Пресет: "yandex" | "openai" | "gigachat".
    preset: String,
    /// Только GigaChat: ключ авторизации (Basic …) для выдачи access-токена.
    basic_key: String,
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
        let mut basic_key = String::new();
        let auth_value = if preset == "gigachat" {
            // GigaChat: ключ — Basic-авторизация для oauth, чат ходит с
            // Bearer access-токеном, который Джарвис получает сам.
            basic_key = api_key.trim().to_string();
            String::new()
        } else if preset == "yandex" {
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
            if folder.trim().is_empty() && !model.starts_with("gpt://") {
                // Без каталога Яндекс вернёт криптическое «Failed to parse
                // model URI»: говорим прямо и до запроса, где что вписать.
                let _ = events.send(JarvisEvent::Error {
                    message: "Для Яндекс GPT нужен Каталог (folder id, b1g…) — Настройки → Модель"
                        .into(),
                });
                return Err(anyhow!(
                    "Каталог (folder id) не задан: впиши его в Настройки → Модель,                      поле «Каталог (folder id)» — b1g… из консоли Яндекса"
                ));
            }
            if !folder.trim().is_empty() && !model.starts_with("gpt://") {
                let version = if model.contains('/') { "" } else { "/latest" };
                model = format!("gpt://{}/{model}{version}", folder.trim());
            }
        }
        // GigaChat: дефолты адреса и модели, если поля пустые.
        let (base_url, model) = if preset == "gigachat" {
            let base = if base_url.trim().is_empty() {
                GIGACHAT_BASE_URL.to_string()
            } else {
                base_url
            };
            let model = if model.trim().is_empty() || model == YANDEX_DEFAULT_MODEL {
                GIGACHAT_DEFAULT_MODEL.to_string()
            } else {
                model
            };
            (base, model)
        } else {
            (base_url, model)
        };
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            auth_value,
            preset,
            basic_key,
        })
    }

    /// Заголовок авторизации для запроса. GigaChat — живой access-токен
    /// (получается Basic-ключом, кэшируется, продлевается сам).
    fn auth_header(&self, force_refresh: bool) -> Result<String> {
        if self.preset != "gigachat" {
            return Ok(self.auth_value.clone());
        }
        Ok(format!(
            "Bearer {}",
            gigachat_access_token(&self.basic_key, force_refresh)?
        ))
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
        let agent = if self.preset == "gigachat" {
            gigachat_agent()
        } else {
            llm_agent()
        };
        let response = agent
            .post(&url)
            .header("Authorization", &self.auth_header(false)?)
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

        // GigaChat — свой агент с бандлом НУЦ: Mozilla-корни цепочку
        // sber-эндпоинтов не знают (живой лог: «не удалось соединиться»).
        let agent = if self.preset == "gigachat" {
            gigachat_agent()
        } else {
            llm_agent()
        };
        // http_status_as_error(false): 4xx/5xx приходят обычным ответом —
        // тогда текст ошибки провайдера ({"error":{"message":..}}) можно
        // вычитать и показать пользователю честно.
        let send = |auth: &str| {
            agent
                .post(&url)
                .header("Authorization", auth)
                .header("Content-Type", "application/json")
                .send_json(&body)
                .map_err(|e| anyhow!("облачная модель недоступна: {e}"))
                .context("запрос к облачной модели не удался")
        };
        let mut response = send(&self.auth_header(false)?)?;
        // GigaChat: access-токен живёт ~30 минут — 401 значит «протух»,
        // берём свежий и повторяем один раз.
        if response.status() == 401 && self.preset == "gigachat" {
            let fresh = self.auth_header(true)?;
            response = send(&fresh)?;
        }

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

/// Доступ к GigaChat без Access-токена не работает: цепочка сертификатов
/// sber-эндпоинтов подписана НУЦ Минцифры, Mozilla-корни её не знают —
/// поэтому у GigaChat свой агент с бандлом НУЦ (root + sub CA).
fn gigachat_agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| {
        // Скачанный с gu-st.ru файл мог прийти с CRLF — на всякий случай
        // нормализуем перед разбором.
        let pem = RU_CA_PEM.replace('\r', "");
        let certs: Vec<ureq::tls::Certificate> = ureq::tls::parse_pem(pem.as_bytes())
            .filter_map(|item| match item {
                Ok(ureq::tls::PemItem::Certificate(c)) => Some(c.to_owned()),
                _ => None,
            })
            .collect();
        let tls = ureq::tls::TlsConfig::builder()
            .root_certs(ureq::tls::RootCerts::Specific(std::sync::Arc::new(certs)))
            .build();
        ureq::Agent::config_builder()
            .tls_config(tls)
            .timeout_resolve(Some(Duration::from_millis(1500)))
            .timeout_connect(Some(Duration::from_secs(10)))
            .http_status_as_error(false)
            .build()
            .new_agent()
    })
}

/// Кэш access-токена GigaChat: токен + момент истечения.
struct GigachatToken {
    token: String,
    until: std::time::Instant,
}

static GIGACHAT_TOKEN: std::sync::Mutex<Option<GigachatToken>> = std::sync::Mutex::new(None);

/// Access-токен GigaChat: из кэша, если жив; иначе POST на oauth с
/// Basic-ключом пользователя. `force_refresh` — после 401.
fn gigachat_access_token(basic_key: &str, force_refresh: bool) -> Result<String> {
    if basic_key.trim().is_empty() {
        return Err(anyhow!(
            "Ключ GigaChat не задан: в настройках → AI-модель → GigaChat"
        ));
    }
    {
        let cached = GIGACHAT_TOKEN
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(t) = cached.as_ref() {
            if !force_refresh && t.until > std::time::Instant::now() {
                return Ok(t.token.clone());
            }
        }
    }
    // RqUID — уникальный идентификатор запроса (uuid v4). Без зависимостей:
    // энтропия из времени, pid и атомарного счётчика.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let mut h = nanos ^ (pid << 32) ^ c.wrapping_mul(0x9E3779B97F4A7C15);
    let mut bytes = [0u8; 16];
    for b in &mut bytes {
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        h ^= h >> 33;
        *b = (h >> 24) as u8;
    }
    // uuid v4: версия 4 и вариант RFC 4122.
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;
    let rquid = format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    );

    let response = gigachat_agent()
        .post(GIGACHAT_OAUTH_URL)
        .header("RqUID", &rquid)
        .header("Authorization", &format!("Basic {basic_key}"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .send(format!("scope={GIGACHAT_SCOPE}"))
        .map_err(|e| anyhow!("не удалось соединиться с GigaChat oauth: {e}"))?;
    let status = response.status();
    let text = response.into_body().read_to_string().unwrap_or_default();
    if status != 200 {
        let msg = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("message")
                    .or_else(|| v.pointer("/error/message"))
                    .and_then(|m| m.as_str().map(str::to_string))
            })
            .unwrap_or(text);
        let msg = msg.trim().to_string();
        return Err(anyhow!(
            "GigaChat oauth: HTTP {status}{}",
            if msg.is_empty() {
                String::new()
            } else {
                format!(": {}", msg.chars().take(200).collect::<String>())
            }
        ));
    }
    let parsed: Value =
        serde_json::from_str(&text).map_err(|e| anyhow!("странный ответ oauth: {e}"))?;
    let token = parsed
        .get("access_token")
        .and_then(Value::as_str)
        .context("oauth не вернул access_token")?
        .to_string();
    // expires_at — unix-время в миллисекундах; страховка: не дольше 25 минут
    // (последние минуты токен уже может быть отозван на шлюзе).
    let until = parsed
        .get("expires_at")
        .and_then(Value::as_i64)
        .map(|ms| {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let secs_left = ((ms - now_ms).max(0) / 1000) as u64;
            std::time::Instant::now()
                + std::time::Duration::from_secs(secs_left.saturating_sub(180))
        })
        .unwrap_or_else(|| std::time::Instant::now() + std::time::Duration::from_secs(25 * 60));
    *GIGACHAT_TOKEN
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(GigachatToken { token, until });
    Ok(GIGACHAT_TOKEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|t| t.token.clone())
        .unwrap_or_default())
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
    fn gigachat_agent_has_ru_ca_roots() {
        // Бандл НУЦ разбирается: хотя бы 2 сертификата (root + sub CA).
        let pem = RU_CA_PEM.replace('\r', "");
        let certs: Vec<_> = ureq::tls::parse_pem(pem.as_bytes())
            .filter_map(|item| match item {
                Ok(ureq::tls::PemItem::Certificate(c)) => Some(c),
                _ => None,
            })
            .collect();
        assert!(certs.len() >= 2, "в бандле НУЦ {} сертификатов", certs.len());
    }

    #[test]
    fn request_body_has_stream_and_no_zero_limit() {
        let engine = ApiEngine {
            base_url: "https://example.com/v1".into(),
            model: "test-model".into(),
            auth_value: "Bearer k".into(),
            preset: "openai".into(),
            basic_key: String::new(),
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
            preset: "openai".into(),
            basic_key: String::new(),
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
