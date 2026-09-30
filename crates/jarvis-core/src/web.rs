//! Поиск в интернете: DuckDuckGo Lite (без ключей и JavaScript) →
//! отрывки результатов → контекст для локальной LLM, которая строит
//! короткий понятный ответ.

use anyhow::{bail, Context, Result};
use std::time::Duration;

const SEARCH_URL: &str = "https://lite.duckduckgo.com/lite/";
const USER_AGENT: &str = "JarvisVoiceAssistant/2.0 (personal assistant)";

/// Результат поиска: заголовок, адрес, отрывок.
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Загружает страницу результатов и вытаскивает заголовки/отрывки.
///
/// Разметка Lite-версии простая: ссылки `<a class="result-link">` и
/// отрывки `<td class="result-snippet">`. Разбираем без тяжёлых крейтов.
pub fn search(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    if query.trim().is_empty() {
        bail!("Пустой поисковый запрос");
    }

    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(6)))
        .build()
        .new_agent();

    let url = format!("{}?q={}", SEARCH_URL, urlencoding::encode(query));
    let response = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept-Language", "ru-RU,ru;q=0.9,en;q=0.8")
        .call()
        .with_context(|| format!("Поиск не удался: {url}"))?;

    let html = response.into_body().read_to_string()?;

    Ok(parse_results(&html, limit))
}

fn parse_results(html: &str, limit: usize) -> Vec<SearchHit> {
    let mut hits = Vec::new();

    let mut rest = html;
    while hits.len() < limit {
        // DDG Lite использует и одинарные, и двойные кавычки в атрибутах.
        let Some(link_pos) = rest.find("result-link") else {
            break;
        };
        let Some(tag_start) = rest[..link_pos].rfind('<') else {
            break;
        };
        let tag = &rest[tag_start..];
        let Some(tag_end_rel) = tag.find('>') else {
            break;
        };
        let href = extract_href(&tag[..=tag_end_rel]);
        let after_tag = &tag[tag_end_rel + 1..];
        let Some(close) = after_tag.find("</a>") else {
            break;
        };
        let title = strip_tags(&after_tag[..close]);
        rest = &after_tag[close..];

        // Отрывок идёт следом за ссылкой.
        let snippet = rest
            .find("result-snippet")
            .and_then(|s| {
                let tail = &rest[s..];
                let o = tail.find('>')?;
                let c = tail[o..].find("</td>")?;
                Some(strip_tags(&tail[o + 1..o + c]))
            })
            .unwrap_or_default();

        if title.is_empty() {
            continue;
        }
        hits.push(SearchHit {
            title,
            url: href,
            snippet,
        });
    }

    hits
}

fn extract_href(tag: &str) -> String {
    tag.split("href=\"")
        .nth(1)
        .map(|tail| tail.split('"').next().unwrap_or("").to_string())
        .unwrap_or_default()
}

/// Убирает теги и разворачивает основные HTML-сущности.
fn strip_tags(fragment: &str) -> String {
    let mut out = String::with_capacity(fragment.len());
    let mut in_tag = false;
    let mut chars = fragment.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            '&' if !in_tag => {
                // Собираем сущность до ';', разворачиваем известные.
                let mut ent = String::new();
                let mut closed = false;
                for e in chars.by_ref() {
                    if e == ';' {
                        closed = true;
                        break;
                    }
                    if !e.is_ascii_alphanumeric() || ent.len() >= 8 {
                        break;
                    }
                    ent.push(e);
                }
                match (closed, ent.as_str()) {
                    (true, "amp") => out.push('&'),
                    (true, "lt") => out.push('<'),
                    (true, "gt") => out.push('>'),
                    (true, "quot") => out.push('"'),
                    (true, "nbsp") => out.push(' '),
                    (true, "#39") | (true, "apos") => out.push('\''),
                    _ => {
                        out.push('&');
                        out.push_str(&ent);
                    }
                }
            }
            c if !in_tag => out.push(c),
            _ => {}
        }
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Строит промпт для LLM: результаты поиска как контекст + вопрос.
/// Ответ должен быть коротким и устным — он пойдёт в озвучку.
pub fn build_summary_prompt(query: &str, hits: &[SearchHit]) -> Option<String> {
    if hits.is_empty() {
        return None;
    }

    let mut context = String::new();
    for (i, hit) in hits.iter().enumerate().take(5) {
        context.push_str(&format!(
            "[{}] {}\n{}\n\n",
            i + 1,
            hit.title,
            if hit.snippet.is_empty() {
                "(без описания)"
            } else {
                &hit.snippet
            }
        ));
    }

    Some(format!(
        "Результаты поиска в интернете по запросу «{query}»:\n\n{context}На основе этих результатов ответь на вопрос пользователя. Кратко: 1-3 предложения, только суть, на русском языке. Если результатов мало для ответа — честно скажи об этом."
    ))
}

/// Эвристика: вопрос про свежие/меняющиеся данные — стоит поискать в интернете.
pub fn looks_like_fresh_info(text: &str) -> bool {
    let t = text.to_lowercase();
    [
        "новост",
        "погод",
        "курс ",
        "курсова",
        "сегодня",
        "сейчас",
        "актуальн",
        "свеж",
        "последн",
        "который час",
        "сколько времени",
        "расписан",
        "когда выйдет",
        "когда выйдут",
        "цена",
        "стоит сейчас",
        "результат матча",
        "счет матча",
        "счёт матча",
        "кто выиграл",
        "что нового",
        "какие новости",
        "индекс",
        "биткоин",
        "биткоина",
        "ethereum",
    ]
    .iter()
    .any(|k| t.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_tags_basic() {
        assert_eq!(strip_tags("<b>Привет</b> мир"), "Привет мир");
        assert_eq!(strip_tags("A &amp; B"), "A & B");
        assert_eq!(strip_tags("1&nbsp;2"), "1 2");
    }

    #[test]
    fn parse_lite_results() {
        let html = r#"<tr><td><a rel="nofollow" href="https://a.example/1" class='result-link'>Первый результат</a></td></tr>
<tr><td class='result-snippet'>Описание &amp; первого</td></tr>
<tr><td><a rel="nofollow" href="https://a.example/2" class='result-link'>Второй</a></td></tr>
<tr><td class='result-snippet'>Второй отрывок</td></tr>"#;
        let hits = parse_results(html, 5);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "Первый результат");
        assert_eq!(hits[0].url, "https://a.example/1");
        assert_eq!(hits[0].snippet, "Описание & первого");
    }

    #[test]
    fn fresh_info_detection() {
        assert!(looks_like_fresh_info("Джарвис, какая погода сегодня"));
        assert!(looks_like_fresh_info("какие новости про металлургию"));
        assert!(!looks_like_fresh_info("расскажи сказку про кота"));
    }

    #[test]
    fn summary_prompt_contains_context() {
        let hits = vec![SearchHit {
            title: "Т".into(),
            url: "u".into(),
            snippet: "С".into(),
        }];
        let p = build_summary_prompt("запрос", &hits).unwrap();
        assert!(p.contains("запрос"));
        assert!(p.contains("Т"));
    }
}
