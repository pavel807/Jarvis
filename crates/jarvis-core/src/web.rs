//! Поиск в интернете: DuckDuckGo Lite (без ключей и JavaScript) →
//! отрывки результатов → контекст для локальной LLM, которая строит
//! короткий понятный ответ.

use anyhow::{Context, Result, bail};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const SEARCH_URL: &str = "https://lite.duckduckgo.com/lite/";
const USER_AGENT: &str = "JarvisVoiceAssistant/2.0 (personal assistant)";

/// Сколько живёт закэшированная выдача. Повторные вопросы («погода»,
/// «новости») отвечают мгновенно, но данные не успевают устареть.
const CACHE_TTL: Duration = Duration::from_secs(120);
/// Размер кэша выдач (LRU).
const CACHE_CAP: usize = 64;

/// Результат поиска: заголовок, адрес, отрывок.
#[derive(Clone)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Общий HTTP-агент: соединение и TLS-сессия переиспользуются между
/// запросами, поэтому поиск не тратит время на рукопожатие каждый раз.
fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(800)))
            .timeout_connect(Some(Duration::from_millis(1500)))
            .timeout_global(Some(Duration::from_secs(4)))
            .max_idle_connections_per_host(4)
            .build()
            .new_agent()
    })
}

/// Агент с коротким бюджетом времени — для случаев, где ждать долго
/// нельзя (например, «открой сайт …»: не нашли — откроем выдачу поиска).
fn agent_fast() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(600)))
            .timeout_connect(Some(Duration::from_millis(1000)))
            .timeout_global(Some(Duration::from_millis(1600)))
            .max_idle_connections_per_host(4)
            .build()
            .new_agent()
    })
}

/// Поиск с жёстким лимитом времени: нужен там, где решение «открыть сайт»
/// или «показать выдачу» должно приниматься быстро.
pub fn search_fast(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    search_with(query, limit, agent_fast())
}

/// Прогревает DNS/TLS до поисковика в фоне, чтобы первый запрос
/// пользователя не платил за рукопожатие.
pub fn warm_up() {
    std::thread::spawn(|| {
        let started = Instant::now();
        match agent().get(SEARCH_URL).call() {
            Ok(_) => println!("🔥 Поиск прогрет за {:.0?}", started.elapsed()),
            Err(e) => eprintln!("⚠️ Прогрев поиска не удался: {e}"),
        }
    });
}

struct CacheEntry {
    key: String,
    hits: Vec<SearchHit>,
    at: Instant,
}

static CACHE: Mutex<VecDeque<CacheEntry>> = Mutex::new(VecDeque::new());

fn cache_key(query: &str) -> String {
    query
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn cache_get(key: &str, limit: usize) -> Option<Vec<SearchHit>> {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let pos = cache.iter().position(|e| e.key == key)?;
    let fresh = cache[pos].at.elapsed() < CACHE_TTL;
    if !fresh {
        cache.remove(pos);
        return None;
    }
    let mut entry = cache.remove(pos)?;
    entry.hits.truncate(limit);
    let hits = entry.hits.clone();
    cache.push_back(entry);
    Some(hits)
}

fn cache_put(key: String, hits: &[SearchHit]) {
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pos) = cache.iter().position(|e| e.key == key) {
        cache.remove(pos);
    }
    cache.push_back(CacheEntry {
        key,
        hits: hits.to_vec(),
        at: Instant::now(),
    });
    while cache.len() > CACHE_CAP {
        cache.pop_front();
    }
}

/// Загружает страницу результатов и вытаскивает заголовки/отрывки.
///
/// Разметка Lite-версии простая: ссылки `<a class="result-link">` и
/// отрывки `<td class="result-snippet">`. Разбираем без тяжёлых крейтов.
pub fn search(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    if query.trim().is_empty() {
        bail!("Пустой поисковый запрос");
    }

    let key = cache_key(query);
    if let Some(hits) = cache_get(&key, limit) {
        println!("⚡ Поиск из кэша: {query:?} ({} результатов)", hits.len());
        return Ok(hits);
    }

    let hits = search_with(query, limit, agent())?;
    if !hits.is_empty() {
        cache_put(key, &hits);
    }
    Ok(hits)
}

/// Общая часть поиска: GET + разбор. Кэш — на уровне вызывающих.
fn search_with(query: &str, limit: usize, agent: &ureq::Agent) -> Result<Vec<SearchHit>> {
    let started = Instant::now();
    let url = format!("{}?q={}", SEARCH_URL, urlencoding::encode(query));
    let response = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept-Language", "ru-RU,ru;q=0.9,en;q=0.8")
        .call()
        .with_context(|| format!("Поиск не удался: {url}"))?;

    let html = response.into_body().read_to_string()?;
    let hits = parse_results(&html, limit);
    println!(
        "🌐 Поиск за {:.0?} ({} результатов)",
        started.elapsed(),
        hits.len()
    );
    Ok(hits)
}

/// Ссылка на страницу результатов поиска — открываем её, когда не уверены,
/// какой сайт нужен. Лучше реальная выдача, чем случайный «левый» сайт.
pub fn search_url(query: &str) -> String {
    format!("https://duckduckgo.com/?q={}", urlencoding::encode(query))
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
        let raw_tag = &tag[..=tag_end_rel];
        let after_tag = &tag[tag_end_rel + 1..];
        let Some(close) = after_tag.find("</a>") else {
            break;
        };
        let title = strip_tags(&after_tag[..close]);
        // Рекламу определяем до продвижения курсора.
        let sponsored = is_sponsored(&rest[..tag_start], raw_tag, &extract_href_raw(raw_tag));
        rest = &after_tag[close..];
        if sponsored {
            continue;
        }

        let href = extract_href(raw_tag);

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

/// Рекламные результаты DuckDuckGo идут через `y.js`-редирект и/или помечены
/// классом `result-sponsored`/`ad_domain` — их пропускаем, чтобы брать органику.
fn is_sponsored(prior: &str, tag: &str, href_raw: &str) -> bool {
    href_raw.contains("y.js")
        || tag.contains("result-sponsored")
        || prior.contains("result-sponsored")
        || prior.contains("ad_domain")
}

fn extract_href_raw(tag: &str) -> String {
    tag.split("href=\"")
        .nth(1)
        .map(|tail| tail.split('"').next().unwrap_or("").to_string())
        .unwrap_or_default()
}

fn extract_href(tag: &str) -> String {
    normalize_href(&extract_href_raw(tag))
}

/// DuckDuckGo Lite отдаёт ссылки через свой редирект
/// `//duckduckgo.com/l/?uddg=<urlencoded>`. Возвращаем настоящий адрес.
fn normalize_href(href: &str) -> String {
    let h = href.replace("&amp;", "&");
    let bare = h
        .strip_prefix("https://")
        .or_else(|| h.strip_prefix("http://"))
        .unwrap_or(&h);
    let bare = bare.strip_prefix("//").unwrap_or(bare);
    if bare.starts_with("duckduckgo.com/l/")
        && let Some(enc) = bare.split("uddg=").nth(1)
    {
        let enc = enc.split('&').next().unwrap_or("");
        if let Ok(dec) = urlencoding::decode(enc) {
            return dec.into_owned();
        }
    }
    h
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
    // Вопрос о человеке — это факты, у модели они устаревшие или выдуманные:
    // «кто такая лариса долина» ищем в сети. Про саму модель («кто ты
    // такая») спрашивать сеть не нужно.
    let person_question = (t.contains("кто такой") || t.contains("кто такая"))
        && !t.contains(" ты")
        && !t.starts_with("ты");
    if person_question {
        return true;
    }
    // Факт-вопросы «когда» и «в каком году»: 2B по памяти путает даты
    // (Айфон «в октябре 2007»), а поиск отвечает верно.
    if t.contains("когда ") || t.contains("в каком году") {
        return true;
    }
    // Рекомендации: «во что поиграть с колодой карт» — без поиска модель
    // выдумывает несуществующие игры. «Что посмотреть/почитать» — то же.
    if t.contains("поиграть") || t.contains("что посмотреть") || t.contains("что почитать")
    {
        return true;
    }
    // «Сколько …» — факты про вещи (игроков, вес, длину), но не про
    // устройство и не про то, что считаем сами (проценты).
    if t.contains("сколько")
        && ![
            "диск",
            "памя",
            "заряд",
            "места",
            "тебе",
            "времени",
            "время",
            "процент",
        ]
        .iter()
        .any(|w| t.contains(w))
    {
        return true;
    }
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
    fn sponsored_results_are_skipped() {
        let html = r#"<tr><td class="result-sponsored"><a rel="nofollow" href="//duckduckgo.com/y.js?ad=1" class='result-link'>Реклама магазина</a></td></tr>
<tr><td class='result-snippet'>Купи сейчас</td></tr>
<tr><td><a rel="nofollow" href="https://www.apple.com/ipad-pro/" class='result-link'>iPad Pro</a></td></tr>
<tr><td class='result-snippet'>Официальный сайт</td></tr>"#;
        let hits = parse_results(html, 5);
        assert_eq!(hits.len(), 1, "реклама должна быть отфильтрована");
        assert_eq!(hits[0].url, "https://www.apple.com/ipad-pro/");
    }

    #[test]
    fn ddg_redirect_is_unwrapped() {
        let href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fcdek.shopping%2F&amp;rut=abc";
        assert_eq!(normalize_href(href), "https://cdek.shopping/");
        assert_eq!(
            normalize_href("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.cdek.ru%2Fru%2F"),
            "https://www.cdek.ru/ru/"
        );
        assert_eq!(normalize_href("https://a.example/x"), "https://a.example/x");
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