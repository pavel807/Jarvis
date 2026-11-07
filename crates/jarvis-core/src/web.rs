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
    // Быстрый путь — один DDG-прогон коротким агентом, без цепочки: решение
    // «открыть сайт» принимается быстро или не принимается вовсе.
    let url = format!("{}?q={}", SEARCH_URL, urlencoding::encode(query));
    let response = agent_fast().get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept-Language", "ru-RU,ru;q=0.9,en;q=0.8")
        .call()
        .with_context(|| format!("Поиск не удался: {url}"))?;
    let html = response.into_body().read_to_string()?;
    Ok(parse_results(&html, limit))
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
    let hits = search_chain(query, limit)?;
    if !hits.is_empty() {
        cache_put(key, &hits);
    }
    Ok(hits)
}

/// Цепочка движков: DDG нестабилен (анти-бот заглушки, таймауты из РФ),
/// поэтому при пустой/упавшей выдаче пробуем Bing — первый непустой побеждает.
fn search_chain(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let mut last_err: Option<String> = None;
    for attempt in 0..2 {
        match search_ddg(query, limit) {
            Ok(hits) if !hits.is_empty() => return Ok(hits),
            Ok(_) => last_err = Some("DDG: пустая выдача".into()),
            Err(e) => last_err = Some(format!("DDG: {e:#}")),
        }
        // Между попытками — пауза, DDG иногда отдаёт результат со второй.
        if attempt == 0 {
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    }
    match search_bing(query, limit) {
        Ok(hits) if !hits.is_empty() => {
            println!("🔍 DDG не дал результатов — ответил Bing ({} шт.)", hits.len());
            Ok(hits)
        }
        Ok(_) => Err(anyhow::anyhow!(
            "{}; Bing: пустая выдача",
            last_err.unwrap_or_default()
        )),
        Err(e) => Err(anyhow::anyhow!("{}; Bing: {e:#}", last_err.unwrap_or_default())),
    }
}

/// Вопрос-определение: «что такое X», «кто такой/кто такая X».
pub fn looks_like_definitional(text: &str) -> bool {
    let t = text.to_lowercase();
    t.contains("что такое") || t.contains("кто такой") || t.contains("кто такая")
}

/// Wikipedia как первый источник для вопросов-определений
/// («что такое X», «кто такой X»): надёжнее поисковиков для энциклопедического.
pub fn wikipedia_hit(query: &str) -> Option<SearchHit> {
    let api = format!(
        "https://ru.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=1&srsearch={}",
        urlencoding::encode(query)
    );
    let response = agent()
        .get(&api)
        .header("User-Agent", USER_AGENT)
        .call()
        .ok()?;
    let text = response.into_body().read_to_string().ok()?;
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    let first = data.pointer("/query/search/0")?;
    let title = first["title"].as_str()?.to_string();
    let snippet = strip_tags(first["snippet"].as_str().unwrap_or(""));
    let url = format!(
        "https://ru.wikipedia.org/wiki/{}",
        urlencoding::encode(&title.replace(' ', "_"))
    );
    Some(SearchHit { title, url, snippet })
}

/// Поиск через Bing HTML (запасной движок).
/// DDG lite: GET + разбор.
fn search_ddg(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    search_with(query, limit, agent())
}

/// Общая часть DDG-поиска: GET + разбор. Кэш — на уровне вызывающих.
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

fn search_bing(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let url = format!(
        "https://www.bing.com/search?q={}&setlang=ru",
        urlencoding::encode(query)
    );
    let response = agent_bing()
        .get(&url)
        .header("User-Agent", BING_UA)
        .call()
        .map_err(|e| anyhow::anyhow!("Bing не отвечает: {e}"))?;
    let html = response.into_body().read_to_string()?;
    let hits = parse_bing_results(&html, limit);
    Ok(hits)
}

const BING_UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36";

fn agent_bing() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(800)))
            .timeout_connect(Some(Duration::from_millis(2000)))
            .timeout_global(Some(Duration::from_secs(5)))
            .build()
            .new_agent()
    })
}

/// Разбор Bing-выдачи: блоки `<li class="b_algo">`, заголовок в `<h2><a>`,
/// сниппет в первом `<p>`; реальные адреса зашиты в redirect `u=a1<base64>`.
fn parse_bing_results(html: &str, limit: usize) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut rest = html;
    while hits.len() < limit {
        let Some(block_pos) = rest.find("<li class=\"b_algo\"") else {
            break;
        };
        rest = &rest[block_pos..];
        let Some(block_end) = rest.find("</li>") else { break };
        let block = &rest[..block_end];
        rest = &rest[block_end..];

        // Заголовок: <h2><a ... href="...">Заголовок</a></h2>
        let (title, raw_href) = match find_h2_link(block) {
            Some(pair) => pair,
            None => continue,
        };
        let raw_href: &str = &raw_href;
        // Сниппет: первый <p ...>…</p> после заголовка.
        let snippet = block
            .split("<h2")
            .nth(1)
            .and_then(|after| after.split("<p").nth(1))
            .and_then(|p| p.split_once('>'))
            .and_then(|(_, body)| body.split("</p>").next())
            .map(strip_tags)
            .unwrap_or_default();
        let url = unwrap_bing_url(raw_href).unwrap_or_else(|| raw_href.to_string());
        if title.is_empty() || url.is_empty() {
            continue;
        }
        hits.push(SearchHit {
            title: title.trim().to_string(),
            url,
            snippet: snippet.trim().to_string(),
        });
    }
    hits
}

fn find_h2_link(block: &str) -> Option<(String, String)> {
    let h2 = block.find("<h2")?;
    let after = &block[h2..];
    let a_start = after.find("<a")?;
    let tag_end = after[a_start..].find('>')? + a_start;
    let tag = &after[a_start..=tag_end];
    let href = tag
        .split("href=\"")
        .nth(1)
        .and_then(|v| v.split('"').next())
        .unwrap_or("")
        .to_string();
    let body = &after[tag_end + 1..];
    let title = body.split("</a>").next().map(strip_tags).unwrap_or_default();
    Some((title, href))
}

/// Bing-редирект `…/ck/a?…&u=a1<base64url>` → настоящий адрес.
fn unwrap_bing_url(href: &str) -> Option<String> {
    let u = href.split("u=a1").nth(1)?;
    let encoded: String = u
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '=')
        .collect();
    if encoded.len() < 8 {
        return None;
    }
    // base64url → base64: -_ заменяем, паддинг дописываем.
    let mut std_b64 = encoded.replace('-', "+").replace('_', "/");
    while std_b64.len() % 4 != 0 {
        std_b64.push('=');
    }
    decode_base64(&std_b64).map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// Минимальный base64-декодер (без внешних крейтов).
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0u32;
    for ch in input.chars() {
        if ch == '=' {
            break;
        }
        let val = TABLE.iter().position(|t| *t as char == ch)? as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
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
    // Хотя бы один содержательный сниппет: выдача из одних заголовков без
    // описаний не даёт модели материала — честный отказ лучше отсебятины.
    if hits.iter().all(|h| h.snippet.trim().is_empty()) {
        return None;
    }
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
    // Определения: «что такое Дональд Трамп» — про реальный мир, короткое
    // содержание из поиска ценнее открывания браузера.
    if t.contains("что такое") || t.contains("что за") {
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
        assert!(looks_like_fresh_info("что такое Дональд Трамп"));
        assert!(looks_like_fresh_info("кто такой Илон Маск"));
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