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

/// Универсальный факт-вопрос — не обязательно про свежие данные: «почему
/// море солёное», «что такое квазар», «чем отличается SQL от NoSQL».
/// Свежесть не важна: маленькая локальная модель на такие вопросы часто
/// выдумывает, а поиск отвечает точно. Вспомогательный гейт к
/// [`looks_like_fresh_info`]: она ловит данные, эта — знания.
pub fn looks_like_factual_question(text: &str) -> bool {
    let t = text.to_lowercase().replace('ё', "е");
    // Вопрос про самого ассистента («почему ты молчишь», «что ты умеешь»,
    // «что тебе нужно») — не поиск.
    if t.split_whitespace().any(|w| w == "ты" || w == "тебе" || w == "тебя") {
        return false;
    }
    [
        "почему",
        "что такое",
        "что такое ",
        "кто такой",
        "кто такая",
        "кто такие",
        "где наход",
        "где род",
        "где жив",
        "где расположен",
        "чем отличается",
        "чем различ",
        "какая разница",
        "в чем разница",
        "в чём разница",
        "кто придумал",
        "кто изобр",
        "кто открыл",
        "кто написал",
        "кто создал",
        "кто первым",
        "самый ",
        "самая ",
        "самое ",
        "самые ",
        "расскажи про ",
        "расскажи о ",
        "расскажи кратко",
        "как работает",
        "как устроен",
        "из чего состоит",
        "для чего нуж",
        "зачем нуж",
        "в чем смысл",
        "в чём смысл",
    ]
    .iter()
    .any(|k| t.contains(k))
}

// ============================================================
// Чтение статей («почитай статью про …», «прочитай по ссылке»)
// ============================================================

/// Сколько символов статьи читается за один заход («читай дальше»):
/// примерно минута речи. Статья длиннее — остаётся на следующий заход.
pub const ARTICLE_CHUNK_CHARS: usize = 1400;
/// Верхний предел извлечённого текста: дальше страница превращается в
/// мусор из меню и комментариев, а контекст модели не резиновый.
const ARTICLE_MAX_CHARS: usize = 20_000;

/// Прочитанная статья: заголовок, полный текст и сколько уже прочитано.
struct ArticleState {
    title: String,
    text: String,
    offset: usize,
}

/// Последняя прочитанная статья — «читай дальше» берёт следующий кусок.
static ARTICLE: Mutex<Option<ArticleState>> = Mutex::new(None);

/// Агент под страницы: у поиска бюджет 4 с — заголовки сниппетов приходят
/// быстро; целая страница тяжелее, ей нужен свой таймаут.
fn agent_page() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(1500)))
            .timeout_connect(Some(Duration::from_secs(6)))
            .timeout_global(Some(Duration::from_secs(20)))
            .max_idle_connections_per_host(2)
            .build()
            .new_agent()
    })
}

/// Похоже ли сказанное на просьбу прочитать статью/страницу вслух.
/// Возвращает предмет чтения: URL из текста либо тему для поиска.
/// Пустая тема («просто почитай что-нибудь») — None: непонятно, что искать.
pub fn read_article_target(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let has_read_word = ["почитай", "прочитай", "прочти", "зачитай", "читай"]
        .iter()
        .any(|w| lower.contains(w));
    if !has_read_word {
        return None;
    }
    // «Не читай», «перестань читать» — не просьба.
    if lower.contains("не читай") || lower.contains("перестань") || lower.contains("хватит") {
        return None;
    }

    // Явная ссылка в тексте — читаем её.
    for token in text.split_whitespace() {
        let token = token.trim_matches(|c: char| {
            c == '"' || c == '\'' || c == ')' || c == '(' || c == ',' || c == '.' || c == '!'
        });
        if token.starts_with("http://") || token.starts_with("https://") || token.contains("://")
        {
            return Some(token.to_string());
        }
        if token.ends_with(".ru") || token.ends_with(".com") {
            return Some(format!("https://{token}"));
        }
    }

    // Предмет: то, что осталось после слов-обёрток («почитай статью про …»).
    let mut subject = text.to_string();
    for filler in [
        "джарвис",
        "пожалуйста",
        "мне",
        "вслух",
        "статью",
        "статьи",
        "статья",
        "почитай",
        "прочитай",
        "прочти",
        "зачитай",
        "читай",
        "про",
        "о ",
        "об ",
        "ссылке",
        "ссылку",
        "по ссылке",
        "из интернета",
        "в интернете",
        "что-нибудь",
        "что нибудь",
        "что-то",
        "сейчас",
        "давай",
        "а",
    ] {
        subject = subject
            .to_lowercase()
            .replace(filler, " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        // replace выше стирает регистр темы — приемлемо: тема уходит в поиск.
    }
    let subject = subject.trim();
    if subject.is_empty() || subject.chars().count() < 3 {
        return None;
    }
    Some(subject.to_string())
}

/// Скачать страницу и вернуть (заголовок, читаемый текст).
/// Скрипты, стили, меню и подвалы вычищаются — синтезу речь нужна, а не HTML.
pub fn fetch_article(url: &str) -> Result<(String, String)> {
    let response = agent_page()
        .get(url)
        .header("User-Agent", USER_AGENT)
        .call()
        .context("страница не открылась")?;
    let mut html = response
        .into_body()
        .read_to_string()
        .context("не удалось прочитать страницу")?;
    // Дальше страницы 2 МБ — это уже не статья, обрежем сырой HTML, чтобы
    // не держать в памяти мегабайты мусора. Граница — по символу UTF-8,
    // а не по байту, иначе строка становится невалидной.
    if html.len() > 2_000_000 {
        let mut end = 2_000_000;
        while !html.is_char_boundary(end) {
            end -= 1;
        }
        html.truncate(end);
    }
    let (title, text) = extract_article_text(&html);
    if text.chars().count() < 120 {
        bail!("на странице не нашлось текста статьи");
    }
    let text = text.chars().take(ARTICLE_MAX_CHARS).collect::<String>();
    Ok((title, text))
}

/// Запомнить статью и отдать первый кусок для озвучки.
pub fn start_reading(title: &str, text: &str) -> String {
    let mut article = ARTICLE.lock().unwrap_or_else(|e| e.into_inner());
    *article = Some(ArticleState {
        title: title.to_string(),
        text: text.to_string(),
        offset: 0,
    });
    next_article_chunk()
        .unwrap_or_else(|| "Статья слишком короткая.".to_string())
}

/// Следующий кусок статьи для «читай дальше». None — нечего читать: статья
/// не открывалась или дочитана до конца.
pub fn next_article_chunk() -> Option<String> {
    let mut article = ARTICLE.lock().unwrap_or_else(|e| e.into_inner());
    let state = article.as_mut()?;
    let chars: Vec<char> = state.text.chars().collect();
    if state.offset >= chars.len() {
        return None;
    }
    let end = (state.offset + ARTICLE_CHUNK_CHARS).min(chars.len());
    let mut chunk: String = chars[state.offset..end].iter().collect();
    // Кусок заканчивается на полуслове — отступаем до последней границы.
    if end < chars.len() {
        if let Some(pos) = chunk.rfind(|c| c == '.' || c == '!' || c == '?' || c == '\n') {
            chunk.truncate(pos + 1);
            state.offset += chunk.chars().count();
        } else {
            state.offset = end;
        }
    } else {
        state.offset = end;
    }
    Some(chunk.trim_start().to_string())
}

/// Сколько ещё осталось читать (символов). 0 — статья дочитана.
pub fn article_remaining() -> usize {
    let article = ARTICLE.lock().unwrap_or_else(|e| e.into_inner());
    article
        .as_ref()
        .map(|a| a.text.chars().count().saturating_sub(a.offset))
        .unwrap_or(0)
}

/// Заголовок последней статьи (для «о чём мы читали»).
pub fn last_article_title() -> Option<String> {
    let article = ARTICLE.lock().unwrap_or_else(|e| e.into_inner());
    article.as_ref().map(|a| a.title.clone())
}

/// Вытащить (заголовок, текст) из HTML: скрипты/стили/навигация прочь,
/// абзацы разделяются переносом строки, теги и сущности раскрываются.
pub fn extract_article_text(html: &str) -> (String, String) {
    let title = html
        .to_ascii_lowercase()
        .find("<title")
        .and_then(|start| {
            let open_end = html[start..].find('>')? + start + 1;
            let close = html[open_end..].find("</title>")? + open_end;
            Some(html[open_end..close].trim().to_string())
        })
        .unwrap_or_default();

    let mut cleaned = html.to_string();
    // Скрипты и стили — целиком, вместе с содержимым.
    for tag in ["script", "style", "noscript", "svg", "head"] {
        cleaned = remove_block(&cleaned, tag);
    }
    // Навигация и подвалы — лучшие усилия: не на всех сайтах семантика есть.
    for tag in ["nav", "header", "footer", "aside", "form", "button"] {
        cleaned = remove_block(&cleaned, tag);
    }
    // HTML-комментарии.
    while let Some(start) = cleaned.find("<!--") {
        match cleaned[start..].find("-->") {
            Some(len) => cleaned.replace_range(start..start + len + 3, ""),
            None => break,
        }
    }
    // Блочные теги дают границу абзаца.
    for tag in [
        "</p>", "</h1>", "</h2>", "</h3>", "</h4>", "</li>", "<br", "</div>", "</tr>",
    ] {
        cleaned = cleaned.replace(tag, "\n");
    }
    let text = strip_tags(&cleaned);
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.chars().count() < 2 {
            continue;
        }
        // Мусорные строки-обёртки («Поделиться», подписки) — лучшие усилия.
        const NOISE: &[&str] = &["поделиться", "подписат", "cookie", "реклам", "комментар"];
        if NOISE.iter().any(|n| line.to_lowercase().contains(n)) && line.chars().count() < 60 {
            continue;
        }
        out.push_str(&line);
        out.push('\n');
    }
    (title, out.trim().to_string())
}

/// Удалить блок `<tag …>…</tag>` целиком (регистронезависимо, без вложенных
/// одноимённых тегов — для script/style этого достаточно).
fn remove_block(html: &str, tag: &str) -> String {
    let mut out = html.to_string();
    let close = format!("</{tag}>");
    let open_a = format!("<{tag}>");
    let open_b = format!("<{tag} ");
    loop {
        let lower = out.to_ascii_lowercase();
        let Some(start) = lower.find(&open_a).or_else(|| lower.find(&open_b)) else {
            break;
        };
        match lower[start..].find(&close) {
            Some(offset) => {
                let end = start + offset + close.len();
                out.replace_range(start..end, " ");
            }
            None => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factual_question_detection() {
        for phrase in [
            "почему море солёное",
            "что такое квазар",
            "где родился Пушкин",
            "чем отличается SQL от NoSQL",
            "кто придумал телефон",
            "самая высокая гора в мире",
            "расскажи про чёрные дыры",
        ] {
            assert!(looks_like_factual_question(phrase), "факт-вопрос: {phrase}");
        }
        for phrase in [
            "расскажи сказку про кота",
            "открой ютуб",
            "почему ты такой задумчивый",
            "что ты умеешь",
            "какая погода сегодня",
        ] {
            assert!(!looks_like_factual_question(phrase), "не факт-вопрос: {phrase}");
        }
    }

    #[test]
    fn article_text_is_cleaned() {
        let html = "<html><head><title>Тест</title><style>body{color:red}</style></head>\
<body><script>alert(1)</script><nav>меню</nav><article><h1>Заголовок</h1>\
<p>Первый &amp; второй абзац.</p><p>Второй абзац.</p></article><footer>подвал</footer></body></html>";
        let (title, text) = extract_article_text(html);
        assert_eq!(title, "Тест");
        assert!(text.contains("Первый & второй абзац"));
        assert!(text.contains("Второй абзац"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("меню"));
        assert!(!text.contains("подвал"));
        assert!(!text.contains("color:red"));
    }

    #[test]
    fn article_chunking_advances() {
        let text = "абзац один. ".repeat(200);
        {
            let mut article = ARTICLE.lock().unwrap_or_else(|e| e.into_inner());
            *article = Some(ArticleState {
                title: "Тест".into(),
                text: text.clone(),
                offset: 0,
            });
        }
        let first = next_article_chunk().expect("первый кусок");
        let second = next_article_chunk().expect("второй кусок");
        assert_ne!(first, second);
        assert!(first.chars().count() <= ARTICLE_CHUNK_CHARS + 200);
    }

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