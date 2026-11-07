//! Погода: детерминированный ответ без нейросети.
//!
//! Источник — открытый API MET Norway (locationforecast 2.0, без ключа,
//! доступен из РФ, требует User-Agent); координаты города — встроенный
//! словарь крупных городов, для остальных — Nominatim (OpenStreetMap).
//!
//! Почему не поиск: DuckDuckGo отдаёт анти-бот заглушку (202 без результатов),
//! а после неё 2B-модель отвечала отсебятиной («Завтра будет ноябрь»).
//! Вопрос «какая погода» — это данные, а не рассуждение: запрос → ответ.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::time::Duration;

/// Агент для погодных запросов: короткие таймауты, честный User-Agent
/// (требование API MET Norway).
fn agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_millis(1500)))
            .timeout_connect(Some(Duration::from_millis(2500)))
            .timeout_global(Some(Duration::from_secs(6)))
            .build()
            .new_agent()
    })
}

const USER_AGENT: &str = "JarvisAssistant/0.2 (github.com/pavel807/Jarvis)";

/// Погода сейчас: то, что произносится вслух.
#[derive(Debug, Clone, PartialEq)]
pub struct Weather {
    /// Отображаемое имя города (в именительном падеже).
    pub city: String,
    /// Температура, °C.
    pub temperature: f64,
    /// Скорость ветра, м/с.
    pub wind: f64,
    /// Описание («облачно», «небольшой снег»…).
    pub description: String,
}

/// Текущая погода в городе. `city` — слово из фразы пользователя
/// (в любом падеже) или значение `weather_city` из настроек.
pub fn fetch(city: &str) -> Result<Weather> {
    let (name, lat, lon) = locate(city)?;
    let url = format!("https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat:.4}&lon={lon:.4}");
    let response = agent()
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| anyhow!("сервис погоды недоступен: {e}"))?;
    let text = response
        .into_body()
        .read_to_string()
        .context("пустой ответ сервиса погоды")?;
    let data: Value = serde_json::from_str(&text).context("странный ответ сервиса погоды")?;
    let now = data
        .pointer("/properties/timeseries/0")
        .ok_or_else(|| anyhow!("в ответе погоды нет текущих данных"))?;

    let temperature = now
        .pointer("/data/instant/details/air_temperature")
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("в ответе погоды нет температуры"))?;
    let wind = now
        .pointer("/data/instant/details/wind_speed")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let symbol = now
        .pointer("/data/next_1_hours/summary/symbol_code")
        .and_then(Value::as_str)
        .unwrap_or("");

    Ok(Weather {
        city: name,
        temperature,
        wind,
        description: describe_symbol(symbol),
    })
}

/// Ответ для озвучки: короткий, числа словами. Город — в именительном
/// падеже в начале фразы: склонение имени из геокодера ненадёжно.
pub fn speak_answer(weather: &Weather) -> String {
    format!(
        "Сейчас {} — {}, температура {}, ветер {} в секунду.",
        weather.city,
        weather.description,
        degrees(weather.temperature),
        wind_words(weather.wind),
    )
}

/// Температура словами: «минус 3 градуса» / «плюс 11 градусов».
fn degrees(t: f64) -> String {
    let rounded = t.round() as i64;
    let sign = if rounded < 0 { "минус" } else { "плюс" };
    format!("{sign} {} {}", rounded.abs(), plural(rounded.abs(), ("градус", "градуса", "градусов")))
}

/// Ветер словами: «3 метра в секунду» (последнее слово подставит вызывающий).
fn wind_words(w: f64) -> String {
    let rounded = w.round().max(0.0) as i64;
    format!("{} {}", rounded, plural(rounded, ("метр", "метра", "метров")))
}

/// Русское склонение: 1 градус, 2 градуса, 5 градусов (11–14 — градусов).
fn plural(n: i64, forms: (&str, &str, &str)) -> String {
    let last10 = n % 10;
    let last100 = n % 100;
    if last10 == 1 && last100 != 11 {
        forms.0.to_string()
    } else if (2..=4).contains(&last10) && !(12..=14).contains(&last100) {
        forms.1.to_string()
    } else {
        forms.2.to_string()
    }
}

/// Код погоды MET Norway → человеческое описание.
/// Суффиксы _day/_night/_polartwist отбрасываются, «andthunder» добавляет грозу.
fn describe_symbol(symbol: &str) -> String {
    let base = symbol
        .trim()
        .trim_end_matches("_polartwist")
        .trim_end_matches("_day")
        .trim_end_matches("_night");
    let thunder = base.contains("thunder");
    let base = base.replace("andthunder", "");
    let mut description: String = match base.as_str() {
        "clearsky" => "ясно".into(),
        "fair" => "ясно".into(),
        "partlycloudy" => "переменная облачность".into(),
        "cloudy" => "облачно".into(),
        "rain" | "rainshowers" => "дождь".into(),
        "lightrain" | "lightrainshowers" => "небольшой дождь".into(),
        "heavyrain" | "heavyrainshowers" => "сильный дождь".into(),
        "snow" | "snowshowers" => "снег".into(),
        "lightsnow" | "lightsnowshowers" => "небольшой снег".into(),
        "heavysnow" | "heavysnowshowers" => "сильный снег".into(),
        "sleet" | "sleetshowers" => "мокрый снег".into(),
        "lightsleet" | "lightsleetshowers" => "небольшой мокрый снег".into(),
        "heavysleet" | "heavysleetshowers" => "сильный мокрый снег".into(),
        // Составные коды MET Norway: дождь со снегом и их интенсивности.
        "rainsnow" | "rainsnowshowers" => "мокрый снег".into(),
        "lightrainsnow" | "lightrainsnowshowers" => "небольшой мокрый снег".into(),
        "heavyrainsnow" | "heavyrainsnowshowers" => "сильный мокрый снег".into(),
        "fog" => "туман".into(),
        "" => "данные о погоде неполные".into(),
        other => return other.to_string(),
    };
    if thunder {
        description.push_str(" с грозой");
    }
    description
}

/// Это вопрос про погоду?
pub fn is_weather_question(text: &str) -> bool {
    text.to_lowercase().contains("погод")
}

/// Слова-не-города: «на улице» значит «снаружи», а не город.
const NOT_A_CITY: &[&str] = &[
    "улице", "улицы", "дворе", "доме", "дома", "квартире", "комнате",
    "офисе", "интернете", "экране", "сайте", "карте", "окне", "балу",
    "работе", "кухне", "мире",
];

/// Вытащить город из фразы: слово после «в/во/на» («в Москве», «на улице»).
/// `None` — города в фразе нет (используется город по умолчанию).
pub fn extract_city(text: &str) -> Option<String> {
    let t = text.to_lowercase();
    let words: Vec<&str> = t
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    let prepositions = ["в", "во", "на"];
    for (i, word) in words.iter().enumerate() {
        if !prepositions.contains(word) || i + 1 >= words.len() {
            continue;
        }
        // «в городе Москве» — берём слово после «городе».
        let candidate = if words[i + 1] == "городе" && i + 2 < words.len() {
            Some(words[i + 2])
        } else {
            Some(words[i + 1])
        };
        if let Some(city) = candidate {
            // «на улице в Москве»: первое слово-не-город не отменяет поиск —
            // город может быть за следующим предлогом.
            if NOT_A_CITY.contains(&city) || city.len() < 3 {
                continue;
            }
            return Some(city.to_string());
        }
    }
    None
}

/// Словарь крупных городов: формы в косвенных падежах → (имя, широта, долгота).
const CITIES: &[(&str, &str, f64, f64)] = &[
    ("москве", "Москва", 55.7558, 37.6173),
    ("москву", "Москва", 55.7558, 37.6173),
    ("питере", "Санкт-Петербург", 59.9398, 30.3146),
    ("петербурге", "Санкт-Петербург", 59.9398, 30.3146),
    ("спб", "Санкт-Петербург", 59.9398, 30.3146),
    ("новгороде", "Нижний Новгород", 56.3269, 44.0059),
    ("новосибирске", "Новосибирск", 55.0084, 82.9357),
    ("екатеринбурге", "Екатеринбург", 56.8389, 60.6057),
    ("казани", "Казань", 55.7963, 49.1088),
    ("сочи", "Сочи", 43.5855, 39.7231),
    ("самаре", "Самара", 53.1959, 50.1002),
    ("уфе", "Уфа", 54.7388, 55.9721),
    ("краснодаре", "Краснодар", 45.0355, 38.9753),
    ("воронеже", "Воронеж", 51.6720, 39.1843),
    ("перми", "Пермь", 58.0105, 56.2502),
    ("владивостоке", "Владивосток", 43.1155, 131.8855),
    ("омске", "Омск", 54.9885, 73.3242),
    ("челябинске", "Челябинск", 55.1644, 61.4368),
    ("ростове", "Ростов-на-Дону", 47.2357, 39.7015),
    ("тюмени", "Тюмень", 57.1522, 65.5272),
    ("иркутске", "Иркутск", 52.2871, 104.3050),
    ("волгограде", "Волгоград", 48.7080, 44.5133),
    ("саратове", "Саратов", 51.5336, 46.0343),
    ("тольятти", "Тольятти", 53.5303, 49.3461),
];

/// Координаты города: словарь форм → Nominatim (OpenStreetMap).
fn locate(city: &str) -> Result<(String, f64, f64)> {
    let key = city.trim().to_lowercase();
    if let Some((_, name, lat, lon)) = CITIES.iter().find(|(form, ..)| *form == key) {
        return Ok((name.to_string(), *lat, *lon));
    }
    // Словарь не знает города — геокодер. Nominatim понимает косвенные
    // падежи неточно, поэтому фильтруем результаты по городским типам.
    let url = format!(
        "https://nominatim.openstreetmap.org/search?q={}&format=json&limit=5&accept-language=ru",
        urlencoding::encode(city)
    );
    let response = agent()
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| anyhow!("геокодер недоступен: {e}"))?;
    let text = response
        .into_body()
        .read_to_string()
        .context("пустой ответ геокодера")?;
    let results: Vec<Value> = serde_json::from_str(&text).context("странный ответ геокодера")?;
    let best = results
        .iter()
        .find(|r| {
            matches!(
                r["addresstype"].as_str(),
                Some("city") | Some("town") | Some("village") | Some("municipality")
            )
        })
        .or_else(|| results.first());
    let best = best.ok_or_else(|| anyhow!("город «{city}» не найден"))?;
    let lat = best["lat"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("геокодер не дал координат"))?;
    let lon = best["lon"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("геокодер не дал координат"))?;
    // Имя — из display_name до первой запятой, из косвенного падежа
    // восстанавливать не пытаемся: «в Казани» → OSM сам отдаст имя.
    let name = best["display_name"]
        .as_str()
        .and_then(|s| s.split(',').next())
        .unwrap_or(city)
        .to_string();
    Ok((name, lat, lon))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_question_detection() {
        assert!(is_weather_question("какая сейчас погода на улице в Москве"));
        assert!(is_weather_question("Погода в Сочи"));
        assert!(!is_weather_question("открой сафари"));
    }

    #[test]
    fn city_extraction_from_phrase() {
        assert_eq!(
            extract_city("Какая сейчас погода на улице в Москве?"),
            Some("москве".into())
        );
        assert_eq!(
            extract_city("погода в городе Сочи"),
            Some("сочи".into())
        );
        // «на улице» — снаружи, не город: берём город по умолчанию.
        assert_eq!(extract_city("какая погода на улице"), None);
        assert_eq!(extract_city("погода"), None);
    }

    #[test]
    fn symbols_translate_to_speech() {
        assert_eq!(describe_symbol("cloudy"), "облачно");
        assert_eq!(describe_symbol("partlycloudy_day"), "переменная облачность");
        assert_eq!(describe_symbol("lightrainsnowshowers_night"), "небольшой мокрый снег");
        assert_eq!(describe_symbol("rainshowersandthunder_day"), "дождь с грозой");
        assert_eq!(describe_symbol("clearsky"), "ясно");
    }

    #[test]
    fn degrees_and_wind_speak_naturally() {
        assert_eq!(degrees(-1.4), "минус 1 градус");
        assert_eq!(degrees(3.0), "плюс 3 градуса");
        assert_eq!(degrees(11.0), "плюс 11 градусов");
        assert_eq!(wind_words(3.7), "4 метра");
        assert_eq!(wind_words(5.1), "5 метров");
        assert_eq!(wind_words(1.2), "1 метр");
    }

    #[test]
    fn known_city_resolves_without_network() {
        // Словарь форм покрывает крупные города без обращения к геокодеру.
        let (name, lat, lon) = locate("москве").unwrap();
        assert_eq!(name, "Москва");
        assert!((lat - 55.75).abs() < 0.1);
        assert!((lon - 37.61).abs() < 0.1);
        let (name, ..) = locate("питере").unwrap();
        assert_eq!(name, "Санкт-Петербург");
    }

    #[test]
    fn answer_is_short_and_spoken() {
        let w = Weather {
            city: "Москва".into(),
            temperature: -2.6,
            wind: 3.7,
            description: "облачно".into(),
        };
        let speech = speak_answer(&w);
        assert_eq!(speech, "Сейчас Москва — облачно, температура минус 3 градуса, ветер 4 метра в секунду.");
    }
}
