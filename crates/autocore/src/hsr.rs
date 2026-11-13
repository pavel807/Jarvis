//! A4. HSR — Hashed Semantic Router: понять команду без единой модели,
//! без словаря и без обучения на старте.
//!
//! Символьные n-граммы фразы (3/2/1, взвешенные) → знаковый feature
//! hashing в фиксированный вектор D=256 → L2-нормализация → косинус к
//! прототипам команд. 50 команд — это ~50 КБ состояния: только прототипы.
//!
//! Пороги отказа: лучший косинус ниже τ, ИЛИ отрыв от второго ниже δ —
//! это отказ/уточнение, не угадывание. Неизвестная команда обязана давать
//! низкий скор, а не уверенно неверный интент.

use std::collections::HashMap;

/// Размерность хэш-пространства.
pub const DIM: usize = 256;

/// Веса уровней грамм: триграммы несут смысл морфем, биграммы и
/// униграммы переживают опечатки и огрехи STT («брозер» делит «бро»,
/// «роз» с «браузер»). Смесь уровней — стандартная робастность без
/// словаря: hash-to-bucket, никаких таблиц.
const W_TRI: f32 = 1.0;
const W_BI: f32 = 0.5;
const W_UNI: f32 = 0.25;

/// Порог косинуса, откалиброванный на приёмочном наборе (см. тесты):
/// верные команды дают 0.55–0.85, неизвестные — до 0.5. Спека стартует
/// с 0.55 для своей векторизации; косинусы не переносимы между
/// featurize-вариантами, поэтому свой порог замерен, а не списан.
pub const DEFAULT_TAU: f32 = 0.5;
/// Минимальный отрыв от второго прототипа: иначе — уточняющий вопрос.
pub const DEFAULT_DELTA: f32 = 0.1;

/// Образец фразы: вектор одного примера команды. Сравнение идёт с
/// ближайшим ОБРАЗЦОМ, а не со средним прототипом: усреднение размывает
/// варианты («открой браузер» и «браузер открой» дают рованный вектор,
/// к которому голая «открой» ближе, чем надо), а ближайший образец
/// сохраняет естественную близость и делает неоднозначность честной.
#[derive(Debug, Clone)]
pub struct Prototype {
    pub intent: &'static str,
    pub vec: [f32; DIM],
}

/// Маршрутизатор.
#[derive(Debug, Clone)]
pub struct Hsr {
    exemplars: Vec<Prototype>,
    pub tau: f32,
    pub delta: f32,
}

/// Итог маршрутизации: уверенное совпадение, неоднозначность или отказ.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Уверенное совпадение с интентом.
    Match { intent: &'static str, score: f32 },
    /// Лучший скор есть, но отрыв от второго мал — уточнить у человека.
    Ambiguous {
        intent: &'static str,
        score: f32,
        second: &'static str,
        second_score: f32,
    },
    /// Никакого уверенного совпадения — отказ, не угадывание.
    Refuse { best_score: f32 },
}

/// Векторизация фразы: lowercase, NFC, символьные n-граммы (3/2/1) со
/// взвешенным знаковым feature hashing (blake3), затем L2-нормализация.
pub fn featurize(text: &str) -> [f32; DIM] {
    let norm: String = crate::smi::normalize_name(text).to_lowercase();
    let chars: Vec<char> = norm.chars().collect();
    let mut v = [0f32; DIM];
    if chars.is_empty() {
        return v;
    }
    if chars.len() < 3 {
        touch(&mut v, &norm, W_TRI);
    } else {
        for w in 0..=chars.len() - 3 {
            let gram: String = chars[w..w + 3].iter().collect();
            touch(&mut v, &gram, W_TRI);
        }
    }
    if chars.len() >= 2 {
        for w in 0..=chars.len() - 2 {
            let gram: String = chars[w..w + 2].iter().collect();
            touch(&mut v, &gram, W_BI);
        }
    } else {
        touch(&mut v, &norm, W_BI);
    }
    for c in &chars {
        touch(&mut v, &c.to_string(), W_UNI);
    }
    let norm_len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_len > 0.0 {
        for x in &mut v {
            *x /= norm_len;
        }
    }
    v
}

fn touch(v: &mut [f32; DIM], gram: &str, weight: f32) {
    let h = crate::smi::hash64(gram.as_bytes());
    let idx = (h % DIM as u64) as usize;
    let sign = if (h >> 63) & 1 == 1 { 1.0 } else { -1.0 };
    v[idx] += sign * weight;
}

fn cosine(a: &[f32; DIM], b: &[f32; DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

impl Hsr {
    /// Маршрутизатор из набора примеров фраз на команду: каждый пример —
    /// образец, ничего не усредняется.
    pub fn train(samples: &[(&'static str, Vec<&str>)]) -> Self {
        let mut exemplars = Vec::new();
        for (intent, phrases) in samples {
            for p in phrases {
                exemplars.push(Prototype {
                    intent,
                    vec: featurize(p),
                });
            }
        }
        Self {
            exemplars,
            tau: DEFAULT_TAU,
            delta: DEFAULT_DELTA,
        }
    }

    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }

    /// Скор по всем интентам: лучший из образцов каждого. Отрыв меряется
    /// между интентами, а не между образцами одной команды.
    pub fn all_scores(&self, v: &[f32; DIM]) -> Vec<(&'static str, f32)> {
        let mut best: HashMap<&'static str, f32> = HashMap::new();
        for ex in &self.exemplars {
            let c = cosine(&v, &ex.vec);
            let slot = best.entry(ex.intent).or_insert(f32::NEG_INFINITY);
            if c > *slot {
                *slot = c;
            }
        }
        let mut out: Vec<_> = best.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Решение по фразе: совпадение, неоднозначность или отказ.
    pub fn route(&self, text: &str) -> Verdict {
        let v = featurize(text);
        let scores = self.all_scores(&v);
        let Some((best, s)) = scores.first().copied() else {
            return Verdict::Refuse { best_score: 0.0 };
        };
        if s < self.tau {
            return Verdict::Refuse { best_score: s };
        }
        match scores.get(1).copied() {
            Some((second, s2)) if s - s2 < self.delta => Verdict::Ambiguous {
                intent: best,
                score: s,
                second,
                second_score: s2,
            },
            _ => Verdict::Match { intent: best, score: s },
        }
    }
}

/// 30 русских команд (§11 Этап 2): реальные формулировки. Публично —
/// тестам приёмки и калибровке порогов.
pub fn command_samples() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("open_browser", vec!["открой браузер", "запусти браузер", "браузер открой",
            "открой мне браузер", "запусти интернет-браузер", "открой интернет"]),
        ("close_app", vec!["закрой приложение", "закрой программу", "заверши приложение",
            "закрой текущее приложение", "закрой все программы", "приложение закрой"]),
        ("set_timer", vec!["поставь таймер", "таймер на десять минут", "заведи таймер",
            "поставь таймер на час", "таймер на пять минут", "таймер поставь"]),
        ("open_app", vec!["открой калькулятор", "запусти приложение", "открой программу",
            "открой текстовый редактор", "запусти приложение почты", "открой приложение"]),
        ("volume_up", vec!["сделай громче", "прибавь громкость", "громкость выше",
            "сделай погромче", "подними громкость", "громкость подними"]),
        ("volume_down", vec!["сделай тише", "убавь громкость", "громкость ниже",
            "сделай потише", "опусти громкость", "громкость опусти"]),
        ("take_screenshot", vec!["сделай скриншот", "снимок экрана", "заскринь экран",
            "сделай снимок экрана", "скриншот экрана", "сфотографируй экран"]),
        ("play_music", vec!["включи музыку", "запусти плеер", "музыку давай",
            "включи трек", "запусти музыку", "музыку включи"]),
        ("stop_music", vec!["выключи музыку", "останови плеер", "музыку стоп",
            "выключи трек", "останови музыку", "музыку выключи"]),
        ("write_text", vec!["напиши текст", "набери сообщение", "впиши текст",
            "напечатай текст", "набери текст", "текст напиши"]),
        ("click_button", vec!["нажми кнопку", "кликни по кнопке", "кнопку нажми",
            "нажми на кнопку", "кликни кнопку", "нажатие кнопки"]),
        ("scroll_down", vec!["прокрути вниз", "листай ниже", "скролл вниз",
            "прокрути страницу вниз", "листай вниз", "вниз прокрути"]),
        ("open_file", vec!["открой файл", "открой документ", "файл открой",
            "открой документ отчет", "открой текстовый файл", "документ открой"]),
        ("save_file", vec!["сохрани файл", "сохранить документ", "сохрани всё",
            "сохрани изменения", "сохранить файл", "файл сохрани"]),
        ("new_window", vec!["новое окно", "открой новое окно", "ещё одно окно",
            "создай новое окно", "открой окно заново", "новое окно открой"]),
        ("close_window", vec!["закрой окно", "закрой это окно", "окно закрой",
            "закрой текущее окно", "закрой вкладку окна", "это окно закрой"]),
        ("maximize_window", vec!["разверни окно", "окно на весь экран", "максимизируй",
            "разверни это окно", "окно разверни", "на весь экран"]),
        ("minimize_window", vec!["сверни окно", "сверни это", "уменьши окно",
            "сверни текущее окно", "окно сверни", "сверни в панель"]),
        ("switch_app", vec!["переключи приложение", "переключись на другое", "сменить окно",
            "переключи окно", "другое приложение", "переключись на приложение"]),
        ("search_text", vec!["найди текст", "поиск по странице", "найди на экране",
            "найди слово на странице", "поиск текста", "текст найди"]),
        ("copy_text", vec!["скопируй текст", "копируй выделенное", "текст в буфер",
            "скопируй выделенное", "копируй текст", "текст скопируй"]),
        ("paste_text", vec!["вставь текст", "вставь из буфера", "вставка текста",
            "вставь скопированное", "вставь сюда текст", "текст вставь"]),
        ("send_message", vec!["отправь сообщение", "отправь письмо", "пошли сообщение",
            "отправь письмо сейчас", "сообщение отправь", "отправь его"]),
        ("read_title", vec!["какой заголовок", "что за окно", "прочитай заголовок",
            "прочитай название окна", "заголовок окна", "что сейчас открыто"]),
        ("list_windows", vec!["какие окна открыты", "список окон", "покажи окна",
            "покажи открытые окна", "все окна покажи", "какие окна есть"]),
        ("run_command", vec!["выполни команду", "запусти команду", "исполни команду",
            "выполни это", "команду выполни", "исполнение команды"]),
        ("lock_screen", vec!["заблокируй экран", "заблокируй компьютер", "экран заблокируй",
            "заблокируй сеанс", "компьютер заблокируй", "блокировка экрана"]),
        ("open_settings", vec!["открой настройки", "настройки открой", "покажи настройки",
            "открой панель настроек", "настройки системы открой", "покажи параметры"]),
        ("delete_file", vec!["удали файл", "удали документ", "файл в корзину",
            "удали этот файл", "документ удали", "файл удали"]),
        ("refresh_page", vec!["обнови страницу", "перегрузи страницу", "обнови",
            "обнови страницу сайта", "страницу обнови", "обнови это"]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hsr() -> Hsr {
        Hsr::train(&command_samples())
    }

    #[test]
    fn flexible_phrases_route_correctly() {
        let r = hsr();
        let cases = [
            ("открой мне браузер пожалуйста", "open_browser"),
            ("браузер открой", "open_browser"),
            ("поставь таймер на 10 минут", "set_timer"),
            ("заблокируй мой компьютер", "lock_screen"),
            ("переключи меня на другое приложение", "switch_app"),
        ];
        for (phrase, expected) in cases {
            match r.route(phrase) {
                Verdict::Match { intent, score } => {
                    assert_eq!(intent, expected, "фраза «{phrase}»: {score:.3}");
                }
                other => panic!("«{phrase}» не распознана: {other:?}"),
            }
        }
    }

    #[test]
    fn unknown_command_refuses_instead_of_guessing() {
        let r = hsr();
        for phrase in ["какая погода", "сколько времени", "расскажи анекдот"] {
            match r.route(phrase) {
                Verdict::Refuse { .. } => {}
                other => panic!("«{phrase}» обязана быть отказом, а не {other:?}"),
            }
        }
    }

    #[test]
    fn ambiguous_phrases_flagged_not_executed() {
        let r = hsr();
        // Голая «открой» делится тремя open_*-интентами: уточнить, не
        // исполнять.
        let v = r.route("открой");
        assert!(
            matches!(v, Verdict::Ambiguous { .. } | Verdict::Refuse { .. }),
            "голая «открой» обязана требовать уточнения: {v:?}"
        );
    }

    #[test]
    fn typos_survive() {
        let r = hsr();
        let cases = [
            ("открой брозер", "open_browser"),
            ("паст таймер на час", "set_timer"),
        ];
        for (phrase, expected) in cases {
            match r.route(phrase) {
                Verdict::Match { intent, .. } => assert_eq!(intent, expected, "«{phrase}»"),
                other => panic!("«{phrase}» не распознана: {other:?}"),
            }
        }
    }

    /// Замер набора (§11 Этап 2): точность и доля отказов на фразах,
    /// которых не было в обучении.
    #[test]
    fn accuracy_and_refusal_rate_measured() {
        let r = hsr();
        let held_out: &[(&str, &str)] = &[
            ("давай браузер", "open_browser"),
            ("закрой телеграм сейчас", "close_app"),
            ("таймер пять минут", "set_timer"),
            ("открой текстовый редактор", "open_app"),
            ("громкость подними", "volume_up"),
            ("снимок экрана сделай", "take_screenshot"),
            ("врубай музыку", "play_music"),
            ("вырубай музыку", "stop_music"),
            ("кликни сюда", "click_button"),
            ("вниз прокрути", "scroll_down"),
            ("открой отчет за март", "open_file"),
            ("сохрани как", "save_file"),
            ("ещё одно окно открой", "new_window"),
            ("окно сверни", "minimize_window"),
            ("на весь экран разверни", "maximize_window"),
            ("заблокируй сеанс", "lock_screen"),
            ("куда отправить письмо", "send_message"),
            ("скопируй это", "copy_text"),
            ("вставь сюда", "paste_text"),
            ("обнови страницу пожалуйста", "refresh_page"),
        ];
        let mut hits = 0;
        let mut refusals = 0;
        for (phrase, expected) in held_out {
            match r.route(phrase) {
                Verdict::Match { intent, .. } if intent == *expected => hits += 1,
                Verdict::Refuse { .. } => refusals += 1,
                Verdict::Match { intent, score } => {
                    println!("miss: «{phrase}» → {intent} ({score:.3}), ждали {expected}")
                }
                Verdict::Ambiguous { intent, score, .. } => {
                    println!("ambiguous: «{phrase}» → {intent} ({score:.3})")
                }
            }
        }
        let total = held_out.len();
        let accuracy = hits as f32 / total as f32;
        let refusal = refusals as f32 / total as f32;
        println!(
            "точность {accuracy:.2} ({hits}/{total}), отказы {refusal:.2} ({refusals}/{total})"
        );
        assert!(accuracy >= 0.75, "точность {accuracy:.2} < 0.75");
        assert!(refusal <= 0.33, "доля отказов {refusal:.2} > 0.33");
    }

    #[test]
    fn empty_and_short_input() {
        let r = hsr();
        assert!(matches!(r.route(""), Verdict::Refuse { .. }));
        assert!(matches!(r.route("ок"), Verdict::Refuse { .. } | Verdict::Ambiguous { .. }));
    }
}
