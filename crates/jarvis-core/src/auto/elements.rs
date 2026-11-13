//! Единая модель элементов экрана и нечёткий матч.
//!
//! Снимок экрана — это не текст для красоты, а список элементов, по которым
//! агент действует. Модель видит элементы с номерами и говорит «кликни 12»,
//! а не угадывает точное имя: точное имя в интерфейсе живёт своей жизнью
//! («Сохранить» против «Сохранить…», «Новая вкладка» против «+»).
//!
//! Матч нечёткий и двуязычный: «кнопку настроек» должен найти элемент
//! «Настройки», «поле поиска» — «Поиск». Сравнение идёт по основе слова
//! (`ru_stem`) и по общему началу, потому что русские падежи дают разные
//! основы: «настроек» → «настроек», «настройки» → «настройк».
//!
//! Один и тот же матч работает для macOS (Accessibility) и Linux (AT-SPI):
//! платформенный слой отдаёт элементы, решение принимает здесь.

use crate::platform::ru_stem;

/// Элемент интерфейса, прочитанный из дерева доступности.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Element {
    /// Порядковый номер в обходе дерева. Внутри одного снимка стабилен и
    /// однозначен: по нему элемент находят после действия.
    pub id: usize,
    /// Нормализованная роль: `button`, `text field`, `link`, `menu bar item`…
    pub role: String,
    /// Роль в терминах ОС (`AXButton`, `push button`): по ней действие
    /// обращается к элементу напрямую, минуя перевод ролей.
    pub raw_role: String,
    /// Имя элемента — то, что видно пользователю.
    pub name: String,
    /// Описание (`description` в AX, `description` в AT-SPI): у GTK-полей
    /// часто только оно и содержит подпись.
    pub description: String,
    /// Текущее значение (текст в поле, выбранный пункт списка).
    pub value: String,
    /// Глубина в дереве — для отладки и отбраковки «мусорных» узлов.
    pub depth: u32,
    /// Номер окна (с единицы), в котором найден элемент.
    pub window: usize,
    /// Приложение, в котором найден элемент.
    pub app: String,
}

impl Element {
    /// Подпись элемента для показа: имя, иначе описание.
    pub fn label(&self) -> &str {
        if !self.name.trim().is_empty() {
            &self.name
        } else {
            &self.description
        }
    }

    /// Всё, чем элемент может назваться: имя, описание, значение.
    /// По любому из них элемент ищется.
    pub fn haystacks(&self) -> [String; 3] {
        [
            self.name.clone(),
            self.description.clone(),
            self.value.clone(),
        ]
    }
}

/// Снимок экрана: приложение, окно и плоский список элементов.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScreenSnapshot {
    pub app: String,
    pub window: String,
    pub elements: Vec<Element>,
}

impl ScreenSnapshot {
    pub fn new(app: impl Into<String>, window: impl Into<String>, elements: Vec<Element>) -> Self {
        Self {
            app: app.into(),
            window: window.into(),
            elements,
        }
    }

    /// Элемент по номеру из снимка.
    pub fn by_id(&self, id: usize) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    /// Текст для модели: элементы с номерами, чтобы кликать по номеру.
    ///
    /// Формат строки — `[12] кнопка «Настройки»`: номер, роль в родительном
    /// падеже не нужен, роль машинная и английская — так её понимает словарь
    /// ролей инструментов.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Экран: {}{}\n",
            if self.app.trim().is_empty() {
                "приложение неизвестно".to_string()
            } else {
                self.app.clone()
            },
            if self.window.trim().is_empty() {
                String::new()
            } else {
                format!(" — «{}»", self.window)
            }
        ));
        let mut by_role: Vec<(String, Vec<&Element>)> = Vec::new();
        for el in &self.elements {
            if el.label().trim().is_empty() {
                continue;
            }
            match by_role.iter_mut().find(|(r, _)| *r == el.role) {
                Some((_, list)) => list.push(el),
                None => by_role.push((el.role.clone(), vec![el])),
            }
        }
        if by_role.is_empty() {
            out.push_str("Элементов не видно: окно пустое или у приложения отключён доступ.\n");
            return out;
        }
        for (role, list) in by_role {
            let items: Vec<String> = list
                .iter()
                .map(|el| {
                    let mut line = format!("[{}] {role} «{}»", el.id, el.label());
                    if !el.value.trim().is_empty() {
                        line.push_str(&format!(" = «{}»", el.value));
                    }
                    line
                })
                .collect();
            out.push_str(&items.join("\n"));
            out.push('\n');
        }
        out
    }
}

// =========================================================================
// Нормализация и сравнение
// =========================================================================

/// Нормализация текста для сравнения: нижний регистр, пунктуация → пробелы,
/// схлопнутые пробелы.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            out.push(ch.to_lowercase().next().unwrap_or(ch));
            space = false;
        } else if ch == '\'' || ch == '\u{2019}' {
            // Апостроф внутри слова («пользовательский») не разрывает слово.
            continue;
        } else {
            if !space && !out.is_empty() {
                out.push(' ');
                space = true;
            }
        }
    }
    out.trim().to_string()
}

/// Слова текста в нормализованном виде.
pub fn tokens(text: &str) -> Vec<String> {
    normalize(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Основы слов — падежи сняты.
fn stems(text: &str) -> Vec<String> {
    tokens(text).iter().map(|w| ru_stem(w)).collect()
}

/// Совпадают ли два слова как «одно и то же слово в разных падежах».
///
/// Совпадение широкое намеренно: основа (`настройк`/`настроек` — разные
/// основы из-за «-ей/-ек»), общий префикс от пяти букв и вхождение длинного
/// слова в короткое. Без этого «кнопку настроек» не находит «Настройки».
pub fn word_match(a: &str, b: &str) -> bool {
    let a = normalize(a);
    let b = normalize(b);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    let (sa, sb) = (ru_stem(&a), ru_stem(&b));
    if sa == sb {
        return true;
    }
    // Общее начало: «настро» в «настроек» и «настройки».
    let common = sa
        .chars()
        .zip(sb.chars())
        .take_while(|(x, y)| x == y)
        .count();
    if common >= 5 {
        return true;
    }
    // Одно слово внутри другого: «поиск» в «поиске».
    let (short, long) = if a.chars().count() <= b.chars().count() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    short.chars().count() >= 4 && long.contains(short)
}

/// Слова, которые описывают роль элемента, а не его имя.
/// «найди кнопку настроек» → цель «настроек», а не «кнопку настроек».
const ROLE_WORDS: &[&str] = &[
    "кнопка",
    "кнопку",
    "кнопки",
    "кнопок",
    "кнопочка",
    "кнопочку",
    "поле",
    "поля",
    "поле",
    "полей",
    "полечку",
    "ссылка",
    "ссылку",
    "ссылки",
    "флажок",
    "флажка",
    "флажки",
    "галочка",
    "галочку",
    "переключатель",
    "переключателя",
    "меню",
    "пункт",
    "пункта",
    "пункты",
    "элемент",
    "элемента",
    "надпись",
    "надписи",
    "текст",
    "текста",
    "поле",
    "окно",
    "окна",
    "вкладка",
    "вкладку",
    "вкладки",
    "ячейка",
    "ячейку",
    "ползунок",
    "ползунка",
    "список",
    "списка",
    "выпадающий",
    "выпадашка",
];

/// Убрать из просьбы слова, описывающие роль: остаётся имя элемента.
///
/// «найди на экране кнопку настроек и нажми её» → «настроек».
pub fn strip_role_words(text: &str) -> String {
    let kept: Vec<String> = tokens(text)
        .into_iter()
        .filter(|w| !ROLE_WORDS.iter().any(|r| *r == w))
        .collect();
    kept.join(" ")
}

/// Насколько элемент подходит под просьбу: 0 — не подходит совсем.
pub fn score_element(el: &Element, want: &str) -> u32 {
    let want = normalize(want);
    if want.is_empty() {
        return 0;
    }
    let [name, description, value] = el.haystacks();
    let name_n = normalize(&name);
    let desc_n = normalize(&description);
    let value_n = normalize(&value);

    // Точное совпадение по имени — сильнее всего.
    if name_n == want {
        return 100;
    }
    if desc_n == want {
        return 96;
    }
    // Совпадение по основе слова: «настроек» ↔ «настройки».
    if word_match(&name, &want) || word_match(&description, &want) {
        return 88;
    }
    // Имя содержит просьбу: «Настройки системы» ⊃ «настроек».
    if name_n.contains(&want) {
        return 80;
    }
    if desc_n.contains(&want) {
        return 74;
    }
    if value_n.contains(&want) {
        return 50;
    }
    // Совпадение по словам: просьба из нескольких слов, элемент содержит их все.
    let want_tokens = tokens(&want);
    if want_tokens.len() > 1 {
        let el_stems = stems(&name);
        let el_stems_desc = stems(&description);
        let hits = want_tokens
            .iter()
            .filter(|w| {
                el_stems.iter().any(|s| word_match(s, w))
                    || el_stems_desc.iter().any(|s| word_match(s, w))
            })
            .count();
        if hits == want_tokens.len() {
            return 66;
        }
        if hits > 0 {
            return (hits * 40 / want_tokens.len()) as u32;
        }
    }
    0
}

/// Найденный элемент с оценкой совпадения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate<'a> {
    pub element: &'a Element,
    pub score: u32,
}

/// Кандидаты по убыванию совпадения.
///
/// Роль — жёсткий фильтр: просили кнопку, а поле — не подходит. Но если роль
/// не задана, берём любую, и оценка решает.
pub fn match_elements<'a>(
    elements: &'a [Element],
    want: &str,
    role: Option<&str>,
) -> Vec<Candidate<'a>> {
    let mut out: Vec<Candidate<'a>> = elements
        .iter()
        .filter_map(|el| {
            if let Some(r) = role
                && !el.role.eq_ignore_ascii_case(r)
            {
                return None;
            }
            let score = score_element(el, want);
            (score > 0).then_some(Candidate { element: el, score })
        })
        .collect();
    // При равной оценке — мельче и ближе к началу дерева: верхние элементы
    // обычно и есть те, что человек видит первыми.
    out.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then(a.element.depth.cmp(&b.element.depth))
            .then(a.element.id.cmp(&b.element.id))
    });
    out
}

/// Лучший кандидат: первый, у кого оценка хотя бы минимальная.
pub fn best_match<'a>(
    elements: &'a [Element],
    want: &str,
    role: Option<&str>,
) -> Option<Candidate<'a>> {
    match_elements(elements, want, role)
        .into_iter()
        .next()
        .filter(|c| c.score >= 50)
}

/// Список кандидатов для ответа пользователю, когда элемент не найден
/// однозначно: «похоже, ты имел в виду вот это».
pub fn candidate_names(elements: &[Element], want: &str, role: Option<&str>, limit: usize) -> Vec<String> {
    match_elements(elements, want, role)
        .into_iter()
        .take(limit)
        .map(|c| format!("{} «{}»", c.element.role, c.element.label()))
        .collect()
}

/// Совпадает ли элемент по «адресу» в дереве: приложение, окно, роль ОС,
/// имя, описание. Одинаковые адреса бывают у дубликатов (две кнопки «Да») —
/// тогда различает номер повтора.
pub fn same_address(a: &Element, b: &Element) -> bool {
    a.app == b.app
        && a.window == b.window
        && a.raw_role == b.raw_role
        && a.name == b.name
        && a.description == b.description
}

/// Номер повтора среди таких же элементов: 0 — первый.
///
/// Нужен, чтобы после снимка экрана найти *тот самый* элемент: у дерева нет
/// устойчивых идентификаторов, но есть порядок обхода — он и есть адрес.
pub fn occurrence_of(elements: &[Element], target: &Element) -> u32 {
    elements
        .iter()
        .filter(|e| same_address(e, target))
        .position(|e| e.id == target.id)
        .unwrap_or(0) as u32
}

/// Элементы с одинаковым адресом — сколько их всего.
pub fn duplicate_count(elements: &[Element], target: &Element) -> usize {
    elements.iter().filter(|e| same_address(e, target)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(id: usize, role: &str, name: &str, description: &str) -> Element {
        Element {
            id,
            role: role.into(),
            raw_role: role.into(),
            name: name.into(),
            description: description.into(),
            value: String::new(),
            depth: 1,
            window: 1,
            app: "Test".into(),
        }
    }

    #[test]
    fn normalizes_case_and_punctuation() {
        assert_eq!(normalize("  Настройки  "), "настройки");
        assert_eq!(normalize("Найти (Сброс)!"), "найти сброс");
        assert_eq!(normalize("пользовательский"), "пользовательский");
    }

    #[test]
    fn matches_russian_case_forms() {
        // Главная боль: «настроек» и «настройки» — разные основы.
        assert!(word_match("настроек", "Настройки"));
        assert!(word_match("кнопку", "кнопка"));
        assert!(word_match("Поиск", "поиска"));
        assert!(!word_match("настроек", "поиск"));
    }

    #[test]
    fn exact_name_beats_partial() {
        let elements = vec![
            el(1, "button", "Настройки системы", ""),
            el(2, "button", "Настройки", ""),
        ];
        let best = best_match(&elements, "настройки", Some("button")).unwrap();
        assert_eq!(best.element.id, 2);
        assert!(best.score >= 90);
    }

    #[test]
    fn matches_by_description_when_name_is_empty() {
        let elements = vec![el(3, "text field", "", "Поиск по сайту")];
        let best = best_match(&elements, "поиск", Some("text field")).unwrap();
        assert_eq!(best.element.id, 3);
    }

    #[test]
    fn role_filters_out_other_kinds() {
        let elements = vec![
            el(1, "text field", "Настройки", ""),
            el(2, "button", "Настройки", ""),
        ];
        let candidates = match_elements(&elements, "настройки", Some("button"));
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].element.id, 2);
    }

    #[test]
    fn multi_word_request_matches_element() {
        let elements = vec![el(1, "button", "Открыть файл", "")];
        let best = best_match(&elements, "открыть файл", Some("button")).unwrap();
        assert_eq!(best.element.id, 1);
    }

    #[test]
    fn strips_role_words_from_request() {
        assert_eq!(strip_role_words("кнопку настроек"), "настроек");
        assert_eq!(strip_role_words("поле поиска"), "поиска");
        assert_eq!(strip_role_words("настройки"), "настройки");
    }

    #[test]
    fn render_lists_ids_for_the_model() {
        let snapshot = ScreenSnapshot::new(
            "Safari",
            "Яндекс Диск",
            vec![
                el(1, "button", "Настройки", ""),
                el(2, "text field", "Поиск", ""),
            ],
        );
        let text = snapshot.render();
        assert!(text.contains("[1] button «Настройки»"));
        assert!(text.contains("[2] text field «Поиск»"));
        assert!(text.contains("Safari"));
    }

    #[test]
    fn empty_snapshot_says_so_instead_of_silence() {
        let snapshot = ScreenSnapshot::new("Safari", "", vec![]);
        assert!(snapshot.render().contains("Элементов не видно"));
    }

    #[test]
    fn by_id_finds_element() {
        let snapshot = ScreenSnapshot::new("App", "Win", vec![el(7, "button", "Go", "")]);
        assert_eq!(snapshot.by_id(7).map(|e| e.name.as_str()), Some("Go"));
        assert!(snapshot.by_id(8).is_none());
    }
}
