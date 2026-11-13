//! Локальный роутер: решает "это действие или разговор" и добавляет
//! подсказки. Не ограничивает набор инструментов — LLM сама выбирает
//! между специализированными и универсальным shell.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent {
    /// Пользователь просит что-то сделать (открыть, запустить, выполнить).
    Action,
    /// Обычный вопрос или болтовня.
    Chat,
}

/// Действие с указанием на конкретный объект на машине пользователя.
const UNAMBIGUOUS: &[&str] = &[
    "открой",
    "открыть",
    "запусти",
    "запустить",
    "включи",
    "включить",
    "выключи",
    "выключить",
    "закрой",
    "закрыть",
    "останови",
    "поставь",
    "поставить",
    "установи",
    "заведи",
    "завести",
    "вруби",
    "удали",
    "удалить",
    "перемести",
    "переименуй",
    "сохрани",
    "сохранить",
    "загрузи",
    "выполни",
    "выполнить",
    "терминал",
    "shell",
    "скрипт",
    "сверни",
    "разверни",
    "заблокируй",
    "усыпи",
    "перезагрузи",
    "скриншот",
    "приглуши",
    "переключись",
    "переключи",
    "отправь",
    "сделай громче",
    "сделай тише",
    "сделай ярче",
    "сделай темнее",
    // Громкость/яркость бывают сформулированы через «убавь/прибавь»: на
    // живом стенде «убавь громкость на двадцать процентов» уходило в
    // разговорную ветку, и модель отказывалась («не имею доступа»).
    "убавь",
    "убавить",
    "прибавь",
    "прибавить",
    "понизь",
    "повысь",
    "поставь таймер",
    "поставь будильник",
    "заведи таймер",
    "заведи будильник",
    "добавь задачу",
    "выполни задачу",
    "что в задачнике",
    "почисти",
    "запомни",
    "прочитай буфер",
    "скопируй в буфер",
    "нажми",
    "нажать",
    "введи",
    "ввести",
    "напечатай",
    "напечатать",
    "набери",
    "вставь",
    "вставить",
    "выдели",
    "напиши текст",
    "кликни",
    "прокрути",
    "выключи компьютер",
    "отправь сообщение",
    "напиши сообщение",
    "запиши задачу",
    "поставь напоминание",
    "заведи напоминание",
];

/// Может быть и действием, и вопросом. Решает окружение.
const AMBIGUOUS_VERBS: &[&str] = &[
    "покажи",
    "показать",
    "напиши",
    "написать",
    "посмотри",
    "посмотреть",
    "найди",
    "найти",
    "проверь",
    "проверить",
    "посчитай",
    "посчитать",
    "вычисли",
    "переведи",
    "перевести",
    "сделай",
    "создай",
    "создать",
    "настрой",
    "настроить",
    "измени",
    "изменить",
    "поменяй",
    "поменять",
    "узнай",
    "определи",
    "запиши",
    "спой",
    "расскажи",
    "прочитай",
    "поищи",
    "глянь",
];

/// Действие, названное существительным: «поставь таймер», «выполни команду».
const ACTION_NOUNS: &[&str] = &[
    "таймер",
    "будильник",
    "напоминание",
    "напоминалку",
    "задачу",
    "задачник",
    "команду",
    "скрипт",
    "терминал",
    "файл",
    "папку",
    "буфер",
    "обмена",
    "скриншот",
    "снимок",
    "громкость",
    "яркость",
    "окно",
    "программу",
    "приложение",
    "сайт",
    "ссылку",
    "сообщение",
    "заметку",
    "событие",
    "встречу",
    "музыку",
    "трек",
    "плеер",
    "число",
    "время",
    "клавишу",
    "клавиатуру",
    "мышь",
    "курсор",
    "текст",
];

/// Абстрактные понятия: с ними действие невозможно физически. Именно из-за
/// них «покажи смысл жизни» раньше превращалось в выдуманный tool-вызов.
const ABSTRACT_OBJECTS: &[&str] = &[
    "смысл",
    "смысла",
    "жизни",
    "жизнь",
    "любовь",
    "любви",
    "счастье",
    "счастья",
    "совесть",
    "совести",
    "идею",
    "идея",
    "идеи",
    "фразу",
    "фраза",
    "фразы",
    "стих",
    "стихотворение",
    "анекдот",
    "шутку",
    "сказку",
    "историю",
    "рассказ",
    "правду",
    "философию",
    "разницу",
    "разница",
    "причину",
    "причина",
    "теорию",
    "теория",
    "мнение",
    "совет",
    "ответ",
    "вопрос",
    "вопрос",
    "аналогию",
    "вывод",
    "итог",
];

/// Вопросительные зачины. Сильнее слабого глагола: «сколько у меня пенсия» —
/// это вопрос, а не просьба что-то выполнить.
const QUESTION_MARKERS: &[&str] = &[
    "сколько",
    "почему",
    "зачем",
    "отчего",
    "какой",
    "какая",
    "какое",
    "какие",
    "каком",
    "какую",
    "который",
    "которая",
    "кто",
    "кого",
    "кому",
    "чем",
    "чём",
    "что",
    "чего",
    "чему",
    "где",
    "куда",
    "когда",
    "как",
    "насколько",
    "правда ли",
    "верно ли",
    "можно ли",
    "стоит ли",
    "нужно ли",
    "есть ли",
    "ли",
];

/// Вопрос о самих внутренностях машины — это настоящее действие (диагностика),
/// хотя по форме похоже на вопрос.
const SYSTEM_QUESTION_NOUNS: &[&str] = &[
    "заряд",
    "батарея",
    "батареи",
    "память",
    "озу",
    "свободного",
    "места",
    "диска",
    "процессор",
    "версия",
    "сборка",
    "система",
    "макос",
    "макоса",
    "macos",
    "uptime",
    "нагрузка",
    "температура",
    "процессора",
    "видеокарта",
    "айпи",
    "ip",
    "сеть",
    "подключение",
    "модель",
    "конфигурация",
];

/// Фразы, у которых действие очевидно по форме целиком.
const ACTION_PHRASES: &[&str] = &[
    "какая у меня версия",
    "какие у меня настройки",
    "что там с",
    "сделай погромче",
    "сделай потише",
    "что в задачнике",
    "что по задачам",
    "покажи список окон",
    "покажи окна",
    "прочитай буфер",
    "что в буфере",
    "какой сегодня день",
    "какое сегодня число",
    "какой сейчас год",
    "что у меня на экране",
    "температура процессора",
    "температура видеокарты",
    "температура cpu",
    "температура gpu",
    "загрузка процессора",
    "загрузка памяти",
];

/// Просьбы, которые звучат как команда, но выполняются ответом, а не
/// инструментом: перевод, объяснение, пересказ.
const ANSWER_PHRASES: &[&str] = &[
    "переведи",
    "перевести",
    "перевод",
    "объясни",
    "объяснить",
    "как будет",
    "как переводится",
    "что значит",
    "растолкуй",
    "поясни",
];

/// Обороты, которые делают просьбу вопросом о человеке или о мире, а не о
/// машине. При них «температура», «загрузка», «версия» и подобные слова не
/// означают диагностику устройства.
const NON_SYSTEM_CONTEXT: &[&str] = &[
    "тела",
    "простуде",
    "болезни",
    "человека",
    "кипения",
    "плавления",
    "воздуха",
    "воды",
    "мозга",
    "сна",
    "печати",
    "английски",
    "английский",
    "английского",
    "немецкий",
    "французский",
];

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .replace(['ё'], "е")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn has_word(words: &[String], set: &[&str]) -> bool {
    words
        .iter()
        .any(|w| set.iter().any(|s| s.replace('ё', "е") == *w))
}

/// Классифицирует запрос как действие или разговор.
///
/// Раньше это был список префиксов с `starts_with`: «покажи мне смысл этой
/// фразы», «сколько у меня будет пенсия», «температура кипения воды» и ещё
/// десять живых фраз уезжали в action-ветку. А там включается принудительный
/// prefill `<tool>{"name":"` — модель лишена возможности ответить текстом и
/// обязана выдумать инструмент. Отсюда «галлюцинирует» и «отвечает не по делу».
///
/// Теперь решает скоринг: вопросительный зачин и абстрактный объект гасят
/// действие, явный объект на машине — включают.
pub fn classify(text: &str) -> Intent {
    classify_detailed(text).intent
}

/// Подробный вердикт: интент и уверенность в нём (0.0..=1.0).
/// Неуверенный `Action` — повод не форсировать tool-блок.
pub fn classify_detailed(text: &str) -> Verdict {
    let compact = normalize(text);
    if compact.is_empty() {
        return Verdict::chat();
    }
    let normalized_phrases: Vec<String> = ACTION_PHRASES.iter().map(|p| normalize(p)).collect();
    let words: Vec<String> = compact.split_whitespace().map(str::to_string).collect();

    // Оборот про человека или про мир, а не про машину: «температура тела»,
    // «загрузка мозга» — это не диагностика устройства.
    let non_system = NON_SYSTEM_CONTEXT
        .iter()
        .any(|p| compact.contains(&normalize(p)));

    // 0. Чтение состояния среды — цель для агента, у которого есть
    // наблюдение и ui_read: «прочитай непрочитанные в телеграме» — это
    // задача, а не болтовня, каким бы ни был глагол.
    if reads_state(text) {
        return Verdict::action(0.9);
    }

    // 1. Явное действие целиком — до всякой аналитики. Короткие фразы
    // («температура тела») уступают контексту: они слишком общие.
    if normalized_phrases
        .iter()
        .any(|p| compact.contains(p.as_str()) && (p.split_whitespace().count() > 2 || !non_system))
    {
        return Verdict::action(1.0);
    }

    // 2. Просьба, которая выполняется ответом, а не инструментом:
    // «переведи это на английский», «объясни, как работает TCP».
    if words.first().is_some_and(|first| {
        ANSWER_PHRASES
            .iter()
            .any(|p| normalize(p).split_whitespace().next() == Some(first.as_str()))
    }) {
        return Verdict::chat();
    }

    // 3. Явный объект на машине плюс слабый глагол — действие.
    let explicit_object = has_word(&words, ACTION_NOUNS);
    let system_question = has_word(&words, SYSTEM_QUESTION_NOUNS) && !non_system;
    let question = has_word(&words, QUESTION_MARKERS);
    let abstract_object = has_word(&words, ABSTRACT_OBJECTS);
    let unambiguous = has_word(&words, UNAMBIGUOUS);
    let ambiguous = has_word(&words, AMBIGUOUS_VERBS);

    // Диагностика машины вопросом: «сколько у меня свободного места».
    // Важнее абстрактных слов: «места» попадает и туда, и сюда.
    if system_question {
        return Verdict::action(0.8);
    }
    // Однозначный глагол — это действие: «открой хром», «поставь таймер»,
    // «выключи музыку». Абстрактный объект его не отменяет.
    if unambiguous {
        return Verdict::action(if explicit_object { 0.95 } else { 0.85 });
    }
    if ambiguous && explicit_object && !non_system {
        return Verdict::action(0.75);
    }
    // С абстрактным объектом действие невозможно физически: «найди смысл жизни».
    if abstract_object {
        return Verdict::chat();
    }
    // Слабый глагол без объекта и с вопросом — это разговор.
    if question {
        return Verdict::chat();
    }
    if ambiguous {
        return Verdict::action(0.5);
    }

    Verdict::chat()
}

/// Вердикт роутера.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    pub intent: Intent,
    /// Насколько роутер уверен. Для `Action` ниже 0.6 — не форсировать
    /// tool-блок: пусть модель может ответить текстом.
    pub confidence: f32,
}

impl Verdict {
    fn action(confidence: f32) -> Self {
        Self {
            intent: Intent::Action,
            confidence,
        }
    }

    fn chat() -> Self {
        Self {
            intent: Intent::Chat,
            confidence: 0.0,
        }
    }
}

/// Явная просьба поискать в интернете. Сильнее эвристик про свежесть:
/// «да посмотри в интернете» не должно снова упираться в ответ из головы.
pub fn explicit_web_request(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    // «Кто такой X» / «что такое X» — вопрос о человеке/сущности: у 2B-модели
    // на незнакомые имена гарантированная галлюцинация, отвечать из весов
    // нельзя — только по выдаче поиска. Проверяем по началу фразы, чтобы не
    // задевать «расскажи другу, кто такой хоккеист» в середине мысли.
    const WHO_PREFIXES: &[&str] = &[
        "кто такой",
        "кто такая",
        "кто такие",
        "что за человек",
        "расскажи про",
        "расскажи о",
    ];
    if WHO_PREFIXES.iter().any(|p| t.starts_with(p)) {
        return true;
    }
    [
        "интернет",
        "погугли",
        "загугли",
        "в гугле",
        "в яндексе",
        "в сети",
    ]
    .iter()
    .any(|k| t.contains(k))
}

/// Слова-обёртки интернет-просьбы. Сам запрос — то, что от них осталось.
const WEB_FILLERS: &[&str] = &[
    "да",
    "ну",
    "а",
    "давай",
    "пожалуйста",
    "быстро",
    "сэр",
    "джарвис",
    "посмотри",
    "посмотреть",
    "поищи",
    "поискать",
    "поискай",
    "найди",
    "найти",
    "погугли",
    "загугли",
    "чекни",
    "гляни",
    "в",
    "во",
    "на",
    "интернете",
    "интернет",
    "инете",
    "гугле",
    "гугл",
    "яндексе",
    "сети",
    "скажи",
    "расскажи",
    "что",
    "там",
    "про",
    "об",
    "чё",
    "че",
];

/// Предмет интернет-просьбы: «да посмотри в интернете кто выиграл матч» →
/// «кто выиграл матч». `None` — предмета во фразе нет («да посмотри
/// в интернете»): движок возьмёт последнюю тему из памяти.
pub fn web_query_from(text: &str) -> Option<String> {
    let subject: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .filter(|w| !WEB_FILLERS.contains(&w.to_lowercase().as_str()))
        .collect();
    (!subject.is_empty()).then(|| subject.join(" "))
}

/// Вопрос про календарь («какой сегодня день недели»): ответ вычисляется из
/// даты, которая уже в системном промпте, — веб-поиск тут трата времени.
pub fn calendar_question(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    [
        "день недели",
        "какое число",
        "сегодня число",
        "какой день",
        "какой год",
        "сейчас год",
        "какой месяц",
    ]
    .iter()
    .any(|k| t.contains(k))
}

/// Инструменты, которые есть в промпте всегда: это «руки» ассистента и его
/// страховка. Даже если срез интента не угадан, сделать работу будет чем.
const CORE_TOOLS: &str = "\
  shell {\"cmd\":\"ls ~/Downloads\"} — терминал: всё, для чего нет своего инструмента
  open_app {\"name\":\"Chrome\"} — запустить приложение; quit_app {\"name\":\"Telegram\"} — закрыть приложение
  open_url {\"url\":\"wikipedia.org\",\"browser\":\"Safari\"} — открыть сайт; open_path {\"path\":\"~/Downloads\"} — папку/файл
  file_read {\"path\":\"src/main.rs\",\"offset\":1,\"limit\":80} — прочитать файл строками
  file_write {\"path\":\"note.txt\",\"content\":\"текст\"} — создать; file_append {\"path\":\"note.txt\",\"content\":\"строка\"} — дописать
  file_edit {\"path\":\"src/main.rs\",\"old\":\"было\",\"new\":\"стало\"} — заменить фрагмент; old бери из file_read
  file_list {\"path\":\".\"} / file_search {\"query\":\"текст\",\"path\":\".\"} / file_glob {\"pattern\":\"*.rs\",\"path\":\".\"} — файлы
  web_search {\"query\":\"курс доллара\"} — найти в интернете
  system_info — система; notify {\"title\":\"..\",\"body\":\"..\"} — уведомление; clipboard_write {\"text\":\"..\"} — буфер
  desktop_input {\"action\":\"type_text\",\"text\":\"Привет\"} — клавиатура/мышь: type_text | press_key | hotkey | click | move | scroll
  ui_read — прочитать экран: приложение, окна, кнопки, поля, меню
  ui_click {\"role\":\"button\",\"name\":\"Продолжить\",\"app\":\"Имя\"} — нажать элемент по имени (роли: button, checkbox, radio button, pop up button, link)
  ui_type {\"name\":\"Поиск\",\"text\":\"запрос\",\"submit\":true} — вписать текст в поле по имени
  ui_menu {\"app\":\"TextEdit\",\"path\":[\"Файл\",\"Сохранить\"]} — пункт меню
  wait {\"seconds\":2} — пауза между шагами";

/// Срезы инструментов: в промпт попадают только те, что нужны по интенту.
/// Полный список из 32 инструментов стоил 940 токенов и был для модели 0.8B
/// шумом — она выбирала инструмент хуже, чем из короткого списка.
const TOOL_SLICES: &[(&str, &[&str], &str)] = &[
    (
        "память и расписание",
        &[
            "запомни",
            "запомнить",
            "помни",
            "занеси в память",
            "я предпочита",
            "расписан",
            "каждое утро",
            "каждый день",
            "каждый час",
            "каждые",
            "напоминай",
            "по расписанию",
            "повторяй",
        ],
        "  remember {\"text\":\"факт о пользователе\"} — долговременная память (предпочтения, проекты, близкие)
  schedule_add {\"prompt\":\"что делать\",\"daily_at\":\"ЧЧ:ММ\"} — каждый день; {\"prompt\":\"…\",\"every_minutes\":N} — каждые N минут
  schedule_list — показать расписание; schedule_remove {\"id\":3} — убрать задачу
  mcp_call {\"server\":\"имя\",\"tool\":\"имя\",\"args\":{…}} — внешние MCP-инструменты, если перечислены ниже",
    ),
    (
        "окна и экран",
        &[
            "сверни",
            "разверни",
            "закрой окно",
            "полный экран",
            "во весь экран",
            "переключись",
            "активное окно",
            "список окон",
            "свернуть",
            "развернуть",
            "громче",
            "тише",
            "громкость",
            "ярче",
            "темнее",
            "яркость",
            "скриншот",
            "снимок экрана",
            "заблокируй",
            "усыпи",
            "спящий режим",
            "утилит",
            "анимаци",
            "реактор",
            "застав",
            "клавиатур",
            "клавиш",
            "сочетание клавиш",
            "горячие клавиши",
            "введи текст",
            "напечатай",
            "набери",
            "вставь",
            "выдели",
            "мышь",
            "мышкой",
            "курсор",
            "прокрути",
            "прокрутка",
        ],
        "  window {\"action\":\"minimize\",\"app\":\"имя\"} — окна: minimize=свернуть, maximize=развернуть, fullscreen=экран, close, focus, list. Без app — Джарвис\n  open_utils {\"tab\":\"timer\"} — открыть окно утилит; tab: timer | stopwatch | calendar (без tab — таймер)\n  close_utils — закрыть окно утилит\n  play_reactor_intro — показать сборку реактора ещё раз (анимация интерфейса)\n  set_volume {\"percent\":50} / set_brightness {\"percent\":70}\n  desktop_input {\"action\":\"type_text\",\"text\":\"текст\"} — ввод текста\n  desktop_input {\"action\":\"press_key\",\"key\":\"ENTER\"} — клавиша\n  desktop_input {\"action\":\"hotkey\",\"keys\":[\"CTRL\",\"S\"]} — сочетание\n  desktop_input {\"action\":\"scroll\",\"direction\":\"down\",\"amount\":3} — прокрутка\n  screenshot / lock_screen / sleep",
    ),
    (
        "таймеры",
        &[
            "таймер",
            "будильник",
            "напомни",
            "напоминание",
            "через час",
            "через минут",
            "выключи будильник",
            "останови будильник",
        ],
        "  set_timer {\"seconds\":600,\"label\":\"чай\"} / set_alarm {…} — встроенный таймер с окном; timer_stop — остановить; alarm_stop — выключить звонящий будильник; НЕ shell sleep",
    ),
    (
        "сообщения и календарь",
        &[
            "сообщение",
            "напиши маме",
            "отправь",
            "imessage",
            "смс",
            "заметку",
            "заметка",
            "календар",
            "встреч",
            "событи",
            "напомни в",
        ],
        "  send_message {\"to\":\"мама\",\"text\":\"..\"} — iMessage\n  add_note {\"title\":\"..\",\"body\":\"..\"} — Заметки\n  create_event {\"title\":\"..\",\"date\":\"ГГГГ-ММ-ДД\",\"time\":\"ЧЧ:ММ\"} — Календарь",
    ),
    (
        "музыка",
        &[
            "музык",
            "плеер",
            "трек",
            "песн",
            "играет",
            "плейлист",
            "переключи трек",
            "пауза",
            "паузу",
            "pause",
            "продолжи",
            "поставь на паузу",
            "останови музыку",
            "выключи музыку",
            "стоп музыку",
            "следующий трек",
            "предыдущий трек",
        ],
        "  music_control {\"action\":\"play|pause|stop|next|previous\"} — музыка: сначала «Музыка», а если она пустая — играющая вкладка браузера (Яндекс Музыка, YouTube)",
    ),
    (
        "задачник",
        &["задач", "задачник", "to-do", "туду", "чеклист"],
        "  task_add {\"text\":\"..\"} / task_list / task_run {\"id\":N} / task_done {\"id\":N} / task_remove {\"id\":N} / task_open — задачник",
    ),
    (
        "экран: элементы и действия",
        &[
            "что на экране",
            "что открыто",
            "что видишь",
            "посмотри на экран",
            "найди на экране",
            "прочитай с экрана",
            "проверь экран",
            "что написано на экране",
            "сделай на экране",
            "на экране",
            "кнопк",
            "флажок",
            "чекбокс",
            "галочку",
            "поле",
            "заполни",
            "впиши",
            "элемент",
            "меню",
            "нажми",
            "нажать",
            "клик",
            "кликни",
            "клацни",
        ],
        "  ui_read / ui_click / ui_type / ui_menu — сперва прочитай экран, затем кликай, вписывай и выбирай меню ПО ИМЕНИ элемента (см. ядро)\n  desktop_input {\"action\":\"click\",\"x\":500,\"y\":300} — клик мышью ТОЛЬКО по точным координатам от пользователя; координаты наугад запрещены\n  desktop_input {\"action\":\"hotkey\",\"keys\":[\"CTRL\",\"S\"]} — сочетание клавиш",
    ),
];

/// Собирает список инструментов под конкретную просьбу: ядро всегда, плюс
/// только те срезы, чьи слова действительно встретились.
fn tool_block_for(prompt: &str) -> String {
    let compact = normalize(prompt);
    let mut out = String::from(CORE_TOOLS);
    for (_name, keywords, body) in TOOL_SLICES {
        if keywords.iter().any(|k| compact.contains(&normalize(k))) {
            out.push('\n');
            out.push_str(body);
        }
    }
    if crate::tools::is_macos() {
        out.push_str("\n  applescript {\"script\":\"..\"} — для прочего в нативных приложениях");
    }
    out
}

/// Компактный системный промпт для режима действий под конкретную просьбу.
///
/// Каждый токен здесь оплачивается префиллом на каждом шаге цикла, а модель
/// маленькая: подробные инструкции её только путают. Поэтому — одна строка на
/// инструмент, примеры аргументов вместо JSON-схем, и только те инструменты,
/// которые нужны по интенту.
/// Просьба о глубоком исследовании (мульти-хоп поисков): «исследуй»,
/// «собери информацию», «проведи расследование». Список узкий намеренно:
/// широкий зацепил бы обычные «найди», которые живут одним web_search.
pub fn is_research_request(prompt: &str) -> bool {
    let t = prompt.to_lowercase();
    [
        "исследуй",
        "исследовать",
        "проведи расследование",
        "проведи исследование",
        "глубокий разбор",
        "глубоко разберись",
        "собери информацию",
        "собери всё про",
        "разберись во",
        "изучи вопрос",
        "мульти-хоп",
    ]
    .iter()
    .any(|k| t.contains(k))
}

#[cfg(test)]
mod research_tests {
    use super::*;

    #[test]
    fn research_phrases_match_and_normal_searches_do_not() {
        assert!(is_research_request("исследуй рынок электрокаров 2026"));
        assert!(is_research_request("собери информацию про FastConformer"));
        assert!(is_research_request("Проведи расследование: куда пропал LLama 4"));
        // Обычный поиск — одна выдача, не исследование.
        assert!(!is_research_request("найди цену на бензин"));
        assert!(!is_research_request("какая погода в Москве"));
        assert!(!is_research_request("открой ютуб"));
    }
}

pub fn action_prompt_for(prompt: &str) -> String {
    let os = crate::tools::os_name();
    let tools = tool_block_for(prompt);
    // Каталог пользовательских команд: блок попадает в промпт только если
    // команды есть — иначе модель зря ищет совпадения с пустым списком.
    let catalog = crate::commands::prompt_catalog();
    let commands_block = if catalog.is_empty() {
        String::new()
    } else {
        format!(
            "\nКОМАНДЫ ПОЛЬЗОВАТЕЛЯ (совпало по смыслу — вызывай run_command):\n{catalog}\n  run_command {{\"name\":\"имя\",\"args\":{{\"параметр\":\"значение\"}}}}"
        )
    };
    // Внешние MCP-инструменты: блок появляется, только если серверы
    // настроены в settings.toml и отдали список инструментов.
    crate::mcp::ensure_connected();
    let mcp_block = crate::mcp::prompt_block();

    // Каталог установленных приложений: модель выбирает open_app из
    // реального списка, а не выдумывает имена («Finder» на Linux). Это
    // универсальная сеть для любой формулировки, которую не поймал
    // нечёткий сопоставитель.
    let catalog = crate::platform::app_catalog();
    let apps_block = if catalog.is_empty() {
        String::new()
    } else {
        format!(
            "\nУСТАНОВЛЕННЫЕ ПРИЛОЖЕНИЯ (open_app — только эти имена):\n{}\n",
            catalog.join(", ")
        )
    };

    format!(
        r#"Ты — агент на компьютере пользователя ({os}). Сегодня {today}. Рабочая папка: {cwd}; относительные пути — от неё.
Просьба — цель, а не имя инструмента. Состояние системы приходит в сообщении «ОБСТАНОВКА» — опирайся на него, не догадывайся.
Действие — ровно один блок без слов вокруг: <tool>{{"name":"ИМЯ","args":{{...}}}}</tool>; имя — только из списка. Ответить — текст без блока, коротко и в мужском роде.

ИНСТРУМЕНТЫ:
{tools}{commands_block}{mcp_block}{apps_block}
После каждого шага придёт свежая «ОБСТАНОВКА»: сверяй её с целью и продолжай, пока цель не подтверждена наблюдением.
Успех — только подтверждённое наблюдением; не подтвердилось — скажи, на чём остановился и почему.
Покупки, отправка наружу, удаление — только после явного подтверждения пользователя текстом."#,
        today = crate::text::today_russian(),
        cwd = crate::tools::cwd(),
    )
}

/// Промпт действий без конкретной просьбы: ядро инструментов плюс каталог.
pub fn action_prompt() -> String {
    action_prompt_for("")
}

// =========================================================================
// Детерминированный план системных команд
// =========================================================================

/// Глаголы, с которых начинается просьба открыть/запустить. Проверяем только
/// первое значимое слово: «что открой ютуб значит» — вопрос, а не команда.
const PLAN_OPEN_TRIGGERS: &[&str] = &[
    "открой",
    "откройте",
    "открыть",
    "открывай",
    "запусти",
    "запустите",
    "запустить",
    "запускай",
    "включи",
    "включите",
    "включить",
    "open",
    "launch",
    "start",
    "run",
];

const PLAN_QUIT_TRIGGERS: &[&str] = &[
    "закрой",
    "закройте",
    "закрыть",
    "закрывай",
    "заверши",
    "завершить",
    "выйди",
    "выйти",
    "выходи",
    "quit",
    "close",
    "exit",
];

/// Обращения и вежливые слова в начале фразы. Не трогаем «да»/«ну»: они
/// тоже могут быть частью вопроса, а команда и без них начинается с глагола.
const PLAN_LEADING_NOISE: &[&str] = &[
    "джарвис",
    "джарвиз",
    "джавиз",
    "джарис",
    "жарвис",
    "ярвис",
    "jarvis",
    "ассистент",
    "эй",
    "слушай",
    "привет",
    "пожалуйста",
    "плиз",
    "сэр",
];

/// Слова, которые не являются частью имени приложения/файла.
const PLAN_TARGET_FILLERS: &[&str] = &[
    "пожалуйста",
    "плиз",
    "сейчас",
    "быстро",
    "мне",
    "сэр",
    "приложение",
    "приложения",
    "приложению",
    "приложением",
    "приложении",
    "программу",
    "программа",
    "программы",
    "программе",
    "программой",
    "app",
    "application",
    "файл",
    "файла",
    "файлу",
    "файлы",
    "файлом",
    "папка",
    "папку",
    "папки",
    "папок",
    "папкой",
    "каталог",
    "сайт",
    "сайта",
    "сайте",
    "страницу",
    "страница",
    "ссылку",
    "ссылку",
    "file",
    "folder",
    "document",
    "documents",
    "site",
    "website",
    "url",
    "окно",
    "окна",
    "окон",
    "окошко",
    "из",
    "работу",
    "работы",
];

/// Токены с сохранением символов пути/URL: обычный `clean_text` вырезает
/// точки и слэши, после чего «example.com» и «~/Downloads» уже не отличить.
fn plan_tokens(prompt: &str) -> Vec<String> {
    let mut tokens: Vec<String> = prompt
        .split_whitespace()
        .filter_map(|raw| {
            let mut word = raw
                .trim_matches(|c: char| {
                    !c.is_alphanumeric() && !matches!(c, '/' | '\\' | '~' | '.' | '-' | '_' | ':')
                })
                .to_string();
            // Точка в конце фразы — пунктуация, а не часть домена/файла.
            if !word.contains('/') && !word.contains('\\') {
                while word.ends_with('.') && !word.starts_with('.') && !word.ends_with("..") {
                    word.pop();
                }
            }
            (!word.is_empty()).then_some(word)
        })
        .collect();
    let mut start = 0;
    while start < tokens.len()
        && PLAN_LEADING_NOISE.contains(&tokens[start].to_lowercase().as_str())
    {
        start += 1;
    }
    tokens.drain(..start);
    tokens
}

fn plan_target(tokens: &[String]) -> String {
    tokens
        .iter()
        .filter(|w| !PLAN_TARGET_FILLERS.contains(&w.to_lowercase().as_str()))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Вопрос, а не команда: «как свернуть окно», «почему закрылось окно».
/// Без этого окно-парсер исполнял бы вопросительные фразы.
fn plan_is_question(prompt: &str) -> bool {
    let text = prompt.trim();
    if text.ends_with('?') {
        return true;
    }
    let first = text
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase();
    matches!(
        first.as_str(),
        "как"
            | "что"
            | "почему"
            | "зачем"
            | "отчего"
            | "где"
            | "куда"
            | "когда"
            | "какой"
            | "какая"
            | "какое"
            | "какие"
            | "который"
            | "которая"
            | "сколько"
            | "кто"
            | "кого"
            | "чем"
            | "можно"
            | "можешь"
            | "нельзя"
            | "стоит"
            | "правда"
    )
}

/// Частые папки: «открой папку загрузки» должно открыть ~/Downloads,
/// а не искать приложение с именем «загрузки».
fn plan_folder_path(target: &str) -> Option<String> {
    let compact = crate::text::compact_name(target);
    let path = match compact.as_str() {
        "загрузки" | "загрузок" | "загрузка" | "downloads" | "download" => {
            "~/Downloads"
        }
        "документы" | "документов" | "documents" | "document" => "~/Documents",
        "рабочийстол" | "рабочемстоле" | "рабочегостола" | "desktop" => {
            "~/Desktop"
        }
        "картинки" | "изображения" | "фотографии" | "pictures" => {
            "~/Pictures"
        }
        "видео" | "фильмы" | "movies" => "~/Movies",
        "музыка" | "музыку" | "music" => "~/Music",
        "домашняяпапка" | "домашнююпапку" | "домашнийкаталог" | "home" => {
            "~"
        }
        _ => return None,
    };
    Some(path.to_string())
}

fn plan_explicit_url(target: &str) -> Option<String> {
    let t = target.trim();
    let lower = t.to_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.") {
        return Some(t.to_string());
    }
    None
}

/// Домен с известным TLD. Без списка «README.md» и «main.rs» выглядели бы
/// сайтами, хотя это файлы.
fn plan_domain(target: &str) -> Option<String> {
    let t = target.trim();
    if t.is_empty()
        || t.contains(char::is_whitespace)
        || t.starts_with('~')
        || t.starts_with('/')
        || t.starts_with('.')
    {
        return None;
    }
    let host = t.split('/').next().unwrap_or("").trim();
    let (name, tld) = host.rsplit_once('.')?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '.')
    {
        return None;
    }
    const TLDS: &[&str] = &[
        "com", "ru", "org", "net", "io", "ai", "dev", "app", "me", "co", "uk", "de", "fr", "info",
        "edu", "gov", "xyz", "tech", "online", "site", "store", "shop", "moscow", "su", "by", "kz",
        "ua",
    ];
    TLDS.contains(&tld.to_lowercase().as_str())
        .then(|| t.to_string())
}

fn plan_path_like(target: &str) -> bool {
    target.contains('/')
        || target.contains('\\')
        || target.starts_with('~')
        || target.starts_with('.')
}

fn plan_existing_path(target: &str) -> Option<String> {
    let resolved = crate::tools::resolve_path(target);
    std::path::Path::new(&resolved)
        .exists()
        .then(|| target.to_string())
}

fn plan_open(
    tokens: &[String],
    lookup: &impl Fn(&str) -> Option<String>,
) -> Option<crate::tools::ToolCall> {
    // «открой aviasales.ru в сафари»: «в сафари» — не часть цели, а браузер.
    let (without_browser, browser) = crate::tool_guard::strip_browser_phrase(&tokens.join(" "));
    let stripped: Vec<String>;
    let tokens: &[String] = if browser.is_some() {
        stripped = without_browser.split_whitespace().map(String::from).collect();
        &stripped
    } else {
        tokens
    };
    let target = plan_target(&tokens[1..]);
    if target.is_empty() {
        return None;
    }
    let lower: Vec<String> = tokens.iter().map(|w| w.to_lowercase()).collect();
    // «открой браузер файт фокс» = «открой Firefox»: слово «браузер» —
    // указание категории, а не часть имени. Один «браузер» — браузер
    // по умолчанию (platform::app_open знает этот случай).
    let rest_of_target: Vec<String> = tokens[2..].to_vec();
    if rest_of_target.is_empty()
        && lower
            .get(1)
            .is_some_and(|w| w.starts_with("браузер") || w.starts_with("browser"))
    {
        return Some(crate::tools::ToolCall::OpenApp {
            name: "браузер".to_string(),
        });
    }
    let target = if lower
        .get(1)
        .is_some_and(|w| w.starts_with("браузер") || w.starts_with("browser"))
    {
        plan_target(&rest_of_target)
    } else {
        target
    };
    if target.is_empty() {
        return None;
    }
    // Музыка/плеер — не запуск приложения: «включи музыку» управляет
    // плеером, а не открывает Music.app. Пусть решает LLM со срезом music.
    if lower.iter().any(|w| {
        w.starts_with("музык")
            || w.starts_with("плеер")
            || w.starts_with("трек")
            || w.starts_with("песн")
    }) {
        return None;
    }
    let path_marker = lower.iter().any(|w| {
        w.starts_with("файл")
            || w.starts_with("папк")
            || w.starts_with("каталог")
            || matches!(w.as_str(), "file" | "folder" | "document" | "documents")
    });
    let site_marker = lower.iter().any(|w| {
        w.starts_with("сайт")
            || w.starts_with("страниц")
            || w.starts_with("ссылк")
            || matches!(w.as_str(), "site" | "website" | "url")
    });

    if let Some(url) = plan_explicit_url(&target) {
        return Some(crate::tools::ToolCall::OpenUrl { url, browser: browser.clone() });
    }
    if let Some(url) = plan_domain(&target) {
        return Some(crate::tools::ToolCall::OpenUrl { url, browser: browser.clone() });
    }
    // Браузер назван, а адреса нет: приложение/папку не угадываем, решает
    // разрешение сайта (кэш, выдача поиска) в цикле действий.
    if browser.is_some() {
        return None;
    }
    // «Открой загрузки/документы» — это папка, даже без слова «папка».
    if let Some(path) = plan_folder_path(&target) {
        return Some(crate::tools::ToolCall::OpenPath { path });
    }
    if path_marker || plan_path_like(&target) {
        return Some(crate::tools::ToolCall::OpenPath { path: target });
    }
    // «Открой сайт X» без домена — не повод запускать случайное приложение:
    // пусть решает обучаемый кэш сайтов или LLM.
    if site_marker {
        return None;
    }
    if let Some(app) = lookup(&target) {
        return Some(crate::tools::ToolCall::OpenApp { name: app });
    }
    // Приложение не нашлось, но такой файл/папка реально есть —
    // открываем файл, а не выдумываем запуск.
    if let Some(path) = plan_existing_path(&target) {
        return Some(crate::tools::ToolCall::OpenPath { path });
    }
    None
}

fn plan_quit(
    tokens: &[String],
    lookup: &impl Fn(&str) -> Option<String>,
) -> Option<crate::tools::ToolCall> {
    let target = plan_target(&tokens[1..]);
    if target.is_empty() {
        return None;
    }
    // Закрытие окна/файла/сайта — не выход приложения.
    let wrong_kind = tokens.iter().any(|w| {
        let w = w.to_lowercase();
        w.starts_with("окн")
            || w == "окон"
            || w.starts_with("окошк")
            || w.starts_with("файл")
            || w.starts_with("папк")
            || w.starts_with("сайт")
            || w.starts_with("страниц")
    });
    if wrong_kind {
        return None;
    }
    if crate::platform::is_self_window_target(&target) {
        // Не убиваем сами себя голосовой командой; выход — отдельная команда.
        return None;
    }
    lookup(&target).map(|name| crate::tools::ToolCall::QuitApp { name })
}

/// Однозначная системная команда без LLM. `None` — фраза неоднозначна или
/// это не системное действие: дальше решают каталог команд, кэш сайтов и LLM.
/// Фраза просит ЧИТАТЬ состояние (непрочитанные, содержимое, счётчики),
/// а не выполнить действие запуска/закрытия. Такие фразы не перехватываются
/// детерминированным планом: «прочитай непрочитанные в телеграме» — это не
/// «открой телеграм», агент должен увидеть состояние и ответить числом.
/// Проверка по смыслу всей фразы, а не по первому слову.
pub fn reads_state(prompt: &str) -> bool {
    let t = normalize(prompt);
    let read_verbs = [
        "прочитай", "прочитать", "почитай", "проверь", "проверить", "посмотри",
        "посмотреть", "скажи", "сколько", "какие", "что там", "покажи", "показать",
        "зачитай", "объяви", "перечисли", "напиши что", "есть ли",
    ];
    let read_objects = [
        "непрочитан", "сообщени", "письм", "писем", "почт", "уведомлен", "чат", "диалог",
        "пропущен", "входящ",
    ];
    let has_verb = read_verbs.iter().any(|v| t.contains(v));
    let has_object = read_objects.iter().any(|o| t.contains(o));
    // «Сколько времени» / «какая погода» — не про состояние приложений.
    let not_about_apps = ["времени", "погода", "погоду", "дата", "день"]
        .iter()
        .any(|w| t.contains(w));
    has_verb && (has_object || (t.contains("непрочитан") || t.contains("уведомлен")))
        && !not_about_apps
}

pub fn plan_system_command(prompt: &str) -> Option<crate::tools::ToolCall> {
    plan_system_command_with(prompt, crate::platform::app_lookup)
}

/// [`plan_system_command`] с подменяемым поиском приложений — для тестов на
/// любой ОС и без реально установленных программ.
pub fn plan_system_command_with(
    prompt: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<crate::tools::ToolCall> {
    // Смысл раньше глагола: «прочитай непрочитанные в телеграме» содержит
    // слово «телеграм», но это чтение состояния, а не «открой телеграм».
    if reads_state(prompt) {
        return None;
    }

    let tokens = plan_tokens(prompt);
    let first = tokens.first()?.to_lowercase();
    if tokens.len() < 2 {
        return None;
    }

    // Окна — раньше выхода приложения: «закрой окно X» и «закрой X» различаются.
    // Вопросительные обороты («как свернуть окно») сюда не пускаем, кроме
    // просьбы перечислить окна: «какие окна открыты» — это команда list.
    if let Some((action, app)) = crate::tools::window_request_from_prompt(prompt)
        && (action == "list" || !plan_is_question(prompt))
    {
        let app = app.map(|name| {
            if name == "активное" {
                name
            } else {
                lookup(&name).unwrap_or(name)
            }
        });
        return Some(crate::tools::ToolCall::Window { action, app });
    }

    if PLAN_QUIT_TRIGGERS.contains(&first.as_str()) {
        return plan_quit(&tokens, &lookup);
    }
    if PLAN_OPEN_TRIGGERS.contains(&first.as_str()) {
        return plan_open(&tokens, &lookup);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_marker_is_stripped_from_target() {
        // «открой браузер файт фокс» = «открой Firefox»: маркер не часть имени.
        let lookup = |want: &str| {
            (matches!(want, "файт фокс" | "зум" | "хром"))
                .then(|| format!("{want}-resolved"))
        };
        let call = plan_system_command_with("открой браузер файт фокс", lookup);
        assert_eq!(
            call,
            Some(crate::tools::ToolCall::OpenApp {
                name: "файт фокс-resolved".to_string()
            })
        );
        // Голый «браузер» — браузер по умолчанию.
        let call = plan_system_command_with("открой браузер", |_| None);
        assert_eq!(
            call,
            Some(crate::tools::ToolCall::OpenApp {
                name: "браузер".to_string()
            })
        );
    }

    #[test]
    fn read_state_requests_are_not_intercepted() {
        // Диагноз задания: «прочитай непрочитанные в Telegram» превращался
        // в запуск Telegram. Теперь фраза чтения уходит агенту.
        for phrase in [
            "прочитай непрочитанные в телеграме",
            "сколько непрочитанных в почте",
            "проверь сообщения в телеграме",
            "что там в почте",
            "прочитай письма",
            "сколько уведомлений",
        ] {
            assert!(
                plan_system_command_with(phrase, |_| None).is_none(),
                "чтение состояния не должно перехватываться планом: {phrase}"
            );
        }
    }

    #[test]
    fn plain_open_quit_still_intercepted() {
        // Регрессии прежних сценариев недопустимы: явные действия работают.
        let lookup = |want: &str| (want == "телеграм").then(|| "Telegram".to_string());
        assert!(plan_system_command_with("открой aviasales.ru", |_| None).is_some());
        assert!(plan_system_command_with("закрой телеграм", lookup).is_some());
        assert!(plan_system_command_with("открой телеграм", lookup).is_some());
    }

    #[test]
    fn reads_state_understands_meaning_not_verb() {
        assert!(reads_state("прочитай непрочитанные в телеграме"));
        assert!(reads_state("сколько писем пришло"));
        assert!(reads_state("проверь почту"));
        // Не состояние приложений:
        assert!(!reads_state("сколько времени"));
        assert!(!reads_state("какая погода"));
        // Явное действие без чтения:
        assert!(!reads_state("открой телеграм"));
        assert!(!reads_state("закрой хром"));
    }

    #[test]
    fn weather_question_still_plans_weather_path() {
        // «проверь погоду» — не перехватывается планом (погода в движке).
        assert!(plan_system_command_with("проверь погоду в москве", |_| None).is_none());
    }

    #[test]
    fn action_recognized() {
        assert_eq!(classify("открой хром"), Intent::Action);
        assert_eq!(classify("запусти таймер на 5 минут"), Intent::Action);
        assert_eq!(classify("выполни команду ls"), Intent::Action);
        assert_eq!(classify("напиши файл на рабочий стол"), Intent::Action);
        assert_eq!(classify("посмотри свободное место"), Intent::Action);
    }

    #[test]
    fn action_wins_over_fresh_info_keywords() {
        // «сейчас» и «сайт» не должны превращать команду в веб-вопрос.
        for text in [
            "открой сейчас дипсик сайт дипсика",
            "открой сайт нетология",
            "запусти сейчас таймер на 5 минут",
            "найди сейчас файл с заметками",
            "напиши сейчас файл на рабочий стол",
        ] {
            assert_eq!(classify(text), Intent::Action, "{text}");
        }
    }

    #[test]
    fn chat_recognized() {
        assert_eq!(classify("как дела"), Intent::Chat);
        assert_eq!(classify("расскажи анекдот"), Intent::Chat);
        assert_eq!(classify("почему небо синее"), Intent::Chat);
    }

    #[test]
    fn ambiguous_defaults_to_chat() {
        // Не явное действие — но модель сама решит через tool если нужно.
        assert_eq!(classify("погода в москве"), Intent::Chat);
    }

    /// Регрессия на живую боль: 13 из 29 фраз уезжали в action-ветку, где
    /// принудительный prefill `<tool>{"name":"` не оставлял модели права
    /// ответить текстом. Она обязана была выдумать инструмент — пользователь
    /// слышал чушь и называл это «галлюцинирует».
    #[test]
    fn questions_about_abstract_things_are_not_actions() {
        for text in [
            "покажи мне смысл этой фразы",
            "найди смысл жизни",
            "сколько у меня будет пенсия",
            "какая у меня температура тела при простуде",
            "проверь, правильно ли я говорю по-английски",
            "посмотри, это хорошая идея или нет",
            "температура кипения воды",
            "загрузка мозга во время сна",
            "версия для печати этого документа",
            "напиши мне стихотворение",
            "переведи это на английский",
            "посчитай, сколько лет я проживу",
            "напомни мне, что такое совесть",
            "что такое квантовая запутанность",
            "сколько будет два плюс два",
            "кто написал войну и мир",
        ] {
            assert_eq!(classify(text), Intent::Chat, "{text}");
        }
    }

    /// Обратная сторона: настоящие действия не должны потеряться.
    #[test]
    fn real_actions_still_actions() {
        for text in [
            "открой хром",
            "сверни зум",
            "запиши файл на рабочий стол",
            "выключи музыку",
            "какая у меня версия макоса",
            "сколько у меня свободного места",
            "что в задачнике",
            "сделай потише",
            "прочитай буфер",
            "отправь сообщение маме",
            "поставь таймер на 10 минут",
            "заведи будильник на 7 утра",
            "почисти корзину",
            "открой папку загрузки",
            "что у меня на экране",
        ] {
            assert_eq!(classify(text), Intent::Action, "{text}");
        }
    }

    /// Уверенность важна не меньше интента: ниже порога prefill не форсируется.
    #[test]
    fn confidence_separates_sure_actions_from_guesses() {
        assert!(classify_detailed("открой хром").confidence >= 0.85);
        assert!(classify_detailed("какая у меня версия макоса").confidence >= 0.9);
        // Слабый глагол без объекта — догадка, tool форсировать нельзя.
        assert!(classify_detailed("сделай красиво").confidence < 0.7);
        assert_eq!(
            classify_detailed("покажи мне смысл этой фразы").intent,
            Intent::Chat
        );
    }

    /// Живой случай: «поставь на паузу» — это про музыку, а не про окна.
    /// Раньше срез «окна и экран» не содержал паузы, и модель искала плеер
    /// среди окон, а музыка играла дальше.
    #[test]
    fn pause_requests_pull_the_music_slice() {
        for text in [
            "поставь на паузу",
            "пауза",
            "останови музыку",
            "выключи музыку",
            "следующий трек",
        ] {
            let prompt = action_prompt_for(text);
            assert!(
                prompt.contains("music_control"),
                "для «{text}» в промпте нет music_control"
            );
            assert!(
                prompt.contains("stop|next"),
                "для «{text}» не описаны действия плеера"
            );
        }
    }

    #[test]
    fn explicit_web_request_detected() {
        for text in [
            "да посмотри в интернете",
            "а что скажет интернет",
            "погугли курс доллара",
            "загугли это",
            "поищи в яндексе",
        ] {
            assert!(explicit_web_request(text), "{text}");
        }
        for text in ["какая погода", "расскажи анекдот", "открой сайт дипсик"]
        {
            assert!(!explicit_web_request(text), "{text}");
        }
    }

    #[test]
    fn web_query_strips_command_words() {
        assert_eq!(
            web_query_from("да посмотри в интернете кто выиграл матч"),
            Some("кто выиграл матч".into())
        );
        // Предмета нет — движок возьмёт последнюю тему из памяти.
        assert_eq!(web_query_from("да посмотри в интернете"), None);
        assert_eq!(
            web_query_from("погугли лариса долина"),
            Some("лариса долина".into())
        );
    }

    #[test]
    fn calendar_questions_skip_web() {
        for text in [
            "а какой сегодня день недели",
            "какой день будет послезавтра",
            "какое сегодня число",
            "какой сейчас год",
        ] {
            assert!(calendar_question(text), "{text}");
        }
        for text in ["какая сегодня погода", "кто выиграл вчера"] {
            assert!(!calendar_question(text), "{text}");
        }
    }

    #[test]
    fn action_prompt_has_no_literal_domains() {
        // Реальная ошибка: домен-пример «deepseek.com» в промпте модель
        // копировала на любой запрос про сайт. Та же болезнь с окнами:
        // пример `"app":"зум"` — на «сверни приложение рио» модель писала
        // «зум». В промпте не должно быть конкретных адресов и имён
        // приложений — только слова из просьбы пользователя.
        let prompt = action_prompt();
        for bad in [
            "deepseek",
            "youtube.com",
            "github.com",
            "http://",
            "www.",
            "зум",
        ] {
            assert!(
                !prompt.to_lowercase().contains(bad),
                "в action-промпте не должно быть домена/примера «{bad}»"
            );
        }
    }

    #[test]
    fn tool_slices_appear_only_when_the_request_needs_them() {
        // Оконный инструмент существует, но модель о нём не знала —
        // «сверни программу» не выполнялся. Теперь просьба сама вытягивает
        // нужный срез в промпт.
        let window = action_prompt_for("сверни зум");
        for must in ["window", "minimize", "set_volume", "screenshot"] {
            assert!(window.contains(must), "в промпте нет «{must}»");
        }
        let desktop = action_prompt_for("введи текст и нажми Ctrl+S в редакторе");
        for must in ["desktop_input", "type_text", "hotkey", "wait"] {
            assert!(desktop.contains(must), "в промпте управления ПК нет «{must}»");
        }
        let timer = action_prompt_for("поставь таймер на 10 минут");
        for must in ["set_timer", "set_alarm"] {
            assert!(timer.contains(must), "в промпте нет «{must}»");
        }
        let message = action_prompt_for("отправь сообщение маме");
        for must in ["send_message", "add_note", "create_event"] {
            assert!(message.contains(must), "в промпте нет «{must}»");
        }
        // А для простого «открой хром» срезы не нужны и в промпт не попадают:
        // каждый лишний инструмент — шум для модели 0.8B.
        let simple = action_prompt_for("открой хром");
        assert!(!simple.contains("set_timer"));
        assert!(!simple.contains("music_control"));
        // Ядро инструментов на месте всегда — это страховка от неверного среза.
        for must in ["shell", "open_app", "open_url", "web_search", "file_write"] {
            assert!(simple.contains(must), "в ядре промпта нет «{must}»");
        }
    }

    // ---------------------------------------------------------------------
    // Детерминированный план: команды, которые не должны зависеть от LLM.
    // ---------------------------------------------------------------------

    fn fake_lookup(name: &str) -> Option<String> {
        [
            "Max",
            "Telegram",
            "Google Chrome",
            "Safari",
            "Zoom",
            "Music",
        ]
        .iter()
        .find(|app| crate::platform::process_matches(app, name))
        .map(|app| (*app).to_string())
    }

    fn plan(prompt: &str) -> Option<crate::tools::ToolCall> {
        plan_system_command_with(prompt, fake_lookup)
    }

    fn open_app(prompt: &str, want: &str) {
        match plan(prompt) {
            Some(crate::tools::ToolCall::OpenApp { name }) => assert_eq!(name, want, "{prompt}"),
            other => panic!("для «{prompt}» ждали open_app {want}, получили {other:?}"),
        }
    }

    fn open_path(prompt: &str, want: &str) {
        match plan(prompt) {
            Some(crate::tools::ToolCall::OpenPath { path }) => assert_eq!(path, want, "{prompt}"),
            other => panic!("для «{prompt}» ждали open_path {want}, получили {other:?}"),
        }
    }

    fn open_url(prompt: &str, want: &str) {
        match plan(prompt) {
            Some(crate::tools::ToolCall::OpenUrl { url, .. }) => assert_eq!(url, want, "{prompt}"),
            other => panic!("для «{prompt}» ждали open_url {want}, получили {other:?}"),
        }
    }

    fn window(prompt: &str, want_action: &str, want_app: Option<&str>) {
        match plan(prompt) {
            Some(crate::tools::ToolCall::Window { action, app }) => {
                assert_eq!(action, want_action, "{prompt}");
                assert_eq!(app.as_deref(), want_app, "{prompt}");
            }
            other => {
                panic!("для «{prompt}» ждали окно {want_action}/{want_app:?}, получили {other:?}")
            }
        }
    }

    fn quit_app(prompt: &str, want: &str) {
        match plan(prompt) {
            Some(crate::tools::ToolCall::QuitApp { name }) => assert_eq!(name, want, "{prompt}"),
            other => panic!("для «{prompt}» ждали quit_app {want}, получили {other:?}"),
        }
    }

    #[test]
    fn planner_opens_installed_apps_with_original_case() {
        open_app("Открой Max", "Max");
        open_app("открой приложение Max", "Max");
        open_app("запусти телеграм", "Telegram");
        open_app("запусти телеграма", "Telegram");
        open_app("открой сафари", "Safari");

        // Имя приложения передаётся в поиск с исходным регистром: настоящий
        // app_lookup сам приводит его к нижнему регистру, а терять «Max» до
        // поиска нельзя.
        let exact = plan_system_command_with("Открой Max", |name| {
            (name == "Max").then(|| "Max".to_string())
        });
        assert!(matches!(
            exact,
            Some(crate::tools::ToolCall::OpenApp { name }) if name == "Max"
        ));
    }

    #[test]
    fn planner_distinguishes_folder_file_path_and_site() {
        open_path("открой папку загрузки", "~/Downloads");
        open_path("Открой загрузки", "~/Downloads");
        open_path("открой документы", "~/Documents");
        open_path("Открой файл README.md", "README.md");
        open_path("открой /Applications", "/Applications");
        open_url("открой сайт example.com", "example.com");
        open_url(
            "Открой https://Example.COM/Path",
            "https://Example.COM/Path",
        );
    }

    #[test]
    fn planner_passes_the_named_browser_to_open_url() {
        match plan("открой aviasales.ru в сафари") {
            Some(crate::tools::ToolCall::OpenUrl { url, browser }) => {
                assert_eq!(url, "aviasales.ru");
                assert_eq!(browser.as_deref(), Some("Safari"));
            }
            other => panic!("unexpected: {other:?}"),
        }
        // Без адреса и с названным браузером приложение не угадывается.
        assert!(plan("открой сайт авиасейлс в сафари").is_none());
    }

    #[test]
    fn planner_does_not_invent_app_or_site() {
        // Незнакомое приложение — не повод открывать одноимённый файл/сайт.
        assert_eq!(plan("открой неизвестноеприложение"), None);
        // «Сайт без домена» решает обучаемый кэш/LLM, а не случайный app lookup.
        assert_eq!(plan("открой сайт нетология"), None);
        // Музыка — управление плеером, не запуск Music.app.
        assert_eq!(plan("включи музыку"), None);
        assert_eq!(plan("запусти трек"), None);
        // Вопросы не исполняем как команды.
        assert_eq!(plan("как открыть телеграм"), None);
        assert_eq!(plan("что открой ютуб значит"), None);
    }

    #[test]
    fn planner_routes_window_actions() {
        window("Сверни окно телеграма", "minimize", Some("Telegram"));
        window("сверни окно голосового помощника", "minimize", None);
        window("разверни окно сафари", "maximize", Some("Safari"));
        window("Окно телеграма сверни", "minimize", Some("Telegram"));
        window("восстанови окно джарвиса", "restore", None);
        window("сверни активное окно", "minimize", Some("активное"));
        window("переключись на телеграм", "focus", Some("Telegram"));
        window("покажи список окон", "list", None);
        window("какие окна открыты", "list", None);
        // Вопрос про окно — не команда.
        assert_eq!(plan("как свернуть окно телеграма"), None);
    }

    #[test]
    fn planner_separates_window_close_from_app_quit() {
        window("закрой окно телеграма", "close", Some("Telegram"));
        quit_app("закрой телеграм", "Telegram");
        quit_app("Закрой Telegram", "Telegram");
        quit_app("закрой приложение телеграм", "Telegram");
        quit_app("заверши телеграм", "Telegram");
        // Чужой файл/папку/сайт не «закрываем» выходом приложения.
        assert_eq!(plan("закрой файл README.md"), None);
        assert_eq!(plan("закрой папку загрузки"), None);
        // Собственный процесс голосовой командой не убиваем.
        assert_eq!(plan("закрой джарвис"), None);
        assert_eq!(
            plan("закрой окно джарвиса"),
            Some(crate::tools::ToolCall::Window {
                action: "close".into(),
                app: None,
            })
        );
    }
}
