//! Память тем: помним, о чём говорили, даже если разговор ушёл в сторону.
//!
//! История диалога живёт только внутри окна прямого слушания — с новым
//! «Джарвис» она очищается, и это правильно. Но бывает так: поговорили про
//! одно, отвлеклись на другое, и незакрытая тема пропала. Пользователь
//! возвращается к ней словами «мы это забыли» — и Джарвис должен продолжить
//! с того места, где остановились, а не начать тему заново.
//!
//! Поэтому память тем живёт отдельно от истории диалога: её не чистит ни
//! возврат к wake word, ни истечение окна прямого слушания.

use std::collections::VecDeque;
use std::sync::Mutex;

/// Сколько последних тем помним.
pub const TOPIC_MEMORY_DEPTH: usize = 6;

/// Начало инструкции возврата к теме. По нему же узнаётся продолжение
/// прежней темы, чтобы не плодить темы-дубликаты.
const RESUME_HEAD: &str = "Вернись к теме";

/// Длина метки темы (первая фраза вопроса), которая уходит в память.
const LABEL_CHARS: usize = 160;

/// Сколько символов прошлого ответа возвращаем в историю диалога.
const ANSWER_CHARS: usize = 600;

/// Инструкция для модели. Саму тему подкладываем в историю диалога, поэтому
/// промпт остаётся коротким и не раздувает контекст.
pub const RESUME_INSTRUCTION: &str = concat!(
    "Вернись к теме, которую мы не закончили, и продолжи её с того места, ",
    "где остановились. Не начинай тему заново и не пересказывай её с нуля — ",
    "ответь по существу."
);

/// Что нужно движку, чтобы вернуться к прежней теме.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resume {
    /// Промпт для модели (короткий, тема лежит в истории).
    pub prompt: String,
    /// Метка темы — для журнала и интерфейса.
    pub topic: String,
    /// Прошлый вопрос по теме.
    pub question: String,
    /// Прошлый ответ по теме — его возвращаем в историю диалога.
    pub answer: String,
}

/// Обмен «вопрос → ответ», который стоит запомнить.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topic {
    /// Первая фраза вопроса: по ней тему узнают в журнале и в памяти.
    pub question: String,
    /// Ответ целиком — его вернём в историю при возврате к теме.
    pub answer: String,
}

#[derive(Default)]
pub struct TopicMemory {
    topics: VecDeque<Topic>,
}

impl TopicMemory {
    pub const fn new() -> Self {
        Self {
            topics: VecDeque::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.topics.is_empty()
    }

    /// Запомнить обмен. Пустые реплики игнорируем, глубина ограничена.
    pub fn remember(&mut self, question: &str, answer: &str) {
        if question.trim().is_empty() || answer.trim().is_empty() {
            return;
        }

        // Возврат к теме — это её продолжение, а не новая тема. Иначе после
        // «вернись к теме» в памяти осталась бы запись без вопроса, и
        // следующий возврат уводил бы совсем не туда.
        if question.trim_start().starts_with(RESUME_HEAD)
            && let Some(last) = self.topics.back_mut()
        {
            last.answer = truncate(answer, ANSWER_CHARS);
            return;
        }

        self.topics.push_back(Topic {
            question: first_sentence(question, LABEL_CHARS),
            answer: truncate(answer, ANSWER_CHARS),
        });
        while self.topics.len() > TOPIC_MEMORY_DEPTH {
            self.topics.pop_front();
        }
    }

    /// Индекс темы, от которой ушли. Если тем одна — она и есть «та самая».
    fn abandoned_index(&self) -> Option<usize> {
        match self.topics.len() {
            0 => None,
            1 => Some(0),
            n => Some(n - 2),
        }
    }

    /// Текущая тема — последняя обсуждённая.
    pub fn current(&self) -> Option<&Topic> {
        self.topics.back()
    }

    /// Незакрытая тема: та, от которой разговор ушёл в сторону.
    pub fn abandoned(&self) -> Option<&Topic> {
        self.abandoned_index().map(|i| &self.topics[i])
    }

    /// Забрать незакрытую тему и сделать её текущей.
    ///
    /// Память приходит в порядок «о чём говорили»: всё, что обсуждали после
    /// возвращаемой темы, пользователь считает забытым и забывает его тоже.
    /// Иначе следующий возврат уводил бы в тот самый уход от темы.
    pub fn resume(&mut self) -> Option<Topic> {
        let idx = self.abandoned_index()?;
        let topic = self.topics.remove(idx)?;
        self.topics.truncate(idx);
        self.topics.push_back(topic.clone());
        Some(topic)
    }

    pub fn clear(&mut self) {
        self.topics.clear();
    }
}

/// Обрезает строку по границе слова, чтобы не разорвать слово.
fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.trim().to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    let head = match cut.rsplit_once(' ') {
        Some((head, _)) if !head.is_empty() => head.to_string(),
        _ => cut,
    };
    format!("{head}…")
}

/// Первая фраза текста — метка темы.
fn first_sentence(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    let end = trimmed
        .find(['.', '!', '?', '\n'])
        .filter(|i| *i > 0)
        .unwrap_or(trimmed.len());
    let head = trimmed[..end].trim();
    truncate(head, max_chars)
}

static MEMORY: Mutex<TopicMemory> = Mutex::new(TopicMemory::new());

/// Запомнить обмен «вопрос → ответ» для будущего возврата к теме.
pub fn remember(question: &str, answer: &str) {
    MEMORY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remember(question, answer);
}

/// Вернуться к незакрытой теме. `None` — возвращаться не к чему.
pub fn resume() -> Option<Resume> {
    let mut mem = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let topic = mem.resume()?;
    Some(Resume {
        prompt: RESUME_INSTRUCTION.to_string(),
        topic: topic.question.clone(),
        question: topic.question,
        answer: topic.answer,
    })
}

/// Последний обсуждённый вопрос — предмет для просьб без темы вроде
/// «да посмотри в интернете»: искать будем то, о чём только что говорили.
pub fn last_question() -> Option<String> {
    let mem = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    mem.current().map(|t| t.question.clone())
}

/// Забыть все темы (пауза, сброс диалога).
pub fn clear() {
    MEMORY.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(q: &str, a: &str) -> (String, String) {
        (q.to_string(), a.to_string())
    }

    #[test]
    fn empty_memory_has_nothing_to_resume() {
        let mut m = TopicMemory::new();
        assert!(m.is_empty());
        assert!(m.resume().is_none());
        assert!(m.abandoned().is_none());
        assert!(m.current().is_none());
    }

    #[test]
    fn single_topic_is_itself_the_abandoned_one() {
        let mut m = TopicMemory::new();
        m.remember("расскажи про марс", "Марс — красная планета");
        assert_eq!(m.abandoned().unwrap().question, "расскажи про марс");
        let t = m.resume().unwrap();
        assert_eq!(t.question, "расскажи про марс");
        assert_eq!(m.topics.len(), 1, "тема не должна дублироваться");
    }

    #[test]
    fn resume_keeps_only_the_older_side_of_the_drift() {
        let mut m = TopicMemory::new();
        m.remember("тема А", "ответ А");
        m.remember("тема Б", "ответ Б");
        m.remember("тема В", "ответ В");
        // Возврат забирает предпоследнюю и обрывает всё, что было после неё.
        assert_eq!(m.resume().unwrap().question, "тема Б");
        assert_eq!(m.topics.len(), 2);
        assert_eq!(m.current().unwrap().question, "тема Б");
    }

    #[test]
    fn resume_returns_to_the_topic_before_the_drift() {
        let mut m = TopicMemory::new();
        m.remember("расскажи про марс", "Марс — красная планета");
        // Ответ ушёл в сторону.
        m.remember("а что скажешь про вторник", "Вторник — тяжёлый день");

        assert_eq!(m.current().unwrap().question, "а что скажешь про вторник");
        assert_eq!(m.abandoned().unwrap().question, "расскажи про марс");

        let t = m.resume().unwrap();
        assert_eq!(t.question, "расскажи про марс");
        assert_eq!(
            t.answer, "Марс — красная планета",
            "в историю должен вернуться прошлый ответ целиком"
        );
        assert_eq!(
            m.current().unwrap().question,
            "расскажи про марс",
            "вернувшись, тема становится текущей"
        );
        assert_eq!(
            m.topics.len(),
            1,
            "уход в сторону пользователь забыл вместе с возвратом"
        );
        assert!(
            !m.topics
                .iter()
                .any(|t| t.question == "а что скажешь про вторник"),
            "брошенная тема не должна мешать следующему возврату"
        );
    }

    #[test]
    fn repeated_resume_keeps_landing_on_the_right_topic() {
        let mut m = TopicMemory::new();
        m.remember("тема А", "ответ А");
        m.remember("уход в сторону", "ответ Б");
        assert_eq!(m.resume().unwrap().question, "тема А");
        // Возврат обновил ответ А, но не создал тему-дубликат.
        m.remember(RESUME_INSTRUCTION, "ответ А продолжение");
        assert_eq!(m.topics.len(), 1);
        assert_eq!(m.current().unwrap().answer, "ответ А продолжение");
        assert_eq!(
            m.resume().unwrap().question,
            "тема А",
            "следующий возврат снова находит тему А"
        );
    }

    #[test]
    fn continuation_updates_instead_of_adding_topic() {
        let mut m = TopicMemory::new();
        m.remember("тема А", "ответ А");
        m.remember(RESUME_INSTRUCTION, "ответ А длиннее");
        assert_eq!(m.topics.len(), 1, "продолжение темы не новая тема");
        assert_eq!(m.current().unwrap().question, "тема А");
        assert_eq!(m.current().unwrap().answer, "ответ А длиннее");
    }

    #[test]
    fn depth_is_capped() {
        let mut m = TopicMemory::new();
        for i in 0..TOPIC_MEMORY_DEPTH + 4 {
            m.remember(&format!("вопрос {i}"), "ответ");
        }
        assert_eq!(m.topics.len(), TOPIC_MEMORY_DEPTH);
        assert_eq!(
            m.current().unwrap().question,
            format!("вопрос {}", TOPIC_MEMORY_DEPTH + 3)
        );
        assert!(
            !m.topics.iter().any(|t| t.question == "вопрос 0"),
            "самые старые темы должны вытесняться"
        );
    }

    #[test]
    fn empty_turns_are_not_remembered() {
        let mut m = TopicMemory::new();
        m.remember("", "ответ");
        m.remember("вопрос", "");
        m.remember("   ", "   ");
        assert!(m.is_empty());
    }

    #[test]
    fn topic_label_is_first_sentence() {
        let mut m = TopicMemory::new();
        m.remember("Расскажи про Марс. И про Венеру тоже.", "Две планеты");
        assert_eq!(m.current().unwrap().question, "Расскажи про Марс");
    }

    #[test]
    fn long_answer_is_truncated_without_splitting_word() {
        let mut m = TopicMemory::new();
        let long = (0..400)
            .map(|i| format!("слово{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        m.remember("вопрос", &long);
        let answer = &m.current().unwrap().answer;
        assert!(answer.chars().count() <= ANSWER_CHARS + 1);
        assert!(answer.ends_with('…'));
        assert!(
            !answer.trim_end_matches('…').ends_with("слов"),
            "обрезать нужно по границе слова"
        );
    }

    #[test]
    fn resume_prompt_is_short_and_carries_no_topic_itself() {
        let mut m = TopicMemory::new();
        m.remember("тема А", "ответ А");
        let r = Resume {
            prompt: RESUME_INSTRUCTION.to_string(),
            topic: m.resume().unwrap().question,
            question: "тема А".into(),
            answer: "ответ А".into(),
        };
        assert!(r.prompt.contains("продолжи её"));
        assert!(
            !r.prompt.contains("ответ А"),
            "тема подкладывается в историю, а не в промпт"
        );
        assert_eq!(r.topic, "тема А");
    }

    #[test]
    fn forgotten_exchange_returns_pair_for_history() {
        let mut m = TopicMemory::new();
        let (q, a) = ask("тема А", "ответ А");
        m.remember(&q, &a);
        let topic = m.resume().unwrap();
        assert_eq!((topic.question, topic.answer), (q, a));
    }
}