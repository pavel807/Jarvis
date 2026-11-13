//! Ожидающий выбор, когда у команды несколько равноправных целей.
//!
//! Пример: «сверни окно хрома», а у Chrome открыто три окна. Молча выбрать
//! первое — значит сделать не то, о чём просили. Вместо этого инструмент
//! оставляет здесь запрос, озвучивает варианты и ждёт ответа: голосом
//! («второе», «2», название окна) или кликом в интерфейсе.
//!
//! Запрос живёт ограниченное время и не переживает новый выбор: если
//! пользователь передумал и сказал другую команду, ожидание снимается, а
//! фраза уходит обычным путём.

use crate::events::{ChoiceOptionView, EventTx, JarvisEvent};
use crate::llm;
use crate::tts::AudioMessage;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Сколько ждать ответа. Дольше, чем обычное окно уточнения: человеку нужно
/// прочитать варианты (а с длинными заголовками — ещё и выбрать).
pub const TTL: Duration = Duration::from_secs(60);

/// Больше девяти вариантов голосом всё равно не выбрать.
const MAX_OPTIONS: usize = 9;

/// Один вариант выбора. `label` — то, что видит и произносит пользователь.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoiceOption {
    pub label: String,
}

impl ChoiceOption {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

/// Итог действия, выбранного пользователем.
#[derive(Debug, Clone)]
pub struct ChoiceActionResult {
    pub ok: bool,
    pub speech: String,
}

impl ChoiceActionResult {
    pub fn ok(speech: impl Into<String>) -> Self {
        Self {
            ok: true,
            speech: speech.into(),
        }
    }

    pub fn fail(speech: impl Into<String>) -> Self {
        Self {
            ok: false,
            speech: speech.into(),
        }
    }
}

type Action = Box<dyn FnOnce(usize) -> ChoiceActionResult + Send>;

struct Pending {
    id: u64,
    created: Instant,
    prompt: String,
    options: Vec<ChoiceOption>,
    action: Action,
    audio_tx: Sender<AudioMessage>,
    events: EventTx,
}

/// Публичное описание ожидающего выбора (без действия и каналов).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingChoice {
    pub id: u64,
    pub prompt: String,
    pub options: Vec<String>,
}

/// Что произошло при попытке разрешить выбор голосом.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChoiceResolution {
    /// Ожидающего выбора не было — обычный путь команды.
    NotPending,
    /// Выбор сделан, действие выполнено, ответ уже озвучен.
    Picked {
        index: usize,
        label: String,
        speech: String,
        ok: bool,
    },
    /// Пользователь явно отменил выбор.
    Cancelled { speech: String },
    /// Выбор был, но фраза не похожа ни на вариант, ни на отмену. Запрос
    /// снят, чтобы не перехватывать следующую команду; вызывающий код
    /// продолжает обычную обработку.
    Unrecognized,
}

static SLOT: OnceLock<Mutex<Option<Pending>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn slot() -> &'static Mutex<Option<Pending>> {
    SLOT.get_or_init(|| Mutex::new(None))
}

fn lock_slot() -> std::sync::MutexGuard<'static, Option<Pending>> {
    slot().lock().unwrap_or_else(|e| e.into_inner())
}

/// Предложить выбор. Предыдущий ожидающий выбор снимается: новый запрос
/// важнее старого. `action` получает индекс выбранного варианта (с нуля).
pub fn offer(
    prompt: impl Into<String>,
    options: Vec<ChoiceOption>,
    audio_tx: Sender<AudioMessage>,
    events: EventTx,
    action: impl FnOnce(usize) -> ChoiceActionResult + Send + 'static,
) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let options: Vec<ChoiceOption> = options.into_iter().take(MAX_OPTIONS).collect();
    let prompt = prompt.into();
    let pending = Pending {
        id,
        created: Instant::now(),
        prompt: prompt.clone(),
        options: options.clone(),
        action: Box::new(action),
        audio_tx,
        events: events.clone(),
    };

    let previous = {
        let mut guard = lock_slot();
        guard.replace(pending)
    };
    if let Some(old) = previous {
        let _ = events.send(JarvisEvent::ChoiceResolved {
            choice_id: old.id,
            index: None,
            label: None,
        });
    }
    let _ = events.send(JarvisEvent::ChoiceRequested {
        choice_id: id,
        prompt,
        options: options
            .iter()
            .enumerate()
            .map(|(index, option)| ChoiceOptionView {
                index,
                label: option.label.clone(),
            })
            .collect(),
    });
    id
}

/// Ожидающий выбор, если он ещё не истёк.
pub fn pending() -> Option<PendingChoice> {
    let expired = {
        let mut guard = lock_slot();
        let Some(pending) = guard.as_ref() else {
            return None;
        };
        if pending.created.elapsed() >= TTL {
            guard.take()
        } else {
            return Some(PendingChoice {
                id: pending.id,
                prompt: pending.prompt.clone(),
                options: pending.options.iter().map(|o| o.label.clone()).collect(),
            });
        }
    };
    if let Some(old) = expired {
        let _ = old.events.send(JarvisEvent::ChoiceResolved {
            choice_id: old.id,
            index: None,
            label: None,
        });
    }
    None
}

/// Сколько осталось до истечения ожидания. `None` — выбора нет.
pub fn remaining() -> Option<Duration> {
    let expired = {
        let mut guard = lock_slot();
        let pending = guard.as_ref()?;
        let left = TTL.saturating_sub(pending.created.elapsed());
        if left.is_zero() {
            guard.take()
        } else {
            return Some(left);
        }
    };
    if let Some(old) = expired {
        let _ = old.events.send(JarvisEvent::ChoiceResolved {
            choice_id: old.id,
            index: None,
            label: None,
        });
    }
    None
}

/// Снять истёкшее ожидание (вызывается движком при закрытии окна слушания).
pub fn expire() {
    let expired = lock_slot().take();
    if let Some(old) = expired {
        let _ = old.events.send(JarvisEvent::ChoiceResolved {
            choice_id: old.id,
            index: None,
            label: None,
        });
    }
}

/// Разрешить выбор голосом. Возвращает [`ChoiceResolution::NotPending`],
/// если ждать было нечего.
pub fn resolve_text(text: &str) -> ChoiceResolution {
    let normalized = normalize(text);
    let expired = {
        let guard = lock_slot();
        guard
            .as_ref()
            .is_some_and(|pending| pending.created.elapsed() >= TTL)
    };
    if expired {
        expire();
        return ChoiceResolution::NotPending;
    }
    let selected = {
        let mut guard = lock_slot();
        let Some(pending) = guard.as_ref() else {
            return ChoiceResolution::NotPending;
        };
        if is_cancel(&normalized) {
            guard.take().map(|p| (p, None))
        } else {
            match parse_pick(&normalized, &pending.options) {
                Some(index) => guard.take().map(|p| (p, Some(index))),
                None => guard.take().map(|p| (p, Some(usize::MAX))),
            }
        }
    };
    let Some((pending, choice)) = selected else {
        return ChoiceResolution::NotPending;
    };
    match choice {
        Some(index) if index != usize::MAX => finish_picked(pending, index),
        Some(_) => {
            // Не похоже на вариант: снимаем запрос, но не озвучиваем отмену —
            // следующая фраза должна обработаться как обычная команда.
            let _ = pending.events.send(JarvisEvent::ChoiceResolved {
                choice_id: pending.id,
                index: None,
                label: None,
            });
            ChoiceResolution::Unrecognized
        }
        None => finish_cancelled(pending),
    }
}

/// Разрешить выбор кликом в интерфейсе.
pub fn resolve_index(choice_id: u64, index: usize) -> Result<String, String> {
    let pending = {
        let mut guard = lock_slot();
        let Some(pending) = guard.as_ref() else {
            return Err("Выбор уже неактуален".to_string());
        };
        if pending.id != choice_id {
            return Err("Этот выбор уже неактуален".to_string());
        }
        if index >= pending.options.len() {
            return Err("Такого варианта нет".to_string());
        }
        guard.take()
    };
    let Some(pending) = pending else {
        return Err("Выбор уже неактуален".to_string());
    };
    match finish_picked(pending, index) {
        ChoiceResolution::Picked { speech, .. } => Ok(speech),
        _ => Err("Не удалось выполнить выбранное действие".to_string()),
    }
}

fn finish_picked(pending: Pending, index: usize) -> ChoiceResolution {
    let label = pending
        .options
        .get(index)
        .map(|option| option.label.clone())
        .unwrap_or_default();
    let result = (pending.action)(index);
    let speech = if result.speech.trim().is_empty() {
        "Готово".to_string()
    } else {
        result.speech.trim().to_string()
    };

    let _ = pending.events.send(JarvisEvent::ChoiceResolved {
        choice_id: pending.id,
        index: Some(index),
        label: Some(label.clone()),
    });
    let _ = pending.events.send(JarvisEvent::ActionResult {
        ok: result.ok,
        summary: speech.clone(),
    });
    llm::speak(&pending.audio_tx, &pending.events, &speech);

    ChoiceResolution::Picked {
        index,
        label,
        speech,
        ok: result.ok,
    }
}

fn finish_cancelled(pending: Pending) -> ChoiceResolution {
    let speech = "Отменил выбор.".to_string();
    let _ = pending.events.send(JarvisEvent::ChoiceResolved {
        choice_id: pending.id,
        index: None,
        label: None,
    });
    llm::speak(&pending.audio_tx, &pending.events, &speech);
    ChoiceResolution::Cancelled { speech }
}

/// Нормализация: нижний регистр, ё → е, только буквы/цифры и пробелы.
fn normalize(text: &str) -> String {
    let lower = text.to_lowercase().replace('ё', "е");
    lower
        .split(|c: char| !c.is_alphanumeric() && c != ' ')
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_cancel(text: &str) -> bool {
    const CANCEL: &[&str] = &[
        "отмена",
        "отмени",
        "отменить",
        "стоп",
        "нет",
        "не надо",
        "ничего",
        "передумал",
        "передумала",
        "не сейчас",
        "забудь",
    ];
    CANCEL.iter().any(|word| text == *word || text.starts_with(&format!("{word} ")))
}

/// Индекс варианта из фразы: цифра, порядковое слово или название.
fn parse_pick(text: &str, options: &[ChoiceOption]) -> Option<usize> {
    if options.is_empty() {
        return None;
    }

    // Число отдельным словом: «2», «номер 2», «вариант 2».
    for token in text.split_whitespace() {
        let token = token.trim_matches(|c: char| !c.is_alphanumeric());
        if let Ok(number) = token.parse::<usize>()
            && (1..=options.len()).contains(&number)
        {
            return Some(number - 1);
        }
    }

    // Количественные слова: «два», «три» — естественный ответ на «какое
    // окно?». Голосовой ввод часто пишет число словом, а не цифрой.
    const CARDINALS: &[(&str, usize)] = &[
        ("один", 0),
        ("одно", 0),
        ("одна", 0),
        ("два", 1),
        ("две", 1),
        ("три", 2),
        ("четыре", 3),
        ("пять", 4),
        ("шесть", 5),
        ("семь", 6),
        ("восемь", 7),
        ("девять", 8),
        ("десять", 9),
    ];
    for (word, index) in CARDINALS {
        if *index < options.len() && text.split_whitespace().any(|token| token == *word) {
            return Some(*index);
        }
    }

    // Порядковые слова: «первое», «второй», «третью»…
    const ORDINALS: &[&str] = &[
        "перв", "втор", "трет", "четверт", "пят", "шест", "седьм", "восьм", "девят", "десят",
    ];
    for (index, stem) in ORDINALS.iter().enumerate() {
        if index >= options.len() {
            break;
        }
        if text
            .split_whitespace()
            .any(|word| word.starts_with(stem))
        {
            return Some(index);
        }
    }

    // Название варианта: точное вхождение или вхождение фразы в название.
    // Проверяем от самого длинного варианта, чтобы «окно 1» не выиграло у
    // «окно 10» (в текущем наборе номеров не больше девяти, но правило
    // оставляем безопасным).
    let mut order: Vec<usize> = (0..options.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(options[index].label.chars().count()));
    let mut matched: Vec<usize> = Vec::new();
    for index in order {
        let label = normalize(&options[index].label);
        if label.is_empty() {
            continue;
        }
        if text.contains(&label) || (text.chars().count() >= 3 && label.contains(text)) {
            matched.push(index);
        }
    }
    // Название приложения вместо заголовка: «хром» → Google Chrome.
    // Пользователь часто помнит приложение, а не заголовок окна.
    let aliased: Vec<String> = text
        .split_whitespace()
        .filter_map(crate::platform::alias)
        .map(normalize)
        .filter(|target| target.chars().count() >= 3)
        .collect();
    if !aliased.is_empty() {
        for index in 0..options.len() {
            if matched.contains(&index) {
                continue;
            }
            let label = normalize(&options[index].label);
            if aliased.iter().any(|target| label.contains(target)) {
                matched.push(index);
            }
        }
    }
    // Фраза подошла к нескольким вариантам — это не выбор, а уточнение
    // «хром» при трёх окнах Chrome. Молча брать первое нельзя.
    match matched.as_slice() {
        [index] => Some(*index),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    use std::sync::{Arc, Mutex};

    /// Глобальный слот выбора один на процесс: тесты, которые его трогают,
    /// не должны идти параллельно.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn options() -> Vec<ChoiceOption> {
        vec![
            ChoiceOption::new("Документация — Google Chrome"),
            ChoiceOption::new("Почта — Google Chrome"),
            ChoiceOption::new("Новая вкладка — Google Chrome"),
        ]
    }

    #[test]
    fn parses_number_ordinal_and_title() {
        let opts = options();
        assert_eq!(parse_pick("2", &opts), Some(1));
        assert_eq!(parse_pick("номер 3", &opts), Some(2));
        // Голосовой ввод часто отдаёт число словом.
        assert_eq!(parse_pick("два", &opts), Some(1));
        assert_eq!(parse_pick("две", &opts), Some(1));
        assert_eq!(parse_pick("три окна", &opts), Some(2));
        assert_eq!(parse_pick("второе", &opts), Some(1));
        assert_eq!(parse_pick("третье окно", &opts), Some(2));
        assert_eq!(parse_pick("почта", &opts), Some(1));
        assert_eq!(parse_pick("новая вкладка", &opts), Some(2));
        assert_eq!(parse_pick("не понимаю", &opts), None);
        // Название приложения есть у всех вариантов — это не выбор окна.
        assert_eq!(parse_pick("google chrome", &opts), None);
        assert_eq!(parse_pick("хром", &opts), None);
    }

    #[test]
    fn cancel_words_are_explicit() {
        for phrase in ["отмена", "Отмени", "стоп", "нет", "не надо"] {
            assert!(is_cancel(&normalize(phrase)), "должно отмениться: {phrase}");
        }
        assert!(!is_cancel("второе"));
    }

    #[test]
    fn voice_and_ui_resolution_run_action_once() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (audio_tx, audio_rx) = channel();
        let (events, event_rx) = channel();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let calls_for_action = Arc::clone(&calls);
        let id = offer(
            "Какое окно свернуть?",
            options(),
            audio_tx,
            events,
            Box::new(move |index| {
                calls_for_action.lock().unwrap().push(index);
                ChoiceActionResult::ok(format!("свернул вариант {index}"))
            }),
        );
        assert!(pending().is_some());
        let resolution = resolve_text("второе");
        match resolution {
            ChoiceResolution::Picked { index, ok, .. } => {
                assert_eq!(index, 1);
                assert!(ok);
            }
            other => panic!("ожидался выбор, получено {other:?}"),
        }
        assert_eq!(*calls.lock().unwrap(), vec![1]);
        assert!(pending().is_none());
        // Ответ ушёл в озвучку и в UI.
        assert!(matches!(
            audio_rx.try_recv(),
            Ok(AudioMessage::SynthesizeAndPlay(_))
        ));
        let mut saw_resolved = false;
        while let Ok(event) = event_rx.try_recv() {
            if matches!(event, JarvisEvent::ChoiceResolved { .. }) {
                saw_resolved = true;
            }
        }
        assert!(saw_resolved);

        // Клик по устаревшему выбору — честная ошибка, а не повторное действие.
        assert!(resolve_index(id, 1).is_err());
        assert!(calls.lock().unwrap().len() == 1);
    }

    #[test]
    fn unknown_phrase_clears_choice_without_action() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (audio_tx, _audio_rx) = channel();
        let (events, _event_rx) = channel();
        let calls = Arc::new(Mutex::new(0usize));
        let calls_for_action = Arc::clone(&calls);
        offer(
            "Что выбрать?",
            options(),
            audio_tx,
            events,
            Box::new(move |_| {
                *calls_for_action.lock().unwrap() += 1;
                ChoiceActionResult::ok("готово")
            }),
        );
        assert_eq!(resolve_text("расскажи шутку"), ChoiceResolution::Unrecognized);
        assert!(pending().is_none());
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn index_resolution_works() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (audio_tx, _audio_rx) = channel();
        let (events, _event_rx) = channel();
        let id = offer(
            "Что выбрать?",
            options(),
            audio_tx,
            events,
            Box::new(|index| ChoiceActionResult::ok(format!("выбрано {index}"))),
        );
        assert_eq!(resolve_index(id, 2).unwrap(), "выбрано 2");
        assert!(pending().is_none());
    }
}
