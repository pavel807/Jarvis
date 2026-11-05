//! Задачник: markdown-файл с чекбоксами, который Jarvis и пользователь
//! правят одновременно.
//!
//! Формат строки задачи: `- [ ] текст` / `- [x] текст`. Любые другие
//! строки (заголовки, заметки, списки без чекбоксов) сохраняются как есть —
//! файл остаётся валидным markdown, который удобно открыть в редакторе.
//!
//! Номер задачи (`id`) — её позиция в списке, начиная с 1. Он сквозной,
//! нумерация одинакова для списка голосом и для промпта модели.

use crate::paths;
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const HEADER: &str = "# Задачи\n";

/// Одна задача из задачника.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    /// Сквозной номер, начиная с 1.
    pub id: usize,
    /// Текст задачи без маркера чекбокса.
    pub text: String,
    pub done: bool,
}

/// Разбирает строку markdown в `(готово, текст)`, если это чекбокс.
fn parse_line(line: &str) -> Option<(bool, &str)> {
    let t = line.trim_start();
    let rest = t.strip_prefix("- ").or_else(|| t.strip_prefix("* "))?;
    if let Some(tail) = rest.strip_prefix("[ ]") {
        return Some((false, tail.trim()));
    }
    if let Some(tail) = rest
        .strip_prefix("[x]")
        .or_else(|| rest.strip_prefix("[X]"))
    {
        return Some((true, tail.trim()));
    }
    None
}

/// Читает все задачи из содержимого файла.
pub fn parse(content: &str) -> Vec<Task> {
    let mut tasks = Vec::new();
    for line in content.lines() {
        if let Some((done, text)) = parse_line(line) {
            if text.is_empty() {
                continue;
            }
            tasks.push(Task {
                id: tasks.len() + 1,
                text: text.to_string(),
                done,
            });
        }
    }
    tasks
}

fn read_lines(path: &Path) -> Result<Vec<String>> {
    if !path.exists() {
        return Ok(vec![HEADER.trim_end().to_string(), String::new()]);
    }
    let content =
        fs::read_to_string(path).with_context(|| format!("Не удалось прочитать {:?}", path))?;
    Ok(content.lines().map(str::to_string).collect())
}

fn write_lines(path: &Path, lines: &[String]) -> Result<()> {
    let mut body = lines.join("\n");
    if !body.ends_with('\n') {
        body.push('\n');
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let tmp = path.with_extension("md.tmp");
    fs::write(&tmp, body).with_context(|| format!("Не удалось записать {:?}", tmp))?;
    fs::rename(&tmp, path).with_context(|| format!("Не удалось заменить {:?}", path))?;
    Ok(())
}

/// Путь к файлу задачника.
pub fn path() -> PathBuf {
    paths::tasks_file()
}

/// Все задачи (и готовые, и нет) в порядке файла.
pub fn load() -> Vec<Task> {
    let path = path();
    match fs::read_to_string(&path) {
        Ok(content) => parse(&content),
        Err(_) => Vec::new(),
    }
}

/// Только невыполненные задачи.
pub fn pending() -> Vec<Task> {
    load().into_iter().filter(|t| !t.done).collect()
}

/// Находит задачу по сквозному номеру.
pub fn find(id: usize) -> Option<Task> {
    load().into_iter().find(|t| t.id == id)
}

/// Добавляет новую задачу в конец списка.
pub fn add(text: &str) -> Result<Task> {
    // Если модель уже обернула текст в чекбокс — не дублируем маркер.
    let text = match parse_line(text) {
        Some((_, t)) => t,
        None => text.trim(),
    };
    if text.is_empty() {
        bail!("Пустая задача");
    }
    let path = path();
    let mut lines = read_lines(&path)?;
    lines.push(format!("- [ ] {text}"));
    write_lines(&path, &lines)?;
    let last = load().last().cloned();
    last.context("Не удалось прочитать задачник после записи")
}

/// Ставит или снимает галочку. `None`, если задачи с таким номером нет.
pub fn set_done(id: usize, done: bool) -> Option<Task> {
    let path = path();
    let mut lines = match read_lines(&path) {
        Ok(l) => l,
        Err(_) => return None,
    };

    let mut counter = 0usize;
    let mut updated: Option<Task> = None;
    for line in lines.iter_mut() {
        let Some((was_done, text)) = parse_line(line).map(|(d, t)| (d, t.to_string())) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        counter += 1;
        if counter != id {
            continue;
        }
        let mark = if done { "x" } else { " " };
        let rest = line.trim_start();
        let bullet = if rest.starts_with('*') { "* " } else { "- " };
        *line = format!("{bullet}[{mark}] {text}");
        updated = Some(Task { id, text, done });
        let _ = was_done;
    }

    write_lines(&path, &lines).ok()?;
    updated
}

/// Удаляет строку задачи. `None`, если задачи с таким номером нет.
pub fn remove(id: usize) -> Option<Task> {
    let path = path();
    let mut lines = match read_lines(&path) {
        Ok(l) => l,
        Err(_) => return None,
    };

    let mut counter = 0usize;
    let mut target: Option<usize> = None;
    let mut removed: Option<Task> = None;
    for (idx, line) in lines.iter().enumerate() {
        let Some((done, text)) = parse_line(line).map(|(d, t)| (d, t.to_string())) else {
            continue;
        };
        if text.is_empty() {
            continue;
        }
        counter += 1;
        if counter == id {
            target = Some(idx);
            removed = Some(Task { id, text, done });
            break;
        }
    }

    let idx = target?;
    lines.remove(idx);
    write_lines(&path, &lines).ok()?;
    removed
}

/// Ответ голосом на «что в задачнике»: сколько всего и что дальше делать.
pub fn speak_list(limit: usize) -> String {
    let all = load();
    let open = all.iter().filter(|t| !t.done).count();
    if all.is_empty() {
        return "Задачник пуст. Скажи «добавь задачу».".to_string();
    }
    let mut parts = vec![format!("Всего задач: {}, невыполненных: {open}", all.len())];
    for t in all.iter().filter(|t| !t.done).take(limit) {
        parts.push(format!("Задача {}: {}", t.id, t.text));
    }
    parts.join(". ")
}

/// Ответ голосом на «что делать дальше».
pub fn speak_next() -> String {
    match pending().first() {
        Some(t) => format!("Следующая задача {}: {}", t.id, t.text),
        None => "Невыполненных задач нет".to_string(),
    }
}

// =========================================================================
// Ожидание подтверждения «задача выполнена? да/нет»
// =========================================================================

/// Задача, по которой ждём голосовой ответ.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectDone {
    pub task_id: usize,
    pub task_text: String,
}

/// Слот живёт на уровне процесса: цикл LLM ставит ожидание, а движок речи
/// его проверяет в следующей реплике пользователя.
static EXPECT_DONE: Mutex<Option<ExpectDone>> = Mutex::new(None);

/// Запомнить, что ждём «да/нет» по задаче.
pub fn expect_done(task_id: usize, task_text: &str) {
    let mut slot = EXPECT_DONE.lock().unwrap_or_else(|e| e.into_inner());
    *slot = Some(ExpectDone {
        task_id,
        task_text: task_text.to_string(),
    });
}

/// Забрать ожидание (сбрасывается при любом ответе пользователя).
pub fn take_expect_done() -> Option<ExpectDone> {
    let mut slot = EXPECT_DONE.lock().unwrap_or_else(|e| e.into_inner());
    slot.take()
}

/// Проверить, ждём ли сейчас подтверждение, не сбрасывая его.
pub fn expecting_done() -> bool {
    EXPECT_DONE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

/// Разбор ответа «да/нет» на вопрос о выполнении задачи.
pub fn is_affirmative(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    let t = crate::text::clean_text(&t);
    let words: Vec<&str> = t.split_whitespace().collect();
    words.iter().any(|w| {
        matches!(
            *w,
            "да" | "ага"
                | "дай"
                | "конечно"
                | "точно"
                | "все"
                | "всё"
                | "хорошо"
                | "ок"
                | "окей"
                | "okay"
                | "yes"
                | "y"
                | "сделал"
                | "сделано"
                | "выполнено"
                | "готов"
                | "готово"
        )
    })
}

pub fn is_negative(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    let t = crate::text::clean_text(&t);
    let words: Vec<&str> = t.split_whitespace().collect();
    words.iter().any(|w| {
        matches!(
            *w,
            "нет"
                | "не"
                | "неа"
                | "ненет"
                | "пока"
                | "позже"
                | "no"
                | "n"
                | "отмена"
                | "отменить"
                | "оставить"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_checkboxes() {
        let content = "# Задачи\n\n- [ ] купить кофе\n- [x] помыть посуду\nпросто текст\n";
        let tasks = parse(content);
        assert_eq!(tasks.len(), 2);
        assert_eq!(
            tasks[0],
            Task {
                id: 1,
                text: "купить кофе".into(),
                done: false
            }
        );
        assert_eq!(
            tasks[1],
            Task {
                id: 2,
                text: "помыть посуду".into(),
                done: true
            }
        );
    }

    #[test]
    fn parses_star_bullet_and_tight_marker() {
        let tasks = parse("* [X] сделано\n- [x]тоже сделано\n- [ ] не сделано\n");
        assert_eq!(tasks.len(), 3);
        assert!(tasks[0].done);
        assert_eq!(tasks[0].text, "сделано");
        assert!(tasks[1].done);
        assert_eq!(tasks[1].text, "тоже сделано");
        assert!(!tasks[2].done);
    }

    #[test]
    fn ignores_plain_lists_and_empty_boxes() {
        let tasks = parse("- пункт без чекбокса\n- [ ] \n1. нумерованный\n");
        assert!(tasks.is_empty());
    }

    #[test]
    fn ids_are_stable_across_done_items() {
        let tasks = parse("- [ ] a\n- [x] b\n- [ ] c\n");
        assert_eq!(
            tasks.iter().map(|t| t.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn recognises_yes_and_no() {
        assert!(is_affirmative("да"));
        assert!(is_affirmative("да, всё сделал"));
        assert!(is_affirmative("готово"));
        assert!(!is_affirmative("ещё нет"));

        assert!(is_negative("нет"));
        assert!(is_negative("нет, не закончил"));
        assert!(!is_negative("да"));
    }
}

#[cfg(test)]
mod fs_tests {
    use super::*;

    /// JARVIS_HOME — переменная процесса, поэтому тесты с файлами идут
    /// строго по одному и каждый работает в своей временной директории.
    static FS_LOCK: Mutex<()> = Mutex::new(());

    struct TempHome {
        dir: PathBuf,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl TempHome {
        fn new(tag: &str) -> Self {
            let guard = FS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::temp_dir().join(format!("jarvis-tasks-test-{tag}"));
            let _ = fs::remove_dir_all(&dir);
            unsafe { std::env::set_var("JARVIS_HOME", &dir) };
            TempHome { dir, _guard: guard }
        }
    }

    impl Drop for TempHome {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
            unsafe { std::env::remove_var("JARVIS_HOME") };
        }
    }

    #[test]
    fn add_list_and_close() {
        let _home = TempHome::new("add-list-close");

        assert!(load().is_empty());
        let t1 = add("купить кофе").expect("задача добавлена");
        let t2 = add("позвонить маме").expect("задача добавлена");
        assert_eq!(t1.id, 1);
        assert_eq!(t2.id, 2);

        // Второй вызов не затирает первый, файл читается заново.
        assert_eq!(load().len(), 2);
        assert_eq!(pending().len(), 2);

        let closed = set_done(1, true).expect("галочка стоит");
        assert_eq!(closed.text, "купить кофе");
        assert_eq!(pending().len(), 1);
        assert_eq!(pending()[0].text, "позвонить маме");

        // Снятие галочки.
        assert!(!set_done(1, false).expect("галочка снята").done);
        assert_eq!(pending().len(), 2);
    }

    #[test]
    fn add_keeps_foreign_lines_and_survives_reread() {
        let _home = TempHome::new("foreign-lines");

        add("первая").expect("ok");
        // Пользователь дописал в файл свои строки руками.
        let file = path();
        let mut content = fs::read_to_string(&file).expect("файл есть");
        content.push_str("# Заметки\n\n- пункт без чекбокса\n");
        fs::write(&file, content).expect("запись");

        let t = add("вторая").expect("ok");
        assert_eq!(t.id, 2, "нумерация не сбилась");
        let content = fs::read_to_string(&file).expect("файл есть");
        assert!(content.contains("# Заметки"));
        assert!(content.contains("- пункт без чекбокса"));
    }

    #[test]
    fn add_strips_checkbox_prefix() {
        let _home = TempHome::new("checkbox-prefix");

        let t = add("- [ ] купить кофе").expect("ok");
        assert_eq!(t.text, "купить кофе");
        let content = fs::read_to_string(path()).expect("файл есть");
        assert!(content.contains("- [ ] купить кофе"));
        assert!(!content.contains("[ ] - [ ]"));
    }

    #[test]
    fn remove_deletes_line_and_renumbers() {
        let _home = TempHome::new("remove");

        add("раз").expect("ok");
        add("два").expect("ok");
        add("три").expect("ok");

        let gone = remove(2).expect("удалено");
        assert_eq!(gone.text, "два");
        assert_eq!(load().len(), 2);
        assert_eq!(load()[1].text, "три");
        assert!(remove(99).is_none());
    }

    #[test]
    fn set_done_reports_missing_task() {
        let _home = TempHome::new("missing");

        assert!(set_done(7, true).is_none());
    }
}
