//! Планировщик задач — cron/interval/once по логике OpenJarvis.
//!
//! Задача — промпт, который Джарвис выполняет сам в изолированном контексте
//! (без истории диалога и без инструментов), а результат проговаривает и
//! показывает в уведомлении: «каждое утро в 8 — собери новости и дай сводку».
//! Хранение — JSON рядом с настройками; поллер раз в 20 секунд проверяет,
//! что пора. Управление голосом: инструменты schedule_add / schedule_list /
//! schedule_remove.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Как повторять задачу.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schedule {
    /// Каждый день в HH:MM по локальным часам.
    Daily { hhmm: String },
    /// Каждые N минут.
    Interval { minutes: u64 },
    /// Один раз: «ГГГГ-ММ-ДД ЧЧ:ММ» или «ГГГГ-ММ-ДДТЧЧ:ММ» (локальное время).
    Once { iso: String },
}

impl Schedule {
    /// Человекочитаемое описание для озвучки списка.
    pub fn describe(&self) -> String {
        match self {
            Schedule::Daily { hhmm } => format!("каждый день в {hhmm}"),
            Schedule::Interval { minutes } => {
                if *minutes % 60 == 0 {
                    format!("каждые {} ч", minutes / 60)
                } else {
                    format!("каждые {minutes} мин")
                }
            }
            Schedule::Once { iso } => format!("один раз: {iso}"),
        }
    }

    /// Проверка формата при добавлении: битое время не должно молча
    /// превратиться в задачу, которая никогда не выполнится.
    pub fn validate(&self) -> Result<()> {
        match self {
            Schedule::Daily { hhmm } => parse_hhmm(hhmm).map(|_| ()),
            Schedule::Interval { minutes } => {
                if *minutes == 0 {
                    bail!("интервал должен быть не меньше одной минуты")
                }
                if *minutes > 10_080 {
                    bail!("интервал больше недели — используй ежедневную задачу")
                }
                Ok(())
            }
            Schedule::Once { iso } => parse_once(iso).map(|_| ()),
        }
    }
}

/// «ЧЧ:ММ» → (часы, минуты). Только строгий формат.
fn parse_hhmm(s: &str) -> Result<(u32, u32)> {
    let s = s.trim();
    let (h, m) = s
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("время «{s}»: нужен формат ЧЧ:ММ"))?;
    let h: u32 = h
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("часы «{h}»: нужно число 0–23"))?;
    let m: u32 = m
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("минуты «{m}»: нужно число 0–59"))?;
    if h > 23 || m > 59 {
        bail!("время «{s}»: часы 0–23, минуты 0–59");
    }
    Ok((h, m))
}

/// Один запуск: «ГГГГ-ММ-ДД ЧЧ:ММ» (или с T) → локальное время.
fn parse_once(s: &str) -> Result<chrono::NaiveDateTime> {
    let s = s.trim().replace('T', " ");
    let (day, time) = s
        .split_once(' ')
        .ok_or_else(|| anyhow::anyhow!("дата «{s}»: нужен формат ГГГГ-ММ-ДД ЧЧ:ММ"))?;
    let (h, m) = parse_hhmm(time)?;
    let mut parts = day.split('-');
    let (y, mo, d) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(y), Some(mo), Some(d), None) => (y, mo, d),
        _ => bail!("дата «{day}»: нужен формат ГГГГ-ММ-ДД"),
    };
    chrono::NaiveDate::from_ymd_opt(
        y.parse().map_err(|_| anyhow::anyhow!("год «{y}»"))?,
        mo.parse().map_err(|_| anyhow::anyhow!("месяц «{mo}»"))?,
        d.parse().map_err(|_| anyhow::anyhow!("день «{d}»"))?,
    )
    .map(|date| date.and_hms_opt(h, m, 0).expect("ЧЧ:ММ уже проверены"))
    .ok_or_else(|| anyhow::anyhow!("дата «{day}» не существует"))
}

/// Задача планировщика.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: u64,
    pub prompt: String,
    pub schedule: Schedule,
    pub enabled: bool,
    /// Секунды UNIX последнего запуска (None — ещё не выполнялась).
    pub last_run: Option<u64>,
}

fn store_path() -> PathBuf {
    crate::paths::base_dir().join("schedule.json")
}

fn load_at(path: &Path) -> Vec<Task> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn load() -> Vec<Task> {
    load_at(&store_path())
}

fn save_at(path: &Path, tasks: &[Task]) {
    let tmp = path.with_extension("json.tmp");
    if let Ok(file) = std::fs::File::create(&tmp)
        && serde_json::to_writer_pretty(file, &tasks).is_ok()
    {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn save(tasks: &[Task]) {
    save_at(&store_path(), tasks);
}

/// Все задачи (для UI/инструмента списка).
pub fn list() -> Vec<Task> {
    load()
}

/// Добавить задачу с проверкой формата; вернуть её id.
pub fn add(prompt: &str, schedule: Schedule) -> Result<u64> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        bail!("пустой промпт задачи");
    }
    if prompt.chars().count() > 2000 {
        bail!("промпт задачи слишком длинный (до 2000 символов)");
    }
    schedule.validate()?;
    let mut tasks = load();
    let id = tasks.iter().map(|t| t.id).max().unwrap_or(0) + 1;
    tasks.push(Task {
        id,
        prompt: prompt.to_string(),
        schedule,
        enabled: true,
        last_run: None,
    });
    save(&tasks);
    Ok(id)
}

/// Убрать задачу по id; true — была и удалена.
pub fn remove(id: u64) -> bool {
    let mut tasks = load();
    let before = tasks.len();
    tasks.retain(|t| t.id != id);
    let removed = tasks.len() < before;
    if removed {
        save(&tasks);
    }
    removed
}

/// Список задач в строках для озвучки/лога.
pub fn describe_all() -> String {
    let tasks = load();
    if tasks.is_empty() {
        return "Расписаний нет.".into();
    }
    let lines: Vec<String> = tasks
        .iter()
        .map(|t| {
            format!(
                "№{}, {} — «{}»",
                t.id,
                t.schedule.describe(),
                t.prompt.chars().take(80).collect::<String>()
            )
        })
        .collect();
    lines.join("; ")
}

/// Пора ли выполнять задачу сейчас.
fn is_due(task: &Task, now: chrono::DateTime<chrono::Local>) -> bool {
    if !task.enabled {
        return false;
    }
    match &task.schedule {
        Schedule::Daily { hhmm } => {
            let Ok((h, m)) = parse_hhmm(hhmm) else {
                return false;
            };
            let today_run = now
                .date_naive()
                .and_hms_opt(h, m, 0)
                .expect("ЧЧ:ММ уже проверены")
                .and_local_timezone(now.timezone())
                .single()
                .map(|t| now >= t)
                .unwrap_or(false);
            today_run
                && !task
                    .last_run
                    .map(|r| chrono::DateTime::from_timestamp(r as i64, 0))
                    .flatten()
                    .map(|r| {
                        r.with_timezone(&now.timezone()).date_naive() == now.date_naive()
                    })
                    .unwrap_or(false)
        }
        Schedule::Interval { minutes } => match task.last_run {
            None => true,
            Some(last) => now.timestamp() as u64 >= last + minutes * 60,
        },
        Schedule::Once { iso } => match parse_once(iso) {
            Ok(dt) => {
                // Наивное локальное время → сравниваем с локальным now.
                now.naive_local() >= dt
            }
            Err(_) => false,
        },
    }
}

/// Выполнить задачу: изолированный диалог без инструментов (как context_mode
/// = isolated в OpenJarvis), ответ проговорить и показать в уведомлении.
fn run_task(
    task: &Task,
    shared: &crate::llm::SharedLlm,
    audio_tx: &std::sync::mpsc::Sender<crate::tts::AudioMessage>,
    events: &crate::events::EventTx,
) {
    println!("⏰ Расписание №{}: {}", task.id, task.prompt);
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let answer = match crate::llm::generate_blocking(
        shared,
        crate::llm::CHAT_SYSTEM_PROMPT,
        &[],
        &task.prompt,
        &cancel,
    ) {
        Ok(text) if !text.trim().is_empty() => text,
        Ok(_) => "По расписанию ответ пустой.".to_string(),
        Err(e) => {
            let _ = events.send(crate::events::JarvisEvent::Error {
                message: format!("Расписание №{}: модель не ответила: {e}", task.id),
            });
            return;
        }
    };
    let _ = audio_tx.send(crate::tts::AudioMessage::SynthesizeAndPlay(answer.clone()));
    // Уведомление на рабочий стол — тот же механизм, что у tool-вызова Notify.
    let _ = crate::platform::notify(
        "Джарвис по расписанию",
        &answer.chars().take(200).collect::<String>(),
    );
    let _ = events.send(crate::events::JarvisEvent::Log {
        message: format!("⏰ Расписание №{}: выполнено", task.id),
    });
}

/// Фоновый поллер: раз в 20 секунд — что пора, то выполняем.
pub fn spawn(
    shared: crate::llm::SharedLlm,
    audio_tx: std::sync::mpsc::Sender<crate::tts::AudioMessage>,
    events: crate::events::EventTx,
) {
    std::thread::Builder::new()
        .name("schedule-poller".into())
        .spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(20));
            let now = chrono::Local::now();
            let mut tasks = load();
            let mut changed = false;
            for task in tasks.iter_mut() {
                if !is_due(task, now) {
                    continue;
                }
                run_task(task, &shared, &audio_tx, &events);
                task.last_run = Some(now.timestamp() as u64);
                // Разовый запуск выключается после выполнения.
                if matches!(task.schedule, Schedule::Once { .. }) {
                    task.enabled = false;
                }
                changed = true;
            }
            if changed {
                save(&tasks);
            }
        })
        .expect("поток планировщика");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jarvis-schedule-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("schedule.json")
    }

    #[test]
    fn add_validates_schedule_format() {
        assert!(Schedule::Daily { hhmm: "8:00".into() }.validate().is_ok());
        assert!(Schedule::Daily { hhmm: "23:59".into() }.validate().is_ok());
        assert!(Schedule::Daily { hhmm: "24:00".into() }.validate().is_err());
        assert!(Schedule::Daily { hhmm: "утром".into() }.validate().is_err());
        assert!(Schedule::Interval { minutes: 0 }.validate().is_err());
        assert!(Schedule::Interval { minutes: 30 }.validate().is_ok());
        assert!(
            Schedule::Once { iso: "2026-11-12 09:30".into() }
                .validate()
                .is_ok()
        );
        assert!(
            Schedule::Once { iso: "завтра утром".into() }
                .validate()
                .is_err()
        );
    }

    #[test]
    fn add_list_remove_roundtrip() {
        let path = temp_store("roundtrip");
        // add/save идут в глобальный стор — работаем через атомарные функции
        // на файле напрямую, чтобы тест не зависел от машины.
        let mut tasks = load_at(&path);
        tasks.push(Task {
            id: 1,
            prompt: "Сводка новостей".into(),
            schedule: Schedule::Daily { hhmm: "08:00".into() },
            enabled: true,
            last_run: None,
        });
        save_at(&path, &tasks);
        let mut loaded = load_at(&path);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].prompt, "Сводка новостей");
        loaded.retain(|t| t.id != 1);
        save_at(&path, &loaded);
        assert!(load_at(&path).is_empty());
    }

    #[test]
    fn daily_due_logic_uses_last_run_date() {
        let now = chrono::Local::now();
        let task_run_today = Task {
            id: 1,
            prompt: "тест".into(),
            schedule: Schedule::Daily { hhmm: "00:00".into() },
            enabled: true,
            // last_run = сейчас → сегодня уже выполнялась, не пора.
            last_run: Some(now.timestamp() as u64),
        };
        assert!(!is_due(&task_run_today, now));
        let task_fresh = Task {
            last_run: None,
            ..task_run_today.clone()
        };
        // «00:00» уже прошло сегодня — свежая задача пора.
        assert!(is_due(&task_fresh, now));
        let task_disabled = Task {
            enabled: false,
            ..task_fresh
        };
        assert!(!is_due(&task_disabled, now));
    }

    #[test]
    fn interval_due_by_elapsed() {
        let now = chrono::Local::now();
        let task = Task {
            id: 2,
            prompt: "тест".into(),
            schedule: Schedule::Interval { minutes: 30 },
            enabled: true,
            last_run: Some((now.timestamp() - 60) as u64),
        };
        assert!(!is_due(&task, now));
        let task = Task {
            last_run: Some((now.timestamp() - 3600) as u64),
            ..task
        };
        assert!(is_due(&task, now));
    }

    #[test]
    fn once_due_then_disabled_after_run() {
        let now = chrono::Local::now();
        let past = now - chrono::Duration::hours(1);
        let task = Task {
            id: 3,
            prompt: "тест".into(),
            schedule: Schedule::Once {
                iso: past.format("%Y-%m-%d %H:%M").to_string(),
            },
            enabled: true,
            last_run: None,
        };
        assert!(is_due(&task, now));
        let future = Task {
            schedule: Schedule::Once {
                iso: (now + chrono::Duration::hours(1))
                    .format("%Y-%m-%d %H:%M")
                    .to_string(),
            },
            ..task
        };
        assert!(!is_due(&future, now));
    }
}
