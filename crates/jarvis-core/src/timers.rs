//! Хаб таймеров Джарвиса: голос ставит таймер — окно «Утилиты» тикает,
//! по завершении Джарвис звонит собственным рингтоном и говорит.
//!
//! Таймеры живут в процессе Джарвиса (пользователь попросил встроенные
//! утилиты вместо Напоминаний): закрытие приложения снимает таймеры —
//! зато контроль, окно и отмена голосом полностью у ассистента.
//!
//! Сигнал — собственный WAV-рингтон (раздельный для таймера и будильника),
//! играет циклом до остановки голосом или кнопкой. Никакого AppleScript.

use crate::events::{EventTx, JarvisEvent};
use crate::paths;
use crate::tts::{self, AudioMessage};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Чем является запись: обычный таймер или будильник (разные рингтоны).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimerKind {
    Timer,
    Alarm,
}

#[derive(Debug, Clone)]
pub struct ActiveTimer {
    pub id: u64,
    pub label: String,
    pub remaining_secs: u64,
    pub kind: TimerKind,
    /// true — уже сработал и сейчас звонит.
    pub ringing: bool,
}

struct Timer {
    label: String,
    total: u64,
    started_at: Instant,
    kind: TimerKind,
}

/// Звенящая (уже сработавшая) запись: пока флаг не поднят — рингтон циклится.
struct Ringing {
    stop: Arc<AtomicBool>,
    kind: TimerKind,
    label: String,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static TIMERS: OnceLock<Mutex<HashMap<u64, Timer>>> = OnceLock::new();
static RINGING: OnceLock<Mutex<HashMap<u64, Ringing>>> = OnceLock::new();

fn map() -> &'static Mutex<HashMap<u64, Timer>> {
    TIMERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ringing() -> &'static Mutex<HashMap<u64, Ringing>> {
    RINGING.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Пути рингтонов: дефолты, переопределяемые из commands.toml.
struct Ringtones {
    timer: PathBuf,
    alarm: PathBuf,
}

static RINGTONES: OnceLock<RwLock<Ringtones>> = OnceLock::new();

fn ringtones() -> &'static RwLock<Ringtones> {
    RINGTONES.get_or_init(|| {
        RwLock::new(Ringtones {
            timer: paths::resolve_data_path("resources/audio/ringtone_timer.wav"),
            alarm: paths::resolve_data_path("resources/audio/ringtone_alarm.wav"),
        })
    })
}

/// Обновляет пути рингтонов из конфига (пустые значения игнорируются).
pub fn set_ringtones(timer: &str, alarm: &str) {
    let mut r = ringtones().write().unwrap_or_else(|e| e.into_inner());
    if !timer.trim().is_empty() {
        r.timer = paths::resolve_data_path(timer);
    }
    if !alarm.trim().is_empty() {
        r.alarm = paths::resolve_data_path(alarm);
    }
}

fn ringtone_for(kind: TimerKind) -> String {
    let r = ringtones().read().unwrap_or_else(|e| e.into_inner());
    match kind {
        TimerKind::Timer => r.timer.clone(),
        TimerKind::Alarm => r.alarm.clone(),
    }
    .to_string_lossy()
    .to_string()
}

/// Ставит таймер/будильник: событие для окна «Утилиты» + поток-наблюдатель,
/// который по истечении звонит рингтоном, говорит и держит сигнал до отмены.
pub fn start(
    events: &EventTx,
    audio_tx: &Sender<AudioMessage>,
    label: &str,
    seconds: u64,
    kind: TimerKind,
) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    map().lock().unwrap_or_else(|e| e.into_inner()).insert(
        id,
        Timer {
            label: label.to_string(),
            total: seconds,
            started_at: Instant::now(),
            kind,
        },
    );

    let _ = events.send(JarvisEvent::TimerStarted {
        id,
        label: label.to_string(),
        seconds,
        kind,
    });

    let events = events.clone();
    let audio_tx = audio_tx.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(seconds));
        let Some(timer) = map().lock().unwrap_or_else(|e| e.into_inner()).remove(&id) else {
            return; // отменён до срабатывания
        };
        let _ = events.send(JarvisEvent::TimerFired {
            id,
            label: timer.label.clone(),
            seconds: timer.total,
            kind: timer.kind,
        });
        ring_until_stopped(&audio_tx, id, &timer.label, timer.kind);
    });
    id
}

/// Циклит рингтон, пока не поднимут stop-флаг через [`cancel`]. Один раз
/// озвучивает срабатывание. Регистрируется в [`RINGING`] на всё время звонка.
fn ring_until_stopped(audio_tx: &Sender<AudioMessage>, id: u64, label: &str, kind: TimerKind) {
    let stop = Arc::new(AtomicBool::new(false));
    ringing().lock().unwrap_or_else(|e| e.into_inner()).insert(
        id,
        Ringing {
            stop: Arc::clone(&stop),
            kind,
            label: label.to_string(),
        },
    );

    let ring = ringtone_for(kind);
    let speech = if kind == TimerKind::Alarm {
        if label.is_empty() {
            "Будильник сработал!".to_string()
        } else {
            format!("Будильник «{label}» — время пришло!")
        }
    } else if label.is_empty() {
        "Время вышло!".to_string()
    } else {
        format!("Таймер «{label}» — время вышло!")
    };

    // Сначала звонок, потом голос: синтез речи блокирует очередь плеера,
    // но первый удар рингтона должен прозвучать сразу.
    let _ = audio_tx.send(AudioMessage::PlayWav(ring.clone()));
    let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech));

    let period = tts::wav_duration_secs(std::path::Path::new(&ring))
        .map(Duration::from_secs_f64)
        .filter(|d| !d.is_zero())
        .unwrap_or(Duration::from_secs(2));

    while !wait_stop(&stop, period) {
        let _ = audio_tx.send(AudioMessage::PlayWav(ring.clone()));
    }

    ringing()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id);
}

/// Спит `dur`, просыпаясь каждые 100 мс и проверяя флаг. true — остановка.
fn wait_stop(stop: &AtomicBool, dur: Duration) -> bool {
    let step = Duration::from_millis(100);
    let mut slept = Duration::ZERO;
    while slept < dur {
        if stop.load(Ordering::SeqCst) {
            return true;
        }
        let chunk = step.min(dur - slept);
        std::thread::sleep(chunk);
        slept += chunk;
    }
    stop.load(Ordering::SeqCst)
}

/// Остановка: сначала ищем звенящий (уже сработавший) таймер, затем тикающий.
/// `target` фильтрует по типу («останови будильник»), `id` — конкретную запись.
pub fn cancel(
    events: &EventTx,
    audio_tx: &Sender<AudioMessage>,
    target: Option<TimerKind>,
    id: Option<u64>,
) -> Option<ActiveTimer> {
    if let Some(stopped) = stop_ringing(events, audio_tx, target, id) {
        return Some(stopped);
    }
    cancel_pending(events, target, id)
}

fn stop_ringing(
    events: &EventTx,
    audio_tx: &Sender<AudioMessage>,
    target: Option<TimerKind>,
    id: Option<u64>,
) -> Option<ActiveTimer> {
    let mut m = ringing().lock().unwrap_or_else(|e| e.into_inner());
    let pick = match id {
        Some(i) => m.get(&i).map(|_| i),
        None => m
            .iter()
            .filter(|(_, r)| target.is_none_or(|k| r.kind == k))
            .map(|(i, _)| *i)
            .max(),
    }?;
    let entry = m.remove(&pick)?;
    entry.stop.store(true, Ordering::SeqCst);
    drop(m);

    // Глушим текущий удар рингтона и озвучку в очереди плеера.
    let _ = audio_tx.send(AudioMessage::Stop);
    let _ = events.send(JarvisEvent::TimerStopped {
        id: pick,
        label: entry.label.clone(),
    });
    Some(ActiveTimer {
        id: pick,
        label: entry.label,
        remaining_secs: 0,
        kind: entry.kind,
        ringing: true,
    })
}

fn cancel_pending(
    events: &EventTx,
    target: Option<TimerKind>,
    id: Option<u64>,
) -> Option<ActiveTimer> {
    let mut m = map().lock().unwrap_or_else(|e| e.into_inner());
    let pick = match id {
        Some(i) => m.get(&i).map(|_| i),
        None => m
            .iter()
            .filter(|(_, t)| target.is_none_or(|k| t.kind == k))
            .map(|(i, _)| *i)
            .max(),
    };
    let timer = m.remove(&pick?)?;
    drop(m);
    let _ = events.send(JarvisEvent::TimerStopped {
        id: pick?,
        label: timer.label.clone(),
    });
    Some(ActiveTimer {
        id: pick?,
        label: timer.label,
        remaining_secs: timer
            .total
            .saturating_sub(timer.started_at.elapsed().as_secs()),
        kind: timer.kind,
        ringing: false,
    })
}

/// Список активных с остатком (для окна).
pub fn active() -> Vec<ActiveTimer> {
    map()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(id, t)| ActiveTimer {
            id: *id,
            label: t.label.clone(),
            remaining_secs: t.total.saturating_sub(t.started_at.elapsed().as_secs()),
            kind: t.kind,
            ringing: false,
        })
        .collect()
}

/// Тестовый доступ: снять всё (не используется в проде).
#[doc(hidden)]
pub fn _clear_for_tests() {
    map().lock().unwrap_or_else(|e| e.into_inner()).clear();
    ringing().lock().unwrap_or_else(|e| e.into_inner()).clear();
}
