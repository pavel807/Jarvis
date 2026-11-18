//! Голосовой движок для iOS/iPadOS — настоящий, без заглушек.
//!
//! На телефоне нет wake-слова (нет постоянного слушания — экономия батареи
//! и честная работа с разрешениями): слушание включается кнопкой
//! (tap-to-talk). Полный конвейер тот же, что на десктопе:
//!
//! микрофон (cpal/CoreAudio) → буфер 16 кГц → FastConformer STT
//! (candle-CPU, модель качается в Documents/Jarvis/models) → гейты
//! (дальше/шутка/факт/календарь/почитай) → [`crate::llm::ask`] → ZONOS3 TTS
//! через [`crate::tts::spawn_audio_player`] (настоящий выходной поток,
//! AVAudioSession активируется в audio.rs).
//!
//! Детерминированные гейты работают как на десктопе: мгновенно, без сети
//! и модели. Всё остальное уходит в общий ask() — тот же роутер
//! действия/вопроса, тот же веб-поиск, та же разговорная память.

use crate::audio::{self, resample_audio};
use crate::events::JarvisEvent;
use crate::llm::{self, AudioMessage, ChatTurn};
use crate::stt::Stt;
use anyhow::Context;
use cpal::traits::StreamTrait;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};

/// FastConformer обучен на 16 кГц mono f32 (на десктопе та же константа
/// wake::SAMPLE_RATE — wake на iOS не входит в сборку).
pub const STT_SAMPLE_RATE: usize = 16_000;

/// Всё, что нужно мобильному движку: общее состояние LLM, история
/// разговора, канал событий (UI слушает их через `engine-event`), флаг
/// отмены генерации и настоящий аудио-конвейер ZONOS3.
pub struct MobileHandles {
    pub llm: llm::SharedLlm,
    pub history: Arc<Mutex<Vec<ChatTurn>>>,
    pub events: crate::events::EventTx,
    pub cancel: Arc<AtomicBool>,
    /// Плеер TTS: llm::ask шлёт сюда SynthesizeAndPlay/Stop.
    pub audio_tx: Sender<AudioMessage>,
    /// Говорит ли плеер прямо сейчас (для UI-индикатора).
    pub is_speaking: Arc<AtomicBool>,
    /// Флаг остановки озвучки (кнопка «стоп» во время ответа).
    pub stop_audio: Arc<AtomicBool>,
    /// FastConformer грузится в фоне; None — ещё не готов.
    stt: Arc<Mutex<Option<Stt>>>,
    /// Активная сессия прослушивания (между tap и stop).
    listen: Arc<Mutex<Option<ListenSession>>>,
}

impl MobileHandles {
    /// Собирает хендлы из общего состояния приложения: плеер ZONOS3 и его
    /// флаги уже созданы в инициализации (общие с десктопным пайплайном),
    /// STT и сессия прослушивания появляются по мере работы.
    pub fn new(
        llm: llm::SharedLlm,
        history: Arc<Mutex<Vec<ChatTurn>>>,
        events: crate::events::EventTx,
        cancel: Arc<AtomicBool>,
        audio_tx: Sender<AudioMessage>,
        is_speaking: Arc<AtomicBool>,
        stop_audio: Arc<AtomicBool>,
    ) -> Self {
        Self {
            llm,
            history,
            events,
            cancel,
            audio_tx,
            is_speaking,
            stop_audio,
            stt: Arc::new(Mutex::new(None)),
            listen: Arc::new(Mutex::new(None)),
        }
    }
}

/// Одна сессия записи: открытый стрим, родная частота устройства и
/// накопленный буфер. Sender живёт вместе со стримом — взяв сессию,
/// закрываем и канал (коллектор дочитает хвост до Disconnected).
struct ListenSession {
    stream: cpal::Stream,
    native_sr: usize,
    buf: Arc<Mutex<Vec<f32>>>,
    /// true, когда коллектор дочитал всё после закрытия канала.
    done: Arc<AtomicBool>,
    _mic_tx: SyncSender<Vec<f32>>,
}

/// Разогнать движок: плеер уже создан (его делает инициализация в lib.rs
/// вместе с загрузчиком TTS). Здесь в фоне грузится FastConformer: при
/// первом запуске модель качается (~460 МБ) — печатаем прогресс в лог и
/// событие, по готовности STT появляется сам.
pub fn run(h: &MobileHandles) {
    let stt_slot = Arc::clone(&h.stt);
    let events = h.events.clone();
    std::thread::spawn(move || {
        println!("🧠 Загрузка распознавания речи (FastConformer RU)...");
        match Stt::load(&models_dir_for_stt()) {
            Ok(stt) => {
                *stt_slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(stt);
                println!("✅ Распознавание готово — можно говорить кнопкой");
                let _ = events.send(JarvisEvent::Log {
                    message: "Распознавание речи готово".into(),
                });
            }
            Err(e) => {
                // Честная ошибка: без STT голосовой ввод недоступен, текст
                // продолжает работать. Повтор попытки — по кнопке в UI.
                let message = format!("Распознавание не загрузилось: {e:#}");
                eprintln!("❌ {message}");
                let _ = events.send(JarvisEvent::Log { message });
            }
        }
    });
    let _ = h.events.send(JarvisEvent::Log {
        message: "Мобильный движок готов — спроси о чём угодно".into(),
    });
}

/// Каталог моделей для STT. Пути считает paths: на iOS это
/// Documents/Jarvis/models сэндбокса, переопределяется JARVIS_MODELS_DIR.
fn models_dir_for_stt() -> PathBuf {
    crate::paths::models_dir()
}

/// Начать слушать (tap-to-talk): запрос разрешения при необходимости,
/// активация аудио-сессии, захват с микрофона в буфер. Возвращает статус
/// строкой: "listening" | "pending" (показан системный запрос) | "speaking"
/// (остановили озвучку, попробуй ещё раз) | "denied:<причина>".
pub fn listen_start(h: &MobileHandles) -> String {
    // Прежнюю озвучку глушим: говорить и записывать одновременно нельзя.
    if h.is_speaking.load(Ordering::SeqCst) {
        h.stop_audio.store(true, Ordering::SeqCst);
        let _ = h.audio_tx.send(AudioMessage::Stop);
        return "speaking".into();
    }
    // Уже слушаем — повторный тап просто игнорируем.
    if h.listen.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
        return "listening".into();
    }

    // Разрешение iOS: при notDetermined показываем системный диалог и
    // возвращаем pending — UI предложит тапнуть снова после ответа.
    match audio::mic_status() {
        "denied" => {
            let _ = h.events.send(JarvisEvent::MicPermission { granted: false });
            return "denied:Нет доступа к микрофону — открой Настройки → Jarvis и включи его".into();
        }
        "pending" => {
            audio::mic_request_access();
            let _ = h.events.send(JarvisEvent::MicPermission { granted: false });
            return "pending".into();
        }
        _ => {}
    }

    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let (mic_tx, mic_rx) = sync_channel::<Vec<f32>>(256);
    // Активацию сессии и открытие потока делаем в этом потоке: cpal вернёт
    // честную ошибку, если маршрут не выдан, — UI покажет её как есть.
    let opened = audio::spawn_stream(mic_tx.clone())
        .map(|(stream, sr)| (stream, sr))
        .and_then(|(stream, sr)| {
            stream
                .play()
                .context("Не удалось запустить микрофон")
                .map(|_| (stream, sr))
        });
    let (stream, native_sr) = match opened {
        Ok(pair) => pair,
        Err(e) => return format!("denied:Микрофон не открылся: {e:#}"),
    };

    let done = Arc::new(AtomicBool::new(false));
    {
        let buf = Arc::clone(&buf);
        let done = Arc::clone(&done);
        // Коллектор дочитывает чанки в буфер; после закрытия канала
        // (сессию взяли) сливает хвост и помечает готовность.
        std::thread::spawn(move || {
            loop {
                match mic_rx.recv_timeout(std::time::Duration::from_millis(200)) {
                    Ok(chunk) => buf
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend(chunk),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            while let Ok(chunk) = mic_rx.try_recv() {
                buf.lock().unwrap_or_else(|e| e.into_inner()).extend(chunk);
            }
            done.store(true, Ordering::SeqCst);
        });
    }

    *h.listen.lock().unwrap_or_else(|e| e.into_inner()) = Some(ListenSession {
        stream,
        native_sr,
        buf,
        done,
        _mic_tx: mic_tx,
    });
    println!("🎙️ Слушаю (tap-to-talk)...");
    let _ = h.events.send(JarvisEvent::StateChanged {
        state: "listening".into(),
    });
    "listening".into()
}

/// Остановить запись и распознать сказанное. Распознавание и ответ идут
/// в фоне: UI получает Transcript → LlmChunk/LlmDone, озвучка — через
/// настоящий плеер ZONOS3.
pub fn listen_stop(h: &MobileHandles) {
    let session = h
        .listen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let Some(session) = session else {
        return;
    };
    let ListenSession {
        stream,
        native_sr,
        buf,
        done,
        _mic_tx,
    } = session;
    // Стрим и последний Sender умирают вместе с session — коллектор
    // дочитает хвост до Disconnected и выставит done.
    drop(stream);

    let events = h.events.clone();
    let _ = events.send(JarvisEvent::StateChanged {
        state: "recognizing".into(),
    });
    let llm = Arc::clone(&h.llm);
    let history = Arc::clone(&h.history);
    let cancel = Arc::clone(&h.cancel);
    let audio_tx = h.audio_tx.clone();
    let stt_slot = Arc::clone(&h.stt);

    std::thread::spawn(move || {
        // Ждём коллектор: он дочитывает хвост записи после закрытия канала.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while !done.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let samples_16k = {
            let raw = buf.lock().unwrap_or_else(|e| e.into_inner()).clone();
            resample_audio(&raw, native_sr, STT_SAMPLE_RATE)
        };

        let secs = samples_16k.len() as f32 / STT_SAMPLE_RATE as f32;
        if secs < 0.25 {
            let _ = events.send(JarvisEvent::Log {
                message: "Не услышал речи — нажми и говори подольше".into(),
            });
            return;
        }

        let mut guard = stt_slot.lock().unwrap_or_else(|e| e.into_inner());
        let Some(stt) = guard.as_mut() else {
            drop(guard);
            let _ = events.send(JarvisEvent::Log {
                message: "Модель распознавания ещё загружается — подожди и повтори".into(),
            });
            return;
        };
        println!("🗣️ Распознаю {secs:.1} с...");
        let text = match stt.transcribe(&samples_16k) {
            Ok(t) => t,
            Err(e) => {
                let _ = events.send(JarvisEvent::Log {
                    message: format!("Распознавание не удалось: {e:#}"),
                });
                return;
            }
        };
        drop(guard);

        if text.trim().is_empty() {
            let _ = events.send(JarvisEvent::Log {
                message: "Речь не разобрал — повтори чуть громче".into(),
            });
            return;
        }
        println!("📝 Распознано: {text}");
        let _ = events.send(JarvisEvent::Transcript { text: text.clone() });

        if !gates_with(&events, &cancel, &text) {
            answer_with(&llm, &history, &events, &audio_tx, &cancel, &text);
        }
    });
}

/// Остановить озвучку (кнопка «стоп» во время ответа).
pub fn speak_stop(h: &MobileHandles) {
    h.stop_audio.store(true, Ordering::SeqCst);
    let _ = h.audio_tx.send(AudioMessage::Stop);
}

/// Общий конвейер вопрос→ответ (текст с клавиатуры): ask() с настоящим
/// аудио-каналом TTS. Гейты не здесь — сначала [`handle_text`].
pub fn answer(h: &MobileHandles, text: &str) {
    answer_with(
        &h.llm,
        &h.history,
        &h.events,
        &h.audio_tx,
        &h.cancel,
        text,
    );
}

/// Детерминированные гейты для текстовой реплики с клавиатуры.
/// true — реплика закрыта мгновенным ответом (ask() не нужен).
pub fn handle_text(h: &MobileHandles, text: &str) -> bool {
    gates_with(&h.events, &h.cancel, text)
}

/// Общий конвейер вопрос→ответ: ask() с настоящим аудио-каналом ZONOS3.
fn answer_with(
    llm: &llm::SharedLlm,
    history: &Arc<Mutex<Vec<ChatTurn>>>,
    events: &crate::events::EventTx,
    audio_tx: &Sender<AudioMessage>,
    cancel: &Arc<AtomicBool>,
    text: &str,
) {
    let prompt = text.trim().to_string();
    if prompt.is_empty() {
        return;
    }
    cancel.store(false, Ordering::SeqCst);
    llm::ask(
        Arc::clone(llm),
        llm::CHAT_SYSTEM_PROMPT,
        prompt,
        Arc::clone(history),
        audio_tx.clone(),
        events.clone(),
        Arc::clone(cancel),
    );
}

/// Детерминированные гейты (мгновенные ответы без модели и сети).
/// false — реплика уходит в общий ask() (см. [`answer_with`]).
fn gates_with(
    events: &crate::events::EventTx,
    cancel: &Arc<AtomicBool>,
    text: &str,
) -> bool {
    let prompt = text.trim();
    if prompt.is_empty() {
        return false;
    }
    // «Дальше» при чтении статьи — только если статья открыта.
    let t = prompt.to_lowercase().replace('ё', "е");
    let wants_more = (t.contains("читай дальше")
        || t.contains("продолжай читат")
        || t.contains("читай еще")
        || t.split_whitespace().all(|w| w == "дальше" || w == "пожалуйста") && !t.is_empty())
        && crate::web::last_article_title().is_some();
    if wants_more {
        match crate::web::next_article_chunk() {
            Some(chunk) => llm::speak_events(events, &chunk),
            None => llm::speak_events(events, "Статья дочитана до конца."),
        }
        return true;
    }

    if crate::jokes::is_joke_request(prompt) {
        match crate::jokes::tell() {
            Ok(joke) => llm::speak_events(events, &joke),
            Err(reason) => llm::speak_events(events, &format!("Не могу рассказать шутку: {reason}")),
        }
        return true;
    }
    if crate::facts::is_fact_request(prompt) {
        match crate::facts::tell() {
            Ok(fact) => llm::speak_events(events, &fact),
            Err(reason) => llm::speak_events(events, &format!("Не могу рассказать факт: {reason}")),
        }
        return true;
    }
    if let Some(answer_text) = crate::text::calendar_answer(prompt) {
        llm::speak_events(events, &answer_text);
        return true;
    }

    // «Почитай статью про X» — чтение вслух в фоне, как на десктопе.
    if let Some(target) = crate::web::read_article_target(prompt) {
        let events = events.clone();
        let cancel = Arc::clone(cancel);
        std::thread::spawn(move || {
            let url = if target.starts_with("http") {
                Some(target.clone())
            } else {
                crate::web::search(&target, 1)
                    .ok()
                    .and_then(|hits| hits.into_iter().next().map(|hit| hit.url))
            };
            let Some(url) = url else {
                llm::speak_events(&events, "Не нашёл подходящей статьи.");
                return;
            };
            match crate::web::fetch_article(&url) {
                Ok((title, article)) => {
                    let first = crate::web::start_reading(&title, &article);
                    let lead = if title.is_empty() {
                        "Читаю.".to_string()
                    } else {
                        format!("Читаю: {title}.")
                    };
                    llm::speak_events(&events, &format!("{lead} {first}"));
                }
                Err(_) => llm::speak_events(
                    &events,
                    "Страница не открылась или без текста — не смог прочитать.",
                ),
            }
            let _ = cancel.load(Ordering::SeqCst);
        });
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_gate_answers_without_network() {
        // Гейты мобильного движка работают без LLM: факт достаётся из базы.
        assert!(crate::facts::is_fact_request("расскажи интересный факт"));
        assert!(crate::facts::tell().is_ok());
    }

    #[test]
    fn article_helpers_survive_empty_state() {
        // Хелперы чтения не паникуют, когда статья не открыта.
        let _ = crate::web::next_article_chunk();
        let _ = crate::web::last_article_title();
        let _ = crate::web::article_remaining();
    }

    #[test]
    fn stt_rate_matches_fastconformer() {
        assert_eq!(STT_SAMPLE_RATE, 16_000);
    }
}
