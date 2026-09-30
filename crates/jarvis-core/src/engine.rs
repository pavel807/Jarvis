//! Конечный автомат ассистента: wake word → команда → LLM.
//!
//! Потоки: callback микрофона шлёт чанки в bounded-канал; главный цикл делает
//! VAD/wake word и, не блокируясь, отправляет готовые фразы в STT-воркер
//! (транскрибация идёт параллельно); ответы озвучивает отдельный поток-плеер.

use crate::audio::{calculate_rms, resample_audio, spawn_stream};
use crate::commands::{execute_command, get_global_config, CommandsConfig};
use crate::events::{EventTx, JarvisEvent};
use crate::llm::{self, SharedLlm};
use crate::paths;
use crate::settings::Settings;
use crate::stt::Stt;
use crate::text::{clean_wakewords_and_prompt, extract_repeat_text, is_valid};
use crate::tts::AudioMessage;
use crate::wake;
use anyhow::{Context, Result};
use cpal::traits::StreamTrait;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Сколько секунд аудио держим в pre-roll истории перед началом речи.
const HISTORY_BUFFER_SECS: usize = 1;

#[derive(PartialEq, Debug, Clone, Copy)]
enum AppState {
    /// Слушаем wake word.
    WaitingForWakeWord,
    /// Окно прямого слушания: ждём команду.
    WaitingForCommand,
    /// Фраза captured, ждём результат транскрибации.
    Processing,
}

/// Всё, что нужно движку для работы; общее для CLI и Tauri.
pub struct EngineHandles {
    pub commands: Arc<RwLock<CommandsConfig>>,
    pub settings: Arc<RwLock<Settings>>,
    pub audio_tx: Sender<AudioMessage>,
    pub is_speaking: Arc<AtomicBool>,
    pub stop_audio_flag: Arc<AtomicBool>,
    /// Пауза прослушивания (из UI/трея).
    pub paused: Arc<AtomicBool>,
    /// Флаг горячей подмены модели wake word (после обучения).
    pub wakeword_reload: Arc<AtomicBool>,
    /// Отмена генерации LLM (сбрасывается перед каждым вопросом).
    pub llm_cancel: Arc<AtomicBool>,
    pub llm: SharedLlm,
    pub events: EventTx,
}

pub struct SttJob {
    pub samples: Vec<f32>,
}

/// STT-воркер: транскрибация уходит из микрофонного цикла, ответы приходят
/// по порядку в общий канал.
pub fn spawn_stt_worker(
    mut stt: Stt,
) -> (SyncSender<SttJob>, Receiver<anyhow::Result<String>>) {
    let (tx, rx) = sync_channel::<SttJob>(2);
    let (out_tx, out_rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            let res = stt.transcribe(&job.samples);
            let _ = out_tx.send(res);
        }
    });

    (tx, out_rx)
}

/// Быстрые социальные ответы без LLM.
fn match_quick_social_response(text: &str) -> Option<&'static str> {
    use crate::text::clean_text;

    let cleaned = clean_text(text);
    let words: Vec<&str> = cleaned.split_whitespace().collect();

    if words
        .iter()
        .any(|&w| matches!(w, "спасибо" | "спасибочки" | "благодарю" | "спс" | "мерси"))
    {
        Some("Пожалуйста! Рад помочь.")
    } else if words.iter().any(|&w| {
        matches!(
            w,
            "привет" | "здравствуй" | "здорово" | "приветствую" | "хай"
        )
    }) {
        Some("Привет! Чем могу помочь?")
    } else if words
        .iter()
        .any(|&w| matches!(w, "пока" | "прощай" | "отключись"))
        || cleaned.contains("до свидания")
    {
        Some("До связи!")
    } else if cleaned.contains("как дела") || cleaned.contains("как ты") {
        Some("Всё отлично, готов к работе!")
    } else if cleaned.contains("кто ты") || cleaned.contains("как тебя зовут") {
        Some("Я Джарвис, твой персональный голосовой ассистент.")
    } else {
        None
    }
}

/// Куда двигаться после обработки фразы: режим wake word или окно follow-up.
#[derive(PartialEq)]
enum Outcome {
    Wake,
    Followup,
}

/// Главная функция движка. Блокирует вызывающий поток.
pub fn run_engine(stt: Stt, h: EngineHandles) -> Result<()> {
    let (mic_tx, mic_rx) = sync_channel::<Vec<f32>>(128);

    println!("🎤 Запрашиваю доступ к микрофону...");
    let (stream, native_sr) =
        spawn_stream(mic_tx).context("Не удалось создать входной аудиопоток.")?;
    stream
        .play()
        .context("Не удалось запустить микрофон.")?;

    println!("✅ Микрофон запущен и готов к работе");
    println!("💡 Скажи 'Джарвис' для активации...");

    let (stt_tx, stt_rx) = spawn_stt_worker(stt);

    let wake_threshold = h
        .settings
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .wake_threshold;
    let mut det = wake::create_detector(&paths::wakeword_model(), wake_threshold)?;

    let mut wi16 = VecDeque::<i16>::new();
    let history_capacity = wake::SAMPLE_RATE * HISTORY_BUFFER_SECS;
    let mut audio_history = VecDeque::<f32>::with_capacity(history_capacity);

    let mut state = AppState::WaitingForWakeWord;
    let mut command_buffer = Vec::new();
    let mut followup_deadline: Option<Instant> = None;

    let mut has_speech_started = false;
    let mut silence_samples = 0;
    let mut rng = rand::thread_rng();

    let _ = h
        .events
        .send(JarvisEvent::StateChanged { state: "wake_word".into() });

    loop {
        // --- 0. Горячая подмена модели wake word и снимок настроек ---
        if h.wakeword_reload.load(Ordering::SeqCst) && state == AppState::WaitingForWakeWord {
            h.wakeword_reload.store(false, Ordering::SeqCst);
            match wake::create_detector(
                &paths::wakeword_model(),
                h.settings
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .wake_threshold,
            ) {
                Ok(d) => {
                    det = d;
                    println!("🔄 Модель wake word обновлена");
                }
                Err(e) => eprintln!("❌ Не удалось подменить wake word: {e}"),
            }
        }

        let settings_snap = h
            .settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let min_command_samples =
            (wake::SAMPLE_RATE as f32 * settings_snap.min_command_duration_secs) as usize;
        let max_command_samples =
            (wake::SAMPLE_RATE as f32 * settings_snap.max_command_duration_secs) as usize;
        let silence_trigger_samples =
            (wake::SAMPLE_RATE as f32 * settings_snap.silence_timeout_secs) as usize;

        // Пауза: вычитываем микрофон, копилки сбрасываем.
        if h.paused.load(Ordering::SeqCst) {
            let _ = mic_rx.recv_timeout(Duration::from_millis(100));
            command_buffer.clear();
            wi16.clear();
            audio_history.clear();
            has_speech_started = false;
            silence_samples = 0;
            continue;
        }

        // --- 1. Забираем готовую транскрипцию, если она есть ---
        if state == AppState::Processing {
            match stt_rx.try_recv() {
                Ok(res) => {
                    state = AppState::WaitingForCommand;

                    let text = match res {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("⚠️ Ошибка распознавания: {e}");
                            command_buffer.clear();
                            has_speech_started = false;
                            silence_samples = 0;

                            if followup_expired_or_none(followup_deadline) {
                                let global_cfg = get_global_config(&h.commands);
                                let _ = h
                                    .audio_tx
                                    .send(AudioMessage::PlayWav(global_cfg.reply_error));
                                state = AppState::WaitingForWakeWord;
                                followup_deadline = None;
                                audio_history.clear();
                            }
                            continue;
                        }
                    };

                    println!("📝 Распознано: {:?}", text);
                    let _ = h
                        .events
                        .send(JarvisEvent::Transcript { text: text.clone() });

                    let outcome = if is_valid(&text) {
                        handle_transcript(&text, &h, &mut rng)
                    } else {
                        eprintln!("🔇 Шум или пустой ввод");
                        if followup_expired_or_none(followup_deadline) {
                            let global_cfg = get_global_config(&h.commands);
                            let _ = h.audio_tx.send(AudioMessage::PlayWav(global_cfg.reply_error));
                            Outcome::Wake
                        } else {
                            Outcome::Followup
                        }
                    };

                    command_buffer.clear();
                    has_speech_started = false;
                    silence_samples = 0;

                    match outcome {
                        Outcome::Wake => {
                            state = AppState::WaitingForWakeWord;
                            followup_deadline = None;
                            audio_history.clear();
                            wi16.clear();
                            let _ = h.events.send(JarvisEvent::StateChanged {
                                state: "wake_word".into(),
                            });
                        }
                        Outcome::Followup => {
                            state = AppState::WaitingForCommand;
                            followup_deadline =
                                Some(Instant::now() + Duration::from_secs(settings_snap.followup_window_secs));
                            audio_history.clear();
                            let _ = h.events.send(JarvisEvent::StateChanged {
                                state: "listening".into(),
                            });
                        }
                    }

                    continue;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    return Err(anyhow::anyhow!("STT-воркер остановился"));
                }
            }
        }

        // --- 2. Очередь микрофона ---
        let chunk = match mic_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Пока звучит ответ ассистента — не слушаем себя.
        if h.is_speaking.load(Ordering::SeqCst) {
            audio_history.clear();
            command_buffer.clear();
            wi16.clear();

            has_speech_started = false;
            silence_samples = 0;
            followup_deadline = Some(Instant::now() + Duration::from_secs(settings_snap.followup_window_secs));
            continue;
        }

        let chunk_16k = if native_sr != wake::SAMPLE_RATE {
            resample_audio(&chunk, native_sr, wake::SAMPLE_RATE)
        } else {
            chunk
        };

        for &s in &chunk_16k {
            if audio_history.len() >= history_capacity {
                audio_history.pop_front();
            }
            audio_history.push_back(s);
        }

        // --- 3. Истечение окна follow-up ---
        if let Some(deadline) = followup_deadline {
            if Instant::now() > deadline {
                if state == AppState::WaitingForCommand {
                    println!(
                        "🔒 Окно прямого слушания ({} сек) истекло. Переход в режим вакворда.",
                        settings_snap.followup_window_secs
                    );

                    state = AppState::WaitingForWakeWord;
                    command_buffer.clear();
                    wi16.clear();
                    audio_history.clear();

                    has_speech_started = false;
                    silence_samples = 0;
                    let _ = h
                        .events
                        .send(JarvisEvent::StateChanged { state: "wake_word".into() });
                }
                followup_deadline = None;
            }
        }

        match state {
            AppState::WaitingForWakeWord => {
                // Буфер для детектора наполняется только здесь — в остальных
                // состояниях он не нужен и его рост ограничен.
                for &s in &chunk_16k {
                    wi16.push_back((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
                }

                while wi16.len() >= det.get_samples_per_frame() {
                    let n = det.get_samples_per_frame();
                    let fr: Vec<i16> = wi16.drain(..n).collect();

                    if let Some(d) = det.process_samples(fr) {
                        eprintln!("🎯 Wake word: {} ({:.3})", d.name, d.score);
                        let _ = h.events.send(JarvisEvent::WakeWordDetected {
                            name: d.name.clone(),
                            score: d.score,
                        });

                        let global_cfg = get_global_config(&h.commands);
                        let ack = paths::resolve_data_path(&global_cfg.reply_ack);
                        let _ = h
                            .audio_tx
                            .send(AudioMessage::PlayWav(ack.to_string_lossy().to_string()));

                        std::thread::sleep(Duration::from_millis(150));

                        wi16.clear();
                        audio_history.clear();

                        state = AppState::WaitingForCommand;
                        followup_deadline =
                            Some(Instant::now() + Duration::from_secs(settings_snap.followup_window_secs));

                        command_buffer.clear();
                        has_speech_started = false;
                        silence_samples = 0;
                        let _ = h
                            .events
                            .send(JarvisEvent::StateChanged { state: "listening".into() });

                        break;
                    }
                }
            }

            AppState::WaitingForCommand => {
                let chunk_rms = calculate_rms(&chunk_16k);

                if chunk_rms >= settings_snap.silence_rms_threshold {
                    if !has_speech_started {
                        has_speech_started = true;
                        let preroll_samples = (wake::SAMPLE_RATE as f32 * 0.3) as usize;
                        let hist_len = audio_history.len();
                        let take_samples = hist_len.min(preroll_samples);
                        if take_samples > 0 {
                            let start_idx = hist_len - take_samples;
                            for i in start_idx..hist_len {
                                command_buffer.push(audio_history[i]);
                            }
                        }
                    }
                    silence_samples = 0;
                } else if has_speech_started {
                    silence_samples += chunk_16k.len();
                }

                if has_speech_started {
                    command_buffer.extend_from_slice(&chunk_16k);

                    let reached_max = command_buffer.len() >= max_command_samples;
                    let silence_timeout = silence_samples >= silence_trigger_samples;

                    if reached_max || silence_timeout {
                        if command_buffer.len() < min_command_samples {
                            command_buffer.clear();
                            has_speech_started = false;
                            silence_samples = 0;
                            continue;
                        }

                        let pad = (wake::SAMPLE_RATE as f32 * 0.2) as usize;
                        let mut padded = Vec::with_capacity(command_buffer.len() + pad * 2);

                        padded.extend(std::iter::repeat(0.0f32).take(pad));
                        padded.extend_from_slice(&command_buffer);
                        padded.extend(std::iter::repeat(0.0f32).take(pad));

                        // Транскрибация уходит в воркер; микрофонный цикл не блокируется.
                        if stt_tx.send(SttJob { samples: padded }).is_err() {
                            return Err(anyhow::anyhow!("STT-воркер недоступен"));
                        }

                        state = AppState::Processing;
                    }
                }
            }

            // Ждём результат транскрибации; аудио не копим.
            AppState::Processing => {}
        }
    }
}

fn followup_expired_or_none(deadline: Option<Instant>) -> bool {
    match deadline {
        Some(dl) => Instant::now() >= dl,
        None => true,
    }
}

/// Обработка распознанной фразы: повтор / социальный ответ / команда / LLM.
fn handle_transcript(text: &str, h: &EngineHandles, rng: &mut impl Rng) -> Outcome {
    // 1. РЕЖИМ ПОВТОРЕНИЯ («скажи ...»)
    if let Some(phrase_to_repeat) = extract_repeat_text(text) {
        println!("🗣️ Озвучивание текста: \"{}\"", phrase_to_repeat);
        let _ = h
            .audio_tx
            .send(AudioMessage::SynthesizeAndPlay(phrase_to_repeat));
        return Outcome::Followup;
    }

    // 2. БЫСТРЫЕ СОЦИАЛЬНЫЕ ОТВЕТЫ
    if let Some(social_reply) = match_quick_social_response(text) {
        println!("⚡ Социальный ответ: \"{}\"", social_reply);
        let _ = h
            .audio_tx
            .send(AudioMessage::SynthesizeAndPlay(social_reply.to_string()));
        return Outcome::Followup;
    }

    // 3. ПОИСК ЛОКАЛЬНЫХ КОМАНД ИЗ TOML
    let (matched_cmd, global_cfg) = {
        let lock = h.commands.read().unwrap_or_else(|e| e.into_inner());
        (lock.find_command(text), lock.global.clone())
    };

    if let Some(cmd) = matched_cmd {
        println!("✅ Команда: {}", cmd.name);
        let _ = h.events.send(JarvisEvent::CommandMatched { name: cmd.name.clone() });

        if cmd.action_type == "stop" {
            h.stop_audio_flag.store(true, Ordering::SeqCst);
            h.llm_cancel.store(true, Ordering::SeqCst);
            let _ = h.audio_tx.send(AudioMessage::Stop);

            let reply_error = global_cfg.reply_error.clone();
            let err_path = paths::resolve_data_path(&reply_error).to_string_lossy().to_string();
            execute_command(&cmd, text, &h.audio_tx, &h.events, &h.llm, &err_path);
            println!("🛑 Команда СТОП: речь остановлена.");
            return Outcome::Wake;
        }

        let sound_num = rng.gen_range(1..=4);
        let sound_path = paths::resolve_data_path(&global_cfg.reply_success.replace("{}", &sound_num.to_string()));
        let _ = h
            .audio_tx
            .send(AudioMessage::PlayWav(sound_path.to_string_lossy().to_string()));

        let reply_error = global_cfg.reply_error.clone();
        let err_path = paths::resolve_data_path(&reply_error).to_string_lossy().to_string();
        execute_command(&cmd, text, &h.audio_tx, &h.events, &h.llm, &err_path);
        return Outcome::Followup;
    }

    // 4. ЛОКАЛЬНЫЙ LLM — ответ на любые вопросы
    let prompt = clean_wakewords_and_prompt(text);

    if !prompt.trim().is_empty() {
        // Вопрос про свежие данные? Ищем в интернете и суммаризируем.
        let web_auto = {
            let st = h.settings.read().unwrap_or_else(|e| e.into_inner());
            st.web_search_auto
        } && crate::web::looks_like_fresh_info(&prompt);

        if web_auto {
            println!("🌐 Вопрос требует свежих данных, ищу в интернете: \"{}\"", prompt);
            let sound_num = rng.gen_range(1..=4);
            let sound_path = paths::resolve_data_path(
                &global_cfg.reply_success.replace("{}", &sound_num.to_string()),
            );
            let _ = h
                .audio_tx
                .send(AudioMessage::PlayWav(sound_path.to_string_lossy().to_string()));

            let _ = h
                .audio_tx
                .send(AudioMessage::SynthesizeAndPlay(format!("Ищу в интернете {}", prompt)));

            let audio_tx = h.audio_tx.clone();
            let events = h.events.clone();
            let llm = Arc::clone(&h.llm);
            let query = prompt.clone();

            std::thread::spawn(move || match crate::web::search(&query, 5) {
                Ok(hits) => {
                    if let Some(web_prompt) = crate::web::build_summary_prompt(&query, &hits) {
                        llm::ask(
                            llm,
                            llm::WEB_SYSTEM_PROMPT,
                            web_prompt,
                            audio_tx,
                            events,
                            Arc::new(std::sync::atomic::AtomicBool::new(false)),
                        );
                    }
                }
                Err(e) => eprintln!("⚠️ Веб-поиск не удался: {e}"),
            });

            return Outcome::Followup;
        }

        println!("🤖 Вопрос к LLM: \"{}\"", prompt);

        let sound_num = rng.gen_range(1..=4);
        let sound_path = paths::resolve_data_path(&global_cfg.reply_success.replace("{}", &sound_num.to_string()));
        let _ = h
            .audio_tx
            .send(AudioMessage::PlayWav(sound_path.to_string_lossy().to_string()));

        llm::ask(
            Arc::clone(&h.llm),
            llm::SYSTEM_PROMPT,
            prompt,
            h.audio_tx.clone(),
            h.events.clone(),
            Arc::clone(&h.llm_cancel),
        );
    } else {
        println!("❓ Команда не найдена: \"{}\"", text);
        let _ = h.events.send(JarvisEvent::CommandNotFound { text: text.to_string() });
        let unknown = paths::resolve_data_path(&global_cfg.reply_unknown);
        let _ = h
            .audio_tx
            .send(AudioMessage::PlayWav(unknown.to_string_lossy().to_string()));
    }

    Outcome::Followup
}
