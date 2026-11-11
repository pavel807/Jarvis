//! Headless-запуск Jarvis из терминала: лог в консоль, движок в jarvis-core.

use anyhow::Result;
use jarvis_core::commands::{CommandsConfig, spawn_config_watcher};
use jarvis_core::engine::{EngineHandles, run_engine};
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::{SharedLlmState, spawn_llm_loader};
use jarvis_core::paths;
use jarvis_core::settings::{Settings, spawn_settings_watcher_with};
use jarvis_core::stt::Stt;
use jarvis_core::tts::{self, SharedTtsState};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, RwLock};

fn main() -> Result<()> {
    jarvis_core::init_tuning();

    println!("🎙️ Запуск Jarvis CLI ...");

    // Модели живут в системной папке программы; старые переносятся туда один раз.
    jarvis_core::paths::migrate_models();
    let models_dir = paths::models_dir();
    std::fs::create_dir_all(&models_dir)?;

    tts::prune_tts_cache();

    let settings_path = paths::settings_file();
    let settings = Arc::new(RwLock::new(Settings::load(&settings_path)?));
    settings.read().unwrap_or_else(|e| e.into_inner()).apply();

    let config_path = paths::commands_file();
    let initial_commands = CommandsConfig::load(&config_path)?;
    initial_commands.apply_ringtones();

    println!("📋 Загружено {} команд", initial_commands.commands.len());

    let commands_arc = Arc::new(RwLock::new(initial_commands));
    spawn_config_watcher(config_path, Arc::clone(&commands_arc));

    let is_speaking = Arc::new(AtomicBool::new(false));
    let stop_audio_flag = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    let wakeword_reload = Arc::new(AtomicBool::new(false));
    let llm_cancel = Arc::new(AtomicBool::new(false));

    // Канал событий создаём первым: он нужен плееру (AudioLevel),
    // движку и логгеру.
    let (events_tx, events_rx) = channel::<JarvisEvent>();

    // Лог событий в консоль.
    std::thread::spawn(move || {
        for ev in events_rx {
            match ev {
                JarvisEvent::StateChanged { state } => println!("🔁 Состояние: {}", state),
                JarvisEvent::WakeWordDetected { name, score } => {
                    println!("🎯 Wake word: {} ({:.3})", name, score)
                }
                JarvisEvent::Transcript { text } => println!("📝 Распознано: {}", text),
                JarvisEvent::CommandMatched { name } => println!("✅ Команда: {}", name),
                JarvisEvent::CommandNotFound { text } => println!("❓ Не найдено: {}", text),
                JarvisEvent::ActionStarted { tool, summary } => {
                    print!("🔧 {tool}");
                    if !summary.is_empty() {
                        print!(": {summary}");
                    }
                    println!()
                }
                JarvisEvent::ActionResult { ok, summary } => {
                    if ok {
                        println!("✅ {summary}")
                    } else {
                        println!("❗ {summary}")
                    }
                }
                JarvisEvent::VoiceCheck { ok, overlap, transcript } => {
                    println!(
                        "🎙️ Диктант {}: совпадение {:.0}% — «{}»",
                        if ok { "принят" } else { "отклонён" },
                        overlap * 100.0,
                        transcript
                    )
                }
                JarvisEvent::TopicResumed { topic } => println!("🧠 Возврат к теме: {}", topic),
                JarvisEvent::LlmChunk { text } => print!("🤖 {}", text),
                JarvisEvent::LlmDone { text } => println!("🤖 [готово] {}", text),
                JarvisEvent::LlmStatus { status } => println!("🧠 LLM: {}", status),
                JarvisEvent::TimerStarted { label, seconds, .. } => {
                    let name = if label.is_empty() {
                        String::new()
                    } else {
                        format!(" «{label}»")
                    };
                    println!("⏱ Таймер{name} на {seconds} с — окно утилит открыто")
                }
                JarvisEvent::TimerFired { label, kind, .. } => {
                    let name = if label.is_empty() {
                        String::new()
                    } else {
                        format!(" «{label}»")
                    };
                    let what = match kind {
                        jarvis_core::timers::TimerKind::Alarm => "Будильник",
                        jarvis_core::timers::TimerKind::Timer => "Таймер",
                    };
                    println!("⏰ {what}{name} — время вышло, звонит рингтон")
                }
                JarvisEvent::TimerStopped { .. } => println!("⏹ Таймер остановлен"),
                JarvisEvent::OpenUtils { tab } => println!("🧰 Окно утилит: {tab}"),
                JarvisEvent::CloseUtils => println!("🧰 Окно утилит закрыто"),
                JarvisEvent::MicPermission { granted } => {
                    if granted {
                        println!("✅ Доступ к микрофону получен");
                    } else {
                        println!(
                            "🎤 Жду доступа к микрофону — подтверди системный запрос \
                                  или выдай его в Системных настройках"
                        );
                    }
                }
                JarvisEvent::TrainProgress { epoch, total, loss } => {
                    println!("🏋️ Эпоха {}/{} (loss {:.4})", epoch, total, loss)
                }
                JarvisEvent::TrainDone { path } => println!("🏋️ Модель обучена: {}", path),
                JarvisEvent::RecordSaved { path } => println!("💾 Сэмпл записан: {}", path),
                // AudioLevel приходит ~20 раз в секунду — в консоль не льём.
                JarvisEvent::AudioLevel { .. } => {}
                JarvisEvent::DownloadProgress { what, percent } => {
                    print!("\r⬇️ {what}: {percent}%")
                }
                JarvisEvent::Error { message } => eprintln!("❗ {}", message),
                JarvisEvent::Reply { text } => println!("💬 {}", text),
                JarvisEvent::Log { message } => println!("{}", message),
                JarvisEvent::ReactorIntro => println!("⚛️ Реактор приветствует"),
                JarvisEvent::RouterVerdict {
                    intent,
                    confidence,
                    force_tool,
                } => println!(
                    "🧭 Роутер: {} ({:.2}, force_tool={})",
                    intent, confidence, force_tool
                ),
                JarvisEvent::ToolParseFailed { reason, raw } => {
                    println!("⚠️ Tool не разобран: {reason}; ответ: {raw}")
                }
            }
        }
    });

    let shared_llm = Arc::new(SharedLlmState::new());

    // Вотчер настроек: смена llm_backend/llm_model перезагружает LLM на лету.
    let events_for_watcher = events_tx.clone();
    let shared_for_watcher = Arc::clone(&shared_llm);
    spawn_settings_watcher_with(settings_path, Arc::clone(&settings), move |_| {
        jarvis_core::llm::reload_llm(&shared_for_watcher, &events_for_watcher);
    });

    let shared_tts: Arc<SharedTtsState> = Arc::new(SharedTtsState::new());
    let tts_failed = Arc::new(AtomicBool::new(false));
    let tts_voice = settings
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .tts_voice
        .clone();
    tts::spawn_tts_loader(&shared_tts, &tts_failed, &tts_voice);

    // Плеер: первым аргументом — клон канала событий для AudioLevel.
    let audio_tx = tts::spawn_audio_player(
        events_tx.clone(),
        Arc::clone(&shared_tts),
        Arc::clone(&tts_failed),
        Arc::clone(&is_speaking),
        Arc::clone(&stop_audio_flag),
        Arc::clone(&shared_llm),
        Arc::clone(&settings),
    );

    // LLM: фоновая загрузка Qwen3.
    spawn_llm_loader(&shared_llm, &events_tx);

    println!("📥 Загрузка FastConformer (ASR)...");
    let model = Stt::load(&models_dir)?;

    // Фильтр говорящего: как в Tauri-приложении; профиль строим сразу.
    let speaker_gate = jarvis_core::speaker::load_gate(&events_tx);
    let (speaker_tx, speaker_rx) = jarvis_core::speaker::spawn_speaker_worker(
        speaker_gate,
        Arc::new(AtomicBool::new(false)),
    );
    let _ = speaker_tx.send(jarvis_core::speaker::SpeakerQuery::Rebuild);

    println!("🚀 Джарвис готов к работе!");

    let handles = EngineHandles {
        commands: commands_arc,
        settings,
        audio_tx,
        is_speaking,
        stop_audio_flag,
        paused,
        wakeword_reload,
        llm_cancel,
        llm_history: Arc::new(Mutex::new(Vec::new())),
        llm: shared_llm,
        events: events_tx,
        speaker_tx,
        speaker_rx,
    };

    run_engine(model, handles)
}