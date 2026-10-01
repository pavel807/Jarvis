//! Headless-запуск Jarvis из терминала: лог в консоль, движок в jarvis-core.

use anyhow::Result;
use jarvis_core::commands::{spawn_config_watcher, CommandsConfig};
use jarvis_core::engine::{run_engine, EngineHandles};
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::{spawn_llm_loader, SharedLlmState};
use jarvis_core::paths;
use jarvis_core::settings::{spawn_settings_watcher, Settings};
use jarvis_core::stt::{self, Stt};
use jarvis_core::tts::{self, SharedTtsState};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, RwLock};

fn main() -> Result<()> {
    println!("🎙️ Запуск Jarvis CLI (GigaAM v3 + Qwen3-TTS + Rustpotter + Qwen3)...");

    stt::init_env();
    let models_dir = paths::models_dir();
    std::fs::create_dir_all(&models_dir)?;

    tts::prune_tts_cache();

    let settings_path = paths::settings_file();
    let settings = Arc::new(RwLock::new(Settings::load(&settings_path)?));
    settings.read().unwrap_or_else(|e| e.into_inner()).apply();
    spawn_settings_watcher(settings_path, Arc::clone(&settings));

    let config_path = paths::commands_file();
    let initial_commands = CommandsConfig::load(&config_path)?;

    println!("📋 Загружено {} команд", initial_commands.commands.len());

    let commands_arc = Arc::new(RwLock::new(initial_commands));
    spawn_config_watcher(config_path, Arc::clone(&commands_arc));

    let is_speaking = Arc::new(AtomicBool::new(false));
    let stop_audio_flag = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    let wakeword_reload = Arc::new(AtomicBool::new(false));
    let llm_cancel = Arc::new(AtomicBool::new(false));

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
                JarvisEvent::LlmChunk { text } => print!("🤖 {}", text),
                JarvisEvent::LlmDone { text } => println!("🤖 [готово] {}", text),
                JarvisEvent::LlmStatus { status } => println!("🧠 LLM: {}", status),
                JarvisEvent::TrainProgress { epoch, total, loss } => {
                    println!("🏋️ Эпоха {}/{} (loss {:.4})", epoch, total, loss)
                }
                JarvisEvent::TrainDone { path } => println!("🏋️ Модель обучена: {}", path),
                JarvisEvent::RecordSaved { path } => println!("💾 Сэмпл записан: {}", path),
                JarvisEvent::Error { message } => eprintln!("❗ {}", message),
                JarvisEvent::Reply { text } => println!("💬 {}", text),
                JarvisEvent::Log { message } => println!("{}", message),
            }
        }
    });

    // LLM создаётся до плеера: он использует её для нейро-нормализации.
    let shared_llm = Arc::new(SharedLlmState::new());

    // TTS: ленивая загрузка в фоне + поток-плеер.
    let shared_tts: Arc<SharedTtsState> = Arc::new(SharedTtsState::new());
    let tts_failed = Arc::new(AtomicBool::new(false));
    let (tts_engine, tts_voice) = {
        let s = settings.read().unwrap_or_else(|e| e.into_inner());
        (s.tts_engine.clone(), s.tts_voice.clone())
    };
    tts::spawn_tts_loader(&shared_tts, &tts_failed, &tts_engine, &tts_voice);

    let audio_tx = tts::spawn_audio_player(
        Arc::clone(&shared_tts),
        Arc::clone(&tts_failed),
        Arc::clone(&is_speaking),
        Arc::clone(&stop_audio_flag),
        Arc::clone(&shared_llm),
        Arc::clone(&settings),
    );

    // LLM: фоновая загрузка Qwen3.
    spawn_llm_loader(&shared_llm, &events_tx);

    println!("📥 Загрузка GigaAM v3 в память...");
    let model = Stt::load(&models_dir)?;

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
    };

    run_engine(model, handles)
}
