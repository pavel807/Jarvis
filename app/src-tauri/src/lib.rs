//! Tauri-бэкенд Jarvis: запускает движок из jarvis-core, мостит события в UI
//! и предоставляет IPC-команды (команды, настройки, wake word, TTS, LLM).

use jarvis_core::commands::{spawn_config_watcher, CommandsConfig};
use jarvis_core::engine::{run_engine, EngineHandles};
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::{self, spawn_llm_loader, ChatTurn, SharedLlmState};
use jarvis_core::paths;
use jarvis_core::settings::{spawn_settings_watcher, Settings};
use jarvis_core::stt::{self, Stt};
use jarvis_core::tts::{self, AudioMessage, SharedTtsState};
use jarvis_core::wake_train;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, RwLock};
use tauri::{AppHandle, Emitter, Manager, State};
/// Сколько последних записей журнала держим для UI.
const LOG_CAPACITY: usize = 400;

pub struct Shared {
    pub commands: Arc<RwLock<CommandsConfig>>,
    pub settings: Arc<RwLock<Settings>>,
    pub shared_tts: Arc<SharedTtsState>,
    pub tts_failed: Arc<AtomicBool>,
    pub shared_llm: Arc<SharedLlmState>,
    pub is_speaking: Arc<AtomicBool>,
    pub stop_audio_flag: Arc<AtomicBool>,
    pub paused: Arc<AtomicBool>,
    pub wakeword_reload: Arc<AtomicBool>,
    /// Отмена голосовой генерации LLM (команда «замолчи»).
    pub llm_cancel: Arc<AtomicBool>,
    /// Токен отмены текущего запроса из чата UI.
    pub chat_cancel: Mutex<Option<Arc<AtomicBool>>>,
    /// История диалога чата UI.
    pub chat_history: Arc<Mutex<Vec<ChatTurn>>>,
    pub record_cancel: Arc<AtomicBool>,
    pub engine_running: Arc<AtomicBool>,
    pub audio_tx: std::sync::mpsc::Sender<AudioMessage>,
    logs: Mutex<VecDeque<String>>,
}

impl Shared {
    #[allow(clippy::too_many_arguments)]
    pub fn with_parts(
        audio_tx: std::sync::mpsc::Sender<AudioMessage>,
        shared_tts: Arc<SharedTtsState>,
        tts_failed: Arc<AtomicBool>,
        is_speaking: Arc<AtomicBool>,
        stop_audio_flag: Arc<AtomicBool>,
        shared_llm: Arc<SharedLlmState>,
        settings: Arc<RwLock<Settings>>,
    ) -> Self {
        // Битый commands.toml не должен валить приложение: работаем
        // без команд, ошибку видно в журнале.
        let commands = CommandsConfig::load(&paths::commands_file()).unwrap_or_else(|e| {
            eprintln!("⚠️ {e:#}; использую пустой список команд");
            CommandsConfig::default()
        });

        Self {
            commands: Arc::new(RwLock::new(commands)),
            settings,
            shared_tts,
            tts_failed,
            shared_llm,
            is_speaking,
            stop_audio_flag,
            paused: Arc::new(AtomicBool::new(false)),
            wakeword_reload: Arc::new(AtomicBool::new(false)),
            llm_cancel: Arc::new(AtomicBool::new(false)),
            chat_cancel: Mutex::new(None),
            chat_history: Arc::new(Mutex::new(Vec::new())),
            record_cancel: Arc::new(AtomicBool::new(false)),
            engine_running: Arc::new(AtomicBool::new(false)),
            audio_tx,
            logs: Mutex::new(VecDeque::with_capacity(LOG_CAPACITY)),
        }
    }

    fn push_log(&self, line: String) {
        let mut logs = self.logs.lock().unwrap_or_else(|e| e.into_inner());
        if logs.len() >= LOG_CAPACITY {
            logs.pop_front();
        }
        logs.push_back(line);
    }

    fn logs(&self) -> Vec<String> {
        self.logs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    fn llm_status(&self) -> String {
        if self.shared_llm.failed.load(Ordering::SeqCst) {
            "failed".into()
        } else if self
            .shared_llm
            .engine
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
        {
            "ready".into()
        } else {
            "loading".into()
        }
    }
}

/// Загружает модели и запускает движок ассистента в фоновом потоке.
fn spawn_engine(shared: &Arc<Shared>, events_tx: std::sync::mpsc::Sender<JarvisEvent>) {
    let shared_engine = Arc::clone(shared);

    // Hot-reload конфигов, как в CLI.
    spawn_config_watcher(
        paths::commands_file(),
        Arc::clone(&shared_engine.commands),
    );
    spawn_settings_watcher(
        paths::settings_file(),
        Arc::clone(&shared_engine.settings),
    );

    std::thread::spawn(move || {
        stt::init_env();

        let tts_voice = {
            let s = shared_engine.settings.read().unwrap_or_else(|e| e.into_inner());
            s.apply();
            s.tts_voice.clone()
        };

        tts::spawn_tts_loader(&shared_engine.shared_tts, &shared_engine.tts_failed, &tts_voice);
        spawn_llm_loader(&shared_engine.shared_llm, &events_tx);

        let models_dir = paths::models_dir();
        let Ok(model) = Stt::load(&models_dir) else {
            shared_engine.engine_running.store(false, Ordering::SeqCst);
            let _ = events_tx.send(JarvisEvent::Error {
                message: "Не удалось загрузить STT-модель".into(),
            });
            return;
        };

        let handles = EngineHandles {
            commands: Arc::clone(&shared_engine.commands),
            settings: Arc::clone(&shared_engine.settings),
            audio_tx: shared_engine.audio_tx.clone(),
            is_speaking: Arc::clone(&shared_engine.is_speaking),
            stop_audio_flag: Arc::clone(&shared_engine.stop_audio_flag),
            paused: Arc::clone(&shared_engine.paused),
            wakeword_reload: Arc::clone(&shared_engine.wakeword_reload),
            llm_cancel: Arc::clone(&shared_engine.llm_cancel),
            llm_history: Arc::clone(&shared_engine.chat_history),
            llm: Arc::clone(&shared_engine.shared_llm),
            events: events_tx,
        };

        shared_engine.engine_running.store(true, Ordering::SeqCst);
        if let Err(e) = run_engine(model, handles) {
            eprintln!("❌ Движок остановился: {e}");
            shared_engine.engine_running.store(false, Ordering::SeqCst);
        }
    });
}

// ---------- IPC-команды ----------

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Status {
    engine_running: bool,
    paused: bool,
    llm_status: String,
    tts_ready: bool,
}

#[tauri::command]
fn status(shared: State<'_, Arc<Shared>>) -> Status {
    Status {
        engine_running: shared.engine_running.load(Ordering::SeqCst),
        paused: shared.paused.load(Ordering::SeqCst),
        llm_status: shared.llm_status(),
        tts_ready: shared
            .shared_tts
            .tts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some(),
    }
}

#[tauri::command]
fn set_paused(shared: State<'_, Arc<Shared>>, paused: bool) {
    shared.paused.store(paused, Ordering::SeqCst);
}

#[tauri::command]
fn get_commands(shared: State<'_, Arc<Shared>>) -> CommandsConfig {
    shared
        .commands
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

#[tauri::command]
fn save_commands(shared: State<'_, Arc<Shared>>, config: CommandsConfig) -> Result<(), String> {
    config
        .save(&paths::commands_file())
        .map_err(|e| e.to_string())?;

    let count = config.commands.len();
    *shared
        .commands
        .write()
        .unwrap_or_else(|e| e.into_inner()) = config;
    println!("⚡ Сохранено команд: {count}");
    Ok(())
}

#[tauri::command]
fn get_settings(shared: State<'_, Arc<Shared>>) -> Settings {
    shared
        .settings
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

#[tauri::command]
fn save_settings(shared: State<'_, Arc<Shared>>, settings: Settings) -> Result<(), String> {
    let voice = settings.tts_voice.clone();
    let voice_changed = tts::current_voice() != voice;
    settings
        .save(&paths::settings_file())
        .map_err(|e| e.to_string())?;
    settings.apply();
    *shared
        .settings
        .write()
        .unwrap_or_else(|e| e.into_inner()) = settings;

    if voice_changed {
        tts::set_voice(&shared.shared_tts, &shared.tts_failed, &voice);
    }
    Ok(())
}

#[tauri::command]
fn get_logs(shared: State<'_, Arc<Shared>>) -> Vec<String> {
    shared.logs()
}

#[tauri::command]
fn tts_voices() -> Vec<(String, String)> {
    tts::TTS_VOICES
        .iter()
        .map(|(id, desc)| (id.to_string(), desc.to_string()))
        .collect()
}

#[tauri::command]
fn tts_test(shared: State<'_, Arc<Shared>>, text: String) {
    let _ = shared
        .audio_tx
        .send(AudioMessage::SynthesizeAndPlay(text));
}

#[tauri::command]
fn llm_ask(shared: State<'_, Arc<Shared>>, app: AppHandle, prompt: String) {
    // События этого вопроса уходят и в UI, и в общий мост событий движка.
    let (tx, rx) = channel::<JarvisEvent>();
    std::thread::spawn(move || {
        for ev in rx {
            let _ = app.emit("engine-event", &ev);
        }
    });

    // Свежий токен на каждый запрос: «Стоп» отменяет текущий вопрос,
    // и гонки с обнулением флага внутри генерации нет.
    let cancel = Arc::new(AtomicBool::new(false));
    *shared
        .chat_cancel
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&cancel));

    llm::ask(
        Arc::clone(&shared.shared_llm),
        llm::SYSTEM_PROMPT,
        prompt,
        Arc::clone(&shared.chat_history),
        shared.audio_tx.clone(),
        tx,
        cancel,
    );
}

#[tauri::command]
fn llm_cancel(shared: State<'_, Arc<Shared>>) {
    if let Some(cancel) = shared
        .chat_cancel
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
    {
        cancel.store(true, Ordering::SeqCst);
    }
    let _ = shared.audio_tx.send(AudioMessage::Stop);
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct WwSamples {
    train: Vec<(String, f64)>,
    test: Vec<(String, f64)>,
}

#[tauri::command]
async fn ww_samples() -> WwSamples {
    let (train_dir, test_dir) = wake_train::sample_dirs();
    let fmt = |dir: &std::path::Path| -> Vec<(String, f64)> {
        tts::list_wav_samples(dir)
            .into_iter()
            .map(|(p, d)| (p.to_string_lossy().to_string(), d))
            .collect()
    };
    WwSamples {
        train: fmt(&train_dir),
        test: fmt(&test_dir),
    }
}

#[tauri::command]
fn ww_record(
    shared: State<'_, Arc<Shared>>,
    app: AppHandle,
    target: String,
    kind: String,
) -> Result<(), String> {
    if shared.record_cancel.swap(false, Ordering::SeqCst) {
        // предыдущая запись ещё не завершена — не стартуем новую
        return Err("Запись уже идёт".into());
    }

    let label = match kind.as_str() {
        "wake" => "jarvis",
        "none" => "none",
        other => return Err(format!("Неизвестный тип сэмпла: {other}")),
    };
    let dir = match target.as_str() {
        "train" => wake_train::sample_dirs().0,
        "test" => wake_train::sample_dirs().1,
        other => return Err(format!("Неизвестный каталог: {other}")),
    };

    let shared = Arc::clone(&shared);
    let record_cancel = Arc::clone(&shared.record_cancel);
    let app = app.clone();

    std::thread::spawn(move || {
        let cancel = AtomicBool::new(false);
        let _guard = RecordGuard {
            flag: Arc::clone(&record_cancel),
        };

        match wake_train::record_sample(&dir, label, 6.0, &cancel) {
            Ok(path) => {
                let _ = app.emit(
                    "engine-event",
                    &JarvisEvent::RecordSaved {
                        path: path.to_string_lossy().to_string(),
                    },
                );
            }
            Err(e) => {
                let _ = app.emit(
                    "engine-event",
                    &JarvisEvent::Error {
                        message: format!("Запись не удалась: {e}"),
                    },
                );
            }
        }
    });

    Ok(())
}

struct RecordGuard {
    flag: Arc<AtomicBool>,
}

impl Drop for RecordGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}

/// Проверяет, что путь (после разрешения симлинков) лежит внутри базового
/// каталога. Пути из webview нельзя принимать на веру.
fn ensure_under_dir(path: &Path, base: &Path) -> Result<PathBuf, String> {
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Не удалось разрешить путь {:?}: {e}", path))?;
    let base_canonical = base
        .canonicalize()
        .map_err(|e| format!("Не удалось разрешить каталог {:?}: {e}", base))?;
    if canonical.starts_with(&base_canonical) {
        Ok(canonical)
    } else {
        Err(format!(
            "Путь {:?} вне каталога {:?}",
            path, base_canonical
        ))
    }
}

#[tauri::command]
fn ww_delete(path: String) -> Result<(), String> {
    let file = ensure_under_dir(Path::new(&path), &paths::train_dir())?;
    std::fs::remove_file(&file).map_err(|e| e.to_string())
}

#[tauri::command]
fn ww_build(app: AppHandle) -> Result<(), String> {
    // DTW-модель строится за секунды, но всё равно в отдельном потоке,
    // чтобы не блокировать UI.
    std::thread::spawn(move || {
        let (tx, rx) = channel::<JarvisEvent>();
        std::thread::spawn(move || {
            for ev in rx {
                let _ = app.emit("engine-event", &ev);
            }
        });

        let cancel = Arc::new(AtomicBool::new(false));
        match wake_train::build_reference_wakeword(&tx, &cancel) {
            Ok(path) => {
                let _ = tx.send(JarvisEvent::TrainDone {
                    path: path.to_string_lossy().to_string(),
                });
            }
            Err(e) => {
                let _ = tx.send(JarvisEvent::Error {
                    message: format!("Не удалось построить модель: {e}"),
                });
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn ww_apply(shared: State<'_, Arc<Shared>>, path: Option<String>) -> Result<(), String> {
    let trained = match path {
        Some(p) => ensure_under_dir(Path::new(&p), &paths::models_dir())
            .or_else(|_| ensure_under_dir(Path::new(&p), &paths::train_dir()))?,
        None => {
            // Дефолт: результат последнего обучения (см. ww_build).
            paths::models_dir().join("jarvis_model_trained.rpw")
        }
    };
    if trained.extension().is_some_and(|x| x != "rpw") {
        return Err("Ожидался файл модели .rpw".into());
    }

    // Активный слот задаётся настройкой wakeword_model.
    let active = {
        let s = shared.settings.read().unwrap_or_else(|e| e.into_inner());
        paths::wakeword_model_file(&s.wakeword_model)
    };
    wake_train::apply_wakeword(&trained, &active).map_err(|e| e.to_string())?;
    shared.wakeword_reload.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
fn ww_test(shared: State<'_, Arc<Shared>>, path: String) -> Result<f32, String> {
    let wav = ensure_under_dir(Path::new(&path), &paths::train_dir())?;
    let active = {
        let s = shared.settings.read().unwrap_or_else(|e| e.into_inner());
        paths::wakeword_model_file(&s.wakeword_model)
    };
    wake_train::test_wakeword(&active, &wav).map_err(|e| e.to_string())
}

// ---------- Трей и запуск ----------

const TRAY_OPEN: &str = "open";
const TRAY_TOGGLE: &str = "toggle";
const TRAY_QUIT: &str = "quit";

fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::TrayIconBuilder;

    let open = MenuItem::with_id(app, TRAY_OPEN, "Открыть Jarvis", true, None::<&str>)?;
    let toggle = MenuItem::with_id(app, TRAY_TOGGLE, "Пауза прослушивания", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, TRAY_QUIT, "Выход", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &toggle, &quit])?;

    let mut tray = TrayIconBuilder::with_id("jarvis-tray")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            TRAY_OPEN => show_main(app),
            TRAY_TOGGLE => {
                let shared = app.state::<Arc<Shared>>();
                let paused = !shared.paused.load(Ordering::SeqCst);
                shared.paused.store(paused, Ordering::SeqCst);
            }
            TRAY_QUIT => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::Click { .. } = event {
                show_main(tray.app_handle());
            }
        });
    // Иконка может отсутствовать (например, при сборке без icons/) —
    // не паникуем, трай просто будет без иконки.
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;

    Ok(())
}

fn show_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Каталог данных приложения для случая, когда собранный .app запущен вне
/// репозитория: `~/Library/Application Support/Jarvis` (macOS) или `~/.jarvis`.
fn app_data_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        #[cfg(target_os = "macos")]
        return home
            .join("Library/Application Support/Jarvis");
        #[cfg(not(target_os = "macos"))]
        return home.join(".jarvis");
    }
    PathBuf::from(".")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Корень данных Jarvis. Приоритет: переменная окружения JARVIS_HOME →
    // корень репозитория, известный на этапе сборки (dev-запуск) → каталог
    // данных приложения (собранный .app).
    if std::env::var_os("JARVIS_HOME").is_none() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo = manifest
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());

        let root = match repo {
            Some(r) if r.join("commands.toml").exists() => r,
            _ => app_data_dir(),
        };
        unsafe {
            std::env::set_var("JARVIS_HOME", root);
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let handle = app.handle().clone();

            // Плеер и SharedTtsState создаются до Shared, чтобы всё
            // разделяло один инстанс состояния TTS.
            let shared_tts: Arc<SharedTtsState> = Arc::new(SharedTtsState::new());
            let tts_failed = Arc::new(AtomicBool::new(false));
            let is_speaking = Arc::new(AtomicBool::new(false));
            let stop_audio_flag = Arc::new(AtomicBool::new(false));

            let shared_llm: Arc<SharedLlmState> = Arc::new(SharedLlmState::new());
            // Битый settings.toml не должен валить приложение: откатываемся
            // на значения по умолчанию.
            let settings = Arc::new(RwLock::new(
                Settings::load(&paths::settings_file()).unwrap_or_else(|e| {
                    eprintln!("⚠️ {e:#}; использую настройки по умолчанию");
                    Settings::default()
                }),
            ));
            settings.read().unwrap_or_else(|e| e.into_inner()).apply();

            let audio_tx = tts::spawn_audio_player(
                Arc::clone(&shared_tts),
                Arc::clone(&tts_failed),
                Arc::clone(&is_speaking),
                Arc::clone(&stop_audio_flag),
                Arc::clone(&shared_llm),
                Arc::clone(&settings),
            );

            let shared = Arc::new(Shared::with_parts(
                audio_tx.clone(),
                shared_tts,
                tts_failed,
                is_speaking,
                stop_audio_flag,
                shared_llm,
                settings,
            ));
            app.manage(Arc::clone(&shared));

            // Мост событий движка в UI + журнал.
            let (events_tx, events_rx) = channel::<JarvisEvent>();
            let shared_events = Arc::clone(&shared);
            let bridge = handle.clone();
            std::thread::spawn(move || {
                for ev in events_rx {
                    shared_events.push_log(format!("{ev:?}"));
                    let _ = bridge.emit("engine-event", &ev);
                }
            });

            tts::prune_tts_cache();
            spawn_engine(&shared, events_tx);
            build_tray(&handle)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // Закрытие окна сворачивает приложение в трей, а не завершает его.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            status,
            set_paused,
            get_commands,
            save_commands,
            get_settings,
            save_settings,
            get_logs,
            tts_voices,
            tts_test,
            llm_ask,
            llm_cancel,
            ww_samples,
            ww_record,
            ww_delete,
            ww_build,
            ww_apply,
            ww_test,
        ])
        .run(tauri::generate_context!())
        .expect("Не удалось запустить Jarvis");
}
