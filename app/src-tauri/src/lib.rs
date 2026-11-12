//! Tauri-бэкенд Jarvis: запускает движок из jarvis-core, мостит события в UI
//! и предоставляет IPC-команды (команды, настройки, wake word, TTS, LLM).

use jarvis_core::commands::{CommandsConfig, spawn_config_watcher};
use jarvis_core::engine::{EngineHandles, run_engine};
use jarvis_core::events::JarvisEvent;
use jarvis_core::llm::{self, ChatTurn, SharedLlmState, spawn_llm_loader};
use jarvis_core::paths;
use jarvis_core::settings::{Settings, spawn_settings_watcher_with};
use jarvis_core::stt::Stt;
use jarvis_core::tts::{self, AudioMessage, SharedTtsState};
use jarvis_core::wake_train;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
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
    /// Флаг пересборки голосового профиля (после записи новых образцов).
    pub speaker_invalidate: Arc<AtomicBool>,
    /// Диктант для профиля голоса: запись идёт / пользователь нажал «Стоп».
    pub voice_recording: Arc<AtomicBool>,
    pub voice_stop: Arc<AtomicBool>,
    /// Ленивая инстанция FastConformer для проверки диктанта: движковый STT
    /// принадлежит своему воркеру, а вторая загрузка (mmap-веса) дешёвая.
    pub train_stt: Mutex<Option<jarvis_core::stt::Stt>>,
    pub audio_tx: std::sync::mpsc::Sender<AudioMessage>,
    /// Канал событий движка: нужен хабу таймеров для окна «Утилиты».
    pub utils_events: Mutex<Option<std::sync::mpsc::Sender<JarvisEvent>>>,
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
        let commands = CommandsConfig::load(&paths::commands_file()).unwrap_or_else(|e| {
            eprintln!("⚠️ {e:#}; использую пустой список команд");
            CommandsConfig::default()
        });
        commands.apply_ringtones();

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
            speaker_invalidate: Arc::new(AtomicBool::new(false)),
            voice_recording: Arc::new(AtomicBool::new(false)),
            voice_stop: Arc::new(AtomicBool::new(false)),
            train_stt: Mutex::new(None),
            audio_tx,
            utils_events: Mutex::new(None),
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

/// Распаковывает подготовленный архив из ресурсов приложения в каталог данных
/// при первом запуске (звуки, конфиги, крошечные wake word-модели).
/// Тяжёлые модели (LLM/STT/TTS) в архив не входят и качаются отдельно при
/// первом обращении. Идемпотентно по файлу-маркеру.
fn provision_data(events: &std::sync::mpsc::Sender<JarvisEvent>, resource_dir: Option<&Path>) {
    let home = paths::base_dir();
    // Маркер — собственная метка, которую ставим после успешной распаковки.
    // Привязываться к файлу модели нельзя: тяжёлые модели больше не в архиве
    // (качаются при первом запуске), а сам архив — это звуки, конфиги и
    // крошечные wake word-модели.
    let marker = home.join(".jarvis_provisioned");
    if marker.exists() {
        return;
    }
    let Some(payload) = find_payload(resource_dir) else {
        return;
    };

    let _ = events.send(JarvisEvent::Log {
        message: "Первый запуск: распаковываю ресурсы (один раз)…".into(),
    });
    if let Err(e) = std::fs::create_dir_all(&home) {
        let _ = events.send(JarvisEvent::Error {
            message: format!("Не удалось создать {}: {e}", home.display()),
        });
        return;
    }

    match std::process::Command::new("/usr/bin/tar")
        .arg("-xzf")
        .arg(&payload)
        .arg("-C")
        .arg(&home)
        .status()
    {
        Ok(s) if s.success() => {
            let _ = std::fs::write(&marker, "1");
            let _ = events.send(JarvisEvent::Log {
                message: "Ресурсы распакованы.".into(),
            });
        }
        other => {
            let _ = events.send(JarvisEvent::Error {
                message: format!("Распаковка не удалась: {other:?}"),
            });
        }
    }
}

/// Ищет подготовленный архив в `Contents/Resources/payload`.
fn find_payload(resource_dir: Option<&Path>) -> Option<PathBuf> {
    let candidate = resource_dir?.join("payload/jarvis-data.tar.gz");
    candidate.exists().then_some(candidate)
}

/// Загружает модели и запускает движок ассистента в фоновом потоке.
fn spawn_engine(
    shared: &Arc<Shared>,
    events_tx: std::sync::mpsc::Sender<JarvisEvent>,
    resource_dir: Option<PathBuf>,
) {
    let shared_engine = Arc::clone(shared);

    spawn_config_watcher(paths::commands_file(), Arc::clone(&shared_engine.commands));
    let llm_for_watcher = Arc::clone(&shared_engine.shared_llm);
    let events_for_watcher = events_tx.clone();
    spawn_settings_watcher_with(paths::settings_file(), Arc::clone(&shared_engine.settings), move |_| {
        // Выбрал облако в мастере/настройках — LLM переключается сразу,
        // а не после перезапуска (и не ждёт, пока качается локальная).
        jarvis_core::llm::reload_llm(&llm_for_watcher, &events_for_watcher);
    });

    std::thread::spawn(move || {
        // Канал для хаба таймеров: окно «Утилиты» получает события.
        *shared_engine
            .utils_events
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(events_tx.clone());

        provision_data(&events_tx, resource_dir.as_deref());

        // Конфиги могли появиться только что из архива — перечитываем их.
        if let Ok(cfg) = CommandsConfig::load(&paths::commands_file()) {
            cfg.apply_ringtones();
            *shared_engine
                .commands
                .write()
                .unwrap_or_else(|e| e.into_inner()) = cfg;
        }
        if let Ok(s) = Settings::load(&paths::settings_file()) {
            s.apply();
            *shared_engine
                .settings
                .write()
                .unwrap_or_else(|e| e.into_inner()) = s;
        }

        // Фильтр говорящего: пайплайн speakrs (CoreML на macOS) грузится
        // вместе с движком; профиль строим сразу, а не при первой фразе —
        // иначе первые фразы после запуска проверялись бы с задержкой.
        let speaker_gate = jarvis_core::speaker::load_gate(&events_tx);
        let (speaker_tx, speaker_rx) = jarvis_core::speaker::spawn_speaker_worker(
            speaker_gate,
            Arc::clone(&shared_engine.speaker_invalidate),
        );
        let _ = speaker_tx.send(jarvis_core::speaker::SpeakerQuery::Rebuild);

        let tts_voice = {
            let s = shared_engine
                .settings
                .read()
                .unwrap_or_else(|e| e.into_inner());
            s.apply();
            s.tts_voice.clone()
        };

        tts::spawn_tts_loader(
            &shared_engine.shared_tts,
            &shared_engine.tts_failed,
            &tts_voice,
        );

        let _ = events_tx.send(JarvisEvent::Log {
            message: "TTS-сборка: silero v5 + нейро-ударитель, голос «{voice}»"
                .replace("{voice}", &tts_voice),
        });
        spawn_llm_loader(&shared_engine.shared_llm, &events_tx);

        // Модели живут в системной папке программы; старые (рядом с
        // настройками) переносятся туда один раз.
        paths::migrate_models();
        let models_dir = paths::models_dir();
        let model = match Stt::load(&models_dir) {
            Ok(model) => model,
            Err(e) => {
                shared_engine.engine_running.store(false, Ordering::SeqCst);
                // Причина обязана дойти до журнала: без неё «не удалось
                // загрузить» не отличить битых весов от отсутствующих.
                eprintln!("❌ STT: {e:#}");
                let _ = events_tx.send(JarvisEvent::Error {
                    message: format!("Не удалось загрузить STT-модель: {e:#}"),
                });
                return;
            }
        };

        // Клон канала для ошибки после moves: events_tx уезжает в handles,
        // а причина падения движка должна дойти до UI.
        let error_tx = events_tx.clone();
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
            speaker_tx,
            speaker_rx,
        };

        shared_engine.engine_running.store(true, Ordering::SeqCst);
        if let Err(e) = run_engine(model, handles) {
            eprintln!("❌ Движок остановился: {e:#}");
            shared_engine.engine_running.store(false, Ordering::SeqCst);
            // Раньше причина уходила только в stderr терминала: при падении
            // аудио-потока (например, macOS молча забрал доступ к микрофону
            // после пересборки) интерфейс выглядел живым, а микрофон не
            // работал. Показываем причину и в журнале UI.
            let _ = error_tx.send(JarvisEvent::Error {
                message: format!("Движок остановился: {e:#}"),
            });
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
    /// Рабочая директория агента: от неё считаются относительные пути
    /// файловых инструментов, её же меняет `cd` в shell.
    cwd: String,
}

// =========================================================================
// Модели LLM: статус, явная загрузка (без авто-качания), перезагрузка
// =========================================================================

/// Одновременных загрузок не ведём: повторный клик по «Скачать» игнорируется
/// с понятным сообщением, а не плодит второй поток на тот же архив.
static LLM_DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// Запись о модели для UI: установлена ли, размер, можно ли скачать кнопкой.
#[derive(serde::Serialize)]
struct LlmModelInfo {
    name: String,
    installed: bool,
    /// Размер каталога модели, МБ (None — ещё не скачана).
    size_mb: Option<u64>,
    /// Есть ли для модели кнопка «Скачать» (каталог загрузок приложения).
    downloadable: bool,
}

fn dir_size_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            match entry.metadata() {
                Ok(meta) if meta.is_file() => total += meta.len(),
                Ok(meta) if meta.is_dir() => total += dir_size_bytes(&entry.path()),
                _ => {}
            }
        }
    }
    total
}

fn dir_size_mb(dir: &Path) -> u64 {
    dir_size_bytes(dir) / 1_048_576
}

/// Каталог моделей для UI: качаемый из релиза Qwen3.5-0.8B (candle) плюс
/// всё, что пользователь опустил в models/ вручную (candle-совместимое:
/// config.json + tokenizer.json).
#[tauri::command]
fn llm_models_status() -> Vec<LlmModelInfo> {
    let models_dir = paths::models_dir();
    let mut out = Vec::new();

    let qwen_dir = models_dir.join("qwen3.5-0.8b");
    let qwen_installed = jarvis_core::model_fetch::qwen35_ready(&qwen_dir);
    out.push(LlmModelInfo {
        name: "qwen3.5-0.8b".into(),
        installed: qwen_installed,
        size_mb: qwen_installed.then(|| dir_size_mb(&qwen_dir)),
        downloadable: true,
    });

    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten().map(|e| e.path()) {
            let Some(name) = entry.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name == "qwen3.5-0.8b" || name == "silero" || name == "fastconformer" {
                continue;
            }
            if entry.join("config.json").is_file() && entry.join("tokenizer.json").is_file() {
                out.push(LlmModelInfo {
                    name: name.to_string(),
                    installed: true,
                    size_mb: Some(dir_size_mb(&entry)),
                    downloadable: false,
                });
            }
        }
    }
    out
}

/// Явная загрузка модели по решению пользователя. Прогресс уходит в UI
/// событиями download_progress; по завершении LLM перезагружается сама,
/// без перезапуска приложения.
#[tauri::command]
fn llm_download_model(
    app: tauri::AppHandle,
    shared: State<'_, Arc<Shared>>,
    name: String,
) -> Result<String, String> {
    // Загружать можно только то, что приложение знает откуда: локальные
    // каталоги пользователь наполняет сам.
    if name != "qwen3.5-0.8b" {
        return Err(format!(
            "Модель «{name}» не скачивается из приложения: опусти её каталог с config.json, tokenizer.json и весами в {}",
            paths::models_dir().display()
        ));
    }
    if LLM_DOWNLOADING.swap(true, Ordering::SeqCst) {
        return Err("Загрузка уже идёт — дождись окончания".into());
    }

    let (tx, rx) = channel::<JarvisEvent>();
    let llm = Arc::clone(&shared.shared_llm);
    let app2 = app.clone();
    let name2 = name.clone();
    std::thread::spawn(move || {
        let name = name2;
        // Прогресс — напрямую в UI: событие движка и мост не нужны.
        let mut last: u8 = 0;
        let result = jarvis_core::model_fetch::ensure_qwen35_with_progress(
            &paths::models_dir().join(&name),
            &mut |pct: u8| {
                if pct >= last + 2 || pct >= 100 {
                    last = pct;
                    let _ = app.emit(
                        "engine-event",
                        &JarvisEvent::DownloadProgress {
                            what: format!("Модель {name}"),
                            percent: pct,
                        },
                    );
                }
            },
        );
        let _ = app.emit(
            "engine-event",
            &JarvisEvent::DownloadProgress {
                what: format!("Модель {name}"),
                percent: 100,
            },
        );
        match result {
            Ok(()) => {
                let _ = tx.send(JarvisEvent::Log {
                    message: format!("Модель {name} скачана, загружаю движок"),
                });
                // Движок перечитает модель без перезапуска приложения.
                spawn_llm_loader(&llm, &tx);
                for ev in rx {
                    let _ = app2.emit("engine-event", &ev);
                }
            }
            Err(e) => {
                let _ = tx.send(JarvisEvent::Error {
                    message: format!("Загрузка модели не удалась: {e:#}"),
                });
                for ev in rx {
                    let _ = app2.emit("engine-event", &ev);
                }
            }
        }
        LLM_DOWNLOADING.store(false, Ordering::SeqCst);
    });
    Ok(format!("Скачиваю {name} — прогресс появится в интерфейсе"))
}

/// Перезагрузка LLM без перезапуска приложения: после смены модели в
/// настройках или провала первой загрузки.
#[tauri::command]
fn llm_reload(app: tauri::AppHandle, shared: State<'_, Arc<Shared>>) -> Result<String, String> {
    let (tx, rx) = channel::<JarvisEvent>();
    let llm = Arc::clone(&shared.shared_llm);
    std::thread::spawn(move || {
        spawn_llm_loader(&llm, &tx);
        for ev in rx {
            let _ = app.emit("engine-event", &ev);
        }
    });
    Ok("Перезагружаю модель".into())
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
        cwd: jarvis_core::tools::cwd(),
    }
}

/// Текущая рабочая директория агента.
#[tauri::command]
fn get_cwd() -> String {
    jarvis_core::tools::cwd()
}

/// Сменить рабочую директорию агента. Пустая строка или несуществующий путь
/// возвращают её к корню данных Джарвиса.
#[tauri::command]
fn set_cwd(path: String) -> String {
    jarvis_core::tools::set_cwd(&path)
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
    *shared.commands.write().unwrap_or_else(|e| e.into_inner()) = config;
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
    // Смена фразы-активации, режима детектора или длины окна: детектор
    // пересобирается на ходу, не дожидаясь перезапуска движка.
    let wake_changed = {
        let cur = shared.settings.read().unwrap_or_else(|e| e.into_inner());
        cur.wake_word != settings.wake_word
            || cur.wake_detector != settings.wake_detector
            || cur.wake_window_ms != settings.wake_window_ms
    };
    settings
        .save(&paths::settings_file())
        .map_err(|e| e.to_string())?;
    settings.apply();
    *shared.settings.write().unwrap_or_else(|e| e.into_inner()) = settings;

    if voice_changed {
        tts::set_voice(&shared.shared_tts, &shared.tts_failed, &voice);
    }
    if wake_changed {
        shared.wakeword_reload.store(true, Ordering::SeqCst);
    }
    Ok(())
}

#[tauri::command]
fn get_logs(shared: State<'_, Arc<Shared>>) -> Vec<String> {
    shared.logs()
}

// ---------- Задачник (экран «Память») ----------

/// Задача из задачника для UI.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TaskDto {
    id: usize,
    text: String,
    done: bool,
}

impl From<jarvis_core::tasks::Task> for TaskDto {
    fn from(t: jarvis_core::tasks::Task) -> Self {
        Self {
            id: t.id,
            text: t.text,
            done: t.done,
        }
    }
}

#[tauri::command]
fn tasks_list() -> Vec<TaskDto> {
    jarvis_core::tasks::load().into_iter().map(TaskDto::from).collect()
}

#[tauri::command]
fn tasks_add(text: String) -> Result<TaskDto, String> {
    jarvis_core::tasks::add(&text)
        .map(TaskDto::from)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn tasks_set_done(id: usize, done: bool) -> Option<TaskDto> {
    jarvis_core::tasks::set_done(id, done).map(TaskDto::from)
}

#[tauri::command]
fn tasks_remove(id: usize) -> Option<TaskDto> {
    jarvis_core::tasks::remove(id).map(TaskDto::from)
}

/// Открывает tasks.md в системном редакторе — правки подхватятся движком.
#[tauri::command]
fn tasks_open() -> Result<(), String> {
    open::that(jarvis_core::tasks::path()).map_err(|e| e.to_string())
}

#[tauri::command]
fn tts_voices() -> Vec<(String, String)> {
    tts::SILERO_VOICES
        .iter()
        .map(|(id, desc)| (id.to_string(), desc.to_string()))
        .collect()
}

/// Открывает ссылку из ответа ассистента в системном браузере.
/// Принимаются только http/https — остальное в вебвью не приходит.
#[tauri::command]
fn open_external(url: String) -> Result<(), String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("Открывать можно только ссылки http(s)".into());
    }
    open::that(url).map_err(|e| e.to_string())
}

/// Озвучивает текст. С параметром `voice` — превью другого диктора:
/// выбранный голос приложения не меняет, в кэш речи не пишется.
#[tauri::command]
fn tts_test(shared: State<'_, Arc<Shared>>, text: String, voice: Option<String>) {
    match voice {
        Some(v) if !v.trim().is_empty() && v != tts::current_voice() => {
            let _ = shared.audio_tx.send(AudioMessage::PreviewVoice { text, voice: v });
        }
        _ => {
            let _ = shared.audio_tx.send(AudioMessage::SynthesizeAndPlay(text));
        }
    }
}

/// Открывает панель «Конфиденциальность → Микрофон» в Системных настройках.
/// Начиная с Ventura Настройки переписаны: старый путь панели молча не
/// открывается, поэтому пробуем новый адрес, затем старый, затем само
/// приложение.
fn open_mic_pane() -> Result<(), String> {
    let urls = [
        "x-apple.systempreferences:com.apple.settings.privacy?Privacy_Microphone",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    ];
    for url in urls {
        if open::that(url).is_ok() {
            return Ok(());
        }
    }
    open::that("/System/Applications/System Settings.app").map_err(|e| e.to_string())
}

/// Открывает панель «Конфиденциальность → Микрофон» в Системных настройках —
/// кнопка в меню доступа к микрофону.
#[tauri::command]
fn open_mic_settings() -> Result<(), String> {
    open_mic_pane()
}

/// Статус доступа к микрофону для строки в Настройках.
#[tauri::command]
fn mic_status() -> String {
    jarvis_core::audio::mic_status().to_string()
}

/// Повторный запрос доступа к микрофону из Настроек. После отказа macOS
/// диалог уже не показывает — тогда сразу открываем Системные настройки.
/// Статус пишем в журнал UI: раньше «кнопка ничего не сделала» было не видно.
#[tauri::command]
fn mic_request(app: AppHandle) -> String {
    let status = jarvis_core::audio::mic_request_access();
    let _ = app.emit(
        "engine-event",
        &JarvisEvent::Log {
            message: format!("Запрос доступа к микрофону: статус «{status}»"),
        },
    );
    if status == "denied" {
        let _ = open_mic_pane();
    }
    status.to_string()
}

/// Окно «Утилиты» (таймер/секундомер/календарь): создаём при первом
/// обращении, дальше показываем и поднимаем наверх.
fn show_utils_window(app: &tauri::AppHandle, tab: Option<&str>) {
    use tauri::Manager;
    if let Some(window) = app.get_webview_window("utils") {
        let _ = window.show();
        let _ = window.set_focus();
    } else {
        let _ = tauri::WebviewWindowBuilder::new(
            app,
            "utils",
            tauri::WebviewUrl::App("index.html".into()),
        )
        .title("Джарвис — утилиты")
        .inner_size(340.0, 560.0)
        .build();
    }
    if let Some(tab) = tab {
        let _ = app.emit("utils-tab", tab);
    }
}

/// Таймер из окна утилит (ручная кнопка, не голос).
#[tauri::command]
fn timer_start(
    shared: State<'_, Arc<Shared>>,
    label: Option<String>,
    seconds: u64,
) -> Result<u64, String> {
    let events = shared
        .utils_events
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .ok_or_else(|| "Движок ещё не запущен".to_string())?;
    Ok(jarvis_core::timers::start(
        &events,
        &shared.audio_tx,
        label.as_deref().unwrap_or(""),
        seconds,
        jarvis_core::timers::TimerKind::Timer,
    ))
}

/// Отмена таймера из окна утилит.
#[tauri::command]
fn timer_cancel(shared: State<'_, Arc<Shared>>, id: u64) -> Result<(), String> {
    let events = shared
        .utils_events
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .ok_or_else(|| "Движок ещё не запущен".to_string())?;
    jarvis_core::timers::cancel(&events, &shared.audio_tx, None, Some(id))
        .map(|_| ())
        .ok_or_else(|| "Таймер не найден".to_string())
}

/// Закрытие окна «Утилиты» из интерфейса (крестик).
#[tauri::command]
fn utils_close(app: AppHandle) {
    use tauri::Manager;
    if let Some(window) = app.get_webview_window("utils") {
        let _ = window.hide();
    }
}

/// Проверка облачной модели минимальным запросом — для кнопки «Проверить».
/// Конфиг берётся из статиков, которые `Settings::apply` уже обновил
/// (кнопка живёт после сохранения настроек).
#[tauri::command]
fn llm_api_test() -> Result<String, String> {
    let (tx, _rx) = channel::<JarvisEvent>();
    let engine = jarvis_core::llm::api_backend::ApiEngine::load(&tx).map_err(|e| format!("{e:#}"))?;
    engine.check_connection().map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn llm_ask(shared: State<'_, Arc<Shared>>, app: AppHandle, prompt: String) {    // Номер обращения: фоновый веб-поиск проверит его перед озвучкой.
    let req_id = jarvis_core::llm::new_request();
    // События этого вопроса уходят и в UI, и в общий мост событий движка.
    let (tx, rx) = channel::<JarvisEvent>();
    std::thread::spawn(move || {
        for ev in rx {
            // Текстовый чат минует мост движка: окно утилит открываем здесь,
            // иначе «открой утилиты» с клавиатуры только скажет о команде.
            match &ev {
                JarvisEvent::OpenUtils { tab } => show_utils_window(&app, Some(tab)),
                JarvisEvent::CloseUtils => {
                    if let Some(window) = app.get_webview_window("utils") {
                        let _ = window.hide();
                    }
                }
                _ => {}
            }
            let _ = app.emit("engine-event", &ev);
        }
    });

    // Свежий токен на каждый запрос: «Стоп» отменяет текущий вопрос,
    // и гонки с обнулением флага внутри генерации нет.
    let cancel = Arc::new(AtomicBool::new(false));
    *shared.chat_cancel.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::clone(&cancel));

    // Свежие данные (погода, курсы, «кто такой X») — тот же веб-флоу, что и
    // в голосовом движке: голос этот путь проходит до ask, текстовый чат
    // приходит напрямую, поэтому проверяем здесь.
    let web_search_auto = shared
        .settings
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .web_search_auto;
    let explicit_web = jarvis_core::router::explicit_web_request(&prompt);
    let fresh_info = jarvis_core::web::looks_like_fresh_info(&prompt)
        && !jarvis_core::router::calendar_question(&prompt);
    if explicit_web || (fresh_info && web_search_auto) {
        let query = jarvis_core::router::web_query_from(&prompt)
            .or_else(jarvis_core::topic::last_question)
            .unwrap_or_else(|| prompt.clone());
        println!("🌐 Текстовый чат: ищу в интернете: \"{query}\"");
        let llm = Arc::clone(&shared.shared_llm);
        let history = Arc::clone(&shared.chat_history);
        let audio_tx = shared.audio_tx.clone();
        std::thread::spawn(move || match jarvis_core::web::search(&query, 5) {
            Ok(hits) => {
                if !jarvis_core::llm::is_current_request(req_id)
                    || cancel.load(Ordering::SeqCst)
                {
                    println!("🚫 Поиск устарел — отвечаю молча");
                    return;
                }
                if let Some(web_prompt) = jarvis_core::web::build_summary_prompt(&query, &hits) {
                    llm::ask(
                        llm,
                        llm::WEB_SYSTEM_PROMPT,
                        web_prompt,
                        history,
                        audio_tx,
                        tx,
                        cancel,
                    );
                }
            }
            Err(e) => eprintln!("⚠️ Веб-поиск не удался: {e}"),
        });
        return;
    }

    // `CHAT_SYSTEM_PROMPT` — базовый разговорный промпт. Внутри `ask`
    // роутер сам подменит его на короткий tool-промпт, если запрос
    // требует действия (таймер, приложение и т.п.).
    llm::ask(
        Arc::clone(&shared.shared_llm),
        llm::CHAT_SYSTEM_PROMPT,
        prompt,
        Arc::clone(&shared.chat_history),
        shared.audio_tx.clone(),
        tx,
        cancel,
    );
}

#[tauri::command]
fn llm_cancel(shared: State<'_, Arc<Shared>>) {
    // Тот же порядок, что и у голосовой команды «замолчи» (engine.rs):
    // флаг останавливает уже играющий кусок (плеер — один поток и внутри
    // play_wav/wait_drain не читает очередь, поэтому без флага Stop лежит
    // в очереди до конца фразы), llm_cancel рвёт генерацию и голосового,
    // и текстового пути, chat_cancel — токен текущего чат-запроса.
    shared.stop_audio_flag.store(true, Ordering::SeqCst);
    shared.llm_cancel.store(true, Ordering::SeqCst);
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
    // Диктанты профиля голоса (train/voice) идут в общий список train:
    // статус в настройках считает всё, что строит профиль.
    let mut train = fmt(&train_dir);
    train.extend(fmt(&wake_train::voice_samples_dir()));
    WwSamples {
        train,
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
                // Новый образец — голосовой профиль фильтра говорящего
                // пересобирается при следующей проверке.
                shared
                    .speaker_invalidate
                    .store(true, std::sync::atomic::Ordering::SeqCst);
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
        Err(format!("Путь {:?} вне каталога {:?}", path, base_canonical))
    }
}

#[tauri::command]
fn ww_delete(path: String, shared: State<'_, Arc<Shared>>) -> Result<(), String> {
    let file = ensure_under_dir(Path::new(&path), &paths::train_dir())?;
    std::fs::remove_file(&file).map_err(|e| e.to_string())?;
    // Удалённый образец не должен жить в профиле голоса до следующей записи.
    shared
        .speaker_invalidate
        .store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(())
}

// ---------- Активационная фраза ----------

/// Синтезировать фразу разными дикторами Silero и проверить, слышит ли её
/// модель: сколько вариантов произнесения распознано буквально.
#[tauri::command]
fn wake_phrase_generate(phrase: String) -> Result<jarvis_core::wake_phrase::PhraseReport, String> {
    // Отдельный экземпляр ASR: веса в mmap, второй экземпляр дёшев.
    let stt = jarvis_core::stt::Stt::load(&paths::models_dir()).map_err(|e| e.to_string())?;
    let stt = std::sync::Arc::new(std::sync::Mutex::new(stt));
    jarvis_core::wake_phrase::generate_and_check(&stt, &phrase).map_err(|e| format!("{e:#}"))
}

// ---------- Диктант для профиля голоса ----------

/// Гвардия диктанта: на выходе (успех, ошибка, паника) снимает флаг записи
/// и возвращает движок из паузы — чтение фразы не должно оставлять Джарвиса
/// глухим, даже если что-то упало.
struct VoiceRecordGuard {
    shared: Arc<Shared>,
}

impl Drop for VoiceRecordGuard {
    fn drop(&mut self) {
        self.shared.voice_recording.store(false, Ordering::SeqCst);
        self.shared.paused.store(false, Ordering::SeqCst);
    }
}

/// Начать запись диктанта: микрофон пишет, пока пользователь не нажмёт
/// «Стоп» ([`voice_record_stop`]). Движок на время записи на паузе.
#[tauri::command]
fn voice_record_start(shared: State<'_, Arc<Shared>>, app: AppHandle) -> Result<(), String> {
    if shared.voice_recording.swap(true, Ordering::SeqCst) {
        return Err("Запись уже идёт".into());
    }
    shared.voice_stop.store(false, Ordering::SeqCst);
    shared.paused.store(true, Ordering::SeqCst);
    let _ = app.emit(
        "engine-event",
        &JarvisEvent::Log {
            message: "Диктант: запись пошла — читай фразу и нажми «Стоп»".into(),
        },
    );

    let shared_arc = Arc::clone(&shared);
    std::thread::spawn(move || {
        let _guard = VoiceRecordGuard {
            shared: Arc::clone(&shared_arc),
        };
        let result = check_dictation_take(&shared_arc);
        match result {
            Ok((overlap, transcript)) => {
                shared_arc
                    .speaker_invalidate
                    .store(true, Ordering::SeqCst);
                let _ = app.emit(
                    "engine-event",
                    &JarvisEvent::VoiceCheck {
                        ok: true,
                        overlap,
                        transcript,
                    },
                );
            }
            Err(VoiceTakeError::Mismatch(overlap, transcript)) => {
                let _ = app.emit(
                    "engine-event",
                    &JarvisEvent::VoiceCheck {
                        ok: false,
                        overlap,
                        transcript,
                    },
                );
            }
            Err(VoiceTakeError::Failed(message)) => {
                let _ = app.emit(
                    "engine-event",
                    &JarvisEvent::Error {
                        message: format!("Диктант не удался: {message}"),
                    },
                );
            }
        }
    });
    Ok(())
}

/// Пользователь дочитал: остановить запись и проверить.
#[tauri::command]
fn voice_record_stop(shared: State<'_, Arc<Shared>>) {
    shared.voice_stop.store(true, Ordering::SeqCst);
}

/// Чем закончилась проверка диктанта.
enum VoiceTakeError {
    /// Прочитано не то: совпадение ниже порога, запись удалена.
    Mismatch(f32, String),
    /// Техническая ошибка (микрофон, диск, модель).
    Failed(String),
}

/// Порог совпадения слов с фразой, ниже которого диктант не принимается.
const VOICE_OVERLAP_MIN: f32 = 0.5;

/// Записать до «Стоп», расшифровать FastConformer'ом, сверить с фразой.
/// Успех — файл остаётся в `train/voice/`, профиль готов к пересборке.
fn check_dictation_take(shared: &Shared) -> Result<(f32, String), VoiceTakeError> {
    let path = wake_train::record_until_stop(
        &wake_train::voice_samples_dir(),
        "voice",
        &shared.voice_stop,
    )
    .map_err(|e| VoiceTakeError::Failed(format!("{e:#}")))?;

    // Проверка прочитанного. ASR недоступен — принимаем без сверки, но
    // честно помечаем в расшифровке.
    let mut stt_guard = shared.train_stt.lock().unwrap_or_else(|e| e.into_inner());
    if stt_guard.is_none() {
        println!("📥 Загрузка FastConformer для проверки диктанта…");
        match jarvis_core::stt::Stt::load(&paths::models_dir()) {
            Ok(stt) => *stt_guard = Some(stt),
            Err(e) => {
                eprintln!("⚠️ Проверка диктанта недоступна: {e:#}");
                return Ok((0.0, "проверка недоступна — принято без сверки".into()));
            }
        }
    }
    let transcript = match stt_guard
        .as_mut()
        .expect("STT загружен выше")
        .transcribe(&jarvis_core::speaker::read_wav_16k(&path).map_err(|e| {
            VoiceTakeError::Failed(format!("запись не читается: {e:#}"))
        })?) {
        Ok(t) => t,
        Err(e) => return Err(VoiceTakeError::Failed(format!("расшифровка не удалась: {e:#}"))),
    };

    let overlap =
        wake_train::phrase_overlap(wake_train::VOICE_TRAINING_PHRASE, &transcript);
    if overlap < VOICE_OVERLAP_MIN {
        // Прочитано не то (шум, чужой текст, слишком тихо): файл не храним,
        // профиль не портим.
        let _ = std::fs::remove_file(&path);
        return Err(VoiceTakeError::Mismatch(overlap, transcript));
    }
    println!(
        "🎙️ Диктант принят: совпадение {:.0}% — профиль голоса пересоберётся",
        overlap * 100.0
    );
    Ok((overlap, transcript))
}

#[tauri::command]
fn ww_build(app: AppHandle) -> Result<(), String> {
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
        None => paths::models_dir().join("jarvis_model_trained.rpw"),
    };
    if trained.extension().is_some_and(|x| x != "rpw") {
        return Err("Ожидался файл модели .rpw".into());
    }

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

fn app_data_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        #[cfg(target_os = "macos")]
        return home.join("Library/Application Support/Jarvis");
        #[cfg(not(target_os = "macos"))]
        return home.join(".jarvis");
    }
    PathBuf::from(".")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    jarvis_core::init_tuning();

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

    // Рабочая директория агента — домашний каталог пользователя. Без этого
    // относительные пути файловых инструментов смотрели в каталог запуска
    // приложения внутри .app, где ничего нет и куда нельзя писать.
    if let Some(home) = std::env::var_os("HOME") {
        jarvis_core::tools::set_cwd(&home.to_string_lossy());
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            let handle = app.handle().clone();

            // 1. Канал событий создаём ПЕРВЫМ: он нужен и плееру (для AudioLevel),
            //    и движку, и мосту в UI.
            let (events_tx, events_rx) = channel::<JarvisEvent>();

            // 2. Состояния, которые плеер и движок делят между собой.
            let shared_tts: Arc<SharedTtsState> = Arc::new(SharedTtsState::new());
            let tts_failed = Arc::new(AtomicBool::new(false));
            let is_speaking = Arc::new(AtomicBool::new(false));
            // Клон для наблюдателя фазы «говорит»: сам флаг уедет в состояние.
            let speaking_flag = Arc::clone(&is_speaking);
            let stop_audio_flag = Arc::new(AtomicBool::new(false));

            let shared_llm: Arc<SharedLlmState> = Arc::new(SharedLlmState::new());

            let settings = Arc::new(RwLock::new(
                Settings::load(&paths::settings_file()).unwrap_or_else(|e| {
                    eprintln!("⚠️ {e:#}; использую настройки по умолчанию");
                    Settings::default()
                }),
            ));
            settings.read().unwrap_or_else(|e| e.into_inner()).apply();

            // 3. Плеер: первым аргументом идёт клон events_tx (нужен для
            //    AudioLevel при воспроизведении). Возвращает Sender очереди
            //    AudioMessage — через него движок и UI отправляют фразы.
            let audio_tx = tts::spawn_audio_player(
                events_tx.clone(),
                Arc::clone(&shared_tts),
                Arc::clone(&tts_failed),
                Arc::clone(&is_speaking),
                Arc::clone(&stop_audio_flag),
                Arc::clone(&shared_llm),
                Arc::clone(&settings),
            );

            // 4. Общее состояние приложения.
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

            // 5. Мост событий движка в UI + журнал.
            let shared_events = Arc::clone(&shared);
            let bridge = handle.clone();
            std::thread::spawn(move || {
                for ev in events_rx {
                    shared_events.push_log(format!("{ev:?}"));
                    // Таймеры и утилиты живут в отдельном окне: создаём/показываем.
                    match &ev {
                        JarvisEvent::TimerStarted { .. } | JarvisEvent::TimerStopped { .. } => {
                            show_utils_window(&bridge, None);
                        }
                        JarvisEvent::TimerFired { label, kind, .. } => {
                            show_utils_window(&bridge, None);
                            use tauri_plugin_notification::NotificationExt;
                            let (title, body) = match kind {
                                jarvis_core::timers::TimerKind::Alarm => (
                                    "Будильник",
                                    if label.is_empty() {
                                        "Пора вставать!".to_string()
                                    } else {
                                        label.to_string()
                                    },
                                ),
                                jarvis_core::timers::TimerKind::Timer => (
                                    "Таймер",
                                    if label.is_empty() {
                                        "Время вышло".to_string()
                                    } else {
                                        format!("{label} — время вышло")
                                    },
                                ),
                            };
                            let _ = bridge
                                .notification()
                                .builder()
                                .title(title)
                                .body(body)
                                .show();
                        }
                        JarvisEvent::OpenUtils { tab } => {
                            show_utils_window(&bridge, Some(tab));
                        }
                        JarvisEvent::CloseUtils => {
                            use tauri::Manager;
                            if let Some(window) = bridge.get_webview_window("utils") {
                                let _ = window.hide();
                            }
                        }
                        _ => {}
                    }
                    let _ = bridge.emit("engine-event", &ev);
                }
            });

            // 5б. Фаза «говорит». В движке для неё нет состояния: пока TTS
            //     играет, UI считал, что Джарвис готов и ждёт. Следим за
            //     флагом озвучки и сообщаем интерфейсу только на смене.
            let speaking_flag = Arc::clone(&speaking_flag);
            let speak_bridge = handle.clone();
            std::thread::spawn(move || {
                let mut last = false;
                loop {
                    std::thread::sleep(Duration::from_millis(80));
                    let now = speaking_flag.load(Ordering::SeqCst);
                    if now != last {
                        last = now;
                        let _ = speak_bridge.emit("speech", serde_json::json!({ "speaking": now }));
                    }
                }
            });
            tts::prune_tts_cache();

            let resource_dir = handle.path().resource_dir().ok();
            spawn_engine(&shared, events_tx, resource_dir);
            build_tray(&handle)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            status,
            set_paused,
            get_cwd,
            set_cwd,
            get_commands,
            save_commands,
            llm_api_test,
            voice_record_start,
            voice_record_stop,
            get_settings,
            save_settings,
            get_logs,
            tasks_list,
            tasks_add,
            tasks_set_done,
            tasks_remove,
            tasks_open,
            tts_voices,
            tts_test,
            open_external,
            llm_ask,
            llm_cancel,
            llm_models_status,
            llm_download_model,
            llm_reload,
            open_mic_settings,
            mic_status,
            mic_request,
            timer_start,
            timer_cancel,
            utils_close,
            ww_samples,
            ww_record,
            ww_delete,
            ww_build,
            ww_apply,
            ww_test,
            wake_phrase_generate,
        ])
        .run(tauri::generate_context!())
        .expect("Не удалось запустить Jarvis");
}
