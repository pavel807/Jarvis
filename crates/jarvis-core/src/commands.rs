//! Команды из commands.toml: загрузка, hot-reload, дословный матч, выполнение.

use crate::events::{EventTx, JarvisEvent};
use crate::llm::{self, ChatTurn, SharedLlm};
use crate::paths;
use crate::text::{clean_text, clean_wakewords_and_prompt};
use crate::tts::AudioMessage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

pub const SIMILARITY_THRESHOLD: f32 = 0.72;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Command {
    pub name: String,
    pub trigger_words: Vec<String>,
    pub action_type: String,
    pub action_value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_fallback: Option<String>,
    /// Выключенная команда не матчится голосом и не попадает в каталог
    /// для модели. Отсутствует в старых конфигах — значит включена.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct CommandsConfig {
    pub global: GlobalConfig,
    #[serde(default)]
    pub commands: Vec<Command>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GlobalConfig {
    pub reply_success: String,
    pub reply_error: String,
    pub reply_unknown: String,
    /// Звук-подтверждение при срабатывании wake word.
    #[serde(default = "default_reply_ack")]
    pub reply_ack: String,
    /// Порог схожести для fuzzy-матчинга. Пока не читается кодом
    /// (используется [`SIMILARITY_THRESHOLD`]), но объявлен здесь, чтобы
    /// сохранение конфига из интерфейса не теряло ключ из `commands.toml`.
    #[serde(default = "default_fuzzy_min_score")]
    pub fuzzy_min_score: f32,
    /// Рингтон таймера: играет циклом, пока таймер не остановят.
    #[serde(default = "default_ringtone_timer")]
    pub ringtone_timer: String,
    /// Рингтон будильника: отдельный, более настойчивый сигнал.
    #[serde(default = "default_ringtone_alarm")]
    pub ringtone_alarm: String,
}

fn default_reply_ack() -> String {
    "resources/audio/reply1.wav".to_string()
}

fn default_fuzzy_min_score() -> f32 {
    SIMILARITY_THRESHOLD
}

fn default_ringtone_timer() -> String {
    "resources/audio/ringtone_timer.wav".to_string()
}

fn default_ringtone_alarm() -> String {
    "resources/audio/ringtone_alarm.wav".to_string()
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            reply_success: "resources/audio/ok{}.wav".to_string(),
            reply_error: "resources/audio/reply1.wav".to_string(),
            reply_unknown: "resources/audio/reply1.wav".to_string(),
            reply_ack: "resources/audio/reply1.wav".to_string(),
            fuzzy_min_score: SIMILARITY_THRESHOLD,
            ringtone_timer: "resources/audio/ringtone_timer.wav".to_string(),
            ringtone_alarm: "resources/audio/ringtone_alarm.wav".to_string(),
        }
    }
}

impl CommandsConfig {
    /// Пробрасывает пути рингтонов в хаб таймеров (дефолты — если пусто).
    pub fn apply_ringtones(&self) {
        crate::timers::set_ringtones(&self.global.ringtone_timer, &self.global.ringtone_alarm);
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content =
            fs::read_to_string(path).with_context(|| format!("Не удалось прочитать {:?}", path))?;

        let config: Self =
            toml::from_str(&content).with_context(|| format!("Ошибка парсинга {:?}", path))?;

        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content =
            toml::to_string_pretty(self).context("Не удалось сериализовать конфиг команд")?;
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, content).with_context(|| format!("Не удалось записать {:?}", tmp))?;
        fs::rename(&tmp, path).with_context(|| format!("Не удалось заменить {:?}", path))?;
        Ok(())
    }

    /// Приоритет определяется порядком команд в TOML.
    ///
    /// Матч — только дословный: триггер стоит в начале фразы (после вежливых
    /// префиксов и wake-слов). Никакого fuzzy: приблизительное совпадение
    /// («яндекс карты» ≈ «яндекс»? нет) отдаёт фразу LLM — та решает по
    /// контексту и вызывает run_command по каталогу. Движок детерминированно
    /// выполняет только системные команды ([`Command::action_type`] == "stop"
    /// или "exit"), остальные — через LLM.
    pub fn find_command(&self, text: &str) -> Option<Command> {
        let text_cleaned = crate::text::clean_wakewords_and_prompt(text);

        if text_cleaned.trim().is_empty() {
            return None;
        }

        for cmd in &self.commands {
            if !cmd.enabled {
                continue;
            }
            for trigger in &cmd.trigger_words {
                let joined = clean_text(trigger);
                if !joined.is_empty() && text_cleaned.starts_with(&joined) {
                    return Some(cmd.clone());
                }
            }
        }

        None
    }

    /// Команда для простой голосовой просьбы действия: «открой ютуб»,
    /// «запусти монтаж», «закрой премьер».
    ///
    /// Триггеры в commands.toml — существительные («ютуб», «премиер»), и
    /// впереди них почти всегда стоит глагол. Обычный [`Self::find_command`]
    /// такое не ловит, и просьба уходила в LLM, где маленькая модель на
    /// простых action-фразах регулярно ошибалась: то выбирала не тот
    /// инструмент, то подставляла чужое имя приложения. Здесь направление
    /// (глагол) и объект (дословный триггер) детерминированно определяют
    /// команду, модель на них не тратится.
    ///
    /// Ловим только «глагол первым + голый триггер за ним»: составные фразы
    /// («закрой ютуб и открой телегу») по-прежнему зона LLM.
    pub fn find_action_command(&self, text: &str) -> Option<Command> {
        let cleaned = clean_wakewords_and_prompt(text);
        let mut words: Vec<&str> = cleaned.split_whitespace().collect();

        // Глагол направления обязан стоять первым: «что открой ютуб значит» —
        // вопрос, а не просьба открыть YouTube.
        let direction = words.first()?;
        let closing = STOP_VERBS.contains(direction);
        let opening = !closing && OPEN_VERBS.contains(direction);
        if !closing && !opening {
            return None;
        }
        words.remove(0);

        let target = words.join(" ");
        if target.trim().is_empty() {
            return None;
        }

        // Дальше — тот же дословный матч, что и в find_command: триггер в
        // начале остатка фразы, никакого fuzzy. Вежливые слова
        // («пожалуйста») снимутся внутри find_command сами.
        let cmd = self.find_command(&target)?;

        // Направление и тип действия должны сочетаться: закрыть можно только
        // приложение, ссылку и файл — только открыть.
        if closing {
            (cmd.action_type.trim() == "app").then_some(cmd)
        } else {
            matches!(cmd.action_type.trim(), "app" | "url" | "path").then_some(cmd)
        }
    }
}

/// Глаголы открытия/запуска. Проверяется только первое слово фразы — так же,
/// как [`STOP_VERBS`] решает направление в perform.
const OPEN_VERBS: &[&str] = &[
    "открой",
    "открыть",
    "откройте",
    "открывай",
    "открывайте",
    "запусти",
    "запустить",
    "запускай",
    "запускаем",
    "включи",
    "включить",
    "включим",
    "включайте",
];

/// Поллинг mtime commands.toml раз в секунду; при изменении перезагружает конфиг.
pub fn spawn_config_watcher(config_path: PathBuf, commands_arc: Arc<RwLock<CommandsConfig>>) {
    std::thread::spawn(move || {
        let mut last_modified = fs::metadata(&config_path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        loop {
            std::thread::sleep(Duration::from_secs(1));

            if let Ok(metadata) = fs::metadata(&config_path)
                && let Ok(modified) = metadata.modified()
                && modified > last_modified
            {
                println!("🔄 Изменение файла {:?}, перезагружаем...", config_path);

                match CommandsConfig::load(&config_path) {
                    Ok(new_cfg) => {
                        let count = new_cfg.commands.len();
                        new_cfg.apply_ringtones();
                        let mut lock = commands_arc.write().unwrap_or_else(|e| e.into_inner());
                        *lock = new_cfg;
                        println!("⚡ Обновлено! Загружено команд: {}", count);
                    }
                    Err(e) => eprintln!("❌ Ошибка обновления {:?}: {e}", config_path),
                }

                last_modified = modified;
            }
        }
    });
}

/// Убирает шумовые слова вокруг поискового запроса.
pub fn extract_search_query(full_text: &str, trigger_words: &[String]) -> String {
    let mut clean_str = clean_text(full_text);

    for tw in trigger_words {
        let clean_tw = clean_text(tw);
        if !clean_tw.is_empty() {
            let filtered = clean_str
                .split_whitespace()
                .filter(|w| *w != clean_tw)
                .collect::<Vec<_>>()
                .join(" ");
            clean_str = filtered;
        }
    }

    let noise_words = ["пожалуйста", "джарвис", "открой", "быстро", "да сэр", "сэр"];
    clean_str = clean_str
        .split_whitespace()
        .filter(|word| !noise_words.contains(word))
        .collect::<Vec<_>>()
        .join(" ");

    clean_text(&clean_str)
}

/// Глаголы остановки/закрытия. Триггеры в commands.toml — существительные
/// («премиер», «музыка»), поэтому без проверки глагола «закрой премьер про»
/// запускал Premiere заново, а «останови музыку» открывал Яндекс Музыку.
const STOP_VERBS: &[&str] = &[
    "закрой",
    "закрыть",
    "закрывай",
    "закройте",
    "закрывайте",
    "убий",
    "убей",
    "выключи",
    "выключить",
    "выйди",
    "выходи",
    "выйдите",
    "останови",
    "остановить",
    "стоп",
    "хватит",
    "пауза",
    "quit",
    "close",
    "exit",
    "kill",
];

/// Просит ли пользователь что-то остановить/закрыть, а не открыть.
pub fn is_stop_request(text: &str) -> bool {
    let t = clean_text(text);
    STOP_VERBS.iter().any(|v| t.split(' ').any(|w| w == *v))
}

fn wants_close(text: &str) -> bool {
    is_stop_request(text)
}

/// Что удалось сделать с приложением по запросу «закрой».
// Закрытие приложений — единый путь через tools::execute(QuitApp) с честным
// поллингом; дублирующий оптимистичный quit удалён (аудит: «два quit»).

#[allow(clippy::too_many_arguments)]
/// Исполняет команду пользователя. Возвращает успех: звук-подтверждение
/// играется ТОЛЬКО после фактического исполнения и только при успехе —
/// раньше он стоял до execute_command, и ошибка «url/path» звучала
/// как успех (звук уже прозвучал).
#[allow(clippy::too_many_arguments)]
pub fn execute_command(
    cmd: &Command,
    raw_text: &str,
    audio_tx: &Sender<AudioMessage>,
    events: &EventTx,
    llm: &SharedLlm,
    llm_cancel: Arc<AtomicBool>,
    llm_history: Arc<Mutex<Vec<ChatTurn>>>,
    reply_error: &str,
) -> bool {
    let action_type = cmd.action_type.trim();
    let action_value = cmd.action_value.trim();

    match action_type {
        "url" => match open::that(action_value) {
            Ok(_) => true,
            Err(e) => {
                let msg = format!("Не удалось открыть {action_value}: {e}");
                eprintln!("⚠️ {msg}");
                let _ = events.send(JarvisEvent::Error { message: msg.clone() });
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(msg));
                false
            }
        },

        // Открывает файл/папку в системном приложении (например tasks.md).
        // Относительный путь отсчитывается от корня данных (JARVIS_HOME).
        "path" => {
            let expanded = crate::tools::expand_path(action_value);
            let target = if std::path::Path::new(&expanded).is_absolute() {
                std::path::PathBuf::from(expanded)
            } else {
                paths::resolve_data_path(&expanded)
            };
            match open::that(&target) {
                Ok(_) => true,
                Err(e) => {
                    let msg = format!("Не удалось открыть {}: {e}", target.display());
                    eprintln!("⚠️ {msg}");
                    let _ = events.send(JarvisEvent::Error { message: msg.clone() });
                    let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(msg));
                    false
                }
            }
        }

        // Открывает поиск в браузере. Голосом ничего не подтверждаем —
        // пользователь сразу видит вкладку и результат поиска.
        "search" => {
            let query = extract_search_query(raw_text, &cmd.trigger_words);
            let search_term = if query.is_empty() {
                clean_text(raw_text)
            } else {
                query
            };

            if search_term.is_empty() {
                let _ = open::that("https://www.google.com");
                return true;
            }

            let encoded_query = urlencoding::encode(&search_term);
            let target_url = action_value.replace("{}", &encoded_query);

            println!("🔍 Запуск поиска: `{}` -> {}", search_term, target_url);
            match open::that(&target_url) {
                Ok(_) => true,
                Err(e) => {
                    let msg = format!("Не удалось открыть браузер: {e}");
                    eprintln!("⚠️ {msg}");
                    let _ = events.send(JarvisEvent::Error { message: msg.clone() });
                    let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(msg));
                    false
                }
            }
        }

        // Веб-поиск: результаты → локальная LLM → короткий ответ голосом.
        // Промежуточного «Ищу в интернете ...» нет: пользователь слышит
        // только финальный ответ.
        "web" => {
            let query = extract_search_query(raw_text, &cmd.trigger_words);
            if query.is_empty() {
                let msg = "Не удалось выделить поисковый запрос".to_string();
                let _ = events.send(JarvisEvent::Error { message: msg.clone() });
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(msg));
                return false;
            }

            let audio_tx = audio_tx.clone();
            let events = events.clone();
            let llm = Arc::clone(llm);
            let reply_error = reply_error.to_string();
            let query = query.clone();
            let req_id = llm::request_snapshot();

            std::thread::spawn(move || match crate::web::search(&query, 5) {
                Ok(hits) => {
                    if !llm::is_current_request(req_id) || llm_cancel.load(Ordering::SeqCst) {
                        println!("🚫 Поиск устарел — отвечаю молча");
                        return;
                    }
                    let count = hits.len();
                    match crate::web::build_summary_prompt(&query, &hits) {
                        Some(prompt) => {
                            println!(
                                "🌐 Найдено {} результатов, суммаризирую через LLM...",
                                count
                            );
                            llm::ask(
                                llm,
                                llm::WEB_SYSTEM_PROMPT,
                                prompt,
                                llm_history,
                                audio_tx,
                                events,
                                llm_cancel,
                            );
                        }
                        None => {
                            let _ = audio_tx.send(AudioMessage::PlayWav(reply_error));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("⚠️ Веб-поиск не удался: {e}");
                    let _ = audio_tx.send(AudioMessage::PlayWav(reply_error));
                }
            });
            // Поиск асинхронный: результат придёт ответом LLM.
            true
        }

        "app" => {
            // «закрой премьер про» матчится тем же триггером, что и «открыть»:
            // триггеры — голые существительные («премиер», «монтаж»), и глагол
            // в запросе раньше просто игнорировался. Поэтому направление
            // действия решаем по глаголу, а не по совпадению триггера.
            if wants_close(raw_text) {
                let app_name = if action_value.is_empty() {
                    cmd.name.clone()
                } else {
                    action_value.to_string()
                };
                // Единый путь с action-веткой: честный поллинг выхода
                // (tools::quit_app_impl), без оптимистичного «Закрываю».
                let audio = audio_tx.clone();
                let events = events.clone();
                std::thread::spawn(move || {
                    let outcome = crate::tools::execute(
                        crate::tools::ToolCall::QuitApp { name: app_name.clone() },
                        &audio,
                        &events,
                    );
                    let _ = audio.send(AudioMessage::SynthesizeAndPlay(outcome.speech));
                });
                // Асинхронное закрытие: звук-подтверждение не играем.
                return true;
            }

            {
                // Открытие приложения — только платформенный слой: на macOS
                // это `open -a`, на Linux — .desktop + gtk-launch, на
                // Windows — Start-Process/меню «Пуск». Провал запуска не
                // подменяется файлом, папкой или сайтом.
                match crate::platform::app_open(action_value) {
                    Ok(speech) => {
                        println!("🚀 {speech}");
                        // app_open уже ждёт процесс (постусловие) — успех.
                        return true;
                    }
                    Err(reason) => {
                        eprintln!("⚠️ {reason}");
                        let _ = events.send(JarvisEvent::Error {
                            message: reason.clone(),
                        });
                        let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(reason.clone()));
                        return false;
                    }
                }
            }
        }
        "exit" => {
            std::process::exit(0);
        }
        "stop" => {
            if !action_value.is_empty() {
                // Исторически тип `stop` хранит скрипты: на macOS это
                // AppleScript, на других ОС — команда своей оболочки
                // (spawn_script сам решает по ОС).
                return crate::platform::shell::spawn_script(action_value);
            }
            true
        }

        "command" => {
            // Раньше результат spawn игнорировался — команда «всегда
            // успешна». Теперь ошибка запуска доходит до пользователя.
            if crate::platform::shell::spawn(action_value) {
                true
            } else {
                let msg = format!("Не удалось запустить: {action_value}");
                eprintln!("⚠️ {msg}");
                let _ = events.send(JarvisEvent::Error { message: msg.clone() });
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(msg));
                false
            }
        }

        _ => {
            let msg = format!("Неизвестный тип команды: {}", action_type);
            eprintln!("{}", msg);
            let _ = events.send(JarvisEvent::Error {
                message: msg.clone(),
            });
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(msg));
            false
        }
    }
}

/// Потокобезопасное чтение глобального конфига.
pub fn get_global_config(commands_arc: &Arc<RwLock<CommandsConfig>>) -> GlobalConfig {
    commands_arc
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .global
        .clone()
}

// =========================================================================
// Универсальные команды: LLM выполняет команды из commands.toml по смыслу
// через инструмент run_command, подставляя параметры в плейсхолдеры {…}.
// =========================================================================

/// Подставляет параметры модели в значение действия: `{query}` → значение
/// аргумента. Неизвестные плейсхолдеры остаются как есть — по ним видно,
/// какого параметра не хватило.
pub fn substitute_params(value: &str, args: &HashMap<String, String>) -> String {
    let mut out = value.to_string();
    for (key, val) in args {
        out = out.replace(&format!("{{{key}}}"), val);
    }
    out
}

/// Объявленные параметры команды: `{name}` в значении действия.
/// Пустой `{}` (слот поиска) параметром не считается.
fn placeholders_of(value: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = value;
    while let Some(start) = rest.find('{') {
        let after = &rest[start + 1..];
        match after.find('}') {
            Some(end) => {
                let name = after[..end].trim();
                if !name.is_empty()
                    && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && !out.iter().any(|x| x == name)
                {
                    out.push(name.to_string());
                }
                rest = &after[end + 1..];
            }
            None => break,
        }
    }
    out
}

fn args_query(args: &HashMap<String, String>) -> Option<String> {
    ["query", "q", "запрос", "текст"]
        .iter()
        .find_map(|k| args.get(*k).cloned())
        .map(|q| q.trim().to_string())
        .filter(|q| !q.is_empty())
}

/// Выполнение команды из tool-вызова `run_command` на готовом конфиге.
/// Возвращает честный итог — озвучивает и передаёт модели вызывающий код.
pub fn run_command_from(
    cfg: &CommandsConfig,
    name: &str,
    args: &HashMap<String, String>,
    audio_tx: &Sender<AudioMessage>,
    events: &EventTx,
) -> crate::tools::ToolOutcome {
    use crate::tools::{ToolCall, ToolOutcome};

    let wanted = name.trim();
    let cmd = cfg
        .commands
        .iter()
        .filter(|c| c.enabled)
        .find(|c| c.name.eq_ignore_ascii_case(wanted))
        // Модель могла назвать команду триггером — примем и это.
        .or_else(|| {
            cfg.commands
                .iter()
                .filter(|c| c.enabled)
                .find(|c| {
                    c.trigger_words
                        .iter()
                        .any(|t| t.trim().eq_ignore_ascii_case(wanted))
                })
        });

    let Some(cmd) = cmd.cloned() else {
        let names: Vec<String> = cfg
            .commands
            .iter()
            .map(|c| c.name.clone())
            .take(15)
            .collect();
        let hint = if names.is_empty() {
            "список команд пуст".to_string()
        } else {
            format!("есть: {}", names.join(", "))
        };
        return ToolOutcome::fail(format!("Команда «{wanted}» не найдена, {hint}"));
    };

    println!("🔧 run_command: {} ({})", cmd.name, cmd.action_type);
    let action_type = cmd.action_type.trim().to_string();
    let action_value = substitute_params(cmd.action_value.trim(), args);

    match action_type.as_str() {
        "url" if !action_value.is_empty() => {
            ToolCall::OpenUrl { url: action_value }.execute(audio_tx, events)
        }
        "path" => ToolCall::OpenPath { path: action_value }.execute(audio_tx, events),
        "app" => ToolCall::OpenApp {
            name: action_value.clone(),
        }
        .execute(audio_tx, events),
        "search" => {
            let Some(query) = args_query(args) else {
                return ToolOutcome::fail(format!(
                    "Команде «{}» нужен параметр query — что искать",
                    cmd.name
                ));
            };
            // Слот `{}` из commands.toml и подстановка `{query}` от модели —
            // оба варианта, запрос кодируется один раз.
            let encoded = urlencoding::encode(&query).to_string();
            let mut enc_args = args.clone();
            enc_args.insert("query".to_string(), encoded.clone());
            let mut target = substitute_params(cmd.action_value.trim(), &enc_args);
            target = target.replace("{}", &encoded);
            match open::that(&target) {
                Ok(_) => ToolOutcome::ok(format!("Ищу «{query}» в браузере")),
                Err(e) => ToolOutcome::fail(format!("Поиск не открылся: {e}")),
            }
        }
        // Веб-команда из LLM: выполняем поиск и отдаём выдачу модели — она
        // суммаризирует её следующим шагом цикла. Вложенный llm::ask тут
        // не нужен и опасен (LLM ждёт саму себя).
        "web" => {
            let Some(query) = args_query(args) else {
                return ToolOutcome::fail(format!(
                    "Команде «{}» нужен параметр query — что искать в интернете",
                    cmd.name
                ));
            };
            match crate::web::search(&query, 5) {
                Ok(hits) if hits.is_empty() => ToolOutcome::fail("Ничего не нашлось в интернете"),
                Ok(hits) => {
                    let list: Vec<String> = hits
                        .iter()
                        .map(|h| format!("{} — {}", h.url, h.title))
                        .collect();
                    let count = hits.len();
                    ToolOutcome::with_detail(
                        format!("Нашёл {count} результатов"),
                        format!("Результаты поиска «{query}»:\n{}", list.join("\n")),
                    )
                }
                Err(e) => ToolOutcome::fail(format!("Поиск не удался: {e}")),
            }
        }
        "command" | "stop" if !action_value.is_empty() => {
            ToolCall::Shell { cmd: action_value }.execute(audio_tx, events)
        }
        "exit" => std::process::exit(0),
        other => ToolOutcome::fail(format!(
            "Неизвестный тип действия «{other}» у команды {}",
            cmd.name
        )),
    }
}

/// [`run_command_from`] с чтением свежего commands.toml с диска — учитывает
/// правки, внесённые между стартом и вызовом.
pub fn run_command_tool(
    name: &str,
    args: &HashMap<String, String>,
    audio_tx: &Sender<AudioMessage>,
    events: &EventTx,
) -> crate::tools::ToolOutcome {
    match CommandsConfig::load(&paths::commands_file()) {
        Ok(cfg) => run_command_from(&cfg, name, args, audio_tx, events),
        Err(e) => {
            crate::tools::ToolOutcome::fail(format!("Не удалось прочитать commands.toml: {e}"))
        }
    }
}

/// Каталог команд для системного промпта (по готовому конфигу).
/// Пустая строка — команд нет.
pub fn prompt_catalog_from(cfg: &CommandsConfig) -> String {
    let mut lines: Vec<String> = Vec::new();
    for cmd in cfg.commands.iter().filter(|c| c.enabled).take(40) {
        let mut params = placeholders_of(&cmd.action_value);
        let value: String = cmd.action_value.trim().chars().take(60).collect();
        let action = match cmd.action_type.trim() {
            "url" => format!("открыть {value}"),
            "path" => format!("открыть путь {value}"),
            "app" => format!("запустить приложение {value}"),
            "search" => format!("поиск в браузере {value}"),
            "web" => format!("ответить по данным интернета {value}"),
            "command" | "stop" => format!("выполнить {value}"),
            "exit" => "завершить работу Джарвиса".to_string(),
            other => format!("действие {other}: {value}"),
        };
        // В промпт достаточно трёх триггеров: полный список раздувал префилл,
        // а модель матчит по смыслу, а не по словарю.
        let triggers: Vec<String> = cmd.trigger_words.iter().take(3).cloned().collect();
        // Команды без параметров описываются одной строкой: «триггеры:» и
        // «параметры:» на 17 командах из 20 — чистый расход префилла.
        if params.is_empty() {
            lines.push(format!("{}, {} → {}", cmd.name, triggers.join("/"), action));
        } else {
            params.sort();
            let mut note = params.join(", ");
            if matches!(cmd.action_type.trim(), "search" | "web") {
                note.push_str(" (query — что искать)");
            }
            lines.push(format!(
                "{}, {} → {} | параметры: {}",
                cmd.name,
                triggers.join("/"),
                action,
                note
            ));
        }
    }
    lines.join("\n")
}

/// Каталог из текущего commands.toml. Пустая строка — команд нет или файл
/// не читается (тогда каталог в промпт не попадает).
pub fn prompt_catalog() -> String {
    match CommandsConfig::load(&paths::commands_file()) {
        Ok(cfg) => prompt_catalog_from(&cfg),
        Err(_) => String::new(),
    }
}

/// Есть ли в commands.toml команда, которую движок уже должен выполнять сам.
/// Детерминированный планировщик в этом случае уступает каталогу: явная
/// команда пользователя важнее общей эвристики «открой X».
pub fn matches_user_command(prompt: &str) -> bool {
    match CommandsConfig::load(&paths::commands_file()) {
        Ok(cfg) => cfg.find_action_command(prompt).is_some() || cfg.find_command(prompt).is_some(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CommandsConfig {
        CommandsConfig {
            global: GlobalConfig {
                reply_success: "ok.wav".into(),
                reply_error: "err.wav".into(),
                reply_unknown: "unknown.wav".into(),
                reply_ack: "ack.wav".into(),
                fuzzy_min_score: SIMILARITY_THRESHOLD,
                ringtone_timer: "resources/audio/ringtone_timer.wav".into(),
                ringtone_alarm: "resources/audio/ringtone_alarm.wav".into(),
            },
            commands: vec![
                Command {
                    name: "youtube".into(),
                    trigger_words: vec!["ютуб".into(), "видео".into()],
                    action_type: "url".into(),
                    action_value: "https://youtube.com".into(),
                    action_fallback: None,
                    enabled: true,
                },
                Command {
                    name: "yandex_music".into(),
                    trigger_words: vec!["музыка".into(), "яндекс музыка".into()],
                    action_type: "url".into(),
                    action_value: "https://music.yandex.ru".into(),
                    action_fallback: None,
                    enabled: true,
                },
                Command {
                    name: "kinopoisk".into(),
                    trigger_words: vec!["запусти кинопоиск".into(), "кино".into()],
                    action_type: "url".into(),
                    action_value: "https://kinopoisk.ru".into(),
                    action_fallback: None,
                    enabled: true,
                },
            ],
        }
    }

    #[test]
    fn close_verb_is_detected_only_for_close_words() {
        for text in [
            "закрой премьер про",
            "закрыть премьер про",
            "ЗАКРОЙ ПРЕМЬЕР ПРО",
            "выйди из премьера",
            "выключи премьер",
            "quit premiere pro",
        ] {
            assert!(wants_close(text), "не увидел глагол закрытия в «{text}»");
        }
        // Открытие и обычные вопросы закрытием не считаются.
        for text in [
            "открой премьер про",
            "запусти премьер",
            "что такое монтаж",
            "покажи премьер про",
        ] {
            assert!(!wants_close(text), "ложное закрытие в «{text}»");
        }
    }

    #[test]
    fn exact_trigger_prefix_still_resolves_app() {
        // Дословный триггер в начале фразы находит app-команду, а направление
        // «открыть/закрыть» решает wants_close (или LLM по контексту).
        let cfg = CommandsConfig {
            global: test_config().global,
            commands: vec![Command {
                name: "Adobe Premiere Pro".into(),
                trigger_words: vec!["премиер".into(), "премьер".into(), "монтаж".into()],
                action_type: "app".into(),
                action_value: "Adobe Premiere Pro 2026".into(),
                action_fallback: None,
                enabled: true,
            }],
        };
        let cmd = cfg
            .find_command("премиер про")
            .expect("премиер в начале фразы");
        assert_eq!(cmd.action_type, "app");

        // Глагол перед триггером больше не движковая команда: fuzzy убран,
        // фразу отдаём LLM с каталогом (run_command + wants_close решают).
        assert!(cfg.find_command("закрой премьер про").is_none());
        assert!(cfg.find_command("открой премьер про").is_none());
        assert!(wants_close("закрой премьер про"));
        assert!(!wants_close("открой премьер про"));
    }

    #[test]
    fn stop_requests_are_detected() {
        for text in [
            "останови музыку",
            "выключи музыку",
            "хватит",
            "закрой премьер про",
        ] {
            assert!(is_stop_request(text), "не увидел остановку в «{text}»");
        }
        for text in ["включи музыку", "запусти таймер", "открой сафари"]
        {
            assert!(!is_stop_request(text), "ложная остановка в «{text}»");
        }
    }

    #[test]
    fn long_question_is_not_a_command() {
        let cfg = test_config();
        assert!(
            cfg.find_command("какие есть сервисы для прослушивания музыки")
                .is_none()
        );
    }

    #[test]
    fn trigger_in_the_middle_is_not_a_engine_command() {
        // «включи музыку» — глагол перед триггером: fuzzy убран, фраза уходит
        // в LLM (каталог знает яndex_music и run_command). Дословный триггер
        // в начале — команда.
        let cfg = test_config();
        assert!(cfg.find_command("включи музыку").is_none());
        let cmd = cfg
            .find_command("музыка")
            .expect("дословный триггер в начале");
        assert_eq!(cmd.name, "yandex_music");
    }

    #[test]
    fn trigger_at_start_is_command() {
        let cfg = test_config();
        let cmd = cfg.find_command("запусти кинопоиск").expect("should match");
        assert_eq!(cmd.name, "kinopoisk");
    }

    #[test]
    fn polite_prefix_stripped() {
        let cfg = test_config();
        let cmd = cfg
            .find_command("джарвис запусти кинопоиск пожалуйста")
            .expect("should match");
        assert_eq!(cmd.name, "kinopoisk");
    }

    // --- Универсальные команды (run_command) ---

    fn no_channels() -> (Sender<AudioMessage>, EventTx) {
        let (audio_tx, _audio_rx) = std::sync::mpsc::channel();
        let (event_tx, _event_rx) = std::sync::mpsc::channel();
        (audio_tx, event_tx)
    }

    #[test]
    fn params_are_substituted() {
        let mut args = HashMap::new();
        args.insert("query".to_string(), "лунная соната".to_string());
        assert_eq!(
            substitute_params("https://music.yandex.ru/search?text={query}", &args),
            "https://music.yandex.ru/search?text=лунная соната"
        );
        // Без аргументов шаблон не трогается — по нему видно, чего не хватило.
        assert_eq!(
            substitute_params("https://x.ru/?q={query}", &HashMap::new()),
            "https://x.ru/?q={query}"
        );
    }

    #[test]
    fn unknown_command_fails_honestly() {
        let cfg = test_config();
        let (audio_tx, events) = no_channels();
        let out = run_command_from(&cfg, "нет-такой", &HashMap::new(), &audio_tx, &events);
        assert!(!out.ok);
        assert!(
            out.speech.contains("не найдена"),
            "получено: {}",
            out.speech
        );
        assert!(
            out.speech.contains("youtube"),
            "подсказка без имён: {}",
            out.speech
        );
    }

    #[test]
    fn command_is_found_by_trigger_word() {
        let cfg = test_config();
        let (audio_tx, events) = no_channels();
        // «видео» — триггер команды youtube; url-действие откроет браузер,
        // но канал событий события не валидирует, а open::that в тесте безопасен.
        let cfg_url_only = CommandsConfig {
            global: cfg.global.clone(),
            commands: vec![Command {
                name: "тест-путь".into(),
                trigger_words: vec!["триппер".into()],
                action_type: "path".into(),
                action_value: "~/нет-такого-каталога-jarvis-test".into(),
                action_fallback: None,
                enabled: true,
            }],
        };
        let out = run_command_from(
            &cfg_url_only,
            "триппер",
            &HashMap::new(),
            &audio_tx,
            &events,
        );
        // Команда нашлась по триггеру, но путь не существует — честная неудача,
        // а не «Открываю».
        assert!(!out.ok, "получено: {}", out.speech);
        assert!(out.speech.contains("не найден"), "получено: {}", out.speech);
    }

    #[test]
    fn search_requires_query_param() {
        let cfg = CommandsConfig {
            global: test_config().global,
            commands: vec![Command {
                name: "поиск-музыки".into(),
                trigger_words: vec!["музыка".into()],
                action_type: "search".into(),
                action_value: "https://music.yandex.ru/search?text={}".into(),
                action_fallback: None,
                enabled: true,
            }],
        };
        let (audio_tx, events) = no_channels();
        let out = run_command_from(&cfg, "поиск-музыки", &HashMap::new(), &audio_tx, &events);
        assert!(!out.ok, "получено: {}", out.speech);
        assert!(out.speech.contains("query"), "получено: {}", out.speech);
    }

    #[test]
    fn catalog_lists_names_triggers_and_params() {
        let cfg = CommandsConfig {
            global: test_config().global,
            commands: vec![
                Command {
                    name: "youtube".into(),
                    trigger_words: vec!["ютуб".into()],
                    action_type: "url".into(),
                    action_value: "https://youtube.com".into(),
                    action_fallback: None,
                    enabled: true,
                },
                Command {
                    name: "поиск-музыки".into(),
                    trigger_words: vec!["музыка".into()],
                    action_type: "search".into(),
                    action_value: "https://music.yandex.ru/search?text={query}".into(),
                    action_fallback: None,
                    enabled: true,
                },
            ],
        };
        let catalog = prompt_catalog_from(&cfg);
        assert!(catalog.contains("youtube"), "каталог: {catalog}");
        assert!(catalog.contains("ютуб"), "каталог: {catalog}");
        assert!(catalog.contains("поиск-музыки"), "каталог: {catalog}");
        assert!(catalog.contains("параметры: query"), "каталог: {catalog}");
        // Пустой конфиг — пустой каталог (блок не попадает в промпт).
        assert!(prompt_catalog_from(&CommandsConfig::default()).is_empty());
    }
}
