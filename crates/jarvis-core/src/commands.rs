//! Команды из commands.toml: загрузка, hot-reload, fuzzy-матчинг, выполнение.

use crate::events::{EventTx, JarvisEvent};
use crate::llm::{self, ChatTurn, SharedLlm};
use crate::text::{clean_text, word_similarity};
use crate::tts::AudioMessage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
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
}

#[derive(Deserialize, Serialize, Debug, Clone)]
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
}

fn default_reply_ack() -> String {
    "resources/audio/reply1.wav".to_string()
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            reply_success: "resources/audio/ok{}.wav".to_string(),
            reply_error: "resources/audio/reply1.wav".to_string(),
            reply_unknown: "resources/audio/reply1.wav".to_string(),
            reply_ack: "resources/audio/reply1.wav".to_string(),
        }
    }
}

impl Default for CommandsConfig {
    fn default() -> Self {
        Self {
            global: GlobalConfig::default(),
            commands: Vec::new(),
        }
    }
}

impl CommandsConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content =
            fs::read_to_string(path).with_context(|| format!("Не удалось прочитать {:?}", path))?;

        let config: Self =
            toml::from_str(&content).with_context(|| format!("Ошибка парсинга {:?}", path))?;

        Ok(config)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Не удалось сериализовать конфиг команд")?;
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, content).with_context(|| format!("Не удалось записать {:?}", tmp))?;
        fs::rename(&tmp, path).with_context(|| format!("Не удалось заменить {:?}", path))?;
        Ok(())
    }

    /// Приоритет определяется порядком команд в TOML.
    ///
    /// Контекстная логика: длинные вопросы («какие есть сервисы для
    /// прослушивания музыки») не должны превращаться в команду из-за слова
    /// «музыка» в середине. Команда срабатывает, если:
    /// - триггер стоит в начале фразы (после вежливых префиксов), или
    /// - фраза короткая (<=3 слов) и содержит триггер, или
    /// - триггер занимает >= 2/3 слов длинной фразы в её начале.
    pub fn find_command(&self, text: &str) -> Option<Command> {
        // Вежливые префиксы и wake word не должны мешать распознаванию.
        let text_cleaned = crate::text::clean_wakewords_and_prompt(text);
        let input_words: Vec<&str> = text_cleaned.split_whitespace().collect();

        if input_words.is_empty() {
            return None;
        }

        let total = input_words.len();
        let mut best_cmd: Option<Command> = None;
        let mut max_score = 0.0f32;

        for cmd in &self.commands {
            for trigger in &cmd.trigger_words {
                let trigger_cleaned = clean_text(trigger);

                if trigger_cleaned.is_empty() {
                    continue;
                }

                let trigger_words: Vec<&str> = trigger_cleaned.split_whitespace().collect();
                let n = trigger_words.len();
                let joined = trigger_cleaned.clone();

                // Команда в начале фразы — безусловное совпадение.
                if text_cleaned.starts_with(&joined) {
                    return Some(cmd.clone());
                }

                // Короткая фраза: триггер в любом месте (fuzzy-окна как раньше).
                if total <= 3 && input_words.len() >= n {
                    for window in input_words.windows(n) {
                        let current_window_str = window.join(" ");

                        let score = if n == 1 {
                            word_similarity(window[0], trigger_words[0])
                        } else {
                            word_similarity(&current_window_str, &trigger_cleaned)
                        };

                        if score >= SIMILARITY_THRESHOLD && score > max_score {
                            max_score = score;
                            best_cmd = Some(cmd.clone());
                        }
                    }
                    continue;
                }

                // Длинная фраза: триггер существенной части в начале.
                if total > 3 && n * 3 >= total * 2 && input_words.len() >= n {
                    let head = input_words[..n].join(" ");
                    let score = if n == 1 {
                        word_similarity(input_words[0], trigger_words[0])
                    } else {
                        word_similarity(&head, &trigger_cleaned)
                    };

                    if score >= SIMILARITY_THRESHOLD && score > max_score {
                        max_score = score;
                        best_cmd = Some(cmd.clone());
                    }
                }
            }
        }

        best_cmd
    }
}

/// Поллинг mtime commands.toml раз в секунду; при изменении перезагружает конфиг.
pub fn spawn_config_watcher(config_path: PathBuf, commands_arc: Arc<RwLock<CommandsConfig>>) {
    std::thread::spawn(move || {
        let mut last_modified = fs::metadata(&config_path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        loop {
            std::thread::sleep(Duration::from_secs(1));

            if let Ok(metadata) = fs::metadata(&config_path) {
                if let Ok(modified) = metadata.modified() {
                    if modified > last_modified {
                        println!("🔄 Изменение файла {:?}, перезагружаем...", config_path);

                        match CommandsConfig::load(&config_path) {
                            Ok(new_cfg) => {
                                let count = new_cfg.commands.len();
                                let mut lock =
                                    commands_arc.write().unwrap_or_else(|e| e.into_inner());
                                *lock = new_cfg;
                                println!("⚡ Обновлено! Загружено команд: {}", count);
                            }
                            Err(e) => eprintln!("❌ Ошибка обновления {:?}: {e}", config_path),
                        }

                        last_modified = modified;
                    }
                }
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
            // Удаляем триггер целиком как слово, чтобы не портить вложенные слова.
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
) {
    let action_type = cmd.action_type.trim();
    let action_value = cmd.action_value.trim();

    match action_type {
        "url" => {
            if let Err(e) = open::that(action_value) {
                eprintln!("⚠️ Ошибка открытия URL: {e}");
            }
        }

        "search" => {
            let query = extract_search_query(raw_text, &cmd.trigger_words);
            let search_term = if query.is_empty() {
                clean_text(raw_text)
            } else {
                query
            };

            if search_term.is_empty() {
                let _ = open::that("https://www.google.com");
                return;
            }

            let encoded_query = urlencoding::encode(&search_term);
            let target_url = action_value.replace("{}", &encoded_query);

            println!("🔍 Запуск поиска: `{}` -> {}", search_term, target_url);
            let _ = open::that(&target_url);

            let speech_reply = format!("Ищу в Гугл {}", search_term);
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech_reply));
        }

        // Поиск в интернете: результаты → локальная LLM → короткий ответ голосом.
        "web" => {
            let query = extract_search_query(raw_text, &cmd.trigger_words);
            if query.is_empty() {
                let _ = events.send(JarvisEvent::Error {
                    message: "Не удалось выделить поисковый запрос".into(),
                });
                return;
            }

            let speech_reply = format!("Ищу в интернете {}", query);
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech_reply));

            let audio_tx = audio_tx.clone();
            let events = events.clone();
            let llm = Arc::clone(llm);
            let reply_error = reply_error.to_string();
            let query = query.clone();

            std::thread::spawn(move || match crate::web::search(&query, 5) {
                Ok(hits) => {
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
        }

        "app" => {
            #[cfg(target_os = "macos")]
            {
                let status = std::process::Command::new("open")
                    .arg("-a")
                    .arg(action_value)
                    .status();

                let success = status.map(|s| s.success()).unwrap_or(false);

                if !success {
                    if let Some(fallback) = &cmd.action_fallback {
                        let _ = std::process::Command::new("open")
                            .arg(fallback.trim())
                            .spawn();
                    } else {
                        let _ = open::that(action_value);
                    }
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                if open::that(action_value).is_err() {
                    if let Some(fallback) = &cmd.action_fallback {
                        let _ = open::that(fallback.trim());
                    }
                }
            }
        }
        "exit" => {
            std::process::exit(0);
        }
        "stop" => {
            if !action_value.is_empty() {
                #[cfg(target_os = "macos")]
                let _ = std::process::Command::new("osascript")
                    .arg("-e")
                    .arg(action_value)
                    .spawn();

                #[cfg(target_os = "windows")]
                let _ = std::process::Command::new("cmd")
                    .args(["/C", action_value])
                    .spawn();

                #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
                let _ = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(action_value)
                    .spawn();
            }
        }

        "command" => {
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("cmd")
                .args(["/C", action_value])
                .spawn();

            #[cfg(not(target_os = "windows"))]
            let _ = std::process::Command::new("sh")
                .arg("-c")
                .arg(action_value)
                .spawn();
        }

        _ => {
            let msg = format!("Неизвестный тип команды: {}", action_type);
            eprintln!("{}", msg);
            let _ = events.send(JarvisEvent::Error { message: msg });
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
            },
            commands: vec![
                Command {
                    name: "youtube".into(),
                    trigger_words: vec!["ютуб".into(), "видео".into()],
                    action_type: "url".into(),
                    action_value: "https://youtube.com".into(),
                    action_fallback: None,
                },
                Command {
                    name: "yandex_music".into(),
                    trigger_words: vec!["музыка".into(), "яндекс музыка".into()],
                    action_type: "url".into(),
                    action_value: "https://music.yandex.ru".into(),
                    action_fallback: None,
                },
                Command {
                    name: "kinopoisk".into(),
                    trigger_words: vec!["запусти кинопоиск".into(), "кино".into()],
                    action_type: "url".into(),
                    action_value: "https://kinopoisk.ru".into(),
                    action_fallback: None,
                },
            ],
        }
    }

    #[test]
    fn long_question_is_not_a_command() {
        let cfg = test_config();
        // Триггер «музыка» в конце длинного вопроса — это вопрос к LLM.
        assert!(cfg
            .find_command("какие есть сервисы для прослушивания музыки")
            .is_none());
    }

    #[test]
    fn short_phrase_with_trigger_is_command() {
        let cfg = test_config();
        let cmd = cfg.find_command("включи музыку").expect("should match");
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
}
