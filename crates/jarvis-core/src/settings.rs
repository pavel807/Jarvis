//! Пользовательские настройки ассистента (settings.toml) с hot-reload.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
pub struct Settings {
    // --- Детекция фразы ---
    /// Порог RMS для отделения речи от тишины.
    pub silence_rms_threshold: f32,
    /// Сколько секунд тишины завершают фразу.
    pub silence_timeout_secs: f32,
    /// Минимальная длительность фразы (отсекает щелчки и шорохи).
    pub min_command_duration_secs: f32,
    /// Максимальная длительность фразы.
    pub max_command_duration_secs: f32,
    /// Окно прямого слушания после wake word/команды.
    pub followup_window_secs: u64,

    // --- Wake word ---
    /// Порог срабатывания детектора RustPotter.
    pub wake_threshold: f32,
    /// Имя файла активной модели wake word внутри `models/`.
    pub wakeword_model: String,

    // --- LLM ---
    pub llm_enabled: bool,
    /// Автоматически искать в интернете, когда вопрос про свежие данные.
    pub web_search_auto: bool,
    /// Нейро-нормализация текста перед озвучкой (числа и единицы через LLM).
    pub llm_normalize: bool,
    pub llm_temperature: f64,
    pub llm_max_tokens: usize,

    // --- TTS ---
    /// Движок синтеза: piper (быстрый, CPU, VITS) | qwen3 (нейросетевой, Metal).
    pub tts_engine: String,
    /// Голос активного движка: для piper — ruslan | denis | dmitri | irina,
    /// для qwen3 — ryan | aiden | eric | dylan | uncle_fu | serena | vivian | ono_anna | sohee.
    pub tts_voice: String,
    /// Скорость речи: 1.0 — естественно, меньше — медленнее.
    pub tts_speed: f32,

    // --- Отклик ---
    /// Звук-подтверждение перед ответом LLM. Выключен по умолчанию:
    /// он стоит в очереди плеера перед речью и отодвигает первый звук ответа.
    pub ack_before_llm: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            silence_rms_threshold: 0.018,
            silence_timeout_secs: 0.7,
            min_command_duration_secs: 0.5,
            max_command_duration_secs: 8.0,
            followup_window_secs: 8,
            wake_threshold: 0.6,
            wakeword_model: "jarvis_model_2.rpw".to_string(),
            llm_enabled: true,
            web_search_auto: true,
            llm_normalize: true,
            llm_temperature: 0.6,
            llm_max_tokens: 256,
            tts_engine: "piper".to_string(),
            tts_voice: "ruslan".to_string(),
            tts_speed: 1.0,
            ack_before_llm: false,
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(raw) => {
                toml::from_str(&raw).with_context(|| format!("Ошибка парсинга {:?}", path))
            }
            Err(_) => Ok(Self::default()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content =
            toml::to_string_pretty(self).context("Не удалось сериализовать настройки")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, content)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Применяет настройки к подсистемам (движок/голос TTS, параметры LLM).
    pub fn apply(&self) {
        crate::tts::set_current_engine(&self.tts_engine);
        crate::tts::set_current_voice(self.tts_voice.clone());
        crate::tts::set_current_speed(self.tts_speed);
        crate::llm::config::update_generation_params(crate::llm::config::GenerationParams {
            temperature: self.llm_temperature,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_window: 64,
            max_tokens: self.llm_max_tokens,
        });
    }
}

/// Горячая перезагрузка settings.toml раз в секунду.
pub fn spawn_settings_watcher(settings_path: PathBuf, settings: Arc<RwLock<Settings>>) {
    std::thread::spawn(move || {
        let mut last_modified = fs::metadata(&settings_path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        loop {
            std::thread::sleep(Duration::from_secs(1));

            if let Ok(metadata) = fs::metadata(&settings_path) {
                if let Ok(modified) = metadata.modified() {
                    if modified > last_modified {
                        match Settings::load(&settings_path) {
                            Ok(new_cfg) => {
                                let mut lock = settings.write().unwrap_or_else(|e| e.into_inner());
                                *lock = new_cfg;
                                println!("⚡ Настройки перезагружены");
                            }
                            Err(e) => eprintln!("❌ Ошибка обновления настроек: {e}"),
                        }
                        last_modified = modified;
                    }
                }
            }
        }
    });
}
