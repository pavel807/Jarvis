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
    /// Сколько секунд подтверждённой VAD тишины завершают фразу.
    /// Пока speakrs слышит человеческую речь, пауза фразу не закрывает.
    pub silence_timeout_secs: f32,
    /// Минимальная длительность фразы (отсекает щелчки и шорохи).
    pub min_command_duration_secs: f32,
    /// Мягкий предел длительности фразы. Пока VAD слышит речь, фраза
    /// продолжается до жёсткого предела движка (30 с).
    pub max_command_duration_secs: f32,
    /// Окно прямого слушания после wake word/команды.
    pub followup_window_secs: u64,

    // --- Wake word ---
    /// Порог срабатывания детектора RustPotter.
    pub wake_threshold: f32,
    /// Имя файла активной модели wake word внутри `models/`.
    pub wakeword_model: String,
    /// Детектор wake word: "ctc" — по CTC-постерам ASR-модели (любое
    /// слово-активация, дикторонезависимо, эталон не нужен),
    /// "metric" — метрический, по облаку эмбеддингов (эталон .jref).
    pub wake_detector: String,
    /// Слово-активация для CTC-детектора (в любом регистре).
    pub wake_word: String,
    /// Длина окна CTC-детектора, мс: окно должно накрывать фразу целиком.
    /// Подбирается автоматически при генерации фразы.
    pub wake_window_ms: usize,

    // --- STT (FastConformer) ---
    /// Декодер ASR: "rnnt" — транзьюсер жадно (по умолчанию, точнее),
    /// "ctc" — CTC жадно, "ctc_beam" — CTC beam search со словарём Jarvis.
    pub asr_decoder: String,
    /// Ширина луча для "ctc_beam".
    pub asr_beam_width: usize,
    /// Словарь Jarvis для beam search: названия приложений и команд из
    /// commands.toml повышают точность их распознавания.
    pub asr_dict: bool,

    // --- LLM ---
    /// Бэкенд LLM: "uzu" — движок Mirai на Metal (только macOS, модель
    /// скачивается сама) или "candle" — собственные реализации на candle.
    /// Применяется при следующем запуске движка.
    pub llm_backend: String,
    /// Каталог модели LLM внутри `models/`: "qwen3.5-0.8b" (гибрид
    /// Gated DeltaNet + attention, ~1.7 ГБ BF16) или "minicpm5-1b-ru".
    /// Для бэкенда uzu — имя из каталога Mirai: "qwen3.5-2b-mirai-m"
    /// (4-бит), "qwen3.5-2b-mirai-l" (8-бит) и т.д.
    /// Применяется при следующем запуске движка.
    pub llm_model: String,
    pub llm_temperature: f64,
    /// Страховочный лимит длины ответа LLM в токенах. `0` — без лимита:
    /// ответы и команды досказываются до конца, обрывы держат только защита
    /// от зацикливания и конец контекста модели.
    pub llm_max_tokens: usize,

    // --- Облачная модель (OpenAI-совместимый API) ---
    /// Мастер первого запуска пройден. false — при старте показывается
    /// онбординг: загрузка моделей, выбор мозга, диктант для профиля голоса.
    pub setup_done: bool,
    /// Пресет провайдера: "yandex" (Api-Key + URI модели) или "openai"
    /// (Bearer, адрес и модель как введены).
    pub api_preset: String,
    /// Базовый адрес OpenAI-совместимого API без хвостового слэша.
    pub api_base_url: String,
    /// API-ключ провайдера. Живёт в settings.toml — файл личных данных
    /// Джарвиса, права на запись ограничиваются 0600.
    pub api_key: String,
    /// Имя модели у провайдера: `yandexgpt-lite`, `gpt-4o-mini`, `meta-llama/...`.
    pub api_model: String,
    /// Идентификатор каталога Яндекс.Облака (только для пресета "yandex"):
    /// из него собирается URI модели `gpt://<каталог>/<модель>/latest`.
    pub api_folder_id: String,
    /// Город погоды по умолчанию («какая погода» без города). Названный в
    /// вопросе город всегда сильнее настройки.
    pub weather_city: String,

    // --- TTS ---
    /// Голос Silero v5: ru_eduard | ru_dmitriy | ru_alexandr | ru_bogdan | ... (29 голосов).
    pub tts_voice: String,
    /// Скорость речи: 1.0 — естественно, меньше — медленнее.
    pub tts_speed: f32,

    // --- Отклик ---
    /// Звук-подтверждение перед ответом LLM. Выключен по умолчанию:
    /// он стоит в очереди плеера перед речью и отодвигает первый звук ответа.
    pub ack_before_llm: bool,
    /// Перебивание: голос пользователя во время ответа Джарвиса глушит
    /// речь и уходит в разбор как обычная фраза. Выключатель только здесь,
    /// в интерфейсе не дублируется.
    pub barge_in: bool,

    // --- Личность ---
    /// Имя ассистента: как он представляется в интерфейсе и разговоре.
    /// Фраза-активация (`wake_word`) независима — её меняют отдельно.
    pub assistant_name: String,

    // --- Память ---
    /// Фильтр говорящего: в LLM-цепочку идёт только голос владельца.
    pub speaker_filter: bool,
    /// Порог косинусного сходства с голосовым профилем владельца (0..1).
    pub speaker_threshold: f32,

    // --- Каналы и внешние инструменты (по логике OpenJarvis) ---
    /// Токен Telegram-бота (BotFather). Пусто — канал не запускается.
    pub telegram_bot_token: String,
    /// Разрешённые chat_id через запятую: кому бот отвечает.
    pub telegram_chat_ids: String,
    /// Telegram может выполнять команды Джарвиса (весь список из
    /// commands.toml и системные планы). false — только разговор.
    pub telegram_execute: bool,
    /// Внешние MCP-серверы: [[mcp_servers]] name/cmd/args в settings.toml.
    pub mcp_servers: Vec<McpServerConfig>,
}

/// Один внешний MCP-сервер: процесс stdio, инструменты которого появляются
/// у модели как mcp_call {"server":"имя","tool":"...","args":{...}}.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Короткое имя: попадает в tool-вызов модели.
    pub name: String,
    /// Исполняемый файл (в PATH или полный путь).
    pub cmd: String,
    /// Аргументы запуска.
    #[serde(default)]
    pub args: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            silence_rms_threshold: 0.018,
            silence_timeout_secs: 0.6,
            min_command_duration_secs: 0.5,
            max_command_duration_secs: 8.0,
            followup_window_secs: 8,
            wake_threshold: 0.6,
            wakeword_model: "jarvis_model_2.rpw".to_string(),
            wake_detector: "ctc".to_string(),
            wake_word: "Джарвис".to_string(),
            wake_window_ms: 1200,
            asr_decoder: "rnnt".to_string(),
            asr_beam_width: 8,
            asr_dict: true,
            llm_backend: if cfg!(target_os = "macos") {
                "uzu".to_string()
            } else {
                "candle".to_string()
            },
            // Модель по умолчанию — родная для бэкенда: uzu живёт в каталоге
            // Mirai (macOS), candle качается из релиза jarvis-models и на
            // Linux выбран по умолчанию.
            llm_model: if cfg!(target_os = "macos") {
                // 8-бит (mirai-l) вместо 4-бит: та же Qwen3.5 2B, но вдвое меньше
                // квантизационных повреждений весов — заметно меньше выдумывает.
                "qwen3.5-2b-mirai-l".to_string()
            } else {
                crate::llm::qwen35::LOCAL_DIR_NAME.to_string()
            },
            llm_temperature: 0.6,
            // 0 — без искусственного лимита (длинные ответы не обрываются).
            llm_max_tokens: 0,
            // Мастер первого запуска: показывается, пока не пройден.
            setup_done: false,
            api_preset: "yandex".to_string(),
            api_base_url: "https://llm.api.cloud.yandex.net/v1".to_string(),
            api_key: String::new(),
            api_model: "yandexgpt-lite".to_string(),
            api_folder_id: String::new(),
            weather_city: "Москва".to_string(),
            tts_voice: "ru_eduard".to_string(),
            tts_speed: 1.0,
            ack_before_llm: false,
            barge_in: true,
            // Фильтр включён по умолчанию: если профиль ещё не построен,
            assistant_name: "Джарвис".to_string(),
            // он просто пропускает всё и ни на что не влияет.
            speaker_filter: true,
            speaker_threshold: 0.6,
            telegram_bot_token: String::new(),
            telegram_chat_ids: String::new(),
            telegram_execute: true,
            mcp_servers: Vec::new(),
        }
    }
}

impl Settings {
    pub fn load(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(raw) => toml::from_str(&raw).with_context(|| format!("Ошибка парсинга {:?}", path)),
            Err(_) => Ok(Self::default()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).context("Не удалось сериализовать настройки")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, content)?;
        fs::rename(&tmp, path)?;
        // В settings.toml живёт API-ключ облачной модели: ограничиваем
        // доступ личным пользователем (0600). Не критично при ошибке —
        // файловые системы без UNIX-прав (FAT) не умеют.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = fs::metadata(path) {
                let mut perms = meta.permissions();
                perms.set_mode(0o600);
                let _ = fs::set_permissions(path, perms);
            }
        }
        Ok(())
    }

    /// Применяет настройки к подсистемам (голос TTS, параметры LLM).
    pub fn apply(&self) {
        crate::tts::set_current_voice(self.tts_voice.clone());
        crate::tts::set_current_speed(self.tts_speed);
        crate::llm::config::set_backend(&self.llm_backend);
        crate::llm::config::set_model_name(&self.llm_model);
        crate::llm::config::update_generation_params(self.llm_params());
        crate::llm::config::set_api_config(
            &self.api_preset,
            &self.api_base_url,
            &self.api_key,
            &self.api_model,
            &self.api_folder_id,
        );
    }

    /// Применяет настройки, безопасные на живую (hot-reload settings.toml):
    /// скорость речи, параметры генерации LLM и конфиг облачного провайдера.
    /// Голос и бэкенд не трогаем: смена голоса требует перезагрузки
    /// Silero-движка (её делает UI-команда сохранения), бэкенд/модель —
    /// перезапуска движка.
    pub fn apply_hot(&self) {
        crate::tts::set_current_speed(self.tts_speed);
        crate::llm::config::update_generation_params(self.llm_params());
        crate::llm::config::set_api_config(
            &self.api_preset,
            &self.api_base_url,
            &self.api_key,
            &self.api_model,
            &self.api_folder_id,
        );
    }

    fn llm_params(&self) -> crate::llm::config::GenerationParams {
        crate::llm::config::GenerationParams {
            temperature: self.llm_temperature,
            top_p: 0.95,
            repeat_penalty: 1.1,
            repeat_window: 64,
            max_tokens: self.llm_max_tokens,
        }
    }
}

/// Горячая перезагрузка settings.toml раз в секунду.
pub fn spawn_settings_watcher(settings_path: PathBuf, settings: Arc<RwLock<Settings>>) {
    spawn_settings_watcher_with(settings_path, settings, |_| {})
}

/// То же, что [`spawn_settings_watcher`], но с колбэком: смена
/// llm_backend/llm_model (мастер первого запуска, настройки) перезагружает
/// LLM на лету — выбранное облако не должно ждать перезапуска приложения,
/// пока в фоне качается локальная модель.
pub fn spawn_settings_watcher_with(
    settings_path: PathBuf,
    settings: Arc<RwLock<Settings>>,
    on_llm_change: impl Fn(&Settings) + Send + 'static,
) {
    std::thread::spawn(move || {
        let mut last_modified = fs::metadata(&settings_path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mut previous = Settings::load(&settings_path).ok();

        loop {
            std::thread::sleep(Duration::from_secs(1));

            if let Ok(metadata) = fs::metadata(&settings_path)
                && let Ok(modified) = metadata.modified()
                && modified > last_modified
            {
                match Settings::load(&settings_path) {
                    Ok(new_cfg) => {
                        let llm_changed = previous.as_ref().is_none_or(|prev| {
                            prev.llm_backend != new_cfg.llm_backend
                                || prev.llm_model != new_cfg.llm_model
                        });
                        previous = Some(new_cfg.clone());
                        let mut lock = settings.write().unwrap_or_else(|e| e.into_inner());
                        *lock = new_cfg.clone();
                        drop(lock);
                        // Применяем горячие настройки сразу: без этого правка
                        // settings.toml вручную (например, llm_max_tokens = 0)
                        // обновляла только копию в памяти, а движок до
                        // перезапуска работал по старым параметрам.
                        new_cfg.apply_hot();
                        println!("⚡ Настройки перезагружены");
                        if llm_changed {
                            println!("🧠 Сменился бэкенд/модель LLM — перезагружаю");
                            on_llm_change(&new_cfg);
                        }
                    }
                    Err(e) => eprintln!("❌ Ошибка обновления настроек: {e}"),
                }
                last_modified = modified;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::Settings;

    #[test]
    fn default_model_matches_default_backend() {
        // Дефолт согласован внутри: candle (не-macOS) — модель из релиза
        // jarvis-models, uzu (macOS) — модель каталога Mirai. Раньше на
        // Linux дефолт указывал на mirai-имя, которое candle не грузит.
        let s = Settings::load(&std::path::PathBuf::from("/nonexistent/settings.toml"))
            .unwrap_or_else(|_| Settings::load(&crate::paths::settings_file()).unwrap());
        if cfg!(target_os = "macos") {
            assert_eq!(s.llm_backend, "uzu");
            assert_eq!(s.llm_model, "qwen3.5-2b-mirai-l");
        } else {
            assert_eq!(s.llm_backend, "candle");
            assert_eq!(s.llm_model, crate::llm::qwen35::LOCAL_DIR_NAME);
        }
    }

    #[test]
    fn zero_max_tokens_stays_uncapped_in_llm_params() {
        // 0 — без лимита: не подменяется дефолтом и не клампится.
        let uncapped = Settings {
            llm_max_tokens: 0,
            ..Settings::default()
        };
        assert_eq!(uncapped.llm_params().max_tokens, 0);

        // Ненулевое — страховочный кап, проходит как есть.
        let capped = Settings {
            llm_max_tokens: 256,
            ..Settings::default()
        };
        assert_eq!(capped.llm_params().max_tokens, 256);
    }
}
