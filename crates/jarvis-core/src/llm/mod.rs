//! Локальный LLM: диспетчер бэкендов + фоновая загрузка, потоковая генерация
//! с озвучкой по предложениям и отменой.
//!
//! Два бэкенда за общим интерфейсом [`LlmEngine`]:
//! - [`candle_backend`] — чистый candle (MiniCPM5-1B-ru, Qwen3.5-0.8B),
//!   реализации архитектур свои, работает на любой платформе;
//! - [`uzu_backend`] — движок Uzu (Mirai) на Metal, только macOS, модели
//!   из каталога Mirai (Qwen3.5 0.8B–9B в 4/8-бит).
//!
//! Выбор — `llm_backend` в settings.toml (`"uzu"` | `"candle"`). Если uzu
//! недоступен (нет сети для каталога, модель не скачалась) — автоматический
//! откат на candle.

pub mod candle_backend;
pub mod config;
pub mod model;
pub mod qwen35;
pub mod sampler;
pub mod streaming;
pub mod template;
#[cfg(target_os = "macos")]
pub mod uzu_backend;

use crate::events::{EventTx, JarvisEvent};
use crate::router;
use crate::tools;
use crate::tts::AudioMessage;
use anyhow::{Context, Result};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

pub use template::ChatTurn;

/// Сколько последних реплик (вопрос + ответ = 2 записи) держим в контексте.
pub const MAX_HISTORY_ENTRIES: usize = 6;

/// Номер последнего обращения пользователя.
///
/// Веб-поиск идёт секунды и больше. Без сверки его результат озвучивается
/// тогда, когда пользователь уже задал следующий вопрос или сказал «стоп» —
/// и слышит он чужой, давно не нужный ответ. Фоновые ветки снимают номер
/// на старте и сравнивают перед озвучкой.
static REQUEST_GEN: AtomicU64 = AtomicU64::new(0);

/// Пользователь обратился снова: поднимает номер запроса.
pub fn new_request() -> u64 {
    REQUEST_GEN.fetch_add(1, Ordering::SeqCst) + 1
}

/// Актуален ли ещё запрос `req_id` — ничего нового пользователь не говорил.
pub fn is_current_request(req_id: u64) -> bool {
    REQUEST_GEN.load(Ordering::SeqCst) == req_id
}

/// Текущий номер запроса — его снимают перед `spawn`, чтобы потом сравнить.
pub fn request_snapshot() -> u64 {
    REQUEST_GEN.load(Ordering::SeqCst)
}

/// Промпт для ответов по результатам веб-поиска.
pub const WEB_SYSTEM_PROMPT: &str = concat!(
    "Ты Джарвис, голосовой ассистент (мужской род: «готов», «понял»). Ниже свежие результаты поиска. ",
    "Ответь по-русски одним-двумя предложениями, только суть, без вступлений и без ссылок. ",
    "Используй только факты из результатов; чего в них нет — того не говори. ",
    "Без markdown и эмодзи: ответ озвучивается синтезом речи."
);

/// Системный промпт нейро-нормализации текста для TTS.
pub const NORMALIZE_SYSTEM_PROMPT: &str = "Ты нормализатор текста для синтеза речи на русском. Перепиши текст, заменив все цифры, числа, единицы измерения, сокращения и символы на их произнесение русскими словами с правильными падежами. Примеры: «2 м/с» → «два метра в секунду»; «21°C» → «двадцать один градус Цельсия»; «1000» → «тысяча»; «12:30» → «двенадцать тридцать». Ничего кроме этих замен не меняй. Верни только готовый текст одной строкой.";

/// Базовый системный промпт для разговорного режима (без tool-инструкций).
///
/// Коротко намеренно. Раньше здесь был абзац о том, как ассистент должен
/// материться в ответ на оскорбления, и требование «5-10 развёрнутых
/// предложений»: он занимал треть промпта, прямо противоречил первой строке
/// про краткость и для модели такого размера был чистым шумом.
pub const CHAT_SYSTEM_PROMPT: &str = concat!(
    "Ты Джарвис — голосовой ассистент, мужской род. ",
    "По-русски коротко: одно-три предложения, без markdown и эмодзи — тебя озвучивает синтез. ",
    "Не выдумывай: даты, числа, факты и правила игр проверяй web_search; не нашёл — «не знаю». ",
    "Умеешь открывать приложения и сайты, искать в интернете, ставить таймеры, ",
    "работать с файлами, окнами и музыкой. ",
    "Действие = один блок ",
    "<tool>{\"name\":\"имя_инструмента\",\"args\":{...}}</tool>; на вопросы tool не нужен."
);
/// Вариант для uzu-модели 2B. Жёсткое «не знаешь — скажи „не знаю“» писали
/// против галлюцинаций 0.8B, а 2B начинает отказываться даже там, где знает
/// ответ («8×13», «что такое Rust»), и лезет в ответ шаблон «ИМЯ» из примера.
/// Для неё честность мягче, а заполнитель примера менее «копируемый».
pub const UZU_CHAT_SYSTEM_PROMPT: &str = concat!(
    "Ты Джарвис — голосовой ассистент, мужской род. ",
    "По-русски коротко: одно-три предложения, без markdown и эмодзи — тебя озвучивает синтез. ",
    "Погода, курсы, цены, новости, люди, события, свежие данные — вызывай web_search ",
    "и отвечай строго по его результатам; поиска нет или он пуст — «не знаю». ",
    "Числа и имена без поиска не выдумывай. ",
    "Общеизвестное (столицы, классики, таблица умножения) отвечай уверенно. ",
    "Умеешь открывать приложения и сайты, искать в интернете, ставить таймеры, ",
    "работать с файлами, окнами и музыкой. Звонки, email, умный дом, покупки — не умеешь. ",
    "Действие = один блок ",
    "<tool>{\"name\":\"имя_инструмента\",\"args\":{...}}</tool>; на вопросы tool не нужен."
);
/// Системный промпт разговорной ветки под активный бэкенд.
fn chat_system_prompt() -> &'static str {
    if config::backend() == "uzu" {
        UZU_CHAT_SYSTEM_PROMPT
    } else {
        CHAT_SYSTEM_PROMPT
    }
}

const LLM_WAIT_TIMEOUT_SECS: u64 = 300;

/// Диспетчер LLM-бэкендов: одинаковый потоковый интерфейс поверх candle и uzu.
pub struct LlmEngine {
    backend: Backend,
}

enum Backend {
    Candle(Box<candle_backend::CandleEngine>),
    #[cfg(target_os = "macos")]
    Uzu(Box<uzu_backend::UzuEngine>),
}

impl LlmEngine {
    /// Загрузка бэкенда по настройке `llm_backend`. Uzu существует только
    /// на macOS; если он недоступен — откат на candle, чтобы ассистент
    /// не оставался без мозга.
    pub fn load(models_dir: &std::path::Path, events: &EventTx) -> Result<Self> {
        if config::backend() == "uzu" {
            #[cfg(target_os = "macos")]
            match uzu_backend::UzuEngine::load(models_dir, events) {
                Ok(engine) => {
                    return Ok(Self {
                        backend: Backend::Uzu(Box::new(engine)),
                    });
                }
                Err(e) => {
                    eprintln!("⚠️ Uzu недоступен, откатываюсь на candle: {e:#}");
                    let _ = events.send(JarvisEvent::Log {
                        message: format!("Uzu недоступен, включаю candle: {e}"),
                    });
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("⚠️ llm_backend = \"uzu\" работает только на macOS — использую candle");
            }
        }
        Ok(Self {
            backend: Backend::Candle(Box::new(candle_backend::CandleEngine::load(models_dir)?)),
        })
    }

    pub fn generate_streaming(
        &mut self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        cancel: &AtomicBool,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_streaming_with_params(
            system,
            history,
            user,
            None,
            cancel,
            on_sentence,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_params(
        &mut self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_streaming_inner(
            system,
            history,
            user,
            params_override,
            cancel,
            None,
            on_sentence,
            on_progress,
        )
    }

    /// Генерация с «вставленным началом ответа ассистента»: у candle `prefill`
    /// дописывается в промпт, у uzu форсирование tool-вызова делается
    /// JSON-грамматикой.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_prefill(
        &mut self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        prefill: &str,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_streaming_inner(
            system,
            history,
            user,
            params_override,
            cancel,
            Some(prefill),
            on_sentence,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_streaming_inner(
        &mut self,
        system: &str,
        history: &[ChatTurn],
        user: &str,
        params_override: Option<config::GenerationParams>,
        cancel: &AtomicBool,
        prefill: Option<&str>,
        on_sentence: impl FnMut(&str),
        on_progress: impl FnMut(&str),
    ) -> Result<String> {
        match &mut self.backend {
            Backend::Candle(engine) => engine.generate_streaming_inner(
                system,
                history,
                user,
                params_override,
                cancel,
                prefill,
                on_sentence,
                on_progress,
            ),
            #[cfg(target_os = "macos")]
            Backend::Uzu(engine) => engine.generate_streaming_with_params(
                system,
                history,
                user,
                params_override,
                cancel,
                prefill,
                on_sentence,
                on_progress,
            ),
        }
    }
}

pub struct SharedLlmState {
    pub engine: Mutex<Option<LlmEngine>>,
    pub condvar: Condvar,
    pub failed: AtomicBool,
}

pub type SharedLlm = Arc<SharedLlmState>;

impl SharedLlmState {
    pub fn new() -> Self {
        Self {
            engine: Mutex::new(None),
            condvar: Condvar::new(),
            failed: AtomicBool::new(false),
        }
    }
}

impl Default for SharedLlmState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn spawn_llm_loader(shared: &SharedLlm, events: &EventTx) {
    let shared = Arc::clone(shared);
    let events = events.clone();

    std::thread::spawn(move || {
        let _ = events.send(JarvisEvent::LlmStatus {
            status: "loading".into(),
        });

        let loaded = LlmEngine::load(&crate::paths::models_dir(), &events);
        match loaded {
            Ok(mut engine) => {
                // Прогрев Metal-шейдеров на коротком промпте.
                let cancel = Arc::new(AtomicBool::new(false));
                let warm_params = config::GenerationParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    max_tokens: 16,
                    ..Default::default()
                };
                let started = std::time::Instant::now();
                let _ = engine.generate_streaming_with_params(
                    CHAT_SYSTEM_PROMPT,
                    &[],
                    "Привет",
                    Some(warm_params),
                    &cancel,
                    |_| {},
                    |_| {},
                );
                println!("🔥 LLM прогрета за {:.1?}", started.elapsed());

                let mut guard = shared.engine.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(engine);
                drop(guard);
                shared.condvar.notify_all();
                let _ = events.send(JarvisEvent::LlmStatus {
                    status: "ready".into(),
                });
            }
            Err(e) => {
                eprintln!("❌ Ошибка загрузки LLM: {e:#}");
                shared.failed.store(true, Ordering::SeqCst);
                shared.condvar.notify_all();
                let _ = events.send(JarvisEvent::LlmStatus {
                    status: "failed".into(),
                });
            }
        }
    });
}

fn wait_for_engine(
    shared: &SharedLlm,
    timeout: Duration,
) -> Option<MutexGuard<'_, Option<LlmEngine>>> {
    let guard = shared.engine.lock().unwrap_or_else(|e| e.into_inner());
    let (guard, _) = shared
        .condvar
        .wait_timeout_while(guard, timeout, |engine| {
            engine.is_none() && !shared.failed.load(Ordering::SeqCst)
        })
        .unwrap_or_else(|e| e.into_inner());

    if guard.is_none() {
        eprintln!("❌ Не дождался готовности LLM за {:?}", timeout);
        None
    } else {
        Some(guard)
    }
}
/// Задать вопрос LLM. Не блокирует: генерация идёт в фоне.
pub fn ask(
    shared: SharedLlm,
    system: &'static str,
    prompt: String,
    history: Arc<Mutex<Vec<ChatTurn>>>,
    audio_tx: Sender<AudioMessage>,
    events: EventTx,
    cancel: Arc<AtomicBool>,
) {
    // Детерминированный план системной команды: «открой Max», «сверни окно
    // голосового помощника», «закрой телеграм». Считаем ДО кэша сайтов,
    // иначе установленное приложение проигрывает запомненному домену.
    // Явные команды пользователя из commands.toml важнее общей эвристики.
    let plan = if system == CHAT_SYSTEM_PROMPT
        && crate::tools::utils_tab_for_prompt(&prompt).is_none()
        && !crate::commands::matches_user_command(&prompt)
    {
        router::plan_system_command(&prompt)
    } else {
        None
    };

    // Обучаемый кэш: этот сайт уже разрешали раньше — открываем мгновенно,
    // вообще не поднимая модель.
    if plan.is_none()
        && system == CHAT_SYSTEM_PROMPT
        && router::classify(&prompt) == router::Intent::Action
        && let Some(url) = crate::site_cache::lookup(&crate::tool_guard::strip_browser_phrase(&prompt).0)
    {
        println!("⚡ Кэш сайта: {prompt} → {url}");
        // Через общий путь исполнения: события ActionStarted/ActionResult,
        // та же речь и тот же формат, что у open_url из модели.
        let outcome = tools::execute(
            tools::ToolCall::OpenUrl {
                url,
                browser: crate::tool_guard::browser_in_text(&prompt),
            },
            &audio_tx,
            &events,
        );
        let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(outcome.speech.clone()));
        let _ = events.send(JarvisEvent::LlmDone {
            text: outcome.speech,
        });
        return;
    }

    std::thread::spawn(move || {
        let _ = events.send(JarvisEvent::LlmStatus {
            status: "thinking".into(),
        });

        // Календарные вопросы считаем сами: мгновенно и без кривых tool-блоков
        // от модели («какой сейчас год» → «Сегодня 2026 год.»).
        if let Some(answer) = crate::text::calendar_answer(&prompt) {
            println!("📅 Календарь: {answer}");
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(answer.clone()));
            let _ = events.send(JarvisEvent::LlmDone { text: answer });
            return;
        }

        // Проценты тоже считаем сами: арифметика 2B плавает
        // (то 360, то 636 на одном и том же «15 процентов от 2400»).
        if let Some(answer) = crate::text::percent_answer(&prompt) {
            println!("🧮 Проценты: {answer}");
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(answer.clone()));
            let _ = events.send(JarvisEvent::LlmDone { text: answer });
            return;
        }

        // «Поставь таймер на двадцать секунд» — мгновенно и без модели:
        // длительность разбираем сами (числа-слова STT пишет словами),
        // хаб ставит таймер, окно утилит открывается по событию.
        // Стоп-глаголы («останови таймер на 5 минут») не должны ставить
        // новый таймер: падаем в обычный путь, где остановка обрабатывается.
        let stop_verbs = ["останови", "отмени", "сними", "убери", "выключи", "стоп"];
        let stops_timer = stop_verbs.iter().any(|w| prompt.to_lowercase().contains(w));
        if system == CHAT_SYSTEM_PROMPT
            && !stops_timer
            && prompt.to_lowercase().contains("таймер")
            && let Some(secs) = crate::text::parse_duration(&prompt)
        {
            let speech = format!(
                "Поставил таймер на {}.",
                crate::tools::humanize_seconds(secs)
            );
            println!("⏱ Быстрый таймер: {secs} с");
            crate::timers::start(
                &events,
                &audio_tx,
                "",
                secs,
                crate::timers::TimerKind::Timer,
            );
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech.clone()));
            let _ = events.send(JarvisEvent::LlmDone { text: speech });
            return;
        }

        // «Открой утилиты» / «закрой утилиты» — окно Джарвиса, решаем сами.
        // Без этого фраза уезжала в LLM, а в каталоге у неё был только
        // close_utils — и «открой» выполнялось наоборот.
        if system == CHAT_SYSTEM_PROMPT
            && let Some(closing) = crate::tools::utils_window_request(&prompt)
        {
            let speech = if closing {
                let _ = events.send(JarvisEvent::CloseUtils);
                "Закрыл утилиты."
            } else {
                let tab = crate::tools::utils_tab_for_prompt(&prompt)
                    .map(crate::tools::UtilsTab::parse)
                    .unwrap_or(crate::tools::UTILS_DEFAULT_TAB);
                let _ = events.send(JarvisEvent::OpenUtils {
                    tab: tab.as_str().to_string(),
                });
                "Открыл утилиты."
            };
            println!("🧰 {speech}");
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech.to_string()));
            let _ = events.send(JarvisEvent::LlmDone {
                text: speech.to_string(),
            });
            return;
        }

        // «Проиграй анимацию реактора» — просьба к интерфейсу: сыграть
        // заставку ещё раз. Здесь нечего спрашивать у модели.
        if system == CHAT_SYSTEM_PROMPT && crate::tools::reactor_intro_request(&prompt) {
            let speech = "Проиграю сборку реактора.";
            let _ = events.send(JarvisEvent::ReactorIntro);
            println!("⚙️ {speech}");
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech.to_string()));
            let _ = events.send(JarvisEvent::LlmDone {
                text: speech.to_string(),
            });
            return;
        }

        // Быстрый путь утилит: календарь/секундомер/таймер — окно Джарвиса,
        // не системные приложения и не LLM. Стоп-глаголы отменяют таймер.
        if system == CHAT_SYSTEM_PROMPT
            && let Some(tab) = crate::tools::utils_tab_for_prompt(&prompt)
        {
            let t = prompt.to_lowercase();
            let stopping = ["останови", "отмени", "сними", "убери", "выключи"]
                .iter()
                .any(|w| t.contains(w));
            if stopping && tab == "timer" {
                let speech = match crate::timers::cancel(
                    &events,
                    &audio_tx,
                    Some(crate::timers::TimerKind::Timer),
                    None,
                ) {
                    Some(timer) => {
                        if timer.label.is_empty() {
                            "Остановил таймер.".to_string()
                        } else {
                            format!("Остановил таймер «{}».", timer.label)
                        }
                    }
                    None => "Активных таймеров нет.".to_string(),
                };
                println!("⏹ {speech}");
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech.clone()));
                let _ = events.send(JarvisEvent::LlmDone { text: speech });
                return;
            }
            let name = match tab {
                "calendar" => "Календарь",
                "stopwatch" => "Секундомер",
                _ => "Таймер",
            };
            println!("🧰 Открываю утилиты: {name}");
            let _ = events.send(JarvisEvent::OpenUtils {
                tab: tab.to_string(),
            });
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(format!("Открываю {name}")));
            let _ = events.send(JarvisEvent::LlmDone {
                text: format!("Открыл {name}"),
            });
            return;
        }

        // «Останови/выключи будильник» — глушим сигнал будильника.
        if system == CHAT_SYSTEM_PROMPT {
            let t = prompt.to_lowercase();
            let mentions_alarm = t.contains("будильник") || t.contains("alarm");
            let stopping_alarm = [
                "останови",
                "отмени",
                "сними",
                "убери",
                "выключи",
                "заглуши",
                "стоп",
            ]
            .iter()
            .any(|w| t.contains(w));
            if mentions_alarm && stopping_alarm {
                let speech = match crate::timers::cancel(
                    &events,
                    &audio_tx,
                    Some(crate::timers::TimerKind::Alarm),
                    None,
                ) {
                    Some(alarm) if !alarm.label.is_empty() => {
                        format!("Остановил будильник «{}».", alarm.label)
                    }
                    Some(_) => "Остановил будильник.".to_string(),
                    None => "Будильник не звонит.".to_string(),
                };
                println!("⏹ {speech}");
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech.clone()));
                let _ = events.send(JarvisEvent::LlmDone { text: speech });
                return;
            }
        }

        // Детерминированный план: приложение, файл, папка, сайт, окно или
        // выход приложения — выполняем без модели и без подмены объекта.
        if let Some(call) = plan {
            println!("🧭 План: {call:?}");
            let outcome = tools::execute(call, &audio_tx, &events);
            speak(&audio_tx, &events, &outcome.speech);
            return;
        }

        // Быстрый путь «открой X» убран: он разрешал «яндекс карты» → Maps по
        // суффиксу, игнорируя контекст. Приложение vs сайт теперь решает LLM:
        // open_app/open_url/run_command (каталог команд) с контекстом фразы.

        let verdict = router::classify_detailed(&prompt);
        let is_action = system == CHAT_SYSTEM_PROMPT && verdict.intent == router::Intent::Action;

        if is_action {
            let system = router::action_prompt_for(&prompt);
            run_action(
                &shared,
                &system,
                &prompt,
                verdict.confidence,
                &audio_tx,
                &events,
                &cancel,
                &history,
            );
            return;
        }

        let effective_system = if system == WEB_SYSTEM_PROMPT {
            format!(
                "{system}\nСегодняшняя дата: {}.",
                crate::text::today_russian()
            )
        } else {
            // Разговорная ветка: каталог команд подмешиваем снова — дословный
            // fuzzy-матч команд убран из движка, и без имён в контексте модель
            // не узнает, какой run_command вызвать по смыслу. Каталог пустой —
            // блок не добавляем, чтобы не тратить префилл.
            let chat = chat_system_prompt();
            let catalog = crate::commands::prompt_catalog();
            let commands_block = if catalog.is_empty() {
                String::new()
            } else {
                format!(
                    "\n\nКОМАНДЫ ПОЛЬЗОВАТЕЛЯ (совпало по смыслу — вызывай run_command):\n{catalog}\n  run_command {{\"name\":\"имя\",\"args\":{{\"параметр\":\"значение\"}}}}"
                )
            };
            format!(
                "{chat}\nСегодняшняя дата: {}.\n{commands_block}\nЕсли просьба совпадает с командой пользователя, \
                 вызови run_command с её именем; иначе просто ответь.",
                crate::text::today_russian()
            )
        };

        let history_snapshot: Vec<ChatTurn> = {
            let buf = history.lock().unwrap_or_else(|e| e.into_inner());
            buf.iter()
                .rev()
                .take(MAX_HISTORY_ENTRIES)
                .rev()
                .cloned()
                .collect()
        };

        let Some(mut guard) = wait_for_engine(&shared, Duration::from_secs(LLM_WAIT_TIMEOUT_SECS))
        else {
            let _ = events.send(JarvisEvent::Error {
                message: "LLM недоступна".into(),
            });
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(
                "Модель ещё загружается, повтори запрос чуть позже.".into(),
            ));
            return;
        };

        let Some(engine) = guard.as_mut() else {
            return;
        };

        // Лимита на длину нет (Default max_tokens = 0): разбор длинного
        // вопроса по результатам поиска не обрывается на полуслове.
        let web_params = (system == WEB_SYSTEM_PROMPT).then(|| config::GenerationParams {
            temperature: 0.2,
            ..Default::default()
        });

        let started = std::time::Instant::now();
        let mut text = match engine.generate_streaming_with_params(
            &effective_system,
            &history_snapshot,
            &prompt,
            web_params.clone(),
            &cancel,
            |sentence| {
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(sentence.to_string()));
            },
            |full| {
                let _ = events.send(JarvisEvent::LlmChunk {
                    text: full.to_string(),
                });
            },
        ) {
            Ok(t) => t,
            Err(e) => {
                drop(guard);
                eprintln!("⚠️ Ошибка генерации LLM: {e}");
                let _ = events.send(JarvisEvent::Error {
                    message: format!("Ошибка LLM: {e}"),
                });
                let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(
                    "Не получилось ответить: модель вернула ошибку.".into(),
                ));
                return;
            }
        };

        // Модель 2B иногда отвечает на обычный вопрос кривым tool-блоком
        // (без поля name и т.п.). Разговорная ветка — это вопросы: один раз
        // просим ответить текстом, и только при повторной неудаче звучит
        // «не смог разобрать команду».
        if matches!(tools::parse(&text), tools::ToolParse::Failed(_))
            && text.trim_start().starts_with("<tool>")
        {
            println!("↩️ Вопрос отвечен tool-блоком — переспрашиваю текстом");
            let mut nudge_history = history_snapshot.clone();
            nudge_history.push((prompt.clone(), text.clone()));
            if let Ok(second) = engine.generate_streaming_with_params(
                &effective_system,
                &nudge_history,
                "Это вопрос, а не действие. Ответь обычным текстом по-русски, без tool-блоков.",
                web_params,
                &cancel,
                |sentence| {
                    let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(sentence.to_string()));
                },
                |full| {
                    let _ = events.send(JarvisEvent::LlmChunk {
                        text: full.to_string(),
                    });
                },
            ) {
                text = second;
            }
        }
        drop(guard);

        println!(
            "🤖 Ответ LLM за {:.1?} ({} симв.)",
            started.elapsed(),
            text.chars().count()
        );
        // В разговорном режиме модель тоже может вызвать инструмент —
        // пользовательские команды из каталога.
        let parsed = match tools::parse(&text) {
            tools::ToolParse::None => tools::parse_bare(&text),
            other => other,
        };
        if let tools::ToolParse::Call(mut call) = parsed {
            // Как в run_action: web_search без запроса — запрос есть фраза.
            if let tools::ToolCall::WebSearch { query } = &mut call
                && query.trim().is_empty()
            {
                *query = prompt.clone();
            }
            let outcome = tools::execute(call, &audio_tx, &events);
            let speech = if outcome.ok {
                outcome.speech
            } else {
                // Единая семантика с action-веткой: неудача инструмента
                // возвращается модели один раз — попытка починить, а не
                // озвучивание ошибки как готового ответа.
                match wait_for_engine(&shared, Duration::from_secs(LLM_WAIT_TIMEOUT_SECS)) {
                    Some(mut guard) => {
                        let retry = format!(
                            "Инструмент вернул НЕУДАЧУ: {}. Кратко объясни пользователю                              причину по-русски и предложи, что сделать. Без tool-блоков.",
                            outcome.detail.chars().take(500).collect::<String>()
                        );
                        match guard.as_mut().and_then(|eng| {
                            eng.generate_streaming_with_params(
                                &effective_system,
                                &history_snapshot,
                                &retry,
                                None,
                                &cancel,
                                |_| {},
                                |_| {},
                            )
                            .ok()
                        }) {
                            Some(explanation) if !explanation.trim().is_empty() => {
                                tools::strip_tool_blocks(&explanation)
                            }
                            _ => outcome.speech,
                        }
                    }
                    None => outcome.speech,
                }
            };
            {
                let mut buf = history.lock().unwrap_or_else(|e| e.into_inner());
                buf.push((prompt.clone(), speech.clone()));
                let excess = buf.len().saturating_sub(MAX_HISTORY_ENTRIES);
                buf.drain(..excess);
            }
            let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(speech.clone()));
            let _ = events.send(JarvisEvent::LlmDone { text: speech });
            return;
        }
        // Модель 2B иногда прикладывает к ответу tool-блок — выдуманный
        // («ЯЗЫК_РАСТ») или битый. Сырой JSON в чат и озвучку не идёт:
        // если весь ответ был таким блоком — честно объясняем словами,
        // если блок прилип к тексту — остаётся только текст.
        let text = if let tools::ToolParse::Failed(reason) = tools::parse(&text)
            && text.trim_start().starts_with("<tool>")
        {
            tools::failure_speech(&text, &reason)
        } else {
            tools::strip_tool_blocks(&text)
        };
        {
            let mut buf = history.lock().unwrap_or_else(|e| e.into_inner());
            buf.push((prompt.clone(), text.clone()));
            let excess = buf.len().saturating_sub(MAX_HISTORY_ENTRIES);
            buf.drain(..excess);
        }
        crate::topic::remember(&prompt, &text);
        let _ = events.send(JarvisEvent::LlmDone { text });
    });
}

/// Сколько раз за одну фразу агент может вызвать инструмент.
pub const MAX_TOOL_STEPS: usize = 12;

/// С какой уверенности роутера форсировать tool-блок префиллом.
///
/// Ниже порога модель получает свободу ответить текстом: лучше честный
/// текстовый ответ на неуверенно распознанное действие, чем выдуманный
/// инструмент, который пользователь слышит как «галлюцинацию».
pub const FORCE_TOOL_CONFIDENCE: f32 = 0.7;

/// Начало ответа, которое мы подставляем вместо модели на первом шаге.
pub const TOOL_PREFILL: &str = "<tool>{\"name\":\"";

/// Для просьб «открой сайт» имя инструмента и схему можно задать заранее.
pub const SITE_OPEN_PREFILL: &str = "<tool>{\"name\":\"open_url\",\"args\":{\"url\":\"https://";

fn action_params() -> config::GenerationParams {
    config::GenerationParams {
        temperature: 0.2,
        top_p: 0.85,
        repeat_penalty: 1.15,
        repeat_window: 64,
        // Без лимита: длинная команда в tool-JSON должна досказаться целиком —
        // обрыв посередине раньше давал ложное «не могу выполнить».
        max_tokens: 0,
    }
}

fn speak(audio_tx: &Sender<AudioMessage>, events: &EventTx, text: &str) {
    let text = if text.trim().is_empty() {
        "Готово".to_string()
    } else {
        text.trim().to_string()
    };
    let _ = audio_tx.send(AudioMessage::SynthesizeAndPlay(text.clone()));
    let _ = events.send(JarvisEvent::LlmDone { text });
}

/// Озвучка с учётом отмены: пользователь сказал «стоп» — никакого «Готово»
/// поверх отменённой генерации, тишина и есть честный ответ.
fn speak_cancelable(
    audio_tx: &Sender<AudioMessage>,
    events: &EventTx,
    text: &str,
    cancel: &AtomicBool,
) {
    if cancel.load(Ordering::Relaxed) {
        println!("🛫 Отменено пользователем — молчание вместо «Готово»");
        return;
    }
    if text.trim().is_empty() {
        // Пустой ответ без отмены — это сбой генерации, не успех.
        speak(audio_tx, events, "Ответ не получился, повтори запрос.");
        return;
    }
    speak(audio_tx, events, text);
}

fn host_of(url: &str) -> Option<String> {
    let s = url.trim();
    let s = s
        .strip_prefix("https://")
        .or_else(|| s.strip_prefix("http://"))
        .unwrap_or(s);
    let host = s.split(['/', '?', '#']).next().unwrap_or("").trim();
    let host = host
        .trim_start_matches("www.")
        .trim_end_matches('.')
        .to_lowercase();
    if host.is_empty() { None } else { Some(host) }
}

fn root_domain(host: &str) -> String {
    const TWO_LEVEL: &[&str] = &[
        "co.uk", "org.uk", "com.ru", "net.ru", "org.ru", "com.au", "com.br", "co.jp", "co.in",
        "com.tr", "co.kr",
    ];
    let labels: Vec<&str> = host.split('.').filter(|l| !l.is_empty()).collect();
    if labels.len() <= 2 {
        return host.to_string();
    }
    let last2 = labels[labels.len() - 2..].join(".");
    if TWO_LEVEL.contains(&last2.as_str()) {
        labels[labels.len() - 3..].join(".")
    } else {
        last2
    }
}

fn same_site(a: &str, b: &str) -> bool {
    let (ra, rb) = (root_domain(a), root_domain(b));
    ra == rb || a.ends_with(&format!(".{b}")) || b.ends_with(&format!(".{a}"))
}

fn translit_ru(word: &str) -> String {
    crate::text::translit_ru(word)
}

fn verify_site_url(prompt: &str, proposed: &str, hits: &[crate::web::SearchHit]) -> (String, bool) {
    let words = crate::site_cache::content_words(prompt);
    let proposed_host = host_of(proposed);

    let mut same: Vec<&crate::web::SearchHit> = hits
        .iter()
        .filter(|h| {
            proposed_host
                .as_ref()
                .zip(host_of(&h.url).as_ref())
                .is_some_and(|(ph, hh)| same_site(ph, hh))
        })
        .collect();
    same.sort_by_key(|h| {
        (
            std::cmp::Reverse(address_score(&h.url, &words)),
            h.url.len(),
        )
    });
    if let Some(hit) = same.first() {
        return (hit.url.clone(), true);
    }

    if let Some(hit) = hits
        .iter()
        .find(|h| domain_named_in_request(&h.url, &words))
    {
        return (hit.url.clone(), true);
    }

    (proposed.to_string(), false)
}

/// Финальный выбор адреса сайта: подтверждённый моделью — берём; иначе
/// верхний результат поиска (поисковик знает бренд лучше 2B); выдача пуста —
/// остаётся вариант модели.
fn resolve_site_url(proposed: &str, confirmed: bool, hits: &[crate::web::SearchHit]) -> String {
    if confirmed || hits.is_empty() {
        return proposed.to_string();
    }
    hits[0].url.clone()
}

fn address_score(url: &str, words: &[String]) -> usize {
    if words.is_empty() {
        return 0;
    }
    let tokens = crate::site_cache::content_words(url);
    words.iter().filter(|w| word_in_tokens(w, &tokens)).count()
}

fn domain_named_in_request(url: &str, words: &[String]) -> bool {
    let Some(host) = host_of(url) else {
        return false;
    };
    let labels: Vec<String> = root_domain(&host)
        .split('.')
        .filter(|l| !l.is_empty())
        .map(translit_ru)
        .filter(|l| l.chars().count() >= 4)
        .collect();
    if labels.is_empty() {
        return false;
    }
    words.iter().any(|w| {
        let q = translit_ru(w);
        if q.chars().count() < 4 {
            return false;
        }
        let head: String = q.chars().take(4).collect();
        labels.iter().any(|l| {
            crate::text::word_similarity(&q, l) >= 0.72
                || crate::text::word_similarity(&head, l) >= 0.7
        })
    })
}

fn explicit_domain(prompt: &str) -> Option<String> {
    for raw in prompt.split_whitespace() {
        let tok = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '-');
        let Some((name, tld)) = tok.rsplit_once('.') else {
            continue;
        };
        let tld_ok = tld.chars().count() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic());
        let name_ok = !name.is_empty()
            && name.chars().any(|c| c.is_ascii_alphabetic())
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.');
        if tld_ok && name_ok {
            return Some(tok.to_lowercase());
        }
    }
    None
}

fn word_in_tokens(word: &str, tokens: &[String]) -> bool {
    let t = translit_ru(word);
    let sub = t.chars().count() >= 4;
    tokens.iter().any(|tok| {
        tok == &t
            || tok == word
            || (sub && tok.contains(&t))
            || (sub && !tok.is_empty() && t.contains(tok.as_str()))
            || crate::text::word_similarity(&t, tok) >= 0.58
    })
}

fn looks_like_site_request(text: &str) -> bool {
    let t = text.to_lowercase();
    let site = has_site_word(&t);
    let verb = [
        "открой",
        "откройте",
        "зайди",
        "зайти",
        "перейди",
        "перейти",
        "заходи",
        "посети",
        "open",
    ]
    .iter()
    .any(|k| t.contains(k));
    let not_site = [
        "файл",
        "папк",
        "задач",
        "таймер",
        "напомни",
        "заметк",
        "стих",
        "сказк",
        "письм",
        "почт",
        "музык",
        "видео",
        "фото",
        "карт",
        "календар",
        "контакт",
        "приложен",
        "утилит",
        "окно",
        "анимаци",
        "реактор",
    ]
    .iter()
    .any(|k| t.contains(k));
    !not_site && (site || verb)
}

fn has_site_word(text: &str) -> bool {
    ["сайт", "страниц", "интернет", "онлайн"]
        .iter()
        .any(|k| text.contains(k))
}

fn take_search(
    prefetch: &mut Option<std::thread::JoinHandle<Option<Vec<crate::web::SearchHit>>>>,
    prompt: &str,
) -> Vec<crate::web::SearchHit> {
    if let Some(handle) = prefetch.take() {
        return handle.join().ok().flatten().unwrap_or_default();
    }
    let query = crate::site_cache::search_query(prompt);
    if query.is_empty() {
        return Vec::new();
    }
    crate::web::search(&query, 5).unwrap_or_default()
}

fn needs_next_step(call: &tools::ToolCall, outcome: &tools::ToolOutcome) -> bool {
    match call {
        tools::ToolCall::Shell { .. }
        | tools::ToolCall::FileRead { .. }
        | tools::ToolCall::FileList { .. } => true,
        // Результат оконной команды финален: после честного «свернул»/«нет
        // окон» модель получала слово и начинала выдумывать причины ошибки.
        tools::ToolCall::Window { .. } => false,
        // Ввод/ожидание — звенья сценария интерфейса: модель должна продолжить
        // цепочку и сама решить, остались ли действия.
        tools::ToolCall::DesktopInput { .. } | tools::ToolCall::Wait { .. } => true,
        tools::ToolCall::WebSearch { .. } => outcome.ok,
        tools::ToolCall::RunCommand { .. } => {
            !outcome.ok || outcome.detail.starts_with("Результаты поиска")
        }
        // Запись файла, снимок и уведомление раньше обрывали цепочку даже при
        // успехе: «сделай» превращалось в один вызов и стоп. Теперь модель
        // сама говорит, закончила она или продолжает. Защита от повторов и
        // лимит шагов на месте, поэтому цикла это не создаёт.
        tools::ToolCall::FileWrite { .. }
        | tools::ToolCall::Screenshot
        | tools::ToolCall::Notify { .. } => true,
        // Плеер: успех — это готовый ответ («Поставил на паузу»), и лишний
        // круг модели тут только добавлял «Задача завершена» после команды.
        // Неудача — наоборот: пусть модель объяснит причину словами.
        tools::ToolCall::MusicControl { .. } => !outcome.ok,
        // GUI-агент возвращает финальный отчёт: успех финален, провал —
        // модель объясняет причину или переформулирует задачу.
        tools::ToolCall::UiTask { .. } => !outcome.ok,
        // Описание экрана — информация для следующего шага, как file_read.
        tools::ToolCall::ScreenLook { .. } => true,
        // Запуск/закрытие приложения и открытие ссылки — звенья цепочки
        // («открой сафари сайт авиасейлс» = open_app + open_url/ui_task).
        // Раньше цикл обрывался после первого шага, и задача не доводилась.
        tools::ToolCall::OpenApp { .. } | tools::ToolCall::QuitApp { .. } => true,
        tools::ToolCall::OpenUrl { .. } | tools::ToolCall::OpenPath { .. } => true,
        _ => !outcome.ok,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_action(
    shared: &SharedLlm,
    system: &str,
    prompt: &str,
    confidence: f32,
    audio_tx: &Sender<AudioMessage>,
    events: &EventTx,
    cancel: &Arc<AtomicBool>,
    history: &Arc<std::sync::Mutex<Vec<ChatTurn>>>,
) {
    let mut loop_history: Vec<ChatTurn> = Vec::new();
    let mut user_msg = prompt.to_string();
    // Промпт шага не постоянен: после веб-поиска переключаемся на веб-промпт
    // (ответ строго по результатам), иначе модель продолжает «задачу».
    let mut active_system: String = system.to_string();
    let mut active_task: Option<crate::tasks::Task> = None;
    let mut nudged = false;
    let mut done_calls: Vec<String> = Vec::new();
    let mut searched = false;
    // Роутер не был уверен, что это действие. Тогда prefill не форсируем:
    // если модель хочет ответить текстом — пусть ответит, это дешевле, чем
    // выдуманный tool-вызов, из-за которого пользователь слышит чушь.
    let force_tool = confidence >= FORCE_TOOL_CONFIDENCE;
    // Диагностика выбора: вердикт роутера виден в журнале UI.
    let _ = events.send(JarvisEvent::RouterVerdict {
        intent: "action".into(),
        confidence,
        force_tool,
    });
    // «Открой сайт X в сафари»: слово браузера не должно попадать ни в поиск,
    // ни в ключ кэша сайтов, а сам браузер передаётся в open_url.
    let (site_prompt_owned, browser_hint) = crate::tool_guard::strip_browser_phrase(prompt);
    let site_prompt: &str = site_prompt_owned.as_str();
    let site_request = looks_like_site_request(prompt);
    let fast_site = site_request && has_site_word(&prompt.to_lowercase());
    let mut prefetch = if site_request {
        let query = crate::site_cache::search_query(site_prompt);
        if query.is_empty() {
            None
        } else {
            Some(std::thread::spawn(move || {
                crate::web::search(&query, 5).ok()
            }))
        }
    } else {
        None
    };

    // Uzu не умеет prefill, поэтому SITE_OPEN_PREFILL там не срабатывает и
    // модель сама выбирает инструмент («открой сайт X в сафари» превращалось
    // в open_app + ui_task). Адрес для «открой сайт …» не требует LLM: берём
    // его из выдачи поиска, которая уже запущена, и открываем сразу.
    if fast_site && config::backend() == "uzu" {
        let hits = take_search(&mut prefetch, site_prompt);
        if !hits.is_empty() {
            let (candidate, confirmed) = verify_site_url(site_prompt, "", &hits);
            let final_url = resolve_site_url(&candidate, confirmed, &hits);
            println!("🎯 Сайт без LLM (бэкенд без prefill): {final_url}");
            let outcome = tools::execute(
                tools::ToolCall::OpenUrl {
                    url: final_url.clone(),
                    browser: browser_hint.clone(),
                },
                audio_tx,
                events,
            );
            if outcome.ok && confirmed {
                crate::site_cache::remember(site_prompt, &final_url);
            }
            speak_cancelable(audio_tx, events, &outcome.speech, cancel);
            commit_action_history(history, prompt, &outcome.speech);
            return;
        }
    }

    for step in 1..=MAX_TOOL_STEPS {
        let generated = {
            let Some(mut guard) =
                wait_for_engine(shared, Duration::from_secs(LLM_WAIT_TIMEOUT_SECS))
            else {
                let _ = events.send(JarvisEvent::Error {
                    message: "LLM недоступна".into(),
                });
                return;
            };
            let Some(engine) = guard.as_mut() else {
                return;
            };
            let progress = |full: &str| {
                let _ = events.send(JarvisEvent::LlmChunk {
                    text: full.to_string(),
                });
            };

            if step == 1 && force_tool {
                let prefill = if fast_site {
                    SITE_OPEN_PREFILL
                } else {
                    TOOL_PREFILL
                };
                engine.generate_streaming_with_prefill(
                    &active_system,
                    &loop_history,
                    &user_msg,
                    Some(action_params()),
                    cancel,
                    prefill,
                    |_| {},
                    progress,
                )
            } else {
                engine.generate_streaming_with_params(
                    &active_system,
                    &loop_history,
                    &user_msg,
                    Some(action_params()),
                    cancel,
                    |_| {},
                    progress,
                )
            }
        };

        let text = match generated {
            Ok(t) => t,
            Err(e) => {
                eprintln!("⚠️ Ошибка генерации LLM: {e}");
                let _ = events.send(JarvisEvent::Error {
                    message: format!("Ошибка LLM: {e}"),
                });
                return;
            }
        };

        // Отмена обрывает генерацию на полуслове: достраивать обрезанный
        // JSON (parse_lenient) и исполнять его нельзя — усечённый shell.cmd
        // становится другой командой.
        if cancel.load(Ordering::Relaxed) {
            println!("🛫 Отменено пользователем — ответ не разбирается и не исполняется");
            return;
        }
        let parsed = match tools::parse_lenient(&text) {
            tools::ToolParse::None => tools::parse_bare(&text),
            other => other,
        };
        match parsed {
            tools::ToolParse::Call(tools::ToolCall::TaskRun { id }) => {
                if step == MAX_TOOL_STEPS {
                    // Перезапись user_msg на последнем шаге раньше приводила к
                    // «слишком много шагов» без единого выполненного действия.
                    speak_cancelable(
                        audio_tx,
                        events,
                        "Шагов не осталось — запускаю задачу в следующий раз.",
                        cancel,
                    );
                    return;
                }
                let task = id
                    .and_then(crate::tasks::find)
                    .filter(|t| !t.done)
                    .or_else(|| crate::tasks::pending().into_iter().next());
                let Some(task) = task else {
                    speak(audio_tx, events, &crate::tasks::speak_next());
                    return;
                };
                println!("▶️ Задача {}: {}", task.id, task.text);
                loop_history.push((user_msg.clone(), text.clone()));
                user_msg = format!(
                    "Задача {}: {}\nВыполни её инструментами, шаг за шагом. \
                     Когда всё готово — ответь одним коротким предложением без tool.",
                    task.id, task.text
                );
                active_task = Some(task);
            }

            tools::ToolParse::Call(call) => {
                println!("🔧 Tool call (шаг {step}/{MAX_TOOL_STEPS}): {call:?}");
                let mut cache_url: Option<String> = None;
                let mut cache_confirmed = false;
                let call = match call {
                    tools::ToolCall::OpenUrl { url, browser } if !searched => {
                        let browser = browser.or_else(|| browser_hint.clone());
                        match explicit_domain(site_prompt) {
                            Some(domain) => {
                                let exact = format!("https://{domain}/");
                                cache_url = Some(exact.clone());
                                cache_confirmed = true; // домен назван явно
                                tools::ToolCall::OpenUrl { url: exact, browser }
                            }
                            None => {
                                let hits = take_search(&mut prefetch, site_prompt);
                                let (candidate, confirmed) = verify_site_url(site_prompt, &url, &hits);
                                // Модель выдумывает адреса («дипсика» → Dipstick
                                // на вики): если выдача её не подтвердила,
                                // открываем верхний результат поиска — поисковик
                                // понимает «дипсика» лучше маленькой модели.
                                let final_url = resolve_site_url(&candidate, confirmed, &hits);
                                if final_url != url {
                                    println!("🎯 Беру адрес из выдачи: {final_url}");
                                }
                                // Кэш отравлять нельзя: запоминаем только
                                // URL, который подтвердила выдача или явный
                                // домен из фразы.
                                cache_url = confirmed.then(|| final_url.clone());
                                cache_confirmed = confirmed;
                                tools::ToolCall::OpenUrl { url: final_url, browser }
                            }
                        }
                    }
                    other => {
                        let other = match other {
                            tools::ToolCall::OpenUrl { url, browser: None } if browser_hint.is_some() => {
                                tools::ToolCall::OpenUrl { url, browser: browser_hint.clone() }
                            }
                            o => o,
                        };
                        if let tools::ToolCall::OpenUrl { url, .. } = &other {
                            // Повторное открытие сайта в том же цикле — модель
                            // уже назвала конкретный адрес: считаем явным.
                            cache_url = Some(url.clone());
                            cache_confirmed = true;
                        }
                        other
                    }
                };
                // web_search с пустым запросом (частый паттерн 2B): запрос —
                // сама фраза пользователя, а не провал парсинга.
                let call = match call {
                    tools::ToolCall::WebSearch { query }
                        if query.trim().is_empty() && !searched =>
                    {
                        tools::ToolCall::WebSearch {
                            query: prompt.to_string(),
                        }
                    }
                    other => other,
                };
                if let tools::ToolCall::WebSearch { .. } = &call {
                    searched = true;
                }

                let call_key = format!("{call:?}");
                if done_calls.contains(&call_key) {
                    println!("🛑 Повтор команды — цикл прерван: {call_key}");
                    if let Some(task) = active_task.take() {
                        offer_close_task(audio_tx, events, &task);
                    } else {
                        speak_cancelable(
                            audio_tx,
                            events,
                            "Эту команду я уже выполнял. Останавливаюсь.",
                            cancel,
                        );
                    }
                    return;
                }
                done_calls.push(call_key.clone());

                let outcome = tools::execute(call.clone(), audio_tx, events);
                if let (Some(url), true) = (cache_url, cache_confirmed) && outcome.ok {
                    crate::site_cache::remember(site_prompt, &url);
                }
                let last_step = step == MAX_TOOL_STEPS;
                if last_step || !needs_next_step(&call, &outcome) {
                    if let Some(task) = active_task.take() {
                        offer_close_task(audio_tx, events, &task);
                    } else {
                        speak_cancelable(audio_tx, events, &outcome.speech, cancel);
                    }
                    commit_action_history(history, prompt, &outcome.speech);
                    return;
                }
                loop_history.push((user_msg.clone(), text.clone()));
                if matches!(&call, tools::ToolCall::WebSearch { .. }) {
                    // Результаты поиска — не «продолжай задачу», а материал
                    // для ответа: переключаемся на веб-промпт и отдаём
                    // сниппеты целиком. Голые заголовки модель 2B превращает
                    // в выдумку по названию («азартный досоло» про Руми-куб).
                    active_system = format!(
                        "{}\nСегодняшняя дата: {}.",
                        WEB_SYSTEM_PROMPT,
                        crate::text::today_russian()
                    );
                    user_msg = outcome.detail;
                } else {
                    // 800 символов вместо 400: вывод команды и содержимое файла
                    // в 400 знаков не влезали, и модель продолжала задачу вслепую.
                    let brief: String = outcome.detail.chars().take(800).collect();
                    let status_note = if outcome.ok {
                        "успех"
                    } else {
                        "НЕУДАЧА"
                    };
                    let empty_note = if outcome.ok && brief.trim().is_empty() {
                        " Команда ничего не вывела — это нормально, она уже выполнена."
                    } else {
                        ""
                    };
                    // Провал open_app — не повод открывать что-то другое:
                    // ни приложение, ни сайт, ни файл вместо несуществующей
                    // программы. Пользователь должен услышать честную причину.
                    let app_note =
                        if matches!(&call, tools::ToolCall::OpenApp { .. }) && !outcome.ok {
                            " Приложение не найдено. НЕ открывай вместо него другое приложение, \
                             сайт или файл и не выдумывай имена. Скажи пользователю, что это \
                             приложение не найдено."
                        } else if matches!(
                            &call,
                            tools::ToolCall::UiTask { .. } | tools::ToolCall::ScreenLook { .. }
                        ) && !outcome.ok
                        {
                            " Агент зрения не отработал. НЕ вводи текст и не кликай вслепую \
                             (desktop_input), не повторяй тот же ui_task. Скажи пользователю причину."
                        } else {
                            ""
                        };
                    let done_list = done_calls.join("; ");
                    user_msg = format!(
                        "Результат инструмента ({status_note}): {brief}{empty_note}{app_note}\n\
                         Уже выполнено: {done_list}. Не повторяй эти команды.\n\
                         Продолжай задачу. Если всё готово — ответь коротко, без tool."
                    );
                }
            }

            tools::ToolParse::Failed(reason) => {
                eprintln!("⚠️ Tool-вызов не разобран: {reason}");
                eprintln!("   сырой ответ: {text}");
                let _ = events.send(JarvisEvent::ToolParseFailed {
                    reason: reason.clone(),
                    raw: text.chars().take(400).collect(),
                });
                // Роутер не был уверен, а модель выдумала инструмент. Второй
                // раз требовать tool бессмысленно: отвечаем как на вопрос.
                if !force_tool && step == 1 {
                    println!("↩️ Неуверенный action + выдуманный tool — отвечаю текстом");
                    speak(
                        audio_tx,
                        events,
                        "Не понял, что именно сделать. Скажи точнее — что открыть или выполнить?",
                    );
                    return;
                }
                if step == 1 && !nudged {
                    nudged = true;
                    loop_history.push((user_msg.clone(), text.clone()));
                    user_msg = format!(
                        "Твой tool-блок не разобран: {reason}. Выбери имя инструмента \
                         строго из списка системного промпта и ответь ОДНИМ корректным \
                         tool-блоком вида <tool>{{\"name\":\"open_url\",\"args\":{{...}}}}</tool>."
                    );
                    continue;
                }
                speak(audio_tx, events, &tools::failure_speech(&text, &reason));
                return;
            }

            tools::ToolParse::None => {
                println!("🗣 LLM без tool: {text}");
                // Отмена («стоп») превыше всего: никакой дожим, никакая речь.
                if cancel.load(Ordering::Relaxed) {
                    println!("🛫 Отменено пользователем — выходим без озвучки");
                    return;
                }
                // Если роутер сомневался — текстовый ответ и есть правильный
                // исход, дожимать tool-блок не нужно.
                if step == 1 && !nudged && !text.trim().is_empty() && force_tool {
                    nudged = true;
                    loop_history.push((user_msg.clone(), text.clone()));
                    user_msg = "Пользователь просил выполнить действие, а ты ответил \
                        текстом. Ответь ТОЛЬКО одним tool-блоком \
                        <tool>{\"name\":\"...\",\"args\":{...}}</tool> — без единого слова."
                        .to_string();
                    continue;
                }
                match active_task.take() {
                    Some(task) => offer_close_task(audio_tx, events, &task),
                    None => speak_cancelable(audio_tx, events, &text, cancel),
                }
                // Итог действия — в общую историю: уточнения «открой его» и
                // «а теперь закрой» должны видеть, что именно выполнялось.
                commit_action_history(history, prompt, &text);
                return;
            }
        }
    }

    speak_cancelable(
        audio_tx,
        events,
        "Остановился: слишком много шагов.",
        cancel,
    );
}

/// Итог действия в общую историю диалога (Этап «контекст»): фраза
/// пользователя и честный итог действия. Без этого уточнения вроде
/// «открой его» и «а теперь закрой» не видят предыдущего действия.
fn commit_action_history(
    history: &Arc<std::sync::Mutex<Vec<ChatTurn>>>,
    prompt: &str,
    result: &str,
) {
    let result = result.trim();
    if result.is_empty() {
        return;
    }
    let mut buf = history.lock().unwrap_or_else(|e| e.into_inner());
    buf.push((prompt.to_string(), result.to_string()));
    let excess = buf.len().saturating_sub(MAX_HISTORY_ENTRIES);
    buf.drain(..excess);
}

fn offer_close_task(audio_tx: &Sender<AudioMessage>, events: &EventTx, task: &crate::tasks::Task) {
    crate::tasks::expect_done(task.id, &task.text);
    println!("⏳ Жду ответа по задаче {}: {}", task.id, task.text);
    speak(
        audio_tx,
        events,
        &format!(
            "Задача {} — {} — сделана? Скажи да или нет.",
            task.id, task.text
        ),
    );
}

pub fn sanitize_normalized(original: &str, output: &str) -> Option<String> {
    let out = output
        .trim()
        .trim_start_matches(['"', '\u{ab}', '\u{bb}'])
        .trim_end_matches(['"', '\u{ab}', '\u{bb}'])
        .trim()
        .to_string();

    if out.is_empty() {
        return None;
    }
    if !out.chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    if out.contains("пример") || out.contains("замени") {
        return None;
    }
    let orig_chars = original.chars().count();
    let out_chars = out.chars().count();
    if out_chars > orig_chars * 4 + 40 || out_chars < orig_chars / 4 {
        return None;
    }
    Some(out)
}

/// Нейро-нормализация текста для TTS без ожидания: если движок занят —
/// сразу возвращает None, вызывающий откатывается на правила.
pub fn try_normalize(shared: &SharedLlm, text: &str) -> Option<String> {
    let Ok(mut guard) = shared.engine.try_lock() else {
        return None;
    };
    let engine = guard.as_mut()?;
    let cancel = Arc::new(AtomicBool::new(false));
    let params = config::GenerationParams {
        temperature: 0.2,
        top_p: 0.8,
        repeat_penalty: 1.15,
        repeat_window: 64,
        // Без обрыва: порезанная нормализация = порезанная озвучка.
        max_tokens: 0,
    };

    let result = engine.generate_streaming_with_params(
        NORMALIZE_SYSTEM_PROMPT,
        &[],
        text,
        Some(params),
        &cancel,
        |_| {},
        |_| {},
    );

    let raw = result.ok()?;
    sanitize_normalized(text, &raw)
}

/// Синхронный вопрос-ответ (для тестов).
pub fn generate_blocking(
    shared: &SharedLlm,
    system: &str,
    history: &[ChatTurn],
    user: &str,
    cancel: &AtomicBool,
) -> Result<String> {
    let mut guard = wait_for_engine(shared, Duration::from_secs(LLM_WAIT_TIMEOUT_SECS))
        .context("LLM недоступна")?;
    let engine = guard.as_mut().context("LLM недоступна")?;
    engine.generate_streaming(system, history, user, cancel, |_| {}, |_| {})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_sanitizer_rejects_garbage() {
        assert_eq!(
            sanitize_normalized("2 м/с", "два метра в секунду").unwrap(),
            "два метра в секунду"
        );
        assert!(sanitize_normalized("2 м/с", "замени единицы на слова").is_none());
        assert!(sanitize_normalized("2 м/с", "  ").is_none());
        assert!(sanitize_normalized("2 м/с", &"два метра в секунду ".repeat(20)).is_none());
    }

    #[test]
    fn host_extraction_normalizes() {
        assert_eq!(
            host_of("https://www.apple.com/ipad-pro").as_deref(),
            Some("apple.com")
        );
        assert_eq!(host_of("cdek.ru").as_deref(), Some("cdek.ru"));
        assert_eq!(host_of("  ").as_deref(), None);
    }

    #[test]
    fn same_site_matches_subdomains() {
        assert!(same_site("netflix.com", "about.netflix.com"));
        assert!(same_site("www.netflix.com", "netflix.com"));
        assert!(!same_site("sdekshop.com", "cdek.ru"));
    }

    #[test]
    fn verify_keeps_brand_over_aggregator() {
        let hits = vec![
            crate::web::SearchHit {
                title: "Купить Mercedes-Benz".into(),
                url: "https://auto.ru/cars/mercedes/all/".into(),
                snippet: String::new(),
            },
            crate::web::SearchHit {
                title: "Mercedes в России".into(),
                url: "https://mercedes-benz-russia.ru/".into(),
                snippet: String::new(),
            },
        ];
        let (url, confirmed) =
            verify_site_url("открой сайт мерседес", "https://www.mercedes.com", &hits);
        assert_eq!(url, "https://www.mercedes.com");
        assert!(!confirmed);
    }

    #[test]
    fn explicit_domain_is_detected() {
        assert_eq!(
            explicit_domain("открой сайт apple.com"),
            Some("apple.com".to_string())
        );
        assert_eq!(explicit_domain("открой сайт дипсика"), None);
    }
}