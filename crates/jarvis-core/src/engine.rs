//! Конечный автомат ассистента: wake word → команда → LLM.

use crate::audio::{calculate_rms, resample_audio, spawn_stream};
use crate::commands::{CommandsConfig, execute_command, get_global_config};
use crate::events::{EventTx, JarvisEvent};
use crate::llm::{self, ChatTurn, SharedLlm};
use crate::paths;
use crate::router;
use crate::settings::Settings;
use crate::speaker::{SpeechProbeResult, SpeakerQuery, SpeakerReply, Verdict};
use crate::stt::Stt;
use crate::text::{clean_wakewords_and_prompt, extract_repeat_text, is_valid};
use crate::tts::AudioMessage;
use crate::wake;
use crate::{text, topic};
use anyhow::{Context, Result};
use cpal::traits::StreamTrait;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError, sync_channel};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

const HISTORY_BUFFER_SECS: usize = 1;
const MIC_LEVEL_INTERVAL_MS: u64 = 40;
/// Сколько миллисекунд речи нужно принять до того, как она станет фразой,
/// от которой берётся преролл (начало слова). 500 мс: потеря первого слога
/// («проиграй» → «играй») недопустима для команд; буфер истории — 1 с.
const PREROLL_SECS: f32 = 0.5;
/// Сколько миллисекунд после озвучки ещё игнорируем вход: в буфере микрофона
/// остаётся хвост только что сказанного.
const POST_SPEAK_GUARD_MS: u64 = 250;

/// VAD speakrs: сколько последних секунд фразы отправляем на проверку.
/// Короче окно — дешевле инференс, но пауза в начале окна выглядит как конец
/// фразы. 1.5 с хватает, чтобы отличить паузу посреди слова от тишины.
const VAD_WINDOW_SECS: f32 = 1.5;
/// Как часто опрашиваем VAD, пока пользователь говорит.
const VAD_PROBE_INTERVAL_MS: u64 = 400;
/// Сколько ждём ответ VAD, прежде чем считать пайплайн зависшим.
const VAD_PROBE_TIMEOUT_MS: u64 = 2500;
/// Насколько свежим должен быть результат VAD, чтобы закрыть по нему фразу.
/// Запас больше типичных 200 мс инференса: медленный CPU не должен лишать
/// ассистента возможности когда-либо закончить фразу.
const VAD_RESULT_MAX_AGE_MS: u64 = 2000;
/// Минимальный измеренный хвост тишины: у speakrs есть смещение ~0.2 с
/// (речь в окне заканчивается раньше реального конца), поэтому слишком
/// маленький порог закрывал бы фразу прямо посреди слова.
const VAD_MIN_TAIL_SECS: f32 = 0.35;
/// Жёсткий предел фразы, когда VAD подтверждает, что речь ещё идёт.
/// Обычный `max_command_duration_secs` — мягкий: на нём фраза не режется,
/// пока speakrs видит голос.
const HARD_MAX_COMMAND_SECS: f32 = 30.0;

#[derive(PartialEq, Debug, Clone, Copy)]
enum AppState {
    WaitingForWakeWord,
    WaitingForCommand,
    Processing,
}

pub struct EngineHandles {
    pub commands: Arc<RwLock<CommandsConfig>>,
    pub settings: Arc<RwLock<Settings>>,
    pub audio_tx: Sender<AudioMessage>,
    pub is_speaking: Arc<AtomicBool>,
    pub stop_audio_flag: Arc<AtomicBool>,
    pub paused: Arc<AtomicBool>,
    pub wakeword_reload: Arc<AtomicBool>,
    pub llm_cancel: Arc<AtomicBool>,
    pub llm_history: Arc<Mutex<Vec<ChatTurn>>>,
    pub llm: SharedLlm,
    pub events: EventTx,
    /// Голосовой воркер: запросы VAD/верификации и ответы.
    pub speaker_tx: std::sync::mpsc::SyncSender<SpeakerQuery>,
    pub speaker_rx: Receiver<SpeakerReply>,
}

pub struct SttJob {
    pub samples: Vec<f32>,
}

pub fn spawn_stt_worker(
    stt: Arc<Mutex<crate::stt::Stt>>,
    settings: Arc<RwLock<crate::settings::Settings>>,
    commands: Arc<RwLock<CommandsConfig>>,
) -> (SyncSender<SttJob>, Receiver<anyhow::Result<String>>) {
    let (tx, rx) = sync_channel::<SttJob>(2);
    let (out_tx, out_rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        // Словарь Jarvis для beam search: триггерные слова команд.
        // Пересобирается, когда состав команд меняется (hot-reload config).
        let mut dict: Option<crate::asr::ctc::JarvisDictionary> = None;
        let mut dict_len = 0usize;
        while let Ok(job) = rx.recv() {
            let mut stt = stt.lock().unwrap_or_else(|e| e.into_inner());
            let started = Instant::now();
            let (decoder_name, beam_width, dict_enabled) = {
                let s = settings.read().unwrap_or_else(|e| e.into_inner());
                (s.asr_decoder.clone(), s.asr_beam_width, s.asr_dict)
            };
            let cmd_len = commands.read().unwrap_or_else(|e| e.into_inner()).commands.len();
            let res = match decoder_name.as_str() {
                "ctc" => stt
                    .model()
                    .transcribe_with(&job.samples, crate::asr::model::Decoder::CtcGreedy, None)
                    .map(|t| t.text.trim().to_string()),
                "ctc_beam" => {
                    if dict_enabled && cmd_len != dict_len {
                        let phrases: Vec<String> = commands
                            .read()
                            .unwrap_or_else(|e| e.into_inner())
                            .commands
                            .iter()
                            .filter(|c| c.enabled)
                            .flat_map(|c| {
                                let mut variants = c.trigger_words.clone();
                                // Латинские триггеры («premiere pro») ASR пишет
                                // кириллицей — добавляем русскую форму
                                // произношения из TextNormalizer.
                                for w in &c.trigger_words {
                                    let ru = crate::tts_normalizer::apply_tech_pronunciation(
                                        &w.to_lowercase(),
                                    );
                                    if ru != w.to_lowercase() && !variants.contains(&ru) {
                                        variants.push(ru);
                                    }
                                }
                                variants
                            })
                            .collect();
                        dict = stt.model().jarvis_dictionary(&phrases).ok();
                        dict_len = cmd_len;
                    }
                    stt.model()
                        .transcribe_with(
                            &job.samples,
                            crate::asr::model::Decoder::CtcBeam {
                                width: beam_width.max(1),
                            },
                            dict.as_ref(),
                        )
                        .map(|t| t.text.trim().to_string())
                }
                _ => stt.transcribe(&job.samples),
            };
            let secs = job.samples.len() as f32 / wake::SAMPLE_RATE as f32;
            println!(
                "📝 STT[{}]: {:.1?} на {:.1} с аудио",
                decoder_name,
                started.elapsed(),
                secs
            );
            let _ = out_tx.send(res);
        }
    });

    (tx, out_rx)
}

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

#[derive(PartialEq)]
enum Outcome {
    Wake,
    Followup,
}

/// Состояние VAD-проверок одной фразы.
///
/// Запросы идут асинхронно: аудио-цикл не ждёт инференс speakrs, а лишь
/// отправляет очередное окно и забирает готовый ответ. Идентификатор
/// отсекает ответы, пришедшие после сброса фразы.
struct VadTracker {
    id: u64,
    in_flight: bool,
    probe_started_at: Instant,
    probe_audio_end_at: Instant,
    last_probe_at: Instant,
    /// Измеренный хвост тишины после последней речи в окне, секунд.
    tail_secs: Option<f32>,
    /// Момент конца аудио, по которому посчитан `tail_secs`.
    tail_at: Option<Instant>,
    /// Хотя бы один запрос завершился успешно — VAD можно доверять.
    healthy: bool,
    /// Пайплайн недоступен: движок откатывается на RMS-детекцию.
    unavailable: bool,
    /// Подряд идущих ошибок инференса (одна может быть случайной).
    failures: u8,
    /// speakrs хоть раз увидел речь в текущей фразе.
    saw_speech: bool,
}

impl VadTracker {
    fn new() -> Self {
        Self {
            id: 0,
            in_flight: false,
            probe_started_at: Instant::now(),
            probe_audio_end_at: Instant::now(),
            // Первый запрос можно отправлять сразу, как только началась речь.
            last_probe_at: Instant::now() - Duration::from_millis(VAD_PROBE_INTERVAL_MS),
            tail_secs: None,
            tail_at: None,
            healthy: false,
            unavailable: false,
            failures: 0,
            saw_speech: false,
        }
    }

    /// Новая фраза: старые ответы становятся невалидными по id.
    fn reset_phrase(&mut self) {
        self.id = self.id.wrapping_add(1);
        self.in_flight = false;
        self.tail_secs = None;
        self.tail_at = None;
        self.saw_speech = false;
        self.last_probe_at = Instant::now() - Duration::from_millis(VAD_PROBE_INTERVAL_MS);
    }

    /// Принять ответ VAD. `false` — ответ устарел и к текущей фразе не относится.
    fn accept(&mut self, result: SpeechProbeResult) -> bool {
        if result.id != self.id {
            return false;
        }
        self.in_flight = false;
        self.healthy = true;
        self.failures = 0;
        let total = result.total_secs.max(0.0);
        let Some(end) = result.last_speech_end else {
            // Речи в окне нет вовсе — хвост тишины равен всему окну.
            self.tail_secs = Some(total);
            self.tail_at = Some(self.probe_audio_end_at);
            return true;
        };
        self.saw_speech = true;
        self.tail_secs = Some((total - end.clamp(0.0, total)).max(0.0));
        self.tail_at = Some(self.probe_audio_end_at);
        true
    }

    /// Ошибка инференса. VAD отключаем только после нескольких подряд:
    /// одиночный сбой не должен лишать движок VAD до конца сессии.
    fn note_failure(&mut self) {
        self.in_flight = false;
        self.failures = self.failures.saturating_add(1);
        if self.failures >= 3 {
            self.unavailable = true;
            eprintln!("🎙️ VAD speakrs недоступен — завершаю фразы по громкости");
        }
    }

    /// VAD уверенно показывает, что после последней речи прошло достаточно тишины.
    fn confirmed_silence(&self, threshold_secs: f32) -> bool {
        if self.unavailable {
            return false;
        }
        let (Some(tail), Some(at)) = (self.tail_secs, self.tail_at) else {
            return false;
        };
        // Свежесть важна: старый «хвост тишины» не должен закрыть фразу,
        // когда пользователь уже снова говорит.
        at.elapsed() <= Duration::from_millis(VAD_RESULT_MAX_AGE_MS)
            && tail >= threshold_secs.max(VAD_MIN_TAIL_SECS)
    }
}

/// Ждёт вердикт фильтра говорящего с ожидаемым `id`, пропуская ответы VAD
/// и устаревшие проверки (фоновые проверки перебивания). Таймаут — тоже
/// [`Verdict::Unverified`]: при включённом фильтре фраза не проходит без
/// проверки (fail-closed).
fn wait_for_verdict(rx: &Receiver<SpeakerReply>, expected_id: u64, timeout: Duration) -> Verdict {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Verdict::Unverified("фильтр не ответил вовремя");
        }
        match rx.recv_timeout(left) {
            Ok(SpeakerReply::Verified { id, verdict }) => {
                if id == expected_id {
                    return verdict;
                }
                // Ответ чужого запроса (например, фоновой проверки перебивания
                // прошлой реплики): не наш вердикт, ждём свой.
                continue;
            }
            Ok(SpeakerReply::Speech(_)) => continue,
            Err(_) => return Verdict::Unverified("фильтр не ответил вовремя"),
        }
    }
}

pub fn run_engine(stt: Stt, h: EngineHandles) -> Result<()> {
    // Прогреваем соединение с поисковиком заранее: тогда первый запрос
    // пользователя не платит за DNS и TLS-рукопожатие.
    crate::web::warm_up();

    // Каналы и планировщик (по логике OpenJarvis): Telegram-бот и задачи
    // по расписанию живут в фоновых потоках движка и молчат, если не
    // настроены (токен/[[mcp_servers]] в settings.toml).
    crate::telegram::spawn(Arc::clone(&h.llm));
    crate::schedule::spawn(
        Arc::clone(&h.llm),
        h.audio_tx.clone(),
        h.events.clone(),
    );

    let (mic_tx, mic_rx) = sync_channel::<Vec<f32>>(128);

    println!("🎤 Запрашиваю доступ к микрофону...");
    // Цикл ожидания доступа: macOS при запрете не возвращает ошибку, а молча
    // подаёт тишину, поэтому без проверки здесь ассистент «работал» без ушей.
    // Пока доступа нет — показываем системный запрос и событие в UI и ждём;
    // как только пользователь разрешил, микрофон открывается без перезапуска.
    let (stream, native_sr) = {
        #[cfg(target_os = "macos")]
        let mut told_user = false;
        let opened;
        loop {
            #[cfg(target_os = "macos")]
            {
                if !crate::audio::mic_access_step() {
                    if !told_user {
                        told_user = true;
                        let _ = h.events.send(JarvisEvent::MicPermission { granted: false });
                        println!("🎤 Жду доступа к микрофону (открыт системный запрос)…");
                    }
                    std::thread::sleep(Duration::from_millis(500));
                    continue;
                }
                if told_user {
                    let _ = h.events.send(JarvisEvent::MicPermission { granted: true });
                }
            }
            match spawn_stream(mic_tx.clone()) {
                Ok(pair) => {
                    opened = pair;
                    break;
                }
                Err(e) => return Err(e.context("Не удалось создать входной аудиопоток.")),
            }
        }
        opened
    };
    stream.play().context("Не удалось запустить микрофон.")?;

    println!("✅ Микрофон запущен и готов к работе");
    println!("💡 Скажи 'Джарвис' для активации...");

    let stt = Arc::new(Mutex::new(stt));
    let (stt_tx, stt_rx) = spawn_stt_worker(
        Arc::clone(&stt),
        Arc::clone(&h.settings),
        Arc::clone(&h.commands),
    );

    let (wake_threshold, ww_model) = {
        let s = h.settings.read().unwrap_or_else(|e| e.into_inner());
        (s.wake_threshold, s.wakeword_model.clone())
    };
    // Дефолтные модели вакворда качаются из релиза (.jref — дикторонезависимый,
    // .rpw — обученный на голосе владельца); нет ни одной — движок не падает,
    // а ждёт: модель появится после обучения в приложении.
    let _ = crate::model_fetch::ensure_wakeword(&ww_model);
    let _ = crate::model_fetch::ensure_wakeword(wake::EMBEDDING_REFERENCE);
    let mut det = loop {
        match wake::create_engine(&stt, &ww_model, wake_threshold) {
            Ok(det) => break det,
            Err(e) => {
                let message = format!(
                    "Модель вакворда не готова: {e:#}. Открой «Wake Word» в приложении и запиши 3–5 сэмплов фразы — движок подхватит её сам."
                );
                eprintln!("⏳ {message}");
                let _ = h.events.send(JarvisEvent::Error { message });
                // Ждём: файл появится из обучения в приложении или докачки.
                std::thread::sleep(Duration::from_secs(3));
                let _ = crate::model_fetch::ensure_wakeword(wake::EMBEDDING_REFERENCE);
            }
        }
    };

    let history_capacity = wake::SAMPLE_RATE * HISTORY_BUFFER_SECS;
    let mut audio_history = VecDeque::<f32>::with_capacity(history_capacity);

    let mut state = AppState::WaitingForWakeWord;
    let mut command_buffer = Vec::new();
    let mut followup_deadline: Option<Instant> = None;

    let mut has_speech_started = false;
    let mut silence_samples = 0;
    // Выдержка тихой речи ниже жёсткого порога: сколько сэмплов подряд
    // выше мягкого порога — после неё фраза начинается.
    let mut quiet_streak = 0usize;
    // Сколько сэмплов было именно речью (а не тишиной) в текущей фразе.
    let mut voiced_samples = 0usize;
    // Фильтр говорящего включён для текущей фразы: сегмент ушёл на
    // верификацию голоса, и в Processing нужно забрать вердикт.
    let mut speaker_check_sent = false;
    // Метка текущей проверки фразы: ответы фоновых проверок перебивания
    // с другими id не принимаются за вердикт фразы.
    let mut speaker_check_id: u64 = 0;
    let mut verify_id: u64 = 1;
    // Подсказка «образцов голоса нет» звучит один раз за сессию, а не на
    // каждой фразе.
    let mut speaker_hint_spoken = false;
    // Оценка фонового шума комнаты: по ней поднимаем порог речи, чтобы
    // вентилятор/улица не запускали распознавание.
    let mut noise_floor = 0.0f32;
    let mut was_speaking = false;
    // Перебивание: микрофон слушает и во время ответа Джарвиса.
    let barge_enabled = {
        let s = h.settings.read().unwrap_or_else(|e| e.into_inner());
        s.barge_in
    };
    let mut barge = BargeTracker::new(barge_enabled);
    // Перебивание сработало: после конца речи Джарвиса буфер фразы
    // пользователя (уже посеянный из кольца) не выбрасывать.
    let mut barge_triggered = false;
    // Сколько сэмплов в начале фразы — пре-ролл перебивания: для проверки
    // голоса он вырезается (там эхо ответа Джарвиса), для STT остаётся.
    let mut barge_preroll_samples = 0usize;
    // Проверка владельца во время ответа: один запрос в полёте, свежий
    // вердикт «владелец» открывает окно триггера перебивания.
    let mut barge_probe_id: Option<u64> = None;
    let mut barge_last_probe: Option<Instant> = None;
    let mut barge_owner_until: Option<Instant> = None;
    // Профиль отсутствует: перебивание по чистой энергии (как раньше).
    let mut barge_free_gate = false;
    // Последний исход диалога был Followup: после конца ответа окно
    // уточнения начинается заново (раньше оно истекало во время речи).
    let mut last_outcome_followup = false;
    let mut rng = rand::thread_rng();
    // VAD speakrs: пока пользователь говорит, фраза не уходит в STT.
    let mut vad = VadTracker::new();

    let mut last_mic_level_emit = Instant::now() - Duration::from_millis(MIC_LEVEL_INTERVAL_MS);
    let mut mic_held_level = 0.0f32;

    let _ = h.events.send(JarvisEvent::StateChanged {
        state: "wake_word".into(),
    });

    loop {
        if h.wakeword_reload.load(Ordering::SeqCst) && state == AppState::WaitingForWakeWord {
            h.wakeword_reload.store(false, Ordering::SeqCst);
            let snap = h.settings.read().unwrap_or_else(|e| e.into_inner()).clone();
            match wake::create_engine(&stt, &snap.wakeword_model, snap.wake_threshold) {
                Ok(d) => {
                    det = d;
                    println!("🔄 Модель wake word обновлена");
                }
                Err(e) => eprintln!("❌ Не удалось подменить wake word: {e}"),
            }
        }

        let settings_snap = h.settings.read().unwrap_or_else(|e| e.into_inner()).clone();
        let min_command_samples =
            (wake::SAMPLE_RATE as f32 * settings_snap.min_command_duration_secs) as usize;
        let max_command_samples =
            (wake::SAMPLE_RATE as f32 * settings_snap.max_command_duration_secs) as usize;
        // Пока speakrs слышит речь, мягкий лимит не режет фразу; жёсткий
        // предел защищает от вечного слушания (шум, музыка, зависший VAD).
        let hard_max_command_samples = (wake::SAMPLE_RATE as f32
            * settings_snap.max_command_duration_secs.max(HARD_MAX_COMMAND_SECS))
            as usize;
        let silence_trigger_samples =
            (wake::SAMPLE_RATE as f32 * settings_snap.silence_timeout_secs) as usize;
        let vad_window_samples = (wake::SAMPLE_RATE as f32 * VAD_WINDOW_SECS) as usize;
        // Речь короче 0.35 с — почти наверняка щелчок или шумовой всплеск.
        let min_voiced_samples = (wake::SAMPLE_RATE as f32 * 0.25) as usize;
        // Насколько высоко вообще можно поднять порог (чтобы не оглохнуть).
        let noise_ceiling = settings_snap.silence_rms_threshold * 4.0;

        if h.paused.load(Ordering::SeqCst) {
            let _ = mic_rx.recv_timeout(Duration::from_millis(100));
            command_buffer.clear();
            det.reset();
            audio_history.clear();
            has_speech_started = false;
            silence_samples = 0;
            voiced_samples = 0;
            vad.reset_phrase();
            mic_held_level = 0.0;
            let _ = h.events.send(JarvisEvent::AudioLevel {
                level: 0.0,
                source: "mic".into(),
            });
            continue;
        }

        if state == AppState::Processing {
            match stt_rx.try_recv() {
                        Ok(res) => {
                            state = AppState::WaitingForCommand;

                            let text = match res {
                                Ok(t) => t,
                                Err(e) => {
                                    eprintln!("⚠️ Ошибка распознавания: {e}");
                                    speaker_check_sent = false;
                                    command_buffer.clear();
                                    has_speech_started = false;
                                    silence_samples = 0;
                                    voiced_samples = 0;
                                    vad.reset_phrase();

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

                    // «Распознано» печатаем после вердикта фильтра: иначе в
                    // консоли чужая фраза выглядела принятой, хотя она отброшена.

                    // Вердикт фильтра говорящего: чужая речь не идёт ни в
                    // LLM, ни в команды — движок просто ждёт дальше. Запас до
                    // 4 с: первая фраза после записи нового образца может
                    // застать пересборку профиля (без кэша — ~2 с).
                    if speaker_check_sent {
                        let verdict = wait_for_verdict(
                            &h.speaker_rx,
                            speaker_check_id,
                            Duration::from_millis(4000),
                        );
                        speaker_check_sent = false;
                        match verdict {
                            Verdict::Owner => {}
                            Verdict::Stranger => {
                                println!("🎙️ Посторонний голос — фраза не принята");
                                let _ = h.events.send(JarvisEvent::Log {
                                    message: "Посторонний голос на фоне — фраза пропущена".into(),
                                });
                                reset_phrase_state(&mut command_buffer, &mut has_speech_started, &mut silence_samples, &mut voiced_samples, &mut vad);
                                // Followup закрываем: чужая фраза не продолжение
                                // диалога с владельцем.
                                state = AppState::WaitingForWakeWord;
                                followup_deadline = None;
                                det.reset();
                                audio_history.clear();
                                continue;
                            }
                            Verdict::Unverified(reason) => {
                                // Fail-closed: профиль есть, проверить не вышло —
                                // фраза не проходит, но владелец получает честную
                                // просьбу повторить вместо молчаливого игнора.
                                eprintln!("🎙️ Голос не верифицирован ({reason}) — вежливый отказ");
                                let _ = h.events.send(JarvisEvent::Log {
                                    message: format!("Голос не верифицирован: {reason} — фраза не принята"),
                                });
                                let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(
                                    "Не смог проверить голос. Повтори ещё раз, пожалуйста.".into(),
                                ));
                                reset_phrase_state(&mut command_buffer, &mut has_speech_started, &mut silence_samples, &mut voiced_samples, &mut vad);
                                state = AppState::WaitingForWakeWord;
                                followup_deadline = None;
                                det.reset();
                                audio_history.clear();
                                continue;
                            }
                            Verdict::NoProfile => {
                                // Фильтр включён, но образцов нет: пропускаем,
                                // но один раз честно говорим, что он не работает.
                                if !speaker_hint_spoken {
                                    speaker_hint_spoken = true;
                                    println!("🎙️ Фильтр голоса не активен: образцов владельца нет");
                                    let _ = h.events.send(JarvisEvent::Log {
                                        message: "Фильтр «только мой голос» не работает: нет записанных образцов".into(),
                                    });
                                    let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(
                                        "Образцы моего голоса не записаны, фильтр «только мой голос» не работает. Запиши их в разделе «Обучение».".into(),
                                    ));
                                }
                            }
                        }
                    }

                    println!("📝 Распознано: {:?}", text);

                    let _ = h
                        .events
                        .send(JarvisEvent::Transcript { text: text.clone() });

                    let noise_input = !is_valid(&text);
                    let outcome = if !noise_input {
                        handle_transcript(&text, &h, &mut rng)
                    } else {
                        eprintln!("🔇 Шум или пустой ввод");
                        if followup_expired_or_none(followup_deadline) {
                            let global_cfg = get_global_config(&h.commands);
                            let _ = h
                                .audio_tx
                                .send(AudioMessage::PlayWav(global_cfg.reply_error));
                            Outcome::Wake
                        } else {
                            Outcome::Followup
                        }
                    };

                    command_buffer.clear();
                    has_speech_started = false;
                    silence_samples = 0;
                    voiced_samples = 0;
                    vad.reset_phrase();

                    match outcome {
                        Outcome::Wake => {
                            state = AppState::WaitingForWakeWord;
                            followup_deadline = None;
                            last_outcome_followup = false;
                            audio_history.clear();
                            det.reset();
                            h.llm_history
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .clear();
                            let _ = h.events.send(JarvisEvent::StateChanged {
                                state: "wake_word".into(),
                            });
                        }
                        Outcome::Followup => {
                            state = AppState::WaitingForCommand;
                            last_outcome_followup = true;
                            // Пустое распознавание (шум) окно не продлевает:
                            // иначе фоновый шум держит ассистента в режиме
                            // слушания бесконечно, съедая цикл за циклом.
                            if !noise_input || followup_deadline.is_none() {
                                followup_deadline = Some(
                                    Instant::now()
                                        + Duration::from_secs(settings_snap.followup_window_secs),
                                );
                            }
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

        let chunk = match mic_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let chunk_16k = if native_sr != wake::SAMPLE_RATE {
            resample_audio(&chunk, native_sr, wake::SAMPLE_RATE)
        } else {
            chunk
        };

        // Уровень микрофона летит в UI всегда, даже пока Джарвис говорит или
        // LLM обрабатывает запрос: звук принимается, просто не обрабатывается
        // как команда. Без этого спектрограмма замирала на всё время ответа.
        let now = Instant::now();
        if now.duration_since(last_mic_level_emit) >= Duration::from_millis(MIC_LEVEL_INTERVAL_MS) {
            let rms = calculate_rms(&chunk_16k);
            let peak_abs = chunk_16k.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let raw = (rms.max(peak_abs * 0.65) * 6.0).min(1.0);

            mic_held_level = if raw > mic_held_level {
                raw
            } else {
                mic_held_level * 0.75
            };

            let _ = h.events.send(JarvisEvent::AudioLevel {
                level: mic_held_level,
                source: "mic".into(),
            });
            last_mic_level_emit = now;
        }

        // Историю пополняем всегда, даже когда Джарвис говорит: первые ~0.3 с
        // речи пользователя приходятся на момент, когда он ещё не умолк, и
        // без истории фраза начиналась бы с середины слова.
        let history_capacity = wake::SAMPLE_RATE * HISTORY_BUFFER_SECS;
        let preroll_samples = (wake::SAMPLE_RATE as f32 * PREROLL_SECS) as usize;
        for &s in &chunk_16k {
            if audio_history.len() >= history_capacity {
                audio_history.pop_front();
            }
            audio_history.push_back(s);
        }

        // Пока Джарвис говорит, микрофон как команда не слушаем: иначе хвост
        // его же речи будет распознан как ответ пользователя. Но за уровнем
        // следим: пользователь может перебить (barge_in).
        let speaking_now = h.is_speaking.load(Ordering::SeqCst);
        if speaking_now {
            was_speaking = true;
            audio_history.clear();
            if barge_triggered {
                // Перебивание уже сработало: чанк копится в кольцо трекера,
                // а посеянная фраза пользователя не трогается — Джарвис ещё
                // доигрывает последние миллисекунды до стопа.
                let floor = speech_floor(settings_snap.silence_rms_threshold, noise_floor);
                barge.on_playback_chunk(&chunk_16k, now, floor);
                continue;
            }
            command_buffer.clear();
            has_speech_started = false;
            vad.reset_phrase();

            // --- Перебивание по голосу владельца (speakrs) ---
            // Энергетический трекер даёт «кто-то говорит громче эха»;
            // фоновая проверка speakrs подтверждает, что это ВЛАДЕЛЕЦ, а не
            // телевизор или видео: эхо Silero-голоса Джарвиса на профиль
            // владельца не похоже. Нет профиля или фильтр выключен —
            // перебивание работает по энергии, как раньше.
            if settings_snap.speaker_filter
                && !barge_free_gate
                && barge.warmed_up()
                && barge_probe_id.is_none()
                && barge_last_probe
                    .map_or(true, |t| now.duration_since(t) >= Duration::from_millis(BARGE_VERIFY_INTERVAL_MS))
                && barge.ring_secs() >= BARGE_VERIFY_WINDOW_SECS
            {
                barge_last_probe = Some(now);
                barge_probe_id = Some(verify_id);
                verify_id = verify_id.wrapping_add(1);
                let _ = h.speaker_tx.try_send(SpeakerQuery::Verify {
                    id: barge_probe_id.unwrap(),
                    samples: barge.recent_window(BARGE_VERIFY_WINDOW_SECS),
                    threshold: (settings_snap.speaker_threshold - BARGE_VERIFY_DELTA)
                        .max(0.4),
                });
            }

            // Ответы фоновых проверок: свежий «владелец» открывает окно
            // триггера; «нет профиля» — постоянный пропуск проверки.
            while let Ok(reply) = h.speaker_rx.try_recv() {
                match reply {
                    SpeakerReply::Verified { id, verdict } => {
                        if Some(id) == barge_probe_id {
                            barge_probe_id = None;
                            match verdict {
                                Verdict::Owner => {
                                    barge_owner_until = Some(
                                        now + Duration::from_millis(BARGE_OWNER_TTL_MS),
                                    );
                                }
                                Verdict::NoProfile => barge_free_gate = true,
                                _ => {}
                            }
                        }
                        // Чужой id — устаревший ответ, пропускаем.
                    }
                    SpeakerReply::Speech(_) => {}
                }
            }

            let floor = speech_floor(settings_snap.silence_rms_threshold, noise_floor);
            let owner_confirmed = barge_free_gate
                || barge_owner_until.is_some_and(|until| now <= until);
            if owner_confirmed && barge.on_playback_chunk(&chunk_16k, now, floor) {
                barge_triggered = true;
                println!("🎤 Перебивание: голос владельца — глушу речь");
                let _ = h.events.send(JarvisEvent::Log {
                    message: "Перебивание: голос владельца во время ответа".into(),
                });
                // Мгновенная тишина + отмена остатка ответа — тот же путь,
                // что у голосовой команды «замолчи».
                h.stop_audio_flag.store(true, Ordering::SeqCst);
                h.llm_cancel.store(true, Ordering::SeqCst);
                let _ = h.audio_tx.send(AudioMessage::Stop);
                // Сказанное до триггера — начало фразы пользователя.
                command_buffer = barge.take_preroll();
                barge_preroll_samples = command_buffer.len();
                has_speech_started = true;
                voiced_samples = command_buffer.len();
                silence_samples = 0;
                vad.reset_phrase();
                state = AppState::WaitingForCommand;
                followup_deadline = Some(
                    Instant::now() + Duration::from_secs(settings_snap.followup_window_secs),
                );
            }
            continue;
        }

        if was_speaking {
            // Озвучка только что закончилась. Обычный случай: выбрасываем хвост
            // из буфера микрофона и коротко молчим, чтобы пользователь не начал
            // говорить в пустоту. После перебивания: фраза пользователя уже
            // посеяна — сохраняем её и продолжаем захват без стража.
            was_speaking = false;
            let keep_phrase = barge_triggered;
            barge_triggered = false;
            barge.reset_playback();
            // Фоновые проверки перебивания завершены вместе с ответом.
            barge_probe_id = None;
            barge_last_probe = None;
            barge_owner_until = None;
            barge_free_gate = false;
            if keep_phrase {
                while mic_rx.try_recv().is_ok() {}
                // Хвост эха уже вырезан триггером: пре-ролл фразы важнее
                // стража, спать нельзя — потеряем кусок живой речи.
                continue;
            }
            has_speech_started = false;
            silence_samples = 0;
            voiced_samples = 0;
            command_buffer.clear();
            vad.reset_phrase();
            while mic_rx.try_recv().is_ok() {}
            audio_history.clear();
            // Ответ закончился: окно уточнения начинается заново, если
            // предыдущая реплика была вопросом владельца (а не шумом).
            if last_outcome_followup && settings_snap.followup_window_secs > 0 {
                followup_deadline = Some(
                    Instant::now() + Duration::from_secs(settings_snap.followup_window_secs),
                );
            }
            std::thread::sleep(Duration::from_millis(POST_SPEAK_GUARD_MS));
            audio_history.clear();
            continue;
        }

        if let Some(deadline) = followup_deadline
            && Instant::now() > deadline
        {
            if state == AppState::WaitingForCommand && has_speech_started {
                // Пользователь уже говорит: окно прямого слушания больше не
                // актуально. Не обрываем фразу на полуслове — продлеваем
                // до конца речи, итог выставит обработка транскрипта.
                followup_deadline = Some(
                    Instant::now() + Duration::from_secs(settings_snap.followup_window_secs),
                );
            } else if state == AppState::WaitingForCommand {
                println!(
                    "🔒 Окно прямого слушания ({} сек) истекло. Переход в режим вакворда.",
                    settings_snap.followup_window_secs
                );

                state = AppState::WaitingForWakeWord;
                command_buffer.clear();
                det.reset();
                audio_history.clear();
                h.llm_history
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();

                has_speech_started = false;
                silence_samples = 0;
                voiced_samples = 0;
                vad.reset_phrase();
                followup_deadline = None;
                let _ = h.events.send(JarvisEvent::StateChanged {
                    state: "wake_word".into(),
                });
            } else {
                followup_deadline = None;
            }
        }

        match state {
            AppState::WaitingForWakeWord => {
                if let Some(d) = det.process_16k(&chunk_16k) {
                    {
                        eprintln!("🎯 Wake word: {} ({:.3})", d.name, d.score);
                        let _ = h.events.send(JarvisEvent::WakeWordDetected {
                            name: d.name.clone(),
                            score: d.score,
                        });

                        // Обращение НЕ верифицируем по голосу: проверка короткого
                        // «Джарвис» по профилю из диктанта отшивала владельца
                        // (репорт пользователя 2026-11-06). Контракт такой: отклик
                        // получает всякий, кого признал детектор wake word
                        // (обучен на образцах владельца), а строгая проверка —
                        // на фразе ПОСЛЕ обращения; чужая команда там отбрасывается.

                        let global_cfg = get_global_config(&h.commands);
                        let ack = paths::resolve_data_path(&global_cfg.reply_ack);
                        let _ = h
                            .audio_tx
                            .send(AudioMessage::PlayWav(ack.to_string_lossy().to_string()));

                        // Небольшая пауза, чтобы хвост подтверждающего звука
                        // не попал в буфер команды. Было 150 мс — это чистая
                        // задержка перед началом слушания.
                        std::thread::sleep(Duration::from_millis(60));

                        det.reset();
                        audio_history.clear();

                        state = AppState::WaitingForCommand;
                        followup_deadline = Some(
                            Instant::now()
                                + Duration::from_secs(settings_snap.followup_window_secs),
                        );

                        command_buffer.clear();
                        has_speech_started = false;
                        silence_samples = 0;
                        voiced_samples = 0;
                        vad.reset_phrase();
                        let _ = h.events.send(JarvisEvent::StateChanged {
                            state: "listening".into(),
                        });
                    }
                }
            }

            AppState::WaitingForCommand => {
                let chunk_rms = calculate_rms(&chunk_16k);

                // Порог речи: не ниже настройки и заметно выше фонового шума.
                let speech_floor = speech_floor(settings_snap.silence_rms_threshold, noise_floor);
                // Мягкий порог старта: тихая речь (тише жёсткого порога, но
                // заметно выше фона) начинает фразу после короткой выдержки —
                // мусор до конца отсекает VAD speakrs (он нейронный и тихую
                // речь видит) и лимит минимальной фразы.
                let start_floor = quiet_start_floor(
                    speech_floor,
                    noise_floor,
                    settings_snap.silence_rms_threshold,
                );

                if chunk_rms >= speech_floor {
                    quiet_streak = 0;
                    voiced_samples += chunk_16k.len();
                    if !has_speech_started {
                        has_speech_started = true;
                        vad.reset_phrase();
                        let hist_len = audio_history.len();
                        let take_samples = hist_len.min(preroll_samples);
                        if take_samples > 0 {
                            let start_idx = hist_len - take_samples;
                            command_buffer.extend(audio_history.iter().skip(start_idx).copied());
                        }
                    }
                    silence_samples = 0;
                    // Громкий звук после паузы отменяет старый вердикт VAD:
                    // пользователь мог продолжить фразу.
                    vad.tail_secs = None;
                    vad.tail_at = None;
                } else if !has_speech_started && chunk_rms >= start_floor {
                    // Тихая речь: копим выдержку, затем начинаем фразу.
                    quiet_streak += chunk_16k.len();
                    if quiet_streak >= (QUIET_START_SECS * wake::SAMPLE_RATE as f32) as usize {
                        quiet_streak = 0;
                        has_speech_started = true;
                        vad.reset_phrase();
                        let hist_len = audio_history.len();
                        let take_samples = hist_len.min(preroll_samples);
                        if take_samples > 0 {
                            let start_idx = hist_len - take_samples;
                            command_buffer.extend(audio_history.iter().skip(start_idx).copied());
                        }
                        voiced_samples = command_buffer.len();
                        silence_samples = 0;
                    }
                } else {
                    quiet_streak = 0;
                    // Тишина: обновляем оценку фона (только когда не говорим),
                    // чтобы порог подстраивался под комнату.
                    if !has_speech_started {
                        noise_floor = update_noise_floor(noise_floor, chunk_rms, noise_ceiling);
                    }
                    if has_speech_started {
                        // Считаем реальную тишину. Это только запасной путь:
                        // пока speakrs видит речь, фразу не закрываем.
                        silence_samples += chunk_16k.len();
                    }
                }

                if has_speech_started {
                    command_buffer.extend_from_slice(&chunk_16k);

                    // Ответы VAD не блокируют аудио-цикл: отправляем окно,
                    // а результат забираем, когда воркер его посчитает.
                    while let Ok(reply) = h.speaker_rx.try_recv() {
                        match reply {
                            SpeakerReply::Speech(Some(result)) => {
                                vad.accept(result);
                            }
                            SpeakerReply::Speech(None) => vad.note_failure(),
                            // Вердикт фильтра от прошлой фразы: текущая его не ждёт.
                            SpeakerReply::Verified { .. } => {}
                        }
                    }

                    // Очередное окно на проверку. Один запрос в полёте:
                    // воркер не копит очередь, а свежесть важнее частоты.
                    if !vad.unavailable
                        && !vad.in_flight
                        && vad.last_probe_at.elapsed()
                            >= Duration::from_millis(VAD_PROBE_INTERVAL_MS)
                    {
                        let take = command_buffer.len().min(vad_window_samples);
                        if take > 0 {
                            let start = command_buffer.len() - take;
                            vad.id = vad.id.wrapping_add(1);
                            let query = SpeakerQuery::SpeechProbe {
                                id: vad.id,
                                samples: command_buffer[start..].to_vec(),
                            };
                            if h.speaker_tx.try_send(query).is_ok() {
                                vad.in_flight = true;
                                vad.probe_started_at = now;
                                vad.probe_audio_end_at = now;
                                vad.last_probe_at = now;
                            }
                        }
                    }

                    // Зависший инференс не должен держать фразу вечно.
                    if vad.in_flight
                        && now.duration_since(vad.probe_started_at)
                            > Duration::from_millis(VAD_PROBE_TIMEOUT_MS)
                    {
                        eprintln!(
                            "🎙️ VAD speakrs не ответил за {VAD_PROBE_TIMEOUT_MS} мс"
                        );
                        vad.note_failure();
                    }

                    let reached_max = command_buffer.len() >= max_command_samples;
                    let reached_hard_max = command_buffer.len() >= hard_max_command_samples;
                    // Первый ответ VAD ещё в пути — не решаем по старой
                    // RMS-логике, иначе тихая речь оборвётся до проверки.
                    let vad_pending = vad.in_flight && !vad.healthy && !vad.unavailable;
                    // Пока VAD здоров, пауза посреди предложения не закрывает
                    // фразу: нужен подтверждённый хвост тишины speakrs.
                    let endpoint = if vad.healthy && !vad.unavailable {
                        vad.confirmed_silence(settings_snap.silence_timeout_secs)
                            || reached_hard_max
                    } else if vad_pending {
                        false
                    } else {
                        silence_samples >= silence_trigger_samples || reached_max
                    };

                    if endpoint {
                        // Мало и буфера, и собственно речи — это шум, а не фраза.
                        // VAD, увидевший речь, отменяет проверку по громкости:
                        // тихую речь RMS может недосчитать.
                        let too_quiet =
                            voiced_samples < min_voiced_samples && !vad.saw_speech;
                        if command_buffer.len() < min_command_samples || too_quiet {
                            command_buffer.clear();
                            has_speech_started = false;
                            silence_samples = 0;
                            voiced_samples = 0;
                            vad.reset_phrase();
                            continue;
                        }

                        let pad = (wake::SAMPLE_RATE as f32 * 0.2) as usize;
                        // Тихая фраза усиливается до целевого уровня: ASR и
                        // эмбеддинги обучены на речи нормальной громкости.
                        let voiced_phrase = normalize_gain(&command_buffer);
                        let mut padded = Vec::with_capacity(voiced_phrase.len() + pad * 2);

                        padded.extend(std::iter::repeat_n(0.0f32, pad));
                        padded.extend_from_slice(&voiced_phrase);
                        padded.extend(std::iter::repeat_n(0.0f32, pad));

                        // Фильтр говорящего: сегмент уходит на проверку голоса
                        // параллельно с STT — вердикт забирается в Processing.
                        // На проверку идёт фраза БЕЗ пре-ролла перебивания
                        // (там эхо ответа Джарвиса портит сходство).
                        speaker_check_sent = settings_snap.speaker_filter;
                        if speaker_check_sent {
                            let clean = verify_slice(
                                &padded,
                                pad,
                                barge_preroll_samples,
                                (crate::speaker::MIN_SEGMENT_SECS * wake::SAMPLE_RATE as f32)
                                    as usize,
                            )
                            .to_vec();
                            speaker_check_id = verify_id;
                            verify_id = verify_id.wrapping_add(1);
                            let query = SpeakerQuery::Verify {
                                id: speaker_check_id,
                                samples: clean,
                                threshold: settings_snap.speaker_threshold,
                            };
                            if h.speaker_tx.try_send(query).is_err() {
                                // Очередь гейта переполнена — фильтр пропускает фразу.
                                speaker_check_sent = false;
                            }
                        }

                        if stt_tx.send(SttJob { samples: padded }).is_err() {
                            return Err(anyhow::anyhow!("STT-воркер недоступен"));
                        }

                        state = AppState::Processing;
                        barge_preroll_samples = 0;
                        // Фраза ушла в STT: ответы VAD на неё больше не нужны.
                        vad.reset_phrase();
                        // Сообщаем UI, что фраза закончилась и пошла обработка.
                        // Без этого события интерфейс узнавал о работе только
                        // от llm_status, и пауза на распознавание выглядела
                        // зависшей визуализацией.
                        let _ = h.events.send(JarvisEvent::StateChanged {
                            state: "processing".into(),
                        });
                    }
                }
            }

            AppState::Processing => {}
        }
    }
}

/// Во сколько раз речь должна быть громче фона, чтобы считаться речью.
const NOISE_MARGIN: f32 = 1.8;

// --- Перебивание (barge-in) ---

/// Сколько подряд идущих «громких» сэмплов микрофона подтверждают речь
/// пользователя (сек): короче — щелчок, длиннее — заметная задержка реакции.
const BARGE_CONFIRM_SECS: f32 = 0.22;
/// Речь пользователя должна быть во столько раз громче эха голоса Джарвиса
/// (при играющих колонках), чтобы считаться перебиванием.
const BARGE_BASELINE_MARGIN: f32 = 1.8;
/// Разогрев: сколько секунд непрерывной речи Джарвиса слушать, прежде чем
/// разрешить перебивание. За это время базлайн эха выходит на рабочий
/// уровень И короткие системные звуки (отклик на «Джарвис» ~0.6 с, таймеры)
/// не вооружают детектор — иначе ложное перебивание на собственную речь
/// владельца сразу после отклика вклеивало эхо в начало его команды и
/// проверка голоса отвергала чистую фразу.
const BARGE_WARMUP_SECS: f32 = 1.2;
/// Хвост кольца, который становится началом фразы пользователя, сек.
const BARGE_PREROLL_SECS: f32 = 0.35;
/// Глубина кольца микрофона во время речи Джарвиса, сек.
const BARGE_RING_SECS: f32 = 3.0;
/// Пауза между перебиваниями, мс.
const BARGE_COOLDOWN_MS: u64 = 2000;

// --- Подтверждение владельца для перебивания (speakrs) ---

/// Как часто во время ответа проверять окно микрофона на голос владельца, мс.
const BARGE_VERIFY_INTERVAL_MS: u64 = 350;
/// Окно микрофона для фоновой проверки, сек: короче — шумный эмбеддинг.
const BARGE_VERIFY_WINDOW_SECS: f32 = 2.0;
/// Порог для подтверждения владельца: мягче основного (окно содержит эхо
/// ответа вперемешку с голосом), но не ниже разумного пола.
const BARGE_VERIFY_DELTA: f32 = 0.1;
/// Сколько живёт вердикт «владелец» для триггера, мс.
const BARGE_OWNER_TTL_MS: u64 = 1000;

/// Детектор перебивания: во время речи Джарвиса следит за микрофоном поверх
/// эха его собственного голоса и решает, заговорил ли пользователь.
///
/// Эхо (речь Джарвиса из колонок) и голос пользователя неразличимы по
/// природе — только по уровню: пользователь у микрофона громче. Базлайн эха
/// — EMA по чанкам во время ответа; триггер — устойчивое превышение
/// `max(базлайн × поле, порог речи)` в течение [`BARGE_CONFIRM_SECS`].
struct BargeTracker {
    enabled: bool,
    /// Кольцо последних секунд микрофона (источник пре-ролла фразы).
    ring: VecDeque<f32>,
    ring_capacity: usize,
    /// EMA уровня эха во время ответа.
    baseline: f32,
    /// Сэмплы подряд выше порога.
    speech_samples: usize,
    /// Сэмплы с начала текущей речи Джарвиса (для разогрева).
    playback_samples: usize,
    cooldown_until: Option<Instant>,
    /// Перебивание уже сработало в этой реплике.
    triggered: bool,
}

impl BargeTracker {
    fn new(enabled: bool) -> Self {
        let ring_capacity = (BARGE_RING_SECS * wake::SAMPLE_RATE as f32) as usize;
        Self {
            enabled,
            ring: VecDeque::with_capacity(ring_capacity),
            ring_capacity,
            baseline: 0.0,
            speech_samples: 0,
            playback_samples: 0,
            cooldown_until: None,
            triggered: false,
        }
    }

    /// Чанк микрофона во время речи Джарвиса. `true` — перебивание: движок
    /// глушит звук и берёт [`Self::take_preroll`] как начало фразы.
    fn on_playback_chunk(&mut self, chunk: &[f32], now: Instant, speech_floor: f32) -> bool {
        for &s in chunk {
            if self.ring.len() >= self.ring_capacity {
                self.ring.pop_front();
            }
            self.ring.push_back(s);
        }
        if !self.enabled || self.triggered {
            return false;
        }
        if let Some(until) = self.cooldown_until {
            if now < until {
                return false;
            }
            self.cooldown_until = None;
        }

        let rms = calculate_rms(chunk);
        // Разогрев: первые мгновения ответа калибруют базлайн, триггер молчит.
        self.playback_samples += chunk.len();
        if (self.playback_samples as f32 / wake::SAMPLE_RATE as f32) < BARGE_WARMUP_SECS {
            self.baseline = self.baseline * 0.5 + rms * 0.5;
            return false;
        }
        let echo_line = self.baseline * BARGE_BASELINE_MARGIN;
        let trigger_line = echo_line.max(speech_floor);
        // Базлайн поднимает только фоновый уровень: всплески (возможно,
        // пользователь) в калибровку не попадают.
        if rms <= trigger_line {
            self.baseline = self.baseline * 0.9 + rms * 0.1;
            self.speech_samples = self.speech_samples.saturating_sub(chunk.len() / 2);
            return false;
        }
        self.speech_samples += chunk.len();
        if self.speech_samples as f32 / wake::SAMPLE_RATE as f32 >= BARGE_CONFIRM_SECS {
            self.triggered = true;
            self.cooldown_until = Some(now + Duration::from_millis(BARGE_COOLDOWN_MS));
            return true;
        }
        false
    }

    /// Начало фразы пользователя: хвост кольца перед триггером.
    fn take_preroll(&self) -> Vec<f32> {
        let take = ((BARGE_PREROLL_SECS * wake::SAMPLE_RATE as f32) as usize).min(self.ring.len());
        self.ring
            .iter()
            .skip(self.ring.len() - take)
            .copied()
            .collect()
    }

    /// Последние `secs` секунд кольца — окно для фоновой проверки владельца.
    fn recent_window(&self, secs: f32) -> Vec<f32> {
        let take = ((secs * wake::SAMPLE_RATE as f32) as usize).min(self.ring.len());
        self.ring
            .iter()
            .skip(self.ring.len() - take)
            .copied()
            .collect()
    }

    /// Глубина кольца в секундах.
    fn ring_secs(&self) -> f32 {
        self.ring.len() as f32 / wake::SAMPLE_RATE as f32
    }

    /// Разогрев пройден: перебивание разрешено решать.
    fn warmed_up(&self) -> bool {
        (self.playback_samples as f32 / wake::SAMPLE_RATE as f32) >= BARGE_WARMUP_SECS
    }

    /// Речь Джарвиса закончилась без перебивания — сброс калибровки.
    fn reset_playback(&mut self) {
        self.ring.clear();
        self.baseline = 0.0;
        self.speech_samples = 0;
        self.playback_samples = 0;
        self.triggered = false;
    }
}

/// Сброс состояния захвата фразы (после отклонённой/обработанной реплики).
fn reset_phrase_state(
    command_buffer: &mut Vec<f32>,
    has_speech_started: &mut bool,
    silence_samples: &mut usize,
    voiced_samples: &mut usize,
    vad: &mut VadTracker,
) {
    command_buffer.clear();
    *has_speech_started = false;
    *silence_samples = 0;
    *voiced_samples = 0;
    vad.reset_phrase();
}

/// Сегмент для проверки голоса: без пре-ролла перебивания. Префикс фразы,
/// захваченный триггером перебивания, содержит эхо ответа Джарвиса из
/// колонок — смешанный эмбеддинг не дотягивал до порога, и владелец
/// получал «Посторонний голос». STT при этом видит всю фразу целиком.
/// Чистого хвоста меньше минимума — возвращаем всё (решение за фильтром).
fn verify_slice<'a>(padded: &'a [f32], pad: usize, preroll: usize, min_samples: usize) -> &'a [f32] {
    let start = pad.saturating_add(preroll);
    if start < padded.len() && padded.len() - start >= min_samples {
        &padded[start..]
    } else {
        padded
    }
}

/// Порог, выше которого чанк считается речью: настройка пользователя, но не
/// ниже фонового шума комнаты с запасом.
fn speech_floor(threshold: f32, noise_floor: f32) -> f32 {
    threshold.max(noise_floor * NOISE_MARGIN)
}

// --- Тихая речь ---

/// Сколько секунд речи выше мягкого порога начинают фразу: меньше — шорохи,
/// больше — первые слова тихой фразы теряются.
const QUIET_START_SECS: f32 = 0.35;

/// Мягкий порог старта для тихой речи: заметно выше фона комнаты, но ниже
/// жёсткого порога. Мусор, пролезший через него, отсекает VAD speakrs
/// (нейронный: тихую речь видит, шорохи — нет) и лимит минимальной фразы.
fn quiet_start_floor(speech_floor: f32, noise_floor: f32, threshold: f32) -> f32 {
    let by_noise = noise_floor * 1.2;
    let by_setting = threshold * 0.5;
    speech_floor.min(by_noise).max(by_setting).max(noise_floor * 1.05)
}

// --- Усиление тихой фразы (AGC) ---

/// Целевой уровень фразы (RMS) после усиления.
const AGC_TARGET_RMS: f32 = 0.09;
/// Потолок усиления: тише этого не поднимаем, чтобы не вытягивать шум.
const AGC_MAX_GAIN: f32 = 8.0;

/// Приведение громкости фразы к целевому уровню: ASR и эмбеддинги обучены
/// на речи нормальной громкости, тихий вход давал «шум» и слабое сходство.
/// Только усиление (громкая речь не ослабляется), приращение ограничено.
fn normalize_gain(samples: &[f32]) -> Vec<f32> {
    let rms = calculate_rms(samples);
    if !rms.is_finite() || rms < 1e-4 {
        return samples.to_vec();
    }
    let gain = (AGC_TARGET_RMS / rms).min(AGC_MAX_GAIN).max(1.0);
    if gain <= 1.0 + f32::EPSILON {
        return samples.to_vec();
    }
    samples
        .iter()
        .map(|s| (s * gain).clamp(-1.0, 1.0))
        .collect()
}

/// Оценка фонового шума: экспоненциальное сглаживание, обновляется только
/// в тишине и не поднимается выше потолка (иначе ассистент «оглохнет»).
fn update_noise_floor(current: f32, rms: f32, ceiling: f32) -> f32 {
    (current * 0.95 + rms * 0.05).min(ceiling)
}

fn followup_expired_or_none(deadline: Option<Instant>) -> bool {
    match deadline {
        Some(dl) => Instant::now() >= dl,
        None => true,
    }
}

/// Фразы-триггеры детерминированного пути к агенту зрения.
const SCREEN_QUESTION_PATTERNS: &[&str] = &[
    "что на экране",
    "что сейчас на экране",
    "что ты видишь",
    "что видишь",
    "что открыто",
    "что на мониторе",
    "что написано на экране",
    "посмотри на экран",
    "проверь экран",
    "опиши экран",
    "опиши что на экране",
    "что у меня на экране",
];

/// «Что ты видишь на экране?» — детерминированный ответ по Accessibility,
/// без LLM-рулетки. Возвращает `Some(Outcome)`, если фраза распознана.
fn is_screen_question(text: &str) -> bool {
    let lowered = crate::text::clean_wakewords_and_prompt(text).to_lowercase();
    SCREEN_QUESTION_PATTERNS.iter().any(|p| lowered.contains(p))
}

fn try_screen_question(text: &str, h: &EngineHandles) -> Option<Outcome> {
    if !is_screen_question(text) {
        return None;
    }
    // Модель зрения удалена: честный отказ вместо выдуманного описания.
    // Детерминированное описание экрана через Accessibility вернётся вместе
    // с модулем автоматизации.
    println!("👀 Вопрос про экран: модель зрения удалена — честный отказ");
    let reply = "Смотреть экран я пока не умею: модель зрения убрали. Могу выполнять команды вслепую — скажи, что сделать.";
    let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(reply.into()));
    let _ = h.events.send(JarvisEvent::Reply { text: reply.into() });
    Some(Outcome::Followup)
}

/// Ответ на вопрос «задача N сделана?». Молчание и невнятный ответ оставляют
/// задачу открытой — лучше лишняя открытая галочка, чем потерянная задача.
fn handle_task_confirmation(
    expect: &crate::tasks::ExpectDone,
    answer: &str,
    h: &EngineHandles,
) -> Outcome {
    if crate::tasks::is_negative(answer) {
        println!("↩️ Задача {} остаётся открытой", expect.task_id);
        let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(
            "Хорошо, оставляю открытой.".into(),
        ));
        let _ = h.events.send(JarvisEvent::Log {
            message: format!("Задача {} осталась открытой", expect.task_id),
        });
        return Outcome::Followup;
    }

    if !crate::tasks::is_affirmative(answer) {
        println!("🤷 Не понял ответ на вопрос по задаче: {answer:?}");
        let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(
            "Не понял, задача открыта. Скажи да или нет.".into(),
        ));
        return Outcome::Followup;
    }

    match crate::tasks::set_done(expect.task_id, true) {
        Some(task) => {
            println!("✅ Задача {} закрыта: {}", task.id, task.text);
            let _ = h.events.send(JarvisEvent::Log {
                message: format!("✅ Задача {} закрыта: {}", task.id, task.text),
            });
            let next = crate::tasks::speak_next();
            let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(next));
        }
        None => {
            eprintln!("⚠️ Задача {} не найдена при подтверждении", expect.task_id);
            let _ = h
                .audio_tx
                .send(AudioMessage::SynthesizeAndPlay("Закрыл.".into()));
        }
    }

    Outcome::Followup
}

/// Возврат к незакрытой теме: «мы это забыли», «о чём говорили».
///
/// Тема возвращается в историю диалога целиком — вопрос и прошлый ответ, —
/// а модели уходит короткая инструкция «продолжи». Так она продолжает с
/// нужного места и не начинает тему заново. Если возвращаться не к чему,
/// честно говорим об этом, а не выдумываем продолжение.
fn resume_topic(spoken: &str, h: &EngineHandles) -> Outcome {
    let topic_memory = h
        .settings
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .topic_memory;

    let resume = topic_memory.then(topic::resume).flatten();

    let Some(resume) = resume else {
        let reply = if topic_memory {
            "Не помню, о чём мы говорили. Напомни тему."
        } else {
            "Память тем выключена в настройках."
        };
        println!("🧠 Возврат к теме: нечего вспоминать");
        let _ = h.events.send(JarvisEvent::Log {
            message: "Просьба вернуться к теме, но память тем пуста".into(),
        });
        let _ = h
            .audio_tx
            .send(AudioMessage::SynthesizeAndPlay(reply.into()));
        return Outcome::Followup;
    };

    println!("🧠 Возврат к теме: {}", resume.topic);
    let _ = h.events.send(JarvisEvent::Log {
        message: format!("Возврат к теме: {}", resume.topic),
    });
    let _ = h.events.send(JarvisEvent::TopicResumed {
        topic: resume.topic,
    });

    let prompt = resume.prompt;
    {
        let mut buf = h.llm_history.lock().unwrap_or_else(|e| e.into_inner());
        // Место под возвращаемый обмен режем заранее, иначе обрезка по
        // лимиту могла бы отрезать его начало — и продолжить было бы нечего.
        let excess = buf.len().saturating_sub(llm::MAX_HISTORY_ENTRIES - 2);
        buf.drain(..excess);
        // Прошлый обмен кладём в историю целиком: без него модели нечего
        // продолжать, и она выдумает тему с нуля.
        buf.push((resume.question, resume.answer));
        buf.push((
            spoken.to_string(),
            "Пользователь попросил вернуться к этой теме и продолжить её.".to_string(),
        ));
    }

    llm::ask(
        Arc::clone(&h.llm),
        llm::CHAT_SYSTEM_PROMPT,
        prompt,
        Arc::clone(&h.llm_history),
        h.audio_tx.clone(),
        h.events.clone(),
        Arc::clone(&h.llm_cancel),
    );

    Outcome::Followup
}

fn handle_transcript(text: &str, h: &EngineHandles, rng: &mut impl Rng) -> Outcome {
    h.llm_cancel.store(false, Ordering::SeqCst);
    // Номер обращения: фоновый веб-поиск проверит его перед озвучкой.
    let req_id = llm::new_request();

    // Джарвис только что спросил «задача выполнена?» — короткий ответ закрывает
    // галочку, не уходя в LLM.
    if crate::tasks::expecting_done()
        && let Some(expect) = crate::tasks::take_expect_done()
    {
        let cleaned = clean_wakewords_and_prompt(text);
        return handle_task_confirmation(&expect, &cleaned, h);
    }

    if let Some(phrase_to_repeat) = extract_repeat_text(text) {
        println!("🗣️ Озвучивание текста: \"{}\"", phrase_to_repeat);
        let _ = h
            .audio_tx
            .send(AudioMessage::SynthesizeAndPlay(phrase_to_repeat));
        return Outcome::Followup;
    }

    if let Some(spoken) = text::extract_topic_resume(text) {
        return resume_topic(&spoken, h);
    }

    if let Some(social_reply) = match_quick_social_response(text) {
        println!("⚡ Социальный ответ: \"{}\"", social_reply);
        let _ = h
            .audio_tx
            .send(AudioMessage::SynthesizeAndPlay(social_reply.to_string()));
        return Outcome::Followup;
    }

    // «Что ты видишь / что на экране?» — детерминированно агенту зрения.
    // Перездавать это в LLM нельзя: маленькая модель отвечает текстом про
    // «текстовое поле» и галлюцинирует — здесь нужен Holo, который видит
    // пиксели. Описание экрана проговаривается как обычный ответ.
    if let Some(outcome) = try_screen_question(text, h) {
        return outcome;
    }

    let (matched_cmd, global_cfg) = {
        let lock = h.commands.read().unwrap_or_else(|e| e.into_inner());
        // Детерминированно выполняем системные команды — стоп речи и
        // завершение работы обязаны срабатывать мгновенно и надёжно с любым
        // размером модели. Простые action-фразы («открой ютуб», «запусти
        // монтаж», «закрой премьер») тоже решаем без модели: маленькая модель
        // именно на них чаще всего ошибалась. Остальное (url/app/search/
        // web/path с контекстом «яндекс музыка» против «яндекс карты»)
        // по-прежнему отдаём LLM с каталогом run_command.
        let matched_cmd = lock
            .find_command(text)
            .filter(|cmd| matches!(cmd.action_type.trim(), "stop" | "exit"))
            .or_else(|| lock.find_action_command(text));
        (matched_cmd, lock.global.clone())
    };

    if let Some(cmd) = matched_cmd {
        println!("✅ Команда: {}", cmd.name);
        let _ = h.events.send(JarvisEvent::CommandMatched {
            name: cmd.name.clone(),
        });

        if cmd.action_type == "stop" {
            h.stop_audio_flag.store(true, Ordering::SeqCst);
            h.llm_cancel.store(true, Ordering::SeqCst);
            // «Стоп» прерывает и автоматизацию экрана: цикл останавливается
            // между действиями, экран больше не трогается.
            crate::auto::request_cancel();
            let _ = h.audio_tx.send(AudioMessage::Stop);

            let reply_error = global_cfg.reply_error.clone();
            let err_path = paths::resolve_data_path(&reply_error)
                .to_string_lossy()
                .to_string();
            execute_command(
                &cmd,
                text,
                &h.audio_tx,
                &h.events,
                &h.llm,
                Arc::clone(&h.llm_cancel),
                Arc::clone(&h.llm_history),
                &err_path,
            );
            println!("🛑 Команда СТОП: речь остановлена.");
            return Outcome::Wake;
        }

        let reply_error = global_cfg.reply_error.clone();
        let err_path = paths::resolve_data_path(&reply_error)
            .to_string_lossy()
            .to_string();
        // Исполнение сначала, звук-подтверждение после и только при успехе:
        // раньше звук стоял ДО execute_command, и ошибка url/path/command
        // звучала как успех.
        let ok = execute_command(
            &cmd,
            text,
            &h.audio_tx,
            &h.events,
            &h.llm,
            Arc::clone(&h.llm_cancel),
            Arc::clone(&h.llm_history),
            &err_path,
        );
        if ok {
            let sound_num = rng.gen_range(1..=4);
            let sound_path = paths::resolve_data_path(
                &global_cfg
                    .reply_success
                    .replace("{}", &sound_num.to_string()),
            );
            let _ = h.audio_tx.send(AudioMessage::PlayWav(
                sound_path.to_string_lossy().to_string(),
            ));
        } else {
            let _ = h.audio_tx.send(AudioMessage::PlayWav(err_path));
        }
        return Outcome::Followup;
    }

    let prompt = clean_wakewords_and_prompt(text);

    if !prompt.trim().is_empty() {
        // Голый wake word в окне слушания («Джарвис», даже записанный STT как
        // «жарвис») — не вопрос: молча продолжаем слушать.
        if crate::text::is_wake_only(&prompt) {
            return Outcome::Followup;
        }

        // Мусор STT («юч», обрывок слова) в LLM не гоняем: ответ по шуму
        // гарантированно бесполезен. Окно слушания остаётся открытым —
        // фразу можно повторить.
        if prompt.chars().filter(|c| c.is_alphanumeric()).count() < 3 {
            println!("🔇 Слишком короткий ввод, пропускаю: \"{prompt}\"");
            return Outcome::Followup;
        }

        // Огрызки речи из одних слов-связок («а после», «ну вот») — тоже не
        // вопрос: ответ LLM будет выдумкой ни о чём.
        if crate::text::is_filler_only(&prompt) {
            println!("🔇 Только слова-связки, пропускаю: \"{prompt}\"");
            return Outcome::Followup;
        }

        // Календарь считаем сами: маленькая модель не умеет прибавлять дни
        // («какой день будет послезавтра» → «не знаю»), а так ответ
        // мгновенный, без LLM и без веб-поиска.
        if let Some(answer) = crate::text::calendar_answer(&prompt) {
            println!("📅 Календарь: {answer}");
            let _ = h.audio_tx.send(AudioMessage::SynthesizeAndPlay(answer));
            return Outcome::Followup;
        }

        // Авто-поиск только для обычных вопросов. Команды действия («открой
        // сейчас дипсик») не должны перехватываться, даже если в тексте есть
        // «сейчас», «цена» и т.п.
        let intent = router::classify(&prompt);
        let web_search_auto = {
            let st = h.settings.read().unwrap_or_else(|e| e.into_inner());
            st.web_search_auto
        };
        // Явная просьба «посмотри в интернете» — прямая команда, работает
        // всегда, даже с выключенным авто-поиском.
        let explicit_web = intent == router::Intent::Chat && router::explicit_web_request(&prompt);
        let fresh_info = intent == router::Intent::Chat
            && crate::web::looks_like_fresh_info(&prompt)
            // «Какой сегодня день недели» вычисляется из даты в промпте —
            // поиск был бы тратой пары секунд.
            && !router::calendar_question(&prompt);
        let web_auto = (fresh_info && web_search_auto) || explicit_web;

        if web_auto && crate::weather::is_weather_question(&prompt) {
            // Погода — данные, а не рассуждение: детерминированный ответ из
            // API погоды, без поиска (DuckDuckGo отдаёт заглушку, после неё
            // модель отвечала отсебятиной) и без нейросети.
            let city = crate::weather::extract_city(&prompt);
            let default_city = {
                let st = h.settings.read().unwrap_or_else(|e| e.into_inner());
                st.weather_city.clone()
            };
            println!("🌤 Погода: город «{}»", city.as_deref().unwrap_or(&default_city));
            let audio_tx = h.audio_tx.clone();
            let events = h.events.clone();
            let llm_cancel = Arc::clone(&h.llm_cancel);
            let req = req_id;
            std::thread::spawn(move || {
                let target = city.unwrap_or(default_city);
                let answer = match crate::weather::fetch(&target) {
                    Ok(w) => crate::weather::speak_answer(&w),
                    Err(e) => {
                        println!("⚠️ Погода: {e:#}");
                        "Не смог узнать погоду: сервис погоды недоступен.".into()
                    }
                };
                if llm::is_current_request(req) && !llm_cancel.load(Ordering::SeqCst) {
                    llm::speak(&audio_tx, &events, &answer);
                }
            });
            return Outcome::Followup;
        }

        if web_auto {
            // «Да посмотри в интернете» без предмета — ищем то, о чём
            // только что говорили (память тем не чистится wake word'ом).
            let query = router::web_query_from(&prompt)
                .or_else(crate::topic::last_question)
                .unwrap_or_else(|| prompt.clone());
            println!("🌐 Ищу в интернете: \"{query}\"");

            let audio_tx = h.audio_tx.clone();
            let events = h.events.clone();
            let llm = Arc::clone(&h.llm);
            let history = Arc::clone(&h.llm_history);
            let cancel = Arc::clone(&h.llm_cancel);

            std::thread::spawn(move || match crate::web::search(&query, 5) {
                Ok(hits) => {
                    // Пока искали, пользователь мог сказать новое слово или
                    // «стоп»: вчерашний курс доллара вслух посреди нового
                    // разговора звучит как бред.
                    if !llm::is_current_request(req_id) || cancel.load(Ordering::SeqCst) {
                        println!("🚫 Поиск устарел — отвечаю молча");
                        return;
                    }
                    match crate::web::build_summary_prompt(&query, &hits) {
                        Some(web_prompt) => {
                            llm::ask(
                                llm,
                                llm::WEB_SYSTEM_PROMPT,
                                web_prompt,
                                history,
                                audio_tx,
                                events,
                                cancel,
                            );
                        }
                        None => {
                            // Пустая выдача (анти-бот заглушка, блокировка) —
                            // честный отказ вместо выдуманного ответа.
                            println!("🌐 Выдача пуста — честный отказ");
                            let _ = events.send(JarvisEvent::Log {
                                message: "Поиск не дал результатов — честный отказ".into(),
                            });
                            llm::speak(&audio_tx, &events, "Не нашёл ответа в интернете.");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("⚠️ Веб-поиск не удался: {e}");
                    if llm::is_current_request(req_id) && !cancel.load(Ordering::SeqCst) {
                        llm::speak(&audio_tx, &events, "Интернет не отвечает, не смог найти.");
                    }
                }
            });

            return Outcome::Followup;
        }

        println!("🤖 Вопрос к LLM: \"{}\"", prompt);

        if h.settings
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .ack_before_llm
        {
            let sound_num = rng.gen_range(1..=4);
            let sound_path = paths::resolve_data_path(
                &global_cfg
                    .reply_success
                    .replace("{}", &sound_num.to_string()),
            );
            let _ = h.audio_tx.send(AudioMessage::PlayWav(
                sound_path.to_string_lossy().to_string(),
            ));
        }

        llm::ask(
            Arc::clone(&h.llm),
            llm::CHAT_SYSTEM_PROMPT,
            prompt,
            Arc::clone(&h.llm_history),
            h.audio_tx.clone(),
            h.events.clone(),
            Arc::clone(&h.llm_cancel),
        );
    } else {
        println!("❓ Команда не найдена: \"{}\"", text);
        let _ = h.events.send(JarvisEvent::CommandNotFound {
            text: text.to_string(),
        });
        let unknown = paths::resolve_data_path(&global_cfg.reply_unknown);
        let _ = h
            .audio_tx
            .send(AudioMessage::PlayWav(unknown.to_string_lossy().to_string()));
    }

    Outcome::Followup
}

#[cfg(test)]
mod vad_tests {
    use super::*;

    #[test]
    fn barge_does_not_trigger_on_echo_alone() {
        // Колонки играют ответ: эхо стабильно, базлайн калибруется по нему —
        // триггера нет.
        let mut barge = BargeTracker::new(true);
        let now = Instant::now();
        let chunk = vec![0.05f32; 1600]; // 0.1 с эха
        for i in 0..40 {
            assert!(
                !barge.on_playback_chunk(&chunk, now + Duration::from_millis(i * 100), 0.018),
                "эхо не должно перебивать самого Джарвиса"
            );
        }
        assert!(!barge.triggered);
    }

    #[test]
    fn barge_triggers_when_user_is_louder_than_echo() {
        // Эхо ~0.05, пользователь говорит втрое громче: после разогрева
        // (1.2 с — отклик/короткие звуки не вооружают) и накопления
        // BARGE_CONFIRM_SECS — перебивание.
        let mut barge = BargeTracker::new(true);
        let now = Instant::now();
        let echo = vec![0.05f32; 1600];
        for i in 0..15 {
            barge.on_playback_chunk(&echo, now + Duration::from_millis(i * 100), 0.018);
        }
        let voice = vec![0.15f32; 1600]; // 0.1 с голоса
        let mut triggered = false;
        for i in 0..10 {
            if barge.on_playback_chunk(&voice, now + Duration::from_millis(1500 + i * 100), 0.018) {
                triggered = true;
                break;
            }
        }
        assert!(triggered, "громкий пользователь должен перебить эхо 0.05");
        // Хвост кольца стал пре-роллом фразы.
        let preroll = barge.take_preroll();
        assert!(!preroll.is_empty());
        assert!(preroll.len() <= (BARGE_PREROLL_SECS * wake::SAMPLE_RATE as f32) as usize);
    }

    #[test]
    fn barge_does_not_arm_during_short_ack() {
        // Отклик на «Джарвис» ~0.6 с эха: разогрев 1.2 с не пройден —
        // даже громкая речь владельца сразу после отклика не триггерит.
        let mut barge = BargeTracker::new(true);
        let now = Instant::now();
        let ack_echo = vec![0.05f32; 1600];
        for i in 0..6 {
            assert!(!barge.on_playback_chunk(
                &ack_echo,
                now + Duration::from_millis(i * 100),
                0.018
            ));
        }
        let voice = vec![0.3f32; 1600];
        for i in 0..6 {
            assert!(
                !barge.on_playback_chunk(&voice, now + Duration::from_millis(600 + i * 100), 0.018),
                "перебивание не должно срабатывать на отклике"
            );
        }
        assert!(!barge.triggered);
    }

    #[test]
    fn barge_respects_floor_and_cooldown() {
        // Ниже порога речи (в наушниках эха нет) триггера нет.
        let mut barge = BargeTracker::new(true);
        let now = Instant::now();
        let quiet = vec![0.01f32; 1600];
        for i in 0..30 {
            assert!(!barge.on_playback_chunk(
                &quiet,
                now + Duration::from_millis(i * 100),
                0.018
            ));
        }
        // После срабатывания повторный триггер в cooldown невозможен.
        barge.cooldown_until = Some(now + Duration::from_secs(5));
        barge.triggered = false;
        let loud = vec![0.3f32; 1600];
        assert!(!barge.on_playback_chunk(&loud, now + Duration::from_millis(100), 0.018));
    }

    #[test]
    fn barge_disabled_is_always_silent() {
        let mut barge = BargeTracker::new(false);
        let now = Instant::now();
        let loud = vec![0.3f32; 1600];
        for i in 0..30 {
            assert!(!barge.on_playback_chunk(
                &loud,
                now + Duration::from_millis(i * 100),
                0.018
            ));
        }
        assert!(!barge.triggered);
    }

    #[test]
    fn verify_slice_cuts_barge_preroll_with_echo() {
        const RATE: usize = 16_000;
        let pad = 3200; // 0.2 с
        let preroll = 5600; // 0.35 с пре-ролла перебивания
        let speech = vec![1.0f32; RATE * 3]; // 3 с чистой речи после эха
        let mut padded = vec![0.0f32; pad];
        padded.extend(vec![2.0f32; preroll]); // «эхо»
        padded.extend_from_slice(&speech);
        padded.extend(vec![0.0f32; pad]);

        let clean = verify_slice(&padded, pad, preroll, RATE);
        // Срез начинается после эха: внутри нет «эха» (2.0), начинается
        // сразу с чистой речи; хвостовой паддинг допустим — trim_silence
        // внутри верификации его снимет.
        assert_eq!(clean.len(), RATE * 3 + pad);
        assert_eq!(clean[0], 1.0);
        assert!(!clean.iter().any(|&s| s == 2.0), "эхо попало в срез");

        // Хвоста после пре-ролла меньше минимума — отдаём всю фразу
        // (решение примет фильтр, а не молчаливый пропуск).
        let short_speech = vec![1.0f32; RATE / 2];
        let mut short_padded = vec![0.0f32; pad];
        short_padded.extend(vec![2.0f32; preroll]);
        short_padded.extend_from_slice(&short_speech);
        short_padded.extend(vec![0.0f32; pad]);
        assert_eq!(
            verify_slice(&short_padded, pad, preroll, RATE).len(),
            short_padded.len()
        );

        // Без перебивания срез = фраза с хвостовым паддингом (без ведущего).
        assert_eq!(verify_slice(&padded, pad, 0, RATE).len(), padded.len() - pad);
    }

    #[test]
    fn quiet_start_floor_is_between_noise_and_speech() {
        // Тихая комната (фон 0.005): жёсткий порог 0.018, мягкий — заметно
        // ниже него и выше фона: тихая речь 0.012 начинает фразу.
        let floor = quiet_start_floor(0.018, 0.005, 0.018);
        assert!(floor < 0.018, "мягкий порог ниже жёсткого: {floor}");
        assert!(floor > 0.005, "мягкий порог выше фона: {floor}");

        // Шумная комната (фон 0.012): мягкий порог не даёт шуму стартовать.
        let floor = quiet_start_floor(0.0216, 0.012, 0.018);
        assert!(floor > 0.012 * 1.05, "порог выше фона с запасом: {floor}");
    }

    #[test]
    fn agc_amplifies_quiet_and_leaves_loud() {
        // Тихая фраза (RMS ~0.01) усиливается к целевому уровню.
        let quiet = vec![0.01f32; 16_000];
        let boosted = normalize_gain(&quiet);
        let new_rms = calculate_rms(&boosted);
        assert!(
            (new_rms - AGC_TARGET_RMS).abs() < AGC_TARGET_RMS * 0.2,
            "RMS после усиления ≈ целевому: {new_rms}"
        );

        // Громкая речь не ослабляется и не клиппуется принудительно.
        let loud = vec![0.5f32; 16_000];
        assert_eq!(normalize_gain(&loud), loud);

        // Цифровая тишина возвращается как есть (усилить тишину = усилить шум).
        let silence = vec![0.0f32; 16_000];
        assert!(normalize_gain(&silence).iter().all(|s| *s == 0.0));
    }

    #[test]
    fn steady_noise_does_not_trigger_speech() {
        let threshold = 0.018f32;
        let ceiling = threshold * 4.0;
        // Комната «шумит» ровно на уровне порога — типичный случай из лога,
        // когда ассистент бесконечно распознавал пустоту.
        let mut floor = 0.0f32;
        for _ in 0..200 {
            floor = update_noise_floor(floor, 0.020, ceiling);
        }
        assert!(
            speech_floor(threshold, floor) > 0.020,
            "порог {} должен подняться выше шума 0.020",
            speech_floor(threshold, floor)
        );
    }

    #[test]
    fn quiet_room_keeps_user_threshold() {
        let threshold = 0.018f32;
        let ceiling = threshold * 4.0;
        let mut floor = 0.0f32;
        for _ in 0..200 {
            floor = update_noise_floor(floor, 0.002, ceiling);
        }
        assert_eq!(speech_floor(threshold, floor), threshold);
        // Обычная речь проходит с большим запасом.
        assert!(0.15 > speech_floor(threshold, floor));
    }

    #[test]
    fn vad_confirms_tail_silence() {
        let mut vad = VadTracker::new();
        vad.reset_phrase();
        let id = vad.id;
        assert!(vad.accept(SpeechProbeResult {
            id,
            total_secs: 1.5,
            last_speech_end: Some(0.7),
        }));
        assert!(vad.saw_speech);
        // Хвост 0.8 с — речь закончилась, фразу можно закрывать.
        assert!(vad.confirmed_silence(0.6));
    }

    #[test]
    fn vad_keeps_phrase_while_speech_reaches_window_end() {
        let mut vad = VadTracker::new();
        vad.reset_phrase();
        let id = vad.id;
        // Модель обычно заканчивает сегмент за ~0.2 с до конца окна даже
        // при непрерывной речи: это смещение, а не тишина.
        assert!(vad.accept(SpeechProbeResult {
            id,
            total_secs: 1.5,
            last_speech_end: Some(1.3),
        }));
        assert!(!vad.confirmed_silence(0.2));
    }

    #[test]
    fn vad_ignores_stale_reply() {
        let mut vad = VadTracker::new();
        vad.reset_phrase();
        let id = vad.id;
        vad.reset_phrase();
        assert!(!vad.accept(SpeechProbeResult {
            id,
            total_secs: 1.5,
            last_speech_end: Some(0.1),
        }));
        assert!(vad.tail_secs.is_none());
    }

    #[test]
    fn vad_no_speech_means_full_window_of_silence() {
        let mut vad = VadTracker::new();
        vad.reset_phrase();
        let id = vad.id;
        assert!(vad.accept(SpeechProbeResult {
            id,
            total_secs: 1.5,
            last_speech_end: None,
        }));
        assert!(!vad.saw_speech);
        assert_eq!(vad.tail_secs, Some(1.5));
        assert!(vad.confirmed_silence(0.6));
    }

    #[test]
    fn vad_disables_after_three_failures() {
        let mut vad = VadTracker::new();
        vad.note_failure();
        vad.note_failure();
        assert!(!vad.unavailable);
        vad.note_failure();
        assert!(vad.unavailable);
    }

    #[test]
    fn verdict_waits_through_stale_vad_replies() {
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(SpeakerReply::Speech(None)).unwrap();
        tx.send(SpeakerReply::Verified {
            id: 7,
            verdict: Verdict::Stranger,
        })
        .unwrap();
        assert_eq!(
            wait_for_verdict(&rx, 7, Duration::from_millis(100)),
            Verdict::Stranger
        );
    }

    #[test]
    fn verdict_skips_stale_barge_replies_by_id() {
        // Ответ фоновой проверки перебивания (id 3) не принимается за
        // вердикт фразы (id 9): иначе перебивание прошлого ответа «признавало»
        // чужую фразу владельцем.
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(SpeakerReply::Verified {
            id: 3,
            verdict: Verdict::Owner,
        })
        .unwrap();
        tx.send(SpeakerReply::Verified {
            id: 9,
            verdict: Verdict::Stranger,
        })
        .unwrap();
        assert_eq!(
            wait_for_verdict(&rx, 9, Duration::from_millis(100)),
            Verdict::Stranger
        );
        // Ожидаемого id нет — fail-closed, а не чужой устаревший вердикт.
        let (tx2, rx2) = std::sync::mpsc::channel();
        tx2.send(SpeakerReply::Verified {
            id: 3,
            verdict: Verdict::Owner,
        })
        .unwrap();
        assert_eq!(
            wait_for_verdict(&rx2, 9, Duration::from_millis(50)),
            Verdict::Unverified("фильтр не ответил вовремя")
        );
    }

    #[test]
    fn verdict_timeout_is_fail_closed() {
        let (_tx, rx) = std::sync::mpsc::channel::<SpeakerReply>();
        // Воркер молчит — фраза не проходит, а не «пропущена фильтром».
        assert_eq!(
            wait_for_verdict(&rx, 1, Duration::from_millis(30)),
            Verdict::Unverified("фильтр не ответил вовремя")
        );
    }

    #[test]
    fn noise_floor_is_capped() {
        let ceiling = 0.072f32;
        let mut floor = 0.0f32;
        for _ in 0..500 {
            floor = update_noise_floor(floor, 0.5, ceiling);
        }
        assert!(floor <= ceiling);
    }

    #[test]
    fn screen_question_patterns_match_real_phrases() {
        assert!(is_screen_question("что ты видишь на экране?"));
        assert!(is_screen_question("джарвис что на экране"));
        assert!(is_screen_question("посмотри на экран"));
        assert!(is_screen_question("Что открыто у меня сейчас?"));
        // Не вопросы про экран — не перехватываем.
        assert!(!is_screen_question("какая погода в москве"));
        assert!(!is_screen_question("поставь таймер на пять минут"));
        assert!(!is_screen_question("открой ютуб"));
    }
}

