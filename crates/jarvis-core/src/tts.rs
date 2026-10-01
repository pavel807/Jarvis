//! TTS: Qwen3-TTS (0.6B CustomVoice) через крейт jarvis-tts — чистый Rust на
//! candle + Metal, без Python/C++. Русский — один из 10 официальных языков.
//! Модель (~2.5 ГБ) скачивается один раз при отсутствии.
//!
//! Поток-плеер последовательно обрабатывает очередь [`AudioMessage`]:
//! проигрывает WAV или синтезирует речь (с кэшем на диске).

use crate::audio::OutputPlayer;
use crate::llm::{self, SharedLlm};
use crate::paths;
use crate::settings::Settings;
use crate::tts_kokoro::KokoroEngine;
use crate::tts_piper::PiperEngine;
use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use jarvis_tts::{auto_device, Language, Qwen3TTS, Speaker, StreamingSession, SynthesisOptions};
pub use crate::tts_kokoro::KOKORO_VOICES;
pub use crate::tts_piper::PIPER_VOICES;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Доступные голоса Qwen3-TTS (пресеты CustomVoice): (идентификатор, описание).
/// Все голоса мультиязычны и говорят по-русски.
pub const TTS_VOICES: &[(&str, &str)] = &[    ("ryan", "мужской, тёплый"),
    ("aiden", "мужской, средний"),
    ("eric", "мужской, глубокий"),
    ("dylan", "мужской, молодой"),
    ("uncle_fu", "мужской, низкий"),
    ("serena", "женский, мягкий"),
    ("vivian", "женский, тёплый"),
    ("ono_anna", "женский, высокий"),
    ("sohee", "женский, средний"),
];

/// Версия движков синтеза — участвует в хэше кэша. v2: склеенные слова
/// в piper (29b8f28), старые кэш-файлы с межсловными паузами невалидны.
const TTS_ENGINE_VERSION: &str = "jarvis-tts-v2";
/// Скорость речи по умолчанию: 1.0 — естественно; <1 медленнее.
/// Меняется в настройках (settings.tts_speed).
pub const DEFAULT_TTS_SPEED: f32 = 1.0;
/// Модель фиксирована: 0.6B CustomVoice — лучший баланс качества и RTF на Metal.
const TTS_MODEL_ID: &str = "Qwen3-TTS-12Hz-0.6B-CustomVoice";
/// Температура сэмплинга: ниже дефолта — стабильнее речь для ассистента.
const TTS_TEMPERATURE: f64 = 0.7;

static TTS_SPEED: std::sync::RwLock<f32> = std::sync::RwLock::new(1.0);

pub fn set_current_speed(speed: f32) {
    *TTS_SPEED.write().unwrap_or_else(|e| e.into_inner()) = speed.clamp(0.5, 1.5);
}

pub fn current_speed() -> f32 {
    *TTS_SPEED.read().unwrap_or_else(|e| e.into_inner())
}

/// Активный движок синтеза: "piper" | "qwen3".
static TTS_ENGINE: RwLock<String> = RwLock::new(String::new());

pub fn current_engine() -> String {
    TTS_ENGINE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub fn set_current_engine(engine: &str) {
    *TTS_ENGINE.write().unwrap_or_else(|e| e.into_inner()) = engine.to_string();
}

const TTS_CACHE_MAX_CHARS: usize = 200;
/// Первый запуск может скачивать модель (~2.5 ГБ), ждём подольше.
const TTS_WAIT_TIMEOUT_SECS: u64 = 600;
/// Жёсткий лимит дискового кэша синтеза: при превышении старые файлы удаляются.
pub const TTS_CACHE_MAX_BYTES: u64 = 300 * 1024 * 1024;

/// Текущий голос (глобально, применяется к каждому синтезу).
static TTS_VOICE: RwLock<String> = RwLock::new(String::new());

pub fn current_voice() -> String {
    TTS_VOICE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

pub fn set_current_voice(voice: String) {
    *TTS_VOICE.write().unwrap_or_else(|e| e.into_inner()) = voice;
}

fn voice_model_dir() -> PathBuf {
    paths::tts_models_dir().join("qwen3-tts-0.6b-customvoice")
}

/// Скачивает модель Qwen3-TTS (0.6B CustomVoice) и текстовый токенизатор,
/// если их ещё нет. Автозагрузка при первом запуске.
fn ensure_tts_model() -> Result<PathBuf> {
    let dir = voice_model_dir();
    let weights = dir.join("model.safetensors");
    let config = dir.join("config.json");
    let speech_tok = dir.join("speech_tokenizer/model.safetensors");
    let tokenizer = dir.join("tokenizer.json");

    if weights.exists() && config.exists() && speech_tok.exists() && tokenizer.exists() {
        return Ok(dir);
    }

    fs::create_dir_all(dir.join("speech_tokenizer"))?;

    let client = hf_hub::HFClientSync::new().context("Не удалось создать HuggingFace-клиент")?;
    let model_repo = client.model("Qwen".to_string(), TTS_MODEL_ID.to_string());
    let tok_repo = client.model("Qwen".to_string(), "Qwen2-0.5B".to_string());

    let fetch = |repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
                 filename: &str,
                 dest: &Path|
     -> Result<()> {
        if dest.exists() {
            return Ok(());
        }
        println!("📥 Загрузка {filename}...");
        let downloaded = repo
            .download_file()
            .filename(filename.to_string())
            .send()
            .with_context(|| format!("Не удалось скачать {filename}"))?;
        fs::copy(&downloaded, dest)
            .with_context(|| format!("Не удалось сохранить {filename}"))?;
        Ok(())
    };

    fetch(&model_repo, "config.json", &config)?;
    fetch(&model_repo, "model.safetensors", &weights)?;
    fetch(&model_repo, "speech_tokenizer/model.safetensors", &speech_tok)?;
    fetch(&tok_repo, "tokenizer.json", &tokenizer)?;

    println!("✅ Модель Qwen3-TTS готова");
    Ok(dir)
}

pub struct TtsEngine {
    model: Qwen3TTS,
    speaker: Speaker,
    voice: String,
}

impl TtsEngine {
    pub fn new(voice: &str) -> Result<Self> {
        if !TTS_VOICES.iter().any(|(v, _)| *v == voice) {
            bail!("Неизвестный голос TTS: {voice}");
        }
        let speaker: Speaker = voice
            .parse()
            .map_err(|_| anyhow::anyhow!("Неизвестный голос TTS: {voice}"))?;

        println!(
            "🚀 Инициализация TTS (Qwen3-TTS 0.6B, голос «{voice}»)..."
        );
        let started = Instant::now();

        let dir = ensure_tts_model()?;
        let device = auto_device()?;
        let model = Qwen3TTS::from_pretrained(&dir.to_string_lossy(), device)
            .context("Не удалось загрузить Qwen3-TTS")?;

        println!("✅ TTS готов за {:.1?}", started.elapsed());
        Ok(Self {
            model,
            speaker,
            voice: voice.to_string(),
        })
    }

    pub fn voice(&self) -> &str {
        &self.voice
    }

    /// Потоковый синтез: сессия выдаёт чанки аудио по мере генерации,
    /// не дожидаясь готовности всей фразы.
    ///
    /// max_length масштабируется по длине текста (~1 фрейм на символ при
    /// 12 Гц): дефолтные 2048 фреймов (163 с) позволяли изредка «убегающей»
    /// генерации без EOS занимать минуты. Первая порция всегда мала
    /// (см. FIRST_CHUNK_FRAMES), установившийся чанк — 10 фреймов.
    pub fn stream_session(&self, text: &str) -> Result<StreamingSession<'_>> {
        let max_length = ((text.chars().count() as f32 * 1.2) as usize + 32).clamp(96, 384);
        let options = SynthesisOptions {
            temperature: TTS_TEMPERATURE,
            max_length,
            chunk_frames: 10,
            ..Default::default()
        };
        self.model
            .synthesize_streaming(text, self.speaker, Language::Russian, options)
    }

    /// Прогрев: первый forward компилирует Metal-шейдеры (на холодную —
    /// секунды), поэтому делаем это при загрузке, а не на первой фразе
    /// пользователя. Короткая фраза — минимальная генерация, результат
    /// отбрасывается.
    pub fn warmup(&self) -> Result<()> {
        let started = Instant::now();
        let mut session = self.stream_session("Да.")?;
        loop {
            match session.next_chunk() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        println!("🔥 TTS прогрет за {:.1?}", started.elapsed());
        Ok(())
    }

    pub fn synthesize_to_file(&self, text: &str, out_path: &Path) -> Result<()> {
        let options = SynthesisOptions {
            temperature: TTS_TEMPERATURE,
            ..Default::default()
        };
        // Разовые отказы Metal-аллокации ("Failed to create metal resource")
        // случаются при системном давлении на память; повтор обычно проходит.
        let mut audio = None;
        for attempt in 0..2 {
            match self.model.synthesize_with_voice(
                text,
                self.speaker,
                Language::Russian,
                Some(options.clone()),
            ) {
                Ok(a) => {
                    audio = Some(a);
                    break;
                }
                Err(e) => {
                    let transient = e.to_string().contains("Failed to create metal resource");
                    if transient && attempt == 0 {
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                    return Err(anyhow::anyhow!("Ошибка синтеза речи: {e}"));
                }
            }
        }
        let audio = audio.expect("retry loop invariant");

        if audio.samples.is_empty() {
            bail!("Синтез вернул пустое аудио");
        }

        // Скорость реализуется частотой WAV: плеер всё равно ресемплирует под
        // устройство, поэтому <1 замедляет и чуть понижает голос, >1 ускоряет.
        let speed = current_speed().clamp(0.5, 1.5);
        let rate = (audio.sample_rate as f32 * speed).round() as u32;

        let spec = WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(out_path, spec)
            .with_context(|| format!("Не удалось создать {:?}", out_path))?;
        for &s in &audio.samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(v)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

/// Состояние фоновой загрузки TTS: None — ещё грузится, Some — готово.
pub struct SharedTtsState {
    pub tts: Mutex<Option<Arc<SynthEngine>>>,
    pub condvar: Condvar,
}

pub type SharedTts = Arc<SharedTtsState>;

/// Активный движок синтеза. Piper — быстрый CPU-бэкенд (RTF < 0.1),
/// Kokoro-ru — качественнее Piper с корректными ударениями и смешанной
/// речью EN+RU, Qwen3 — нейросетевой на Metal с естественной интонацией.
pub enum SynthEngine {
    Qwen3(TtsEngine),
    Piper(PiperEngine),
    Kokoro(KokoroEngine),
}

impl SynthEngine {
    /// Неавторегрессивные движки (Piper/Kokoro): вся фраза одним проходом,
    /// возвращают (сэмплы, частота источника). Qwen3 стримит чанками.
    pub fn synthesize_whole(&self, text: &str, speed: f32) -> Result<(Vec<f32>, u32)> {
        match self {
            SynthEngine::Piper(p) => p.synthesize(text, speed).map(|a| (a, p.sample_rate())),
            SynthEngine::Kokoro(k) => k.synthesize(text, speed).map(|a| (a, k.sample_rate())),
            SynthEngine::Qwen3(_) => bail!("Qwen3 синтезирует стримингом"),
        }
    }
}

impl SharedTtsState {
    pub fn new() -> Self {
        Self {
            tts: Mutex::new(None),
            condvar: Condvar::new(),
        }
    }
}

/// Фоновая загрузка TTS: по завершении заполняет SharedTtsState и уведомляет.
pub fn spawn_tts_loader(
    shared: &SharedTts,
    tts_failed: &Arc<AtomicBool>,
    engine: &str,
    voice: &str,
) {
    let shared = Arc::clone(shared);
    let tts_failed = Arc::clone(tts_failed);
    let engine = engine.to_string();
    let voice = voice.to_string();
    set_current_engine(&engine);
    set_current_voice(voice.clone());

    std::thread::spawn(move || {
        println!("🧠 Загрузка TTS ({engine}, голос «{voice}») в фоне...");
        let backend = match engine.as_str() {
            "piper" => PiperEngine::new(&voice).map(SynthEngine::Piper),
            "kokoro" => KokoroEngine::new(&voice).map(SynthEngine::Kokoro),
            _ => match TtsEngine::new(&voice) {
                Ok(e) => Ok(SynthEngine::Qwen3(e)),
                Err(first) => {
                    // Повтор: разовые отказы аллокации Metal при старте системы.
                    std::thread::sleep(Duration::from_secs(2));
                    TtsEngine::new(&voice)
                        .map(SynthEngine::Qwen3)
                        .map_err(|second| {
                            anyhow::anyhow!("{first}; повторная попытка тоже не удалась: {second}")
                        })
                }
            },
        };
        match backend {
            Ok(SynthEngine::Qwen3(engine)) => {
                // Прогреваем до публикации движка: wait_for_tts подождёт,
                // зато ни одна фраза не попадёт на холодные шейдеры.
                if let Err(e) = engine.warmup() {
                    eprintln!("⚠️ Прогрев TTS не удался (не критично): {e}");
                }
                let mut guard = shared.tts.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(Arc::new(SynthEngine::Qwen3(engine)));
                shared.condvar.notify_all();
                println!("✅ TTS готов к озвучке");
            }
            Ok(other) => {
                // Piper: короткий прогрев ORT-ядер на реальном синтезе.
                if let SynthEngine::Piper(p) = &other {
                    if let Err(e) = p.synthesize("Да.", 1.0) {
                        eprintln!("⚠️ Прогрев piper не удался (не критично): {e}");
                    }
                }
                let mut guard = shared.tts.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(Arc::new(other));
                shared.condvar.notify_all();
                println!("✅ TTS готов к озвучке");
            }
            Err(e) => {
                eprintln!("❌ Ошибка загрузки TTS: {e}");
                tts_failed.store(true, Ordering::SeqCst);
                shared.condvar.notify_all();
            }
        }
    });
}

/// Смена движка/голоса на лету: пересобирает движок в фоне.
pub fn set_voice(shared: &SharedTts, tts_failed: &Arc<AtomicBool>, engine: &str, voice: &str) {
    spawn_tts_loader(shared, tts_failed, engine, voice);
}

/// Удаляет самые старые WAV из кэша синтеза при превышении лимита.
pub fn prune_tts_cache() {
    let dir = paths::tts_cache_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return;
    };

    let mut files: Vec<(SystemTime, u64, PathBuf)> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            Some((meta.modified().ok()?, meta.len(), e.path()))
        })
        .collect();

    let total: u64 = files.iter().map(|(_, size, _)| size).sum();
    if total <= TTS_CACHE_MAX_BYTES {
        return;
    }

    files.sort_by_key(|(t, _, _)| *t);

    let mut removed = 0u64;
    for (_, size, path) in files {
        if total - removed <= TTS_CACHE_MAX_BYTES {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            removed += size;
        }
    }

    if removed > 0 {
        println!("🧹 Кэш TTS уменьшен на {:.1} МБ", removed as f64 / 1048576.0);
    }
}

fn tts_cache_path(engine: &str, text: &str, voice: &str, speed: f32) -> PathBuf {
    let mut hasher = DefaultHasher::new();

    engine.hash(&mut hasher);
    TTS_ENGINE_VERSION.hash(&mut hasher);
    voice.hash(&mut hasher);
    ((speed * 100.0).round() as i64).hash(&mut hasher);
    text.trim().hash(&mut hasher);

    paths::tts_cache_dir().join(format!("{:016x}.wav", hasher.finish()))
}

/// Пишет моно int16 WAV (кэш синтеза / файлы).
fn write_wav_i16(path: &Path, spec: WavSpec, samples: &[f32]) -> Result<()> {
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Не удалось создать {:?}", path))?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn wait_for_tts(
    shared_tts: &SharedTts,
    tts_failed: &AtomicBool,
    timeout: Duration,
) -> Option<Arc<SynthEngine>> {
    let mut guard = shared_tts.tts.lock().unwrap_or_else(|e| e.into_inner());
    let start = Instant::now();

    while guard.is_none() {
        if tts_failed.load(Ordering::SeqCst) {
            eprintln!("❌ Загрузка TTS завершилась ошибкой.");
            return None;
        }

        let elapsed = start.elapsed();
        if elapsed >= timeout {
            eprintln!("❌ Не дождался готовности TTS за {:?}.", timeout);
            return None;
        }

        let remaining = timeout - elapsed;
        match shared_tts.condvar.wait_timeout(guard, remaining) {
            Ok((g, _)) => guard = g,
            Err(e) => guard = e.into_inner().0,
        }
    }

    guard.clone()
}

pub enum AudioMessage {
    PlayWav(String),
    SynthesizeAndPlay(String),
    Stop,
}

/// Нужна ли нейро-нормализация: есть цифры или символьные обозначения.
fn needs_neural_normalize(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_digit())
        || text.contains(['%', '°', '₽', '/'])
        || text.contains("м/с")
}

/// Малый LRU-кэш «исходный текст -> нормализованный», чтобы не гонять LLM
/// на повторяющихся фразах.
struct NormalizeCache {
    map: std::collections::HashMap<String, String>,
    order: VecDeque<String>,
    cap: usize,
}

impl NormalizeCache {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: VecDeque::new(),
            cap: 200,
        }
    }

    fn get(&mut self, key: &str) -> Option<String> {
        self.map.get(key).cloned()
    }

    fn put(&mut self, key: String, value: String) {
        if self.map.len() >= self.cap {
            if let Some(old) = self.order.front().cloned() {
                self.order.remove(0);
                self.map.remove(&old);
            }
        }
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
    }
}

/// Воспроизведение через общий OutputPlayer (или ошибка, если стрим не открыт).
fn play_via(output: &Option<OutputPlayer>, path: &str, stop_flag: &AtomicBool) -> Result<()> {
    match output {
        Some(player) => player.play_wav(path, stop_flag),
        None => bail!("Выходной аудиопоток недоступен"),
    }
}

/// Потоковый синтез фразы: чанки аудио уходят в очередь вывода по мере
/// генерации, поэтому воспроизведение начинается до завершения синтеза.
/// `wav` — опциональная запись в кэш-файл на лету; при ошибке записи
/// кэш отключается, но воспроизведение продолжается.
fn stream_to_output(
    tts: &TtsEngine,
    text: &str,
    player: &OutputPlayer,
    device_rate: usize,
    speed: f32,
    wav: &mut Option<WavWriter<BufWriter<File>>>,
    stop_flag: &AtomicBool,
) -> Result<()> {
    // Скорость реализуется «виртуальной» частотой источника: чанки
    // генерируются на 24 кГц, но ресемплируются в очередь так, будто они
    // записаны на 24 кГц * speed — тот же приём, что и у кэш-WAV.
    let src_rate = (jarvis_tts::audio::SAMPLE_RATE as f32 * speed).round() as usize;
    let started = Instant::now();
    let mut session = tts.stream_session(text)?;
    let mut first_chunk_logged = false;

    loop {
        let chunk = match session.next_chunk() {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => return Err(e),
        };
        if stop_flag.load(Ordering::SeqCst) {
            return Ok(());
        }
        let audio = chunk;

        if !first_chunk_logged {
            println!("🔊 TTS: первый чанк за {:.0?}", started.elapsed());
            first_chunk_logged = true;
        }

        if wav.is_some() {
            let mut ok = true;
            if let Some(w) = wav.as_mut() {
                for &s in &audio.samples {
                    let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    if w.write_sample(v).is_err() {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                eprintln!("⚠️ Запись кэша речи не удалась, продолжаю без кэша");
                *wav = None;
            }
        }

        let samples = if src_rate != device_rate {
            crate::audio::resample_audio(&audio.samples, src_rate, device_rate)
        } else {
            audio.samples
        };
        player.enqueue(&samples);
    }

    Ok(())
}

/// Отдельный поток-плеер: последовательно обрабатывает очередь сообщений.
pub fn spawn_audio_player(
    tts_shared: SharedTts,
    tts_failed: Arc<AtomicBool>,
    is_speaking: Arc<AtomicBool>,
    stop_audio_flag: Arc<AtomicBool>,
    llm_shared: SharedLlm,
    settings: Arc<RwLock<Settings>>,
) -> Sender<AudioMessage> {
    let (tx, rx) = channel::<AudioMessage>();

    std::thread::spawn(move || {
        let mut normalize_cache = NormalizeCache::new();

        // Один постоянный выходной стрим на всё время жизни плеера.
        let output = match OutputPlayer::new() {
            Ok(o) => Some(o),
            Err(e) => {
                eprintln!("❌ Не удалось открыть выходной аудиопоток: {e}");
                None
            }
        };

        'player: while let Ok(msg) = rx.recv() {
            match msg {
                AudioMessage::Stop => {
                    stop_audio_flag.store(true, Ordering::SeqCst);
                    while rx.try_recv().is_ok() {}
                    if let Some(o) = &output {
                        o.clear();
                    }
                    stop_audio_flag.store(false, Ordering::SeqCst);
                    is_speaking.store(false, Ordering::SeqCst);
                }

                AudioMessage::PlayWav(path) => {
                    if stop_audio_flag.load(Ordering::SeqCst) {
                        continue;
                    }

                    is_speaking.store(true, Ordering::SeqCst);

                    if let Err(e) = play_via(&output, &path, &stop_audio_flag) {
                        eprintln!("⚠️ Ошибка воспроизведения файла ({}): {e}", path);
                    }

                    is_speaking.store(false, Ordering::SeqCst);
                }

                AudioMessage::SynthesizeAndPlay(first) => {
                    if stop_audio_flag.load(Ordering::SeqCst) {
                        continue;
                    }
                    if first.trim().is_empty() {
                        continue;
                    }

                    // Склеиваем уже готовые в очереди фразы в одну партию:
                    // одна сессия синтеза вместо отдельной на каждое
                    // предложение убирает паузы (префилл + первый чанк +
                    // пересоздание кэш-форм) между соседними предложениями.
                    let mut raws: Vec<String> = vec![first.trim().to_string()];
                    let mut deferred_wav: Option<String> = None;
                    while raws.iter().map(|t| t.chars().count()).sum::<usize>() < 350 {
                        match rx.try_recv() {
                            Ok(AudioMessage::SynthesizeAndPlay(more)) => {
                                if !more.trim().is_empty() {
                                    raws.push(more.trim().to_string());
                                }
                            }
                            Ok(AudioMessage::PlayWav(path)) => {
                                // Звук должен прозвучать после речи — откладываем.
                                deferred_wav = Some(path);
                                break;
                            }
                            Ok(AudioMessage::Stop) => {
                                stop_audio_flag.store(true, Ordering::SeqCst);
                                while rx.try_recv().is_ok() {}
                                if let Some(o) = &output {
                                    o.clear();
                                }
                                stop_audio_flag.store(false, Ordering::SeqCst);
                                is_speaking.store(false, Ordering::SeqCst);
                                continue 'player;
                            }
                            Err(_) => break,
                        }
                    }

                    let raw_text = raws.join(" ");

                    // Правила — мгновенно; нейро-нормализация (LLM) — для цифр,
                    // единиц и сокращений, если включена и модель свободна.
                    // try_normalize не ждёт движок: пока LLM отвечает, откат на правила.
                    let mut text = crate::text::normalize_for_tts(&raw_text);
                    let nn_enabled = settings
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .llm_normalize;
                    if nn_enabled && needs_neural_normalize(&raw_text) {
                        let final_text = match normalize_cache.get(&raw_text) {
                            Some(cached) => Some(cached),
                            None => llm::try_normalize(&llm_shared, &raw_text)
                                .inspect(|n| normalize_cache.put(raw_text.clone(), n.clone())),
                        };
                        if let Some(n) = final_text {
                            text = n;
                        }
                    }

                    if text.is_empty() {
                        continue;
                    }

                    let engine_name = current_engine();
                    let voice = current_voice();
                    let speed = current_speed();
                    let can_cache = text.chars().count() <= TTS_CACHE_MAX_CHARS;
                    let cache_path = tts_cache_path(&engine_name, &text, &voice, speed);

                    if can_cache && cache_path.exists() {
                        let cache_path_string = cache_path.to_string_lossy().to_string();
                        is_speaking.store(true, Ordering::SeqCst);

                        if let Err(e) = play_via(&output, &cache_path_string, &stop_audio_flag) {
                            eprintln!(
                                "⚠️ Ошибка воспроизведения кэша речи ({}): {e}",
                                cache_path.display()
                            );
                        }

                        is_speaking.store(false, Ordering::SeqCst);
                        continue;
                    }

                    let Some(backend) = wait_for_tts(
                        &tts_shared,
                        tts_failed.as_ref(),
                        Duration::from_secs(TTS_WAIT_TIMEOUT_SECS),
                    ) else {
                        eprintln!("⚠️ TTS не готов. Фраза не озвучена: \"{}\"", text);
                        continue;
                    };

                    if stop_audio_flag.load(Ordering::SeqCst) {
                        continue;
                    }

                    let Some(player) = &output else {
                        eprintln!("⚠️ Выходной аудиопоток недоступен, фраза пропущена");
                        continue;
                    };
                    let device_rate = player.device_sample_rate();

                    is_speaking.store(true, Ordering::SeqCst);

                    match backend.as_ref() {
                        SynthEngine::Qwen3(tts) => {
                            // Кэш-файл пишется на лету, по мере генерации чанков.
                            let mut wav: Option<WavWriter<BufWriter<File>>> = None;
                            if can_cache {
                                if let Err(e) = fs::create_dir_all(paths::tts_cache_dir()) {
                                    eprintln!("⚠️ Не удалось создать папку кэша речи: {e}");
                                }
                                let spec = WavSpec {
                                    channels: 1,
                                    sample_rate: (jarvis_tts::audio::SAMPLE_RATE as f32 * speed)
                                        .round() as u32,
                                    bits_per_sample: 16,
                                    sample_format: SampleFormat::Int,
                                };
                                match WavWriter::create(&cache_path, spec) {
                                    Ok(w) => wav = Some(w),
                                    Err(e) => {
                                        eprintln!("⚠️ Не удалось открыть кэш-файл речи: {e}")
                                    }
                                }
                            }

                            let result = stream_to_output(
                                tts,
                                &text,
                                player,
                                device_rate,
                                speed,
                                &mut wav,
                                &stop_audio_flag,
                            );

                            if let Err(e) = result {
                                eprintln!("⚠️ Ошибка синтеза речи: {e}");
                                if can_cache {
                                    let _ = fs::remove_file(&cache_path);
                                }
                            } else if let Some(Err(e)) = wav.map(|w| w.finalize()) {
                                eprintln!("⚠️ Не удалось сохранить кэш речи: {e}");
                                let _ = fs::remove_file(&cache_path);
                            }
                        }
                        engine => {
                            // Неавторегрессивный движок: вся фраза за один
                            // проход, RTF < 0.1 — стриминг не нужен.
                            let started = Instant::now();
                            match engine.synthesize_whole(&text, speed) {
                                Ok((samples, src_hz)) if !samples.is_empty() => {
                                    if can_cache {
                                        if let Err(e) = fs::create_dir_all(paths::tts_cache_dir())
                                        {
                                            eprintln!(
                                                "⚠️ Не удалось создать папку кэша речи: {e}"
                                            );
                                        }
                                        let spec = WavSpec {
                                            channels: 1,
                                            sample_rate: (src_hz as f32 * speed).round() as u32,
                                            bits_per_sample: 16,
                                            sample_format: SampleFormat::Int,
                                        };
                                        if let Err(e) =
                                            write_wav_i16(&cache_path, spec, &samples)
                                        {
                                            eprintln!("⚠️ Не удалось сохранить кэш речи: {e}");
                                        }
                                    }

                                    let src_rate = (src_hz as f32 * speed).round() as usize;
                                    let out = if src_rate != device_rate {
                                        crate::audio::resample_audio(
                                            &samples,
                                            src_rate,
                                            device_rate,
                                        )
                                    } else {
                                        samples
                                    };
                                    println!(
                                        "🔊 TTS: {:.0?} на {:.1} с аудио",
                                        started.elapsed(),
                                        out.len() as f32 / device_rate as f32
                                    );
                                    player.enqueue(&out);
                                }
                                Ok(_) => eprintln!("⚠️ Движок вернул пустое аудио"),
                                Err(e) => {
                                    eprintln!("⚠️ Ошибка синтеза речи: {e}");
                                    if can_cache {
                                        let _ = fs::remove_file(&cache_path);
                                    }
                                }
                            }
                        }
                    }

                    // Отложенный звук (например, ack) — после речи.
                    if let Some(p) = deferred_wav {
                        if let Err(e) = play_via(&output, &p, &stop_audio_flag) {
                            eprintln!("⚠️ Ошибка воспроизведения файла ({}): {e}", p);
                        }
                    }

                    // Короткая пауза, чтобы предложения не сливались
                    // в сплошной поток.
                    player.enqueue(&vec![0.0f32; device_rate * 12 / 100]);
                    player.wait_drain(&stop_audio_flag);

                    // Прерванный стопом синтез оставляет неполный кэш-файл.
                    if stop_audio_flag.load(Ordering::SeqCst) && can_cache {
                        let _ = fs::remove_file(&cache_path);
                    }

                    is_speaking.store(false, Ordering::SeqCst);
                }
            }
        }
    });

    tx
}

/// Читает длительность WAV-файла в секундах (используется тренировкой wake word).
pub fn wav_duration_secs(path: &Path) -> Option<f64> {
    let reader = WavReader::open(path).ok()?;
    let spec = reader.spec();
    Some(reader.duration() as f64 / spec.sample_rate as f64)
}

/// Список сэмплов в каталоге с их длительностью.
pub fn list_wav_samples(dir: &Path) -> Vec<(PathBuf, f64)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for e in entries.filter_map(|e| e.ok()) {
        let p = e.path();
        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")) {
            if let Some(d) = wav_duration_secs(&p) {
                out.push((p, d));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Сопоставление ярлыка из имени файла `[label]_...` (конвенция rustpotter).
pub fn label_of_sample(file_name: &str) -> String {
    if let Some(start) = file_name.find('[') {
        if let Some(end) = file_name[start..].find(']') {
            return file_name[start + 1..start + end].to_string();
        }
    }
    "none".to_string()
}

/// Удобный тип для тренировки: ярлык -> список WAV-байтов.
pub type LabelBuffers = HashMap<String, Vec<u8>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_parsing() {
        assert_eq!(label_of_sample("[jarvis]_001.wav"), "jarvis");
        assert_eq!(label_of_sample("noise_002.wav"), "none");
        assert_eq!(label_of_sample("[none]_bg.wav"), "none");
    }

    #[test]
    fn voices_known() {
        assert!(TTS_VOICES.iter().any(|(v, _)| *v == "ryan"));
    }
}
