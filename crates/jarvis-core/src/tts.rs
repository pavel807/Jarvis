//! TTS: Kokoro-ru (82M StyleTTS2-lite) через ONNX Runtime — быстрый
//! качественный русский синтез со смешанной речью EN+RU.
//!
//! Модель (~311 МБ) скачивается один раз при отсутствии в models/kokoro-ru/.
//!
//! Поток-плеер последовательно обрабатывает очередь [`AudioMessage`]:
//! проигрывает WAV или синтезирует речь (с кэшем на диске).

use crate::audio::OutputPlayer;
use crate::llm::{self, SharedLlm};
use crate::paths;
use crate::settings::Settings;
use crate::tts_kokoro::KokoroEngine;
pub use crate::tts_kokoro::KOKORO_VOICES;
use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Версия движков синтеза — участвует в хэше кэша. v4: трёхэшелонный
/// ударитель (переопределения -> словарь -> нейросеть) — старые кэш-файлы
/// с неверными ударениями («какАя») невалидны.
const TTS_ENGINE_VERSION: &str = "jarvis-tts-v4";
/// Скорость речи по умолчанию: 1.0 — естественно; <1 медленнее.
pub const DEFAULT_TTS_SPEED: f32 = 1.0;
const TTS_CACHE_MAX_CHARS: usize = 200;
/// Первый запуск может скачивать модель (~311 МБ), ждём подольше.
const TTS_WAIT_TIMEOUT_SECS: u64 = 600;
/// Жёсткий лимит дискового кэша синтеза: при превышении старые файлы удаляются.
pub const TTS_CACHE_MAX_BYTES: u64 = 300 * 1024 * 1024;

static TTS_SPEED: RwLock<f32> = RwLock::new(1.0);

pub fn set_current_speed(speed: f32) {
    *TTS_SPEED.write().unwrap_or_else(|e| e.into_inner()) = speed.clamp(0.5, 1.5);
}

pub fn current_speed() -> f32 {
    *TTS_SPEED.read().unwrap_or_else(|e| e.into_inner())
}

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

pub struct SharedTtsState {
    pub tts: Mutex<Option<Arc<KokoroEngine>>>,
    pub condvar: Condvar,
}

pub type SharedTts = Arc<SharedTtsState>;

impl SharedTtsState {
    pub fn new() -> Self {
        Self {
            tts: Mutex::new(None),
            condvar: Condvar::new(),
        }
    }
}

/// Фоновая загрузка TTS: по завершении заполняет SharedTtsState и уведомляет.
pub fn spawn_tts_loader(shared: &SharedTts, tts_failed: &Arc<AtomicBool>, voice: &str) {
    let shared = Arc::clone(shared);
    let tts_failed = Arc::clone(tts_failed);
    let voice = voice.to_string();
    set_current_voice(voice.clone());

    std::thread::spawn(move || {
        println!("🧠 Загрузка TTS (kokoro, голос «{voice}») в фоне...");
        println!("🔊 TTS-сборка: jarvis-tts-v4 + нейро-ударитель. Если этой строки нет в консоли — работает устаревший бинарь.");
        match KokoroEngine::new(&voice) {
            Ok(engine) => {
                let mut guard = shared.tts.lock().unwrap_or_else(|e| e.into_inner());
                *guard = Some(Arc::new(engine));
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

/// Смена голоса на лету: пересобирает движок в фоне.
pub fn set_voice(shared: &SharedTts, tts_failed: &Arc<AtomicBool>, voice: &str) {
    spawn_tts_loader(shared, tts_failed, voice);
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

fn tts_cache_path(text: &str, voice: &str, speed: f32) -> PathBuf {
    let mut hasher = DefaultHasher::new();

    TTS_ENGINE_VERSION.hash(&mut hasher);
    voice.hash(&mut hasher);
    ((speed * 100.0).round() as i64).hash(&mut hasher);
    text.trim().hash(&mut hasher);

    paths::tts_cache_dir().join(format!("{:016x}.wav", hasher.finish()))
}

/// Пишет моно int16 WAV (кэш синтеза).
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
) -> Option<Arc<KokoroEngine>> {
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
                    // предложение убирает паузы между соседними фразами.
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

                    // Правила — мгновенно (числа, единицы, даты словами);
                    // нейро-нормализация (LLM) — сверху, если включена и модель
                    // свободна. На вход даём уже подготовленный правилами текст:
                    // LLM не ломает грамматику дат и падежей.
                    let mut text = crate::text::normalize_for_tts(&raw_text);
                    let nn_enabled = settings
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .llm_normalize;
                    if nn_enabled && needs_neural_normalize(&raw_text) {
                        let final_text = match normalize_cache.get(&text) {
                            Some(cached) => Some(cached),
                            None => llm::try_normalize(&llm_shared, &text)
                                .inspect(|n| normalize_cache.put(text.clone(), n.clone())),
                        };
                        if let Some(n) = final_text {
                            text = n;
                        }
                    }

                    if text.is_empty() {
                        continue;
                    }

                    let voice = current_voice();
                    let speed = current_speed();
                    let can_cache = text.chars().count() <= TTS_CACHE_MAX_CHARS;
                    let cache_path = tts_cache_path(&text, &voice, speed);

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

                    let Some(kokoro) = wait_for_tts(
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

                    // Неавторегрессивная модель: вся фраза за один проход,
                    // RTF ~0.15 — воспроизведение начинается по готовности.
                    let started = Instant::now();
                    match kokoro.synthesize(&text, speed) {
                        Ok(samples) if !samples.is_empty() => {
                            if can_cache {
                                if let Err(e) = fs::create_dir_all(paths::tts_cache_dir()) {
                                    eprintln!("⚠️ Не удалось создать папку кэша речи: {e}");
                                }
                                let spec = WavSpec {
                                    channels: 1,
                                    sample_rate: (crate::tts_kokoro::SAMPLE_RATE as f32 * speed)
                                        .round() as u32,
                                    bits_per_sample: 16,
                                    sample_format: SampleFormat::Int,
                                };
                                if let Err(e) = write_wav_i16(&cache_path, spec, &samples) {
                                    eprintln!("⚠️ Не удалось сохранить кэш речи: {e}");
                                }
                            }

                            let src_rate =
                                (crate::tts_kokoro::SAMPLE_RATE as f32 * speed).round() as usize;
                            let out = if src_rate != device_rate {
                                crate::audio::resample_audio(&samples, src_rate, device_rate)
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
        assert!(KOKORO_VOICES.iter().any(|(v, _)| *v == "dima"));
    }
}
