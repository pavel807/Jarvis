//! TTS: Silero v5 на candle. Синтез речи и воспроизведение любых WAV,
//! с эмиссией `AudioLevel` для анимации UI-сферы.

use crate::audio::OutputPlayer;
use crate::events::{EventTx, JarvisEvent};
use crate::llm::{self, SharedLlm};
use crate::paths;
use crate::settings::Settings;
use crate::tts_silero::SAMPLE_RATE as SILERO_SAMPLE_RATE;
pub use crate::tts_silero::SILERO_VOICES;
use crate::tts_silero::SileroEngine;
use anyhow::{Context, Result, bail};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::collections::{HashMap, VecDeque, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// v9: ударитель и омографы переехали с ONNX Runtime на candle
/// (плюс выбор ударной гласной по максимуму вероятности) — кэш v8
/// местами звучал с ударением не на той гласной.
const TTS_ENGINE_VERSION: &str = "jarvis-tts-v9-silero";
const TTS_CACHE_MAX_CHARS: usize = 200;
const TTS_WAIT_TIMEOUT_SECS: u64 = 600;
pub const TTS_CACHE_MAX_BYTES: u64 = 300 * 1024 * 1024;
const LEVEL_WINDOW_MS: u64 = 40;

static TTS_SPEED: RwLock<f32> = RwLock::new(1.0);

pub fn set_current_speed(speed: f32) {
    *TTS_SPEED.write().unwrap_or_else(|e| e.into_inner()) = speed.clamp(0.5, 1.5);
}

pub fn current_speed() -> f32 {
    *TTS_SPEED.read().unwrap_or_else(|e| e.into_inner())
}

static TTS_VOICE: RwLock<String> = RwLock::new(String::new());

pub fn current_voice() -> String {
    TTS_VOICE.read().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn set_current_voice(voice: String) {
    *TTS_VOICE.write().unwrap_or_else(|e| e.into_inner()) = voice;
}

pub struct SharedTtsState {
    pub tts: Mutex<Option<Arc<SileroEngine>>>,
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

impl Default for SharedTtsState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn spawn_tts_loader(shared: &SharedTts, tts_failed: &Arc<AtomicBool>, voice: &str) {
    let shared = Arc::clone(shared);
    let tts_failed = Arc::clone(tts_failed);
    let voice = voice.to_string();
    set_current_voice(voice.clone());

    std::thread::spawn(move || {
        println!("🧠 Загрузка TTS (silero, голос «{voice}») в фоне...");
        match SileroEngine::new(&voice) {
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

pub fn set_voice(shared: &SharedTts, tts_failed: &Arc<AtomicBool>, voice: &str) {
    spawn_tts_loader(shared, tts_failed, voice);
}

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
        println!(
            "🧹 Кэш TTS уменьшен на {:.1} МБ",
            removed as f64 / 1048576.0
        );
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

fn write_wav_i16(path: &Path, spec: WavSpec, samples: &[f32]) -> Result<()> {
    let mut writer =
        WavWriter::create(path, spec).with_context(|| format!("Не удалось создать {:?}", path))?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        WavReader::open(path).with_context(|| format!("Не удалось открыть {:?}", path))?;
    let spec = reader.spec();

    let raw: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / 32768.0)
            .collect(),
        SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    let mono = if spec.channels > 1 {
        crate::audio::downmix_f32(&raw, spec.channels as usize)
    } else {
        raw
    };

    Ok((mono, spec.sample_rate))
}

fn peak_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let win = 2400usize;
    samples
        .chunks(win)
        .map(crate::audio::calculate_rms)
        .fold(0.0f32, f32::max)
}

fn wait_for_tts(
    shared_tts: &SharedTts,
    tts_failed: &AtomicBool,
    timeout: Duration,
) -> Option<Arc<SileroEngine>> {
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
    /// Превью голоса в интерфейсе: фраза синтезируется указанным диктором,
    /// но выбранный голос приложения не меняет. В кэш не пишется.
    PreviewVoice { text: String, voice: String },
    Stop,
}

fn needs_neural_normalize(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_digit())
        || text.contains(['%', '°', '₽', '/'])
        || text.contains("м/с")
}

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
        if self.map.len() >= self.cap
            && let Some(old) = self.order.front().cloned()
        {
            self.order.remove(0);
            self.map.remove(&old);
        }
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
    }
}

fn play_via(output: &Option<OutputPlayer>, path: &str, stop_flag: &AtomicBool) -> Result<()> {
    match output {
        Some(player) => player.play_wav(path, stop_flag),
        None => bail!("Выходной аудиопоток недоступен"),
    }
}

fn spawn_playback_level_emitter(
    events: EventTx,
    samples: Vec<f32>,
    sample_rate: usize,
    stop_flag: Arc<AtomicBool>,
) {
    if samples.is_empty() || sample_rate == 0 {
        return;
    }

    let peak = peak_rms(&samples).max(1e-4);
    let gain = (0.35 / peak).clamp(0.5, 8.0);

    std::thread::spawn(move || {
        let window = (sample_rate as u64 * LEVEL_WINDOW_MS / 1000).max(1) as usize;
        let total = samples.len();
        let start = Instant::now();
        let mut pos = 0usize;
        let mut held = 0.0f32;
        const DECAY: f32 = 0.72;

        while pos < total {
            if stop_flag.load(Ordering::SeqCst) {
                break;
            }

            let end = (pos + window).min(total);
            let slice = &samples[pos..end];

            let rms = crate::audio::calculate_rms(slice);
            let peak_abs = slice.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            let raw = ((rms * gain).max(peak_abs * gain * 0.75)).min(1.0);

            held = if raw > held { raw } else { held * DECAY };

            let _ = events.send(JarvisEvent::AudioLevel {
                level: held,
                source: "tts".into(),
            });

            pos = end;

            let target = Duration::from_millis(
                (pos as u64 * 1000 / sample_rate as u64).max(LEVEL_WINDOW_MS),
            );
            let elapsed = start.elapsed();
            if target > elapsed {
                std::thread::sleep(target - elapsed);
            }
        }

        let _ = events.send(JarvisEvent::AudioLevel {
            level: 0.0,
            source: "tts".into(),
        });
    });
}

fn play_wav_with_level(
    events: &EventTx,
    output: &Option<OutputPlayer>,
    path: &str,
    stop_flag: &Arc<AtomicBool>,
) {
    let Ok((mono, src_rate)) = read_wav_mono(Path::new(path)) else {
        if let Err(e) = play_via(output, path, stop_flag) {
            eprintln!("⚠️ Ошибка воспроизведения файла ({path}): {e}");
        }
        return;
    };

    let Some(player) = output else {
        return;
    };

    let device_rate = player.device_sample_rate();
    let out = if src_rate as usize != device_rate {
        crate::audio::resample_audio(&mono, src_rate as usize, device_rate)
    } else {
        mono
    };

    spawn_playback_level_emitter(
        events.clone(),
        out.clone(),
        device_rate,
        Arc::clone(stop_flag),
    );
    player.enqueue(&out);
    player.wait_drain(stop_flag);
}

/// Доигрывает готовые сэмплы: фейд в конце, ресемплинг под устройство,
/// эмиттер уровня, хвостовая пауза по последнему знаку. Общий финал
/// озвучки ответа и превью голоса — чтобы пауза и затухание звучали
/// одинаково в обоих путях.
fn enqueue_playback(
    events: &EventTx,
    player: &OutputPlayer,
    stop_flag: &Arc<AtomicBool>,
    device_rate: usize,
    mut samples: Vec<f32>,
    speed: f32,
    text: &str,
) {
    let fade_len = (device_rate as f32 * 0.015) as usize;
    let n = samples.len();
    if n > fade_len && fade_len > 0 {
        for i in 0..fade_len {
            let k = 1.0 - (i as f32 / fade_len as f32);
            samples[n - 1 - i] *= k;
        }
    }

    let src_rate = (SILERO_SAMPLE_RATE as f32 * speed).round() as usize;
    let out = if src_rate != device_rate {
        crate::audio::resample_audio(&samples, src_rate, device_rate)
    } else {
        samples
    };

    spawn_playback_level_emitter(
        events.clone(),
        out.clone(),
        device_rate,
        Arc::clone(stop_flag),
    );
    player.enqueue(&out);

    let pause_ms = if text.ends_with(['.', '!', '?']) {
        140
    } else if text.ends_with([',', ';', ':']) {
        80
    } else {
        40
    };
    let tail_samples = (device_rate as f32 * pause_ms as f32 / 1000.0) as usize;
    player.enqueue(&vec![0.0f32; tail_samples]);

    // Небольшой запас тишины в конце куска: плеер проигрывает его перед
    // синтезом следующего, так что 180 мс превращались в заметную паузу.
    let grace = (device_rate as f32 * 0.080) as usize;
    player.enqueue(&vec![0.0f32; grace]);
    player.wait_drain(stop_flag);
}

pub fn spawn_audio_player(
    events: EventTx,
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
                    let _ = events.send(JarvisEvent::AudioLevel {
                        level: 0.0,
                        source: "tts".into(),
                    });
                }

                AudioMessage::PlayWav(path) => {
                    if stop_audio_flag.load(Ordering::SeqCst) {
                        continue;
                    }
                    is_speaking.store(true, Ordering::SeqCst);
                    play_wav_with_level(&events, &output, &path, &stop_audio_flag);
                    is_speaking.store(false, Ordering::SeqCst);
                }

                AudioMessage::SynthesizeAndPlay(first) => {
                    if stop_audio_flag.load(Ordering::SeqCst) {
                        continue;
                    }
                    if first.trim().is_empty() {
                        continue;
                    }

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
                                deferred_wav = Some(path);
                                break;
                            }
                            // Превью во время набора ответа теряется: ответ
                            // важнее, а вернуть сообщение в очередь неоткуда.
                            Ok(AudioMessage::PreviewVoice { .. }) => break,
                            Ok(AudioMessage::Stop) => {
                                stop_audio_flag.store(true, Ordering::SeqCst);
                                while rx.try_recv().is_ok() {}
                                if let Some(o) = &output {
                                    o.clear();
                                }
                                stop_audio_flag.store(false, Ordering::SeqCst);
                                is_speaking.store(false, Ordering::SeqCst);
                                let _ = events.send(JarvisEvent::AudioLevel {
                                    level: 0.0,
                                    source: "tts".into(),
                                });
                                continue 'player;
                            }
                            Err(_) => break,
                        }
                    }

                    let raw_text = raws
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let needs_dot =
                                i + 1 < raws.len() && !s.ends_with(['.', '!', '?', ',', ';', ':']);
                            if needs_dot {
                                format!("{s}.")
                            } else {
                                s.clone()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");

                    // Полный конвейер устной речи: Markdown-структура →
                    // произношения названий/аббревиатур → числа и единицы.
                    let mut text = crate::tts_normalizer::normalize(&raw_text);
                    let nn_enabled = settings
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .llm_normalize;
                    // Нейросетевую нормализацию зовём только если после
                    // правил в тексте остались цифры/символы, которые синтез
                    // прочитает не по-русски. Иначе это лишний проход LLM
                    // (около секунды) прямо перед озвучкой, да ещё и в одном
                    // потоке с плеером.
                    if nn_enabled && needs_neural_normalize(&text) {
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
                        is_speaking.store(true, Ordering::SeqCst);
                        let cache_path_str = cache_path.to_string_lossy().to_string();
                        play_wav_with_level(&events, &output, &cache_path_str, &stop_audio_flag);

                        if let Some(player) = &output {
                            let device_rate = player.device_sample_rate();
                            // Небольшой запас тишины в конце куска: плеер
                            // проигрывает его перед синтезом следующего, так
                            // что 180 мс превращались в заметную паузу.
                            let grace = (device_rate as f32 * 0.080) as usize;
                            player.enqueue(&vec![0.0f32; grace]);
                            player.wait_drain(&stop_audio_flag);
                        }
                        is_speaking.store(false, Ordering::SeqCst);
                        continue;
                    }

                    let Some(engine) = wait_for_tts(
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

                    match engine.synthesize(&text, speed) {
                        Ok(samples) if !samples.is_empty() => {
                            if can_cache {
                                if let Err(e) = fs::create_dir_all(paths::tts_cache_dir()) {
                                    eprintln!("⚠️ Не удалось создать папку кэша речи: {e}");
                                }
                                let spec = WavSpec {
                                    channels: 1,
                                    sample_rate: (SILERO_SAMPLE_RATE as f32 * speed).round() as u32,
                                    bits_per_sample: 16,
                                    sample_format: SampleFormat::Int,
                                };
                                if let Err(e) = write_wav_i16(&cache_path, spec, &samples) {
                                    eprintln!("⚠️ Не удалось сохранить кэш речи: {e}");
                                }
                            }

                            enqueue_playback(
                                &events,
                                player,
                                &stop_audio_flag,
                                device_rate,
                                samples,
                                speed,
                                &text,
                            );
                        }
                        Ok(_) => eprintln!("⚠️ Движок вернул пустое аудио"),
                        Err(e) => {
                            eprintln!("⚠️ Ошибка синтеза речи: {e}");
                            if can_cache {
                                let _ = fs::remove_file(&cache_path);
                            }
                        }
                    }

                    if let Some(p) = deferred_wav {
                        play_wav_with_level(&events, &output, &p, &stop_audio_flag);
                    }

                    if stop_audio_flag.load(Ordering::SeqCst) && can_cache {
                        let _ = fs::remove_file(&cache_path);
                    }

                    is_speaking.store(false, Ordering::SeqCst);
                }

                AudioMessage::PreviewVoice { text, voice } => {
                    if stop_audio_flag.load(Ordering::SeqCst) {
                        continue;
                    }

                    let text = crate::text::normalize_for_tts(text.trim());
                    if text.is_empty() {
                        continue;
                    }

                    let Some(engine) = wait_for_tts(
                        &tts_shared,
                        tts_failed.as_ref(),
                        Duration::from_secs(TTS_WAIT_TIMEOUT_SECS),
                    ) else {
                        eprintln!("⚠️ TTS не готов. Превью не озвучено: \"{text}\"");
                        continue;
                    };
                    let Some(player) = &output else {
                        continue;
                    };

                    let device_rate = player.device_sample_rate();
                    let speed = current_speed();
                    is_speaking.store(true, Ordering::SeqCst);
                    match engine.synthesize_with(&voice, &text, speed) {
                        Ok(samples) if !samples.is_empty() => {
                            enqueue_playback(
                                &events,
                                player,
                                &stop_audio_flag,
                                device_rate,
                                samples,
                                speed,
                                &text,
                            );
                        }
                        Ok(_) => eprintln!("⚠️ Превью голоса вернуло пустое аудио"),
                        Err(e) => eprintln!("⚠️ Ошибка превью голоса: {e}"),
                    }
                    is_speaking.store(false, Ordering::SeqCst);
                }
            }
        }
    });

    tx
}

pub fn wav_duration_secs(path: &Path) -> Option<f64> {
    let reader = WavReader::open(path).ok()?;
    let spec = reader.spec();
    Some(reader.duration() as f64 / spec.sample_rate as f64)
}

pub fn list_wav_samples(dir: &Path) -> Vec<(PathBuf, f64)> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for e in entries.filter_map(|e| e.ok()) {
        let p = e.path();
        if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav"))
            && let Some(d) = wav_duration_secs(&p)
        {
            out.push((p, d));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

pub fn label_of_sample(file_name: &str) -> String {
    if let Some(start) = file_name.find('[')
        && let Some(end) = file_name[start..].find(']')
    {
        return file_name[start + 1..start + end].to_string();
    }
    "none".to_string()
}

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
    fn peak_rms_finds_loudest_window() {
        let mut samples = vec![0.0f32; 4800];
        for s in samples.iter_mut().skip(2400) {
            *s = 0.5;
        }
        let p = peak_rms(&samples);
        assert!(p > 0.4 && p < 0.6, "peak_rms = {p}");
    }
}
