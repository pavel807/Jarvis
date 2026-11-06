//! Голосовой гейт владельца и VAD на пайплайне speakrs.
//!
//! Библиотека speakrs (pyannote community-1 пайплайн на Rust) даёт
//! per-frame speaker embeddings и временные границы речи. Здесь она решает
//! две задачи:
//!
//! 1. **VAD**: движок регулярно спрашивает, когда в последнем окне звучала
//!    человеческая речь. Пока речь не кончилась, фраза не уходит в STT/LLM —
//!    пауза посреди предложения или тихая речь больше не обрывают её.
//! 2. **Фильтр говорящего**: профиль владельца строится из образцов wake word
//!    (train/*[jarvis]*.wav — их записывает обучение «Джарвис»), каждая новая
//!    фраза перед распознаванием сравнивается с профилем косинусным сходством.
//!    Чужой голос не доходит до LLM-цепочки и инструментов.
//!
//! Политика верификации — **fail-closed**: если профиль есть, а проверить
//! фразу не удалось (ошибка, таймаут, короткий сегмент), фраза не проходит.
//! Пропускается только случай без профиля — и тогда движок честно говорит
//! пользователю, что фильтр не активен, вместо молчаливого пропуска чужих.
//!
//! Контракт потоков: инференс занимает десятки–сотни мс, поэтому гейт живёт
//! в собственном потоке-воркере, а аудио-цикл движка никогда не блокируется
//! вычислениями. Пайплайн загружается один раз и обслуживает оба запроса.

use crate::events::{EventTx, JarvisEvent};
use crate::tts;
use crate::wake;
use anyhow::{Context as _, Result};
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

/// Минимальная длительность сегмента для верификации, секунд.
const MIN_SEGMENT_SECS: f32 = 0.8;
/// Частота дискретизации профиля и сегментов.
const TARGET_RATE: usize = wake::SAMPLE_RATE;

/// Вердикт фильтра говорящего.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Владелец — фраза проходит.
    Owner,
    /// Чужой — фраза отбрасывается.
    Stranger,
    /// Профиля нет (образцы не записаны): фильтр не активен. Фраза проходит,
    /// движок один раз за сессию сообщает пользователю о нерабочем фильтре.
    NoProfile,
    /// Проверить не удалось: при включённом фильтре фраза не проходит
    /// (fail-closed), движок просит повторить.
    Unverified(&'static str),
}

/// Запрос к воркеру speakrs.
pub enum SpeakerQuery {
    /// Пересобрать профиль из образцов владельца.
    Rebuild,
    /// Верифицировать сегмент; f32-порог косинусного сходства.
    Verify {
        samples: Vec<f32>,
        threshold: f32,
    },
    /// VAD: где в окне закончилась последняя человеческая речь.
    SpeechProbe { id: u64, samples: Vec<f32> },
}

/// Ответ VAD на [`SpeakerQuery::SpeechProbe`].
pub struct SpeechProbeResult {
    /// Идентификатор запроса: движок отбрасывает устаревшие ответы.
    pub id: u64,
    /// Длительность окна в секундах.
    pub total_secs: f32,
    /// Конец последней речи от начала окна; `None` — речи нет.
    pub last_speech_end: Option<f32>,
}

/// Ответ воркера speakrs.
pub enum SpeakerReply {
    /// Вердикт фильтра говорящего.
    Verified(Verdict),
    /// Результат VAD; `None` — пайплайн недоступен или произошла ошибка.
    Speech(Option<SpeechProbeResult>),
}

/// Голосовой гейт: пайплайн speakrs + профиль владельца.
pub struct SpeakerGate {
    pipeline: OwnedDiarizationPipeline,
    /// Усреднённый нормализованный эмбеддинг владельца.
    profile: Option<Vec<f32>>,
    /// Сколько образцов использовано в профиле (для логов).
    profile_records: usize,
}

impl SpeakerGate {
    /// Загрузка пайплайна. На macOS — CoreML (нативный, FP32), при недоступности — CPU.
    pub fn load(events: &crate::events::EventTx) -> Option<Self> {
        // CoreML variant существует только при feature `coreml`, которая
        // подключается для macOS в Cargo.toml. Остальные ОС компилируют
        // чистый CPU-путь и не тянут Apple-фреймворки.
        #[cfg(target_os = "macos")]
        let modes: &[(&str, ExecutionMode)] = &[
            ("CoreML", ExecutionMode::CoreMl),
            ("CPU", ExecutionMode::Cpu),
        ];
        #[cfg(not(target_os = "macos"))]
        let modes: &[(&str, ExecutionMode)] = &[("CPU", ExecutionMode::Cpu)];
        for &(label, mode) in modes {
            match OwnedDiarizationPipeline::from_pretrained(mode) {
                Ok(pipeline) => {
                    println!("🎙️ speakrs: VAD и фильтр говорящего загружены ({label})");
                    return Some(Self {
                        pipeline,
                        profile: None,
                        profile_records: 0,
                    });
                }
                Err(e) => {
                    println!("🎙️ speakrs: {label} недоступен: {e}");
                }
            }
        }
        let _ = events.send(JarvisEvent::Log {
            message: "VAD speakrs и фильтр говорящего недоступны (пайплайн не загрузился)".into(),
        });
        None
    }

    /// Пересобрать профиль из записей владельца: короткие образцы wake word
    /// (`train/jarvis`, метка `jarvis`) и диктанты (`train/voice`, метка
    /// `voice`). Возвращает число использованных записей.
    pub fn rebuild_profile(&mut self, events: &EventTx) -> usize {
        self.profile = None;
        self.profile_records = 0;

        let (train_dir, _) = crate::wake_train::sample_dirs();
        let mut records = tts::list_wav_samples(&train_dir)
            .into_iter()
            .filter(|(path, _)| {
                tts::label_of_sample(&path.file_name().unwrap_or_default().to_string_lossy())
                    != "none"
            })
            .collect::<Vec<_>>();
        records.extend(
            tts::list_wav_samples(&crate::wake_train::voice_samples_dir())
                .into_iter()
                .filter(|(path, _)| {
                    tts::label_of_sample(
                        &path.file_name().unwrap_or_default().to_string_lossy(),
                    ) == "voice"
                }),
        );

        if records.is_empty() {
            let _ = events.send(JarvisEvent::Log {
                message: "Профиль голоса: образцов владельца нет — фильтр пропускает всё".into(),
            });
            return 0;
        }

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for (path, _) in &records {
            let read = read_wav_16k(path);
            let Ok(samples) = read else { continue };
            // Тишина по краям записи размывает средний эмбеддинг: сравнение
            // идёт только по звучавшей части. У длинного диктанта берём
            // середину: эмбеддинг записи на 3+ минуты нестабилен, а первых
            // 60 секунд звучавшей речи достаточно для точного среднего.
            let voiced = trim_silence(&samples, TARGET_RATE);
            let voiced = if voiced.len() > TARGET_RATE * 60 {
                let mid = voiced.len() / 2;
                let half = TARGET_RATE * 30;
                &voiced[mid - half..mid + half]
            } else {
                voiced
            };
            if voiced.len() < (MIN_SEGMENT_SECS * TARGET_RATE as f32) as usize {
                continue;
            }
            if let Some(mean) = self.segment_embedding(voiced) {
                vectors.push(mean);
            }
        }

        if vectors.is_empty() {
            let _ = events.send(JarvisEvent::Log {
                message: "Профиль голоса: образцы слишком коротки для анализа".into(),
            });
            return 0;
        }

        self.profile = Some(normalize(&mean_of(&vectors)));
        self.profile_records = vectors.len();
        let _ = events.send(JarvisEvent::Log {
            message: format!(
                "Профиль голоса владельца построен: {} записей",
                self.profile_records
            ),
        });
        self.profile_records
    }

    /// Верификация сегмента: [`Verdict::Owner`] — голос владельца,
    /// [`Verdict::Stranger`] — чужой, [`Verdict::Unverified`] — проверить
    /// не удалось (fail-closed), [`Verdict::NoProfile`] — профиля нет.
    pub fn verify(&mut self, samples: &[f32], threshold: f32) -> Verdict {
        let Some(profile) = self.profile.clone() else {
            return Verdict::NoProfile;
        };
        if samples.len() < (MIN_SEGMENT_SECS * TARGET_RATE as f32) as usize {
            return Verdict::Unverified("сегмент короче порога верификации");
        }
        let voiced = trim_silence(samples, TARGET_RATE);
        let Some(mean) = self.segment_embedding(voiced) else {
            return Verdict::Unverified("эмбеддинг сегмента не посчитался");
        };
        let cosine = dot(&profile, &normalize(&mean));
        if cosine >= threshold {
            Verdict::Owner
        } else {
            Verdict::Stranger
        }
    }

    /// VAD через сегментацию speakrs: конец последней речи в окне.
    ///
    /// `None` внутри `Ok` означает, что модель не нашла речи; `Err` —
    /// пайплайн не смог её обработать (движок тогда откатывается на RMS).
    pub fn last_speech_end(&mut self, samples: &[f32]) -> Result<Option<f32>> {
        if samples.is_empty() {
            return Ok(None);
        }
        let result = self
            .pipeline
            .run(samples)
            .context("speakrs: ошибка диаризации VAD-окна")?;
        // `segments` уже склеены и отсортированы по началу; берём самый
        // поздний конец, потому что в окне могут быть пересекающиеся реплики.
        Ok(result
            .segments
            .iter()
            .map(|segment| segment.end)
            .max_by(f64::total_cmp)
            .map(|end| end as f32))
    }

    /// Усреднённый эмбеддинг сегмента через пайплайн speakrs.
    fn segment_embedding(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        let result = match self.pipeline.run(samples) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("🎙️ speakrs: ошибка диаризации сегмента: {e:?}");
                return None;
            }
        };
        // ChunkEmbeddings: Array3 [chunk][frame][dim]. Собираем все фреймы
        // всех чанков в один пул и усредняем.
        let embeddings = &result.embeddings;
        let mut pool: Vec<f32> = Vec::new();
        let mut dim = 0usize;
        for chunk in embeddings.rows() {
            for frame in chunk.rows() {
                if dim == 0 {
                    dim = frame.len();
                }
                if frame.len() == dim {
                    pool.extend(frame.iter().copied());
                }
            }
        }
        if pool.is_empty() || dim == 0 {
            return None;
        }
        let count = pool.len() / dim;
        let mut mean = vec![0.0f32; dim];
        for chunk in pool.chunks(dim) {
            for (m, v) in mean.iter_mut().zip(chunk) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= count as f32;
        }
        Some(mean)
    }

    /// Профиль построен?
    pub fn has_profile(&self) -> bool {
        self.profile.is_some()
    }

    /// Записей в профиле.
    pub fn profile_records(&self) -> usize {
        self.profile_records
    }
}

/// Поток-воркер гейта: единственный владелец пайплайна.
/// `invalidate` — флаг пересборки профиля (после записи новых образцов).
pub fn spawn_speaker_worker(
    gate: Option<SpeakerGate>,
    invalidate: Arc<AtomicBool>,
) -> (SyncSender<SpeakerQuery>, Receiver<SpeakerReply>) {
    let (tx, rx) = std::sync::mpsc::sync_channel::<SpeakerQuery>(4);
    let (out_tx, out_rx) = std::sync::mpsc::channel::<SpeakerReply>();

    std::thread::spawn(move || {
        // gate может отсутствовать (пайплайн не загрузился): верификация
        // невозможна — fail-closed, фразы не проходят (движок вежливо просит
        // повторить или отключить фильтр). VAD при этом тоже недоступен.
        let mut gate = gate;
        while let Ok(query) = rx.recv() {
            let answer = match (&mut gate, query) {
                (None, SpeakerQuery::Rebuild) => SpeakerReply::Verified(Verdict::Unverified(
                    "пайплайн проверки голоса не загрузился",
                )),
                (None, SpeakerQuery::Verify { .. }) => SpeakerReply::Verified(Verdict::Unverified(
                    "пайплайн проверки голоса не загрузился",
                )),
                (None, SpeakerQuery::SpeechProbe { .. }) => SpeakerReply::Speech(None),
                (Some(_), SpeakerQuery::Rebuild) => SpeakerReply::Verified(Verdict::Owner),
                (Some(g), SpeakerQuery::Verify { samples, threshold }) => {
                    if invalidate.swap(false, Ordering::SeqCst) {
                        let events = dummy_events();
                        g.rebuild_profile(&events);
                    }
                    if !g.has_profile() {
                        // Профиль ещё не собран — собираем при первой верификации.
                        let events = dummy_events();
                        g.rebuild_profile(&events);
                    }
                    let verdict = if g.has_profile() {
                        g.verify(&samples, threshold)
                    } else {
                        Verdict::NoProfile
                    };
                    SpeakerReply::Verified(verdict)
                }
                (Some(g), SpeakerQuery::SpeechProbe { id, samples }) => {
                    let total_secs = samples.len() as f32 / TARGET_RATE as f32;
                    match g.last_speech_end(&samples) {
                        Ok(last_speech_end) => SpeakerReply::Speech(Some(SpeechProbeResult {
                            id,
                            total_secs,
                            last_speech_end,
                        })),
                        Err(e) => {
                            eprintln!("🎙️ VAD speakrs: {e:#}");
                            SpeakerReply::Speech(None)
                        }
                    }
                }
            };
            let _ = out_tx.send(answer);
        }
    });

    (tx, out_rx)
}

/// Загрузить гейт (фоново, при старте движка). None — фильтр недоступен.
pub fn load_gate(events: &crate::events::EventTx) -> Option<SpeakerGate> {
    SpeakerGate::load(events)
}

fn dummy_events() -> crate::events::EventTx {
    let (tx, _) = std::sync::mpsc::channel();
    tx
}

/// Читает WAV как моно f32 на 16 кГц (ресемплирование при необходимости).
pub fn read_wav_16k(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Не удалось открыть {:?}", path))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut samples: Vec<f32> = Vec::new();
    for frame in reader.samples::<i32>() {
        let raw = frame.with_context(|| format!("Битый сэмпл в {:?}", path))?;
        // hound возвращает i32 для любых битностей — нормируем по битности.
        let max = 1i64 << (spec.bits_per_sample.saturating_sub(1));
        samples.push(raw as f32 / max as f32);
    }
    // Моно: усредняем каналы.
    if channels > 1 {
        let mono: Vec<f32> = samples
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect();
        samples = mono;
    }
    if spec.sample_rate != TARGET_RATE as u32 {
        samples = crate::audio::resample_audio(&samples, spec.sample_rate as usize, TARGET_RATE);
    }
    Ok(samples)
}

fn mean_of(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x;
        }
    }
    let count = vectors.len().max(1) as f32;
    for m in &mut mean {
        *m /= count;
    }
    mean
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Запас по краям после обрезки тишины, секунд.
const TRIM_MARGIN_SECS: f32 = 0.12;
/// Окно RMS-огибающей для чистки тишины, секунд.
const TRIM_WINDOW_SECS: f32 = 0.03;
/// Доля от пика огибающей, ниже которой окно считается тишиной.
const TRIM_PEAK_SHARE: f32 = 0.06;

/// Обрезает тишину по краям записи по RMS-огибающей.
///
/// Профиль и верифицируемая фраза приходят с паузами (пре-ролл, нулевые
/// паддинги, хвостовая тишина): усреднение эмбеддингов по всему сегменту
/// размывало голос фреймами тишины. Возвращается срез от первого до
/// последнего «звучащего» окна с запасом [`TRIM_MARGIN_SECS`]. Полностью
/// тихая запись возвращается как есть — обрезать до нуля нельзя.
fn trim_silence(samples: &[f32], rate: usize) -> &[f32] {
    if samples.is_empty() {
        return samples;
    }
    let window = ((TRIM_WINDOW_SECS * rate as f32) as usize).max(1);
    if samples.len() <= window {
        return samples;
    }
    let mut peak = 0.0f32;
    let envelope: Vec<f32> = samples
        .chunks(window)
        .map(|chunk| {
            let sum: f32 = chunk.iter().map(|x| x * x).sum();
            let rms = (sum / chunk.len() as f32).sqrt();
            peak = peak.max(rms);
            rms
        })
        .collect();
    // Цифровая тишина/почти тишина: не режем, верификации нечего решать.
    if peak < 1e-4 {
        return samples;
    }
    let threshold = peak * TRIM_PEAK_SHARE;
    let Some(first) = envelope.iter().position(|&rms| rms >= threshold) else {
        return samples;
    };
    let last = envelope
        .iter()
        .rposition(|&rms| rms >= threshold)
        .unwrap_or(first);
    let margin = (TRIM_MARGIN_SECS * rate as f32) as usize;
    let start = first.saturating_mul(window).saturating_sub(margin);
    let end = ((last + 1) * window + margin).min(samples.len());
    &samples[start..end]
}

/// Записи владельца для UI-подсказки: образцы wake word + диктанты.
pub fn profile_sources() -> Vec<(String, f64)> {
    let (train_dir, _) = crate::wake_train::sample_dirs();
    let mut sources = tts::list_wav_samples(&train_dir)
        .into_iter()
        .filter(|(path, _)| {
            tts::label_of_sample(&path.file_name().unwrap_or_default().to_string_lossy()) != "none"
        })
        .map(|(path, duration)| {
            (path.file_name().unwrap_or_default().to_string_lossy().into_owned(), duration)
        })
        .collect::<Vec<_>>();
    sources.extend(
        tts::list_wav_samples(&crate::wake_train::voice_samples_dir())
            .into_iter()
            .filter(|(path, _)| {
                tts::label_of_sample(&path.file_name().unwrap_or_default().to_string_lossy())
                    == "voice"
            })
            .map(|(path, duration)| {
                (path.file_name().unwrap_or_default().to_string_lossy().into_owned(), duration)
            }),
    );
    sources
}

/// Путь к файлу профиля не нужен: профиль пересобирается из образцов.
pub fn profile_available() -> bool {
    let (train_dir, _) = crate::wake_train::sample_dirs();
    paths_exist(&train_dir)
}

fn paths_exist(dir: &Path) -> bool {
    dir.is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = 16_000;

    fn tone(freq: f32, secs: f32) -> Vec<f32> {
        let n = (secs * RATE as f32) as usize;
        (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / RATE as f32).sin())
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * RATE as f32) as usize]
    }

    #[test]
    fn trims_leading_and_trailing_silence() {
        let mut samples = silence(0.5);
        samples.extend(tone(220.0, 1.0));
        samples.extend(silence(0.7));
        let trimmed = trim_silence(&samples, RATE);
        // Запас 120 мс с обеих сторон: 1.0 с звука → ~1.0–1.3 с среза.
        let secs = trimmed.len() as f32 / RATE as f32;
        assert!(
            (0.95..=1.35).contains(&secs),
            "ожидался срез около 1 с с запасом, получено {secs:.2} с"
        );
        // Границы среза — внутри «звуковой» зоны: тишина ушла.
        assert!(trimmed.len() < samples.len());
    }

    #[test]
    fn all_silence_is_not_trimmed_to_nothing() {
        let samples = silence(1.0);
        assert_eq!(trim_silence(&samples, RATE).len(), samples.len());
    }

    #[test]
    fn short_and_empty_inputs_pass_through() {
        assert!(trim_silence(&[], RATE).is_empty());
        let tiny = vec![0.1f32; 100];
        assert_eq!(trim_silence(&tiny, RATE).len(), 100);
    }

    #[test]
    fn verdict_threshold_separates_owner_and_stranger() {
        // Порог — простое сравнение косинуса: сонаправленный вектор проходит,
        // ортогональный — нет.
        let profile = normalize(&[1.0, 0.0, 0.0]);
        let owner = normalize(&[0.9, 0.1, 0.0]);
        let stranger = normalize(&[0.0, 1.0, 0.0]);
        assert!(dot(&profile, &owner) >= 0.6);
        assert!(dot(&profile, &stranger) < 0.6);
    }
}