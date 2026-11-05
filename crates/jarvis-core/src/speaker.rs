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
//!    Чужая речь не доходит ни до FastConformer-транскрипта в LLM-цепочку,
//!    ни до инструментов.
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
    /// Вердикт фильтра говорящего: `Some(true)` — владелец.
    Verified(Option<bool>),
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
        let modes: [(&str, ExecutionMode); 2] = [
            ("CoreML", ExecutionMode::CoreMl),
            ("CPU", ExecutionMode::Cpu),
        ];
        for (label, mode) in modes {
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

    /// Пересобрать профиль из wake word-образцов владельца.
    /// Возвращает число использованных записей.
    pub fn rebuild_profile(&mut self, events: &EventTx) -> usize {
        self.profile = None;
        self.profile_records = 0;

        let (train_dir, _) = crate::wake_train::sample_dirs();
        let records = tts::list_wav_samples(&train_dir)
            .into_iter()
            .filter(|(path, _)| {
                tts::label_of_sample(&path.file_name().unwrap_or_default().to_string_lossy())
                    != "none"
            })
            .collect::<Vec<_>>();

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
            if samples.len() < (MIN_SEGMENT_SECS * TARGET_RATE as f32) as usize {
                continue;
            }
            if let Some(mean) = self.segment_embedding(&samples) {
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

    /// Верификация сегмента: Some(true) — голос владельца, Some(false) — чужой,
    /// None — фильтр не может решить (нет профиля, сегмент слишком короткий).
    pub fn verify(&mut self, samples: &[f32], threshold: f32) -> Option<bool> {
        let profile = self.profile.clone()?;
        if samples.len() < (MIN_SEGMENT_SECS * TARGET_RATE as f32) as usize {
            return None;
        }
        let mean = self.segment_embedding(samples)?;
        let normalized = normalize(&mean);
        let cosine = dot(&profile, &normalized);
        Some(cosine >= threshold)
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
        // gate может отсутствовать (пайплайн не загрузился): тогда всё пропускаем.
        let mut gate = gate;
        while let Ok(query) = rx.recv() {
            let answer = match (&mut gate, query) {
                (None, SpeakerQuery::Rebuild) => SpeakerReply::Verified(None),
                (None, SpeakerQuery::Verify { .. }) => SpeakerReply::Verified(None),
                (None, SpeakerQuery::SpeechProbe { .. }) => SpeakerReply::Speech(None),
                (Some(_), SpeakerQuery::Rebuild) => SpeakerReply::Verified(None),
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
                    SpeakerReply::Verified(g.verify(&samples, threshold))
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
fn read_wav_16k(path: &Path) -> Result<Vec<f32>> {
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

/// Каталог образцов владельца — для UI-подсказки.
pub fn profile_sources() -> Vec<(String, f64)> {
    let (train_dir, _) = crate::wake_train::sample_dirs();
    tts::list_wav_samples(&train_dir)
        .into_iter()
        .filter(|(path, _)| {
            tts::label_of_sample(&path.file_name().unwrap_or_default().to_string_lossy()) != "none"
        })
        .map(|(path, duration)| {
            (path.file_name().unwrap_or_default().to_string_lossy().into_owned(), duration)
        })
        .collect()
}

/// Путь к файлу профиля не нужен: профиль пересобирается из образцов.
pub fn profile_available() -> bool {
    let (train_dir, _) = crate::wake_train::sample_dirs();
    paths_exist(&train_dir)
}

fn paths_exist(dir: &Path) -> bool {
    dir.is_dir()
}
