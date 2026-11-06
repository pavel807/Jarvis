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
pub const MIN_SEGMENT_SECS: f32 = 0.8;
/// Частота дискретизации профиля и сегментов.
const TARGET_RATE: usize = wake::SAMPLE_RATE;

/// Каноническое окно верификации, секунд.
///
/// Эмбеддинги speakrs совместимы ТОЛЬКО между запусками одной длины:
/// две длинные записи одного голоса дают ~0.89, два коротких куска ~0.95,
/// а короткий кусок против длинной записи того же голоса ~0.03 (диагностика
/// `examples/speaker_probe.rs`). Поэтому и профиль, и проверяемая фраза
/// режутся на изолированные окна [`WINDOW_SECS`] — каждый прогон пайплайна
/// получает ровно одну и ту же длину, и все сравнения в одном домене.
const WINDOW_SECS: f32 = 3.0;
const WINDOW_SAMPLES: usize = (WINDOW_SECS * wake::SAMPLE_RATE as f32) as usize;

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
    /// Верифицировать сегмент; f32-порог косинусного сходства. `id` —
    /// метка запроса: движок различает ответы на проверку фразы и фоновые
    /// проверки перебивания (устаревшие ответы пропускаются по id).
    Verify {
        id: u64,
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
    /// Вердикт фильтра говорящего; `id` — метка запроса [`SpeakerQuery::Verify`].
    Verified { id: u64, verdict: Verdict },
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
    /// Загрузка пайплайна. Предпочтение — CPU: вся наша нагрузка после
    /// оконной верификации — короткие 3-с прогоны, а замер
    /// (`examples/speaker_bench.rs`) на этой длине даёт CPU ~98 мс против
    /// CoreML ~229 мс (диспетч-оверхед CoreML на коротком входе, тот же
    /// эффект, что у FastConformer). CoreML остаётся запасным.
    pub fn load(events: &crate::events::EventTx) -> Option<Self> {
        // Обе фичи (cpu/coreml) включены для macOS в Cargo.toml; остальные ОС
        // компилируют чистый CPU-путь и не тянут Apple-фреймворки.
        #[cfg(target_os = "macos")]
        let modes: &[(&str, ExecutionMode)] = &[
            ("CPU", ExecutionMode::Cpu),
            ("CoreML", ExecutionMode::CoreMl),
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
    ///
    /// Профиль кэшируется на диск: если записи не менялись (отпечаток совпал),
    /// пересборка мгновенна — первая фраза после запуска не ждёт десятков
    /// прогонов пайплайна.
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

        // Кэш: тот же набор файлов (имя + размер + mtime) → мгновенная загрузка.
        let fingerprint = sources_fingerprint(&records);
        if self.load_profile_cache(&fingerprint) {
            let _ = events.send(JarvisEvent::Log {
                message: format!(
                    "Профиль голоса: загружен из кэша ({} записей) — пересборка не нужна",
                    self.profile_records
                ),
            });
            return self.profile_records;
        }

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        let mut used_records = 0usize;
        for (path, _) in &records {
            let read = read_wav_16k(path);
            let Ok(samples) = read else { continue };
            // Тишина по краям записи размывает средний эмбеддинг: сравнение
            // идёт только по звучавшей части. У длинного диктанта берём
            // середину: 24 секунд звучавшей речи дают 8 окон — достаточно
            // для точного профиля, а пересборка остаётся быстрой.
            let voiced = trim_silence(&samples, TARGET_RATE);
            let voiced: &[f32] = if voiced.len() > TARGET_RATE * 24 {
                let mid = voiced.len() / 2;
                let half = TARGET_RATE * 12;
                &voiced[mid - half..mid + half]
            } else {
                voiced
            };
            if voiced.len() < (MIN_SEGMENT_SECS * TARGET_RATE as f32) as usize {
                continue;
            }
            // Каждое окно — изолированный запуск пайплайна одной длины.
            let mut record_windows = 0usize;
            for window in windows_of(voiced) {
                if let Some(mean) = self.segment_embedding(&window) {
                    if mean.iter().all(|v| v.is_finite()) {
                        vectors.push(mean);
                        record_windows += 1;
                    }
                }
            }
            if record_windows > 0 {
                used_records += 1;
            } else {
                eprintln!(
                    "🎙️ Профиль: запись без годных окон пропущена: {:?}",
                    path.file_name()
                );
            }
        }

        if vectors.is_empty() {
            let _ = events.send(JarvisEvent::Log {
                message: "Профиль голоса: образцы слишком коротки для анализа".into(),
            });
            return 0;
        }

        let profile = normalize(&mean_of(&vectors));
        if !profile.iter().all(|v| v.is_finite()) {
            // NaN-профиль делал «чужим» любой голос: лучше честно работать
            // без профиля, чем молча отвергать всех.
            let _ = events.send(JarvisEvent::Log {
                message: "Профиль голоса: нечисловые значения — профиль не сохранён".into(),
            });
            return 0;
        }
        self.profile = Some(profile);
        self.profile_records = used_records;
        self.save_profile_cache(&fingerprint, &vectors);
        let _ = events.send(JarvisEvent::Log {
            message: format!(
                "Профиль голоса владельца построен: {} записей, {} окон",
                used_records,
                vectors.len()
            ),
        });
        self.profile_records
    }

    /// Верификация сегмента: [`Verdict::Owner`] — голос владельца,
    /// [`Verdict::Stranger`] — чужой, [`Verdict::Unverified`] — проверить
    /// не удалось (fail-closed), [`Verdict::NoProfile`] — профиля нет.
    ///
    /// Фраза режется на канонические окна [`WINDOW_SECS`] — те же, из которых
    /// собран профиль; вердикт — лучшее сходство окна (короткая фраза даёт
    /// одно окно, и оно само по себе показательно).
    pub fn verify(&mut self, samples: &[f32], threshold: f32) -> Verdict {
        let Some(profile) = self.profile.clone() else {
            return Verdict::NoProfile;
        };
        if samples.len() < (MIN_SEGMENT_SECS * TARGET_RATE as f32) as usize {
            return Verdict::Unverified("сегмент короче порога верификации");
        }
        let voiced = trim_silence(samples, TARGET_RATE);
        let mut best: Option<f32> = None;
        let mut usable_windows = 0usize;
        for window in windows_of(voiced) {
            let Some(mean) = self.segment_embedding(&window) else {
                continue;
            };
            let normalized = normalize(&mean);
            if !normalized.iter().all(|v| v.is_finite()) {
                continue;
            }
            usable_windows += 1;
            let cosine = dot(&profile, &normalized);
            if best.is_none_or(|b| cosine > b) {
                best = Some(cosine);
            }
        }
        let Some(cosine) = best else {
            // Ни одно окно не дало сравнимого эмбеддинга — не «чужой», а
            // честная невозможность сравнить.
            return Verdict::Unverified("ни одно окно не дало эмбеддинга");
        };
        // Сходство в лог: без числа «Посторонний голос» не отличить битый
        // профиль от чуть-чуть не дотянувшего до порога.
        println!(
            "🎙️ Сходство голоса: {:.3} (порог {:.2}, окон {}) — {}",
            cosine,
            threshold,
            usable_windows,
            if cosine >= threshold { "владелец" } else { "чужой" }
        );
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
        // всех чанков в один пул и усредняем. Фреймы с нечисловыми значениями
        // (CoreML иногда отдаёт NaN) выбрасываются: один такой фрейм делал
        // среднее NaN, а NaN-сходство означало «чужой» для ЛЮБОГО голоса,
        // включая владельца.
        let embeddings = &result.embeddings;
        let mut pool: Vec<f32> = Vec::new();
        let mut dim = 0usize;
        let mut bad_frames = 0usize;
        for chunk in embeddings.rows() {
            for frame in chunk.rows() {
                if dim == 0 {
                    dim = frame.len();
                }
                if frame.len() == dim {
                    if frame.iter().all(|v| v.is_finite()) {
                        pool.extend(frame.iter().copied());
                    } else {
                        bad_frames += 1;
                    }
                }
            }
        }
        if bad_frames > 4 {
            // 1–2 битых фрейма на окно — норма (паддинг/граница окна); шум
            // в лог не пишем, аномалии видны.
            eprintln!("🎙️ speakrs: {} нечисловых фреймов эмбеддинга отброшено", bad_frames);
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
                (None, SpeakerQuery::Rebuild) => SpeakerReply::Verified {
                    id: 0,
                    verdict: Verdict::Unverified("пайплайн проверки голоса не загрузился"),
                },
                (None, SpeakerQuery::Verify { id, .. }) => SpeakerReply::Verified {
                    id,
                    verdict: Verdict::Unverified("пайплайн проверки голоса не загрузился"),
                },
                (None, SpeakerQuery::SpeechProbe { .. }) => SpeakerReply::Speech(None),
                (Some(g), SpeakerQuery::Rebuild) => {
                    // Стартовая пересборка: с кэшем мгновенна, без — полная.
                    // Выполняется СЕЙЧАС, чтобы первая фраза владельца не
                    // ждала десятков прогонов пайплайна в очереди.
                    let events = dummy_events();
                    g.rebuild_profile(&events);
                    SpeakerReply::Verified {
                        id: 0,
                        verdict: Verdict::Owner,
                    }
                }
                (Some(g), SpeakerQuery::Verify { id, samples, threshold }) => {
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
                    SpeakerReply::Verified { id, verdict }
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
    // Нечисловые компоненты зануляются: NaN-норма проходила сравнение
    // `norm < 1e-9` как false и делила вектор на NaN.
    let clean: Vec<f32> = v.iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect();
    let norm = clean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-9 {
        return clean;
    }
    clean.iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// --- Кэш профиля на диске ---

/// Версия формата кэша профиля.
const PROFILE_CACHE_VERSION: u32 = 2;

#[derive(serde::Serialize, serde::Deserialize)]
struct ProfileCache {
    version: u32,
    /// Отпечаток источников: (имя файла, размер + mtime) — изменение любого
    /// образца инвалидирует кэш.
    sources: Vec<(String, u64)>,
    vectors: Vec<Vec<f32>>,
    records: usize,
}

/// Отпечаток набора записей владельца: любое добавление/удаление/правка
/// образца меняет его.
fn sources_fingerprint(records: &[(std::path::PathBuf, f64)]) -> Vec<(String, u64)> {
    records
        .iter()
        .map(|(path, _)| {
            let meta = std::fs::metadata(path)
                .map(|m| m.len().wrapping_mul(1_000_003).wrapping_add(m.modified().map(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0)
                }).unwrap_or(0)))
                .unwrap_or(0);
            (
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                meta,
            )
        })
        .collect()
}

fn profile_cache_path() -> std::path::PathBuf {
    crate::wake_train::voice_samples_dir().join("profile.cache")
}

impl SpeakerGate {
    /// Загрузить профиль из кэша, если отпечаток источников совпал.
    fn load_profile_cache(&mut self, fingerprint: &[(String, u64)]) -> bool {
        let Ok(raw) = std::fs::read(profile_cache_path()) else {
            return false;
        };
        let Ok(cache) = serde_json::from_str::<ProfileCache>(&String::from_utf8_lossy(&raw)) else {
            return false;
        };
        if cache.version != PROFILE_CACHE_VERSION || cache.sources != fingerprint {
            return false;
        }
        if cache.vectors.is_empty() || cache.records == 0 {
            return false;
        }
        let profile = normalize(&mean_of(&cache.vectors));
        if !profile.iter().all(|v| v.is_finite()) {
            return false;
        }
        self.profile = Some(profile);
        self.profile_records = cache.records;
        true
    }

    /// Сохранить профиль в кэш (ошибки не критичны: пересборка просто
    /// повторится при следующем старте).
    fn save_profile_cache(&self, fingerprint: &[(String, u64)], vectors: &[Vec<f32>]) {
        let cache = ProfileCache {
            version: PROFILE_CACHE_VERSION,
            sources: fingerprint.to_vec(),
            vectors: vectors.to_vec(),
            records: self.profile_records,
        };
        let Ok(json) = serde_json::to_string(&cache) else {
            return;
        };
        let path = profile_cache_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("cache.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

/// Режет звучавшую часть на канонические окна: полные без перекрытия,
/// хвост включается, если он не короче половины окна; всё добивается
/// нулями до [`WINDOW_SAMPLES`] — длина каждого прогона пайплайна
/// одинаковая, иначе эмбеддинги несравнимы.
fn windows_of(voiced: &[f32]) -> Vec<Vec<f32>> {
    let min_tail = WINDOW_SAMPLES / 2;
    if voiced.len() <= WINDOW_SAMPLES {
        let mut only = voiced.to_vec();
        only.resize(WINDOW_SAMPLES, 0.0);
        return vec![only];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start + WINDOW_SAMPLES <= voiced.len() {
        out.push(voiced[start..start + WINDOW_SAMPLES].to_vec());
        start += WINDOW_SAMPLES;
    }
    let tail = voiced.len() - start;
    if tail >= min_tail {
        let mut last = voiced[start..].to_vec();
        last.resize(WINDOW_SAMPLES, 0.0);
        out.push(last);
    }
    out
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

    #[test]
    fn normalize_never_returns_nan() {
        // NaN-вектор раньше давал NaN-норму (сравнение `norm < 1e-9` ложно)
        // и делил на NaN — сходство становилось NaN, любой голос был «чужим».
        let normalized = normalize(&[f32::NAN, 1.0, f32::INFINITY]);
        assert!(normalized.iter().all(|v| v.is_finite()));
        // Нулевой вектор — тоже конечный результат.
        assert!(normalize(&[0.0; 4]).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mean_of_skips_nonfinite_components() {
        // Среднее с NaN-компонентой после normalize обязано быть конечным.
        let poisoned = vec![f32::NAN, 1.0, 2.0, 3.0];
        let normalized = normalize(&mean_of(&[poisoned, vec![1.0, 1.0, 1.0, 1.0]]));
        assert!(normalized.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn windows_of_pads_short_input_to_canonical_length() {
        // Короткая фраза: одно окно канонической длины (добивка нулями) —
        // домен сравнения единый с профилем.
        let voiced = vec![0.5f32; TARGET_RATE / 2]; // 0.5 с
        let windows = windows_of(&voiced);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len(), WINDOW_SAMPLES);
        assert_eq!(windows[0][0], 0.5);
        assert_eq!(windows[0][TARGET_RATE / 2], 0.0);
    }

    #[test]
    fn windows_of_splits_long_input_and_keeps_big_tail() {
        // 7 с: два полных окна + хвост 1 с (< половины окна) отбрасывается.
        let seven = vec![0.1f32; TARGET_RATE * 7];
        let windows = windows_of(&seven);
        assert_eq!(windows.len(), 2);
        assert!(windows.iter().all(|w| w.len() == WINDOW_SAMPLES));

        // Хвост ≥ половины окна включается с добивкой (4.5 с = 3 с + 1.5 с).
        let four_and_half = vec![0.1f32; TARGET_RATE * 4 + TARGET_RATE / 2];
        let windows = windows_of(&four_and_half);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[1].len(), WINDOW_SAMPLES);
        assert_eq!(windows[1][WINDOW_SAMPLES - 1], 0.0);
    }

    #[test]
    fn profile_cache_fingerprint_tracks_changes() {
        // Отпечаток меняется при изменении набора файлов или их содержимого.
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.wav");
        std::fs::write(&a, b"first").unwrap();
        let records = vec![(a.clone(), 0.0)];
        let before = sources_fingerprint(&records);
        std::fs::write(&a, b"second-longer-content").unwrap();
        let after = sources_fingerprint(&records);
        assert_ne!(before, after, "правка файла должна менять отпечаток");
        let removed = vec![(dir.path().join("b.wav"), 0.0)];
        assert_ne!(
            sources_fingerprint(&removed),
            before,
            "другой набор файлов — другой отпечаток"
        );
    }
}