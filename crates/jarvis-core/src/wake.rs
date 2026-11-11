//! Детектор wake word (RustPotter): создание, горячая подмена модели.

use anyhow::{Context, Result, anyhow, bail};
use rustpotter::{FrameEmbedder, Rustpotter, RustpotterConfig, SampleFormat};
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;

/// Конфигурация детектора по умолчанию для Jarvis.
pub fn detector_config(threshold: f32) -> RustpotterConfig {
    let mut dcfg = RustpotterConfig::default();
    dcfg.fmt.sample_rate = SAMPLE_RATE;
    dcfg.fmt.channels = 1;
    dcfg.fmt.sample_format = SampleFormat::I16;
    dcfg.detector.threshold = threshold;
    dcfg.filters.band_pass.enabled = false;
    dcfg
}

/// Создаёт детектор с одной зарегистрированной моделью wake word.
pub fn create_detector(model_path: &Path, threshold: f32) -> Result<Rustpotter> {
    if !model_path.exists() {
        bail!("Файл модели вакворда {} не найден", model_path.display());
    }

    let dcfg = detector_config(threshold);
    let mut det = Rustpotter::new(&dcfg).map_err(|e| anyhow!(e))?;

    det.add_wakeword_from_file("jarvis", model_path.to_str().context("путь не UTF-8")?)
        .map_err(|e| anyhow!(e))?;

    Ok(det)
}
// =========================================================================
// Дикторонезависимый движок: Few-Shot Metric Learning (RustPotter embedding)
// =========================================================================

use std::sync::{Arc, Mutex};

/// Эмбеддер на FastConformer: окно 16 кГц → mel → энкодер → mean-pool
/// кадров → вектор d_model. Энкодер обучен на сотнях дикторов, поэтому его
/// представление слабо зависит от тембра — то, что нужно для метрики.
/// Делит один экземпляр модели с STT-воркером (состояния движка взаимно
/// исключающие, contention нет).
pub struct FastConformerEmbedder {
    stt: Arc<Mutex<crate::stt::Stt>>,
}

impl FastConformerEmbedder {
    pub fn new(stt: Arc<Mutex<crate::stt::Stt>>) -> Self {
        Self { stt }
    }
}

impl rustpotter::FrameEmbedder for FastConformerEmbedder {
    fn embed(&mut self, window: &[f32]) -> Result<Vec<f32>, String> {
        let stt = self.stt.lock().unwrap_or_else(|e| e.into_inner());
        let model = stt.model();
        let (mel, length) = model
            .frontend
            .compute(window, &model.device)
            .map_err(|e| format!("mel: {e}"))?;
        let (enc, t_valid) = model
            .encoder
            .forward(&mel, length)
            .map_err(|e| format!("encoder: {e}"))?;
        let (t, d) = (
            enc.dim(1).map_err(|e| e.to_string())?,
            enc.dim(2).map_err(|e| e.to_string())?,
        );
        let flat = enc
            .reshape((t * d,))
            .map_err(|e| e.to_string())?
            .to_vec1::<f32>()
            .map_err(|e| e.to_string())?;
        let t_use = t_valid.min(t).max(1);
        let mut out = vec![0_f32; d];
        for frame in flat.chunks(d).take(t_use) {
            for (acc, x) in out.iter_mut().zip(frame) {
                *acc += x;
            }
        }
        for x in out.iter_mut() {
            *x /= t_use as f32;
        }
        Ok(out)
    }

    fn dims(&self) -> usize {
        let stt = self.stt.lock().unwrap_or_else(|e| e.into_inner());
        stt.model().cfg.encoder.d_model
    }
}

/// Единый контракт детектора для главного цикла движка: кусок 16 кГц на
/// входе — срабатывание на выходе.
pub enum WakeEngine {
    /// Классический DTW по .rpw: точен на голосе владельца.
    Dtw {
        det: rustpotter::Rustpotter,
        frame: usize,
        buffer: std::collections::VecDeque<i16>,
    },
    /// Метрический по эмбеддингам (.jref): срабатывает на любой голос.
    Embedding(rustpotter::EmbeddingDetector<FastConformerEmbedder>),
}

pub struct WakeHit {
    pub name: String,
    pub score: f32,
}

impl WakeEngine {
    pub fn process_16k(&mut self, chunk: &[f32]) -> Option<WakeHit> {
        match self {
            WakeEngine::Dtw { det, frame, buffer } => {
                for &s in chunk {
                    buffer.push_back((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
                }
                let mut hit = None;
                while buffer.len() >= *frame {
                    let fr: Vec<i16> = buffer.drain(..*frame).collect();
                    if let Some(d) = det.process_samples(fr) {
                        hit = Some(WakeHit {
                            name: d.name.clone(),
                            score: d.score,
                        });
                    }
                }
                hit
            }
            WakeEngine::Embedding(det) => det.process(chunk).map(|d| WakeHit {
                name: "jarvis".to_string(),
                score: d.score,
            }),
        }
    }

    /// Сброс потокового состояния (после срабатывания/паузы).
    pub fn reset(&mut self) {
        match self {
            WakeEngine::Dtw { buffer, .. } => buffer.clear(),
            WakeEngine::Embedding(det) => det.reset(),
        }
    }
}

/// Эталон метрического детектора — фиксированное имя в папке моделей.
pub const EMBEDDING_REFERENCE: &str = "jarvis_model.jref";

/// Счёт окна аудио против эталона метрического детектора (диагностика,
/// пример wake_check): cos(x, позитив) − cos(x, негатив).
/// None — эталона нет на диске.
pub fn score_reference(
    stt: &Arc<Mutex<crate::stt::Stt>>,
    window: &[f32],
) -> Option<f32> {
    let path = crate::paths::models_dir().join(EMBEDDING_REFERENCE);
    if !path.exists() {
        return None;
    }
    let reference = rustpotter::EmbeddingReference::load_reference_only(&path).ok()?;
    let mut e = FastConformerEmbedder::new(Arc::clone(stt)).embed(window).ok()?;
    rustpotter::l2_normalize(&mut e);
    Some(
        e.iter().zip(&reference.positive).map(|(a, b)| a * b).sum::<f32>()
            - e.iter().zip(&reference.negative).map(|(a, b)| a * b).sum::<f32>(),
    )
}

/// Создать движок вакворда: метрический, если есть эталон .jref (любой
/// голос), иначе DTW по .rpw (голос владельца).
pub fn create_engine(
    stt: &Arc<Mutex<crate::stt::Stt>>,
    rpw_name: &str,
    threshold: f32,
) -> Result<WakeEngine> {
    let jref = crate::paths::models_dir().join(EMBEDDING_REFERENCE);
    if jref.exists() {
        let embedder = FastConformerEmbedder::new(Arc::clone(stt));
        let mut det = rustpotter::EmbeddingDetector::new(
            embedder,
            rustpotter::EmbeddingConfig {
                threshold,
                ..Default::default()
            },
        );
        det.load_reference(&jref).map_err(|e| {
            anyhow::anyhow!("эталон {}: {e}", jref.display())
        })?;
        println!("🎯 Wake word: метрический детектор ({} — дикторонезависимый)", EMBEDDING_REFERENCE);
        return Ok(WakeEngine::Embedding(det));
    }
    let det = create_detector(&crate::paths::wakeword_model_file(rpw_name), threshold)?;
    println!("🎯 Wake word: DTW-детектор ({rpw_name} — голос владельца)");
    let frame = det.get_samples_per_frame();
    Ok(WakeEngine::Dtw {
        det,
        frame,
        buffer: std::collections::VecDeque::new(),
    })
}
