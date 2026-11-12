//! Детектор wake word (RustPotter): создание, горячая подмена модели.

use anyhow::{Result, bail};
use rustpotter::{FrameEmbedder, RustpotterConfig, SampleFormat};

pub const SAMPLE_RATE: usize = 16_000;

/// Конфигурация DTW-детектора (для проверки обученной модели `.rpw`
/// в приложении; из пути распознавания он выведен).
pub fn detector_config(threshold: f32) -> RustpotterConfig {
    let mut dcfg = RustpotterConfig::default();
    dcfg.fmt.sample_rate = SAMPLE_RATE;
    dcfg.fmt.channels = 1;
    dcfg.fmt.sample_format = SampleFormat::I16;
    dcfg.detector.threshold = threshold;
    dcfg.filters.band_pass.enabled = false;
    dcfg
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

/// Метрический детектор (эталон .jref) — прежний путь: окно сравнивается с
/// облаком эмбеддингов, набранным из синтетических голосов.
pub struct MetricWakeEngine {
    det: rustpotter::EmbeddingDetector<FastConformerEmbedder>,
}

pub struct WakeHit {
    pub name: String,
    pub score: f32,
}

/// Детектор wake word движка.
///
/// `Ctc` — по CTC-постерам ASR-модели: эталон это слово-активация, а не
/// облако аудио, поэтому слово любое и голос не важен; срабатывание
/// означает, что модель услышала слово в декоде. `Metric` — прежний
/// метрический путь по эталону `.jref`.
pub enum WakeEngine {
    Ctc(crate::wake_ctc::CtcWakeEngine),
    Metric(MetricWakeEngine),
}

impl WakeEngine {
    pub fn process_16k(&mut self, chunk: &[f32]) -> Option<WakeHit> {
        match self {
            WakeEngine::Ctc(d) => d.process(chunk).map(|c| WakeHit {
                name: d.word().to_string(),
                score: c.score,
            }),
            WakeEngine::Metric(d) => d.process_16k(chunk),
        }
    }

    /// Сброс потокового состояния (после срабатывания/паузы).
    pub fn reset(&mut self) {
        match self {
            WakeEngine::Ctc(d) => d.reset(),
            WakeEngine::Metric(d) => d.reset(),
        }
    }

    /// Счёт последнего окна (тюнение порога в UI/журнале).
    pub fn last_score(&self) -> Option<f32> {
        match self {
            WakeEngine::Ctc(d) => d.last_score(),
            WakeEngine::Metric(d) => d.last_score(),
        }
    }

    /// Человекочитаемая диагностика последнего окна — в журнал.
    pub fn describe(&self) -> String {
        match self {
            WakeEngine::Ctc(d) => match d.last_diag() {
                Some(diag) => format!(
                    "rms {:.4}, пол {:.4}, голос {}, декод «{}», счёт {:?}, попаданий {}/{}",
                    diag.rms, diag.noise_floor, diag.voiced, diag.decoded, diag.score, diag.hits, 0
                ),
                None => "нет диагностики".to_string(),
            },
            WakeEngine::Metric(d) => format!("метрический, счёт {:?}", d.last_score()),
        }
    }
}

impl MetricWakeEngine {
    pub fn process_16k(&mut self, chunk: &[f32]) -> Option<WakeHit> {
        self.det.process(chunk).map(|d| WakeHit {
            name: "jarvis".to_string(),
            score: d.score,
        })
    }

    /// Сброс потокового состояния (после срабатывания/паузы).
    pub fn reset(&mut self) {
        self.det.reset();
    }

    /// Счёт последнего окна (тюнение порога в UI/журнале).
    pub fn last_score(&self) -> Option<f32> {
        self.det.last_raw_score()
    }
}

/// Эталон метрического детектора — фиксированное имя в папке моделей.
pub const EMBEDDING_REFERENCE: &str = "jarvis_model.jref";

/// Счёт окна аудио против эталона (диагностика, wake_check) — та же
/// семантика, что у боевого детектора: центрмирование → max по позитивам
/// минус сходство с негативом. None — эталона нет.
pub fn score_reference(stt: &Arc<Mutex<crate::stt::Stt>>, window: &[f32]) -> Option<f32> {
    let path = crate::paths::models_dir().join(EMBEDDING_REFERENCE);
    let reference = rustpotter::EmbeddingReference::load_reference_only(&path).ok()?;
    let mut e = FastConformerEmbedder::new(Arc::clone(stt))
        .embed(window)
        .ok()?;
    rustpotter::l2_normalize(&mut e);
    // Центрирование тем же фоновым средним, что и в детекторе.
    for (x, m) in e.iter_mut().zip(&reference.mean) {
        *x -= m;
    }
    // Guard вырожденного остатка — как в детекторе (см. MIN_RESIDUAL_NORM).
    let residual: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !residual.is_finite() || residual < rustpotter::MIN_RESIDUAL_NORM {
        return Some(-1.0);
    }
    for x in e.iter_mut() {
        *x /= residual;
    }
    let best_p = reference
        .positives
        .iter()
        .map(|p| e.iter().zip(p).map(|(a, b)| a * b).sum::<f32>())
        .fold(f32::NEG_INFINITY, f32::max);
    let best_n = reference
        .negatives
        .iter()
        .map(|n| e.iter().zip(n).map(|(a, b)| a * b).sum::<f32>())
        .fold(f32::NEG_INFINITY, f32::max);
    // Вес негативов заморожен в эталоне (reference.weight): бой и диагностика
    // обязаны считать по ОДНОЙ шкале. Без веса счёт здесь систематически
    // ниже боевого (best_p - best_n вместо best_p - w*best_n), а порог
    // калибруется по hold-out именно здесь — то есть калибровали бы на одной
    // шкале, а сравнивали в бою на другой: ложные срабатывания вслепую.
    Some(best_p - reference.weight * best_n)
}

/// Создать движок вакворда: метрический, если есть эталон .jref (любой
/// голос), иначе DTW по .rpw (голос владельца).
pub fn create_engine(
    stt: &Arc<Mutex<crate::stt::Stt>>,
    _rpw_name: &str,
    threshold_setting: f32,
    mode: &str,
    word: &str,
    window_ms: usize,
) -> Result<WakeEngine> {
    // CTC-детектор: эталон — слово текстом, поэтому файл .jref не нужен
    // и новое активационное слово не требует пересборки модели.
    if mode.eq_ignore_ascii_case("ctc") {
        let w = crate::wake_ctc::normalize_word(word);
        if w.is_empty() {
            bail!("Пустое слово-активация: задайте wake_word в настройках");
        }
        let mut config = crate::wake_ctc::CtcConfig::default();
        // Окно должно накрывать фразу целиком: короткая фраза помещается
        // в стандартное 1.2 с, длинная — требует более широкого окна.
        config.window_ms = window_ms.max(config.window_ms);
        let engine =
            crate::wake_ctc::CtcWakeEngine::new(Arc::clone(stt), &w, config);
        println!("🎯 Wake word: CTC-детектор, слово «{w}» (эталон не нужен)");
        return Ok(WakeEngine::Ctc(engine));
    }
    let jref = crate::paths::models_dir().join(EMBEDDING_REFERENCE);
    if !jref.exists() {
        anyhow::bail!(
            "Нет эталона {}: скачай его из релиза jarvis-models или построй свой \
             (Настройки → Wake Word → запиши 3–5 сэмплов → «Собрать модель», \
             либо cargo run -p jarvis-core --example wake_enroll)",
            jref.display()
        );
    }
    let embedder = FastConformerEmbedder::new(Arc::clone(stt));
    let mut det =
        rustpotter::EmbeddingDetector::new(embedder, rustpotter::EmbeddingConfig::default())
            .map_err(|e| anyhow::anyhow!("конфигурация wake-детектора: {e}"))?;
    det.load_reference(&jref)
        .map_err(|e| anyhow::anyhow!("эталон {}: {e}", jref.display()))?;
    let calibrated = det.reference().map(|r| r.threshold).unwrap_or(f32::NAN);
    // Порог метрического детектора калибруется при сборке эталона и лежит
    // в самом .jref. Значение из настроек сюда НЕ подставляется: шкала
    // счёта метрики не та же, что у DTW, и подстановка молча сломала бы
    // калибровку. Если значения разошлись — говорим об этом явно.
    if (threshold_setting - calibrated).abs() > f32::EPSILON {
        eprintln!(
            "⚠️ wake_threshold из настроек ({threshold_setting:.4}) не применяется: \
             метрический детектор использует калиброванный порог эталона {calibrated:.4}. \
             Подстрой порог пересборкой эталона (cargo run -p jarvis-core --example wake_enroll)."
        );
    }
    println!(
        "🎯 Wake word: метрический детектор ({} — дикторонезависимый, порог {calibrated:.4})",
        EMBEDDING_REFERENCE
    );
    Ok(WakeEngine::Metric(MetricWakeEngine { det }))
}
