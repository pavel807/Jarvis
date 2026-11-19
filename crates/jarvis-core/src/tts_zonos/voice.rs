//! Голос ZONOS3 как движок синтеза для `tts`.
//!
//! Обёртка превращает `ZonosEngine` в `SpeechEngine`: берёт эмбеддинг
//! голоса (из спикер-энкодера или из ассетов), переводит скорость речи в
//! бакет условия `speaking_rate` и отдаёт готовые сэмплы 44.1 кГц.

use super::codec::ZacDecoder;
use super::engine::{SAMPLE_RATE, ZonosEngine, default_buckets};
use super::sampler::Sampling;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Голос ZONOS3: модель + эмбеддинг голоса + ZAC-декодер.
pub struct ZonosVoice {
    engine: Mutex<ZonosEngine>,
    /// Декодер кодов в звук (общий на все голоса).
    decoder: Option<Arc<ZacDecoder>>,
    /// 192-мерный эмбеддинг голоса (клонирование по опорной записи).
    speaker: Option<Vec<f32>>,
    /// Бакеты условий: громкость, f_max, темп.
    buckets: [usize; 3],
    /// Параметры сэмплирования.
    sampling: Sampling,
    /// Усиление клонирования (CFG). В бою всегда 1.0 (выкл): замеры дали
    /// +0.04–0.065 схожести голоса, но ~2× времени генерации и частые
    /// искажения текста. Голос задаёт эмбеддинг, а не CFG. Поле оставлено
    /// только для офлайн-замеров (`tts_sweep`, `probe_voice_fit`).
    cfg_scale: f32,
    /// Имя голоса (для интерфейса и кэша).
    voice_name: String,
    /// Каталог `voices/` с эмбеддингами для превью других голосов.
    voices_dir: Option<PathBuf>,
    /// Кэш эмбеддингов превью: имя → эмбеддинг (None — голоса нет).
    preview: Mutex<HashMap<String, Option<Vec<f32>>>>,
}

impl ZonosVoice {
    /// Собрать голос из уже загруженного движка.
    ///
    /// `decoder` — ZAC-декодер (общий между голосами); без него синтез вернёт
    /// только коды кадров, звук не соберётся.
    pub fn new(
        engine: ZonosEngine,
        decoder: Option<Arc<ZacDecoder>>,
        speaker: Option<Vec<f32>>,
        buckets: Option<[usize; 3]>,
        voice_name: impl Into<String>,
    ) -> Self {
        Self::with_preview_dir(engine, decoder, speaker, buckets, voice_name, None)
    }

    /// Как [`new`], но с каталогом `voices/` для превью других голосов.
    pub fn with_preview_dir(
        engine: ZonosEngine,
        decoder: Option<Arc<ZacDecoder>>,
        speaker: Option<Vec<f32>>,
        buckets: Option<[usize; 3]>,
        voice_name: impl Into<String>,
        voices_dir: Option<PathBuf>,
    ) -> Self {
        let buckets = buckets.unwrap_or_else(|| default_buckets(&engine.cfg));
        // Боевые дефолты: 0.6 — первая попытка почти всегда без искажений
        // (0.75 чаще заикается и уходит в пересинтез), сид 1 проходит
        // самопроверку с первой попытки чаще, чем 0. CFG не используем:
        // он вдвое дороже и портит текст, а голос задаёт эмбеддинг.
        // Ручки для замеров: JARVIS_TTS_TEMP, JARVIS_TTS_SEED.
        let mut sampling = Sampling::default();
        sampling.temperature = std::env::var("JARVIS_TTS_TEMP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|t| (0.0..=2.0).contains(t))
            .unwrap_or(0.6);
        sampling.seed = 1;
        if let Ok(v) = std::env::var("JARVIS_TTS_SEED") {
            if let Ok(seed) = v.parse() {
                sampling.seed = seed;
            }
        }
        Self {
            engine: Mutex::new(engine),
            decoder,
            speaker,
            buckets,
            sampling,
            cfg_scale: 1.0,
            voice_name: voice_name.into(),
            voices_dir,
            preview: Mutex::new(HashMap::new()),
        }
    }

    /// Заменить эмбеддинг голоса (новый клон).
    pub fn set_speaker(&mut self, speaker: Option<Vec<f32>>) {
        self.speaker = speaker;
    }

    /// Эмбеддинг голоса.
    pub fn speaker(&self) -> Option<&[f32]> {
        self.speaker.as_deref()
    }

    /// Имя голоса.
    pub fn voice_name(&self) -> &str {
        &self.voice_name
    }

    /// Параметры сэмплирования (сид/температура) — для вариаций синтеза.
    pub fn set_sampling(&mut self, sp: Sampling) {
        self.sampling = sp;
    }

    /// Усиление клонирования голоса (CFG scale). Только для офлайн-замеров:
    /// боевой путь всегда работает с 1.0 (см. комментарий к полю).
    pub fn set_cfg_scale(&mut self, scale: f32) {
        self.cfg_scale = scale;
    }

    /// Эмбеддинг голоса по имени: свой — для текущего голоса, чужие —
    /// из каталога `voices/` (кэшируется).
    fn speaker_for(&self, name: &str) -> Option<Vec<f32>> {
        if name == self.voice_name {
            return self.speaker.clone();
        }
        let dir = self.voices_dir.as_ref()?;
        let mut cache = self.preview.lock().ok()?;
        if let Some(cached) = cache.get(name) {
            return cached.clone();
        }
        let emb = load_voice_embedding(dir, name);
        cache.insert(name.to_string(), emb.clone());
        emb
    }

    /// Бакет темпа речи по скорости: 1.0 — естественный темп, меньше — медленнее.
    ///
    /// Скорость в интерфейсе — множитель, а модель управляется абсолютным
    /// темпом в «байтах UTF-8 в секунду». Переводим через середину
    /// диапазона бакетов: 1.0 → средний бакет, 0.5 → нижний, 1.5 → верхний.
    pub fn speed_bucket(&self, speed: f32) -> usize {
        let n = self.engine.lock().map(|e| e.cfg.unknown_bucket(2)).unwrap_or(30);
        if n == 0 {
            return 0;
        }
        // Естественный темп — «неизвестно»: модель стабильнее на unknown,
        // чем на конкретных бакетах (проверено: средний бакет даёт заикания).
        if (speed - 1.0).abs() < 0.04 {
            return n;
        }
        let mid = (n.saturating_sub(1)) as f32 / 2.0;
        let scaled = mid * speed.clamp(0.5, 1.5) / 1.0;
        (scaled.round().max(0.0) as usize).min(n - 1)
    }
}

/// Требование движка синтеза речи: единый интерфейс для `tts`.
pub trait SpeechEngine: Send + Sync {
    /// Частота дискретизации выходного звука.
    fn sample_rate(&self) -> u32;
    /// Синтез речи с текущим голосом.
    fn speak(&self, text: &str, speed: f32) -> Result<Vec<f32>>;
    /// Синтез речи заданным голосом (превью в интерфейсе).
    fn speak_with(&self, voice: &str, text: &str, speed: f32) -> Result<Vec<f32>>;
    /// Имя движка: «zonos».
    fn engine_name(&self) -> &'static str;
}

/// Загрузить эмбеддинг голоса (192 измерения) из `.npy`.
///
/// Ищем `voices/<голос>.embedding.npy`, затем `<голос>.embedding.npy` в
/// каталоге весов. Формат — обычный numpy: заголовок-словарь и сырые f32.
pub fn load_voice_embedding(dir: &Path, voice: &str) -> Option<Vec<f32>> {
    let candidates = [
        dir.join("voices").join(format!("{voice}.embedding.npy")),
        dir.join(format!("{voice}.embedding.npy")),
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            if let Some(values) = parse_npy_f32(&bytes) {
                if values.len() == 192 {
                    return Some(values);
                }
            }
        }
    }
    None
}

/// Минимальный читатель `.npy` с одномерным массивом f32.
fn parse_npy_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(bytes.get(header_start..header_start + header_len)?).ok()?;
    if !header.contains("'<f4'") && !header.contains("\"<f4\"") {
        return None;
    }
    let data = bytes.get(header_start + header_len..)?;
    let count = data.len() / 4;
    let mut out = Vec::with_capacity(count);
    for chunk in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

impl SpeechEngine for ZonosVoice {
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn speak(&self, text: &str, speed: f32) -> Result<Vec<f32>> {
        self.speak_internal(self.speaker.clone(), text, speed)
    }

    fn speak_with(&self, voice: &str, text: &str, speed: f32) -> Result<Vec<f32>> {
        let speaker = self.speaker_for(voice).or_else(|| self.speaker.clone());
        self.speak_internal(speaker, text, speed)
    }

    fn engine_name(&self) -> &'static str {
        "zonos"
    }
}

impl ZonosVoice {
    /// Синтез с самопроверкой: сэмплинг ZONOS3 иногда вставляет или повторяет
    /// слова, поэтому прогон через FastConformer сравнивается с текстом;
    /// при совпадении ниже 0.88 — пересинтез со следующим сидом (по умолчанию
    /// до 2 попыток: первая при temp 0.6 обычно уже годная, а лишние попытки
    /// стоят секунды). В ответ уходит лучший вариант. Без ASR-модели — как
    /// есть, одна попытка.
    fn speak_internal(
        &self,
        speaker: Option<Vec<f32>>,
        text: &str,
        speed: f32,
    ) -> Result<Vec<f32>> {
        let Some(decoder) = self.decoder.as_ref() else {
            anyhow::bail!("ZONOS3: нет ZAC-декодера (коды есть, звук не собирается)");
        };
        let mut buckets = self.buckets;
        buckets[2] = self.speed_bucket(speed);
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| anyhow::anyhow!("ZONOS3: движок заблокирован"))?;
        let asr = voice_asr();
        let max_attempts = std::env::var("JARVIS_TTS_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(2)
            .clamp(1, 4);

        let mut best: Option<(f32, Vec<f32>)> = None;
        for attempt in 0..max_attempts {
            let mut sp = self.sampling.clone();
            sp.seed = self.sampling.seed.wrapping_add(attempt);
            let started = std::time::Instant::now();
            let out = engine
                .synthesize(text, speaker.as_deref(), buckets, &sp, self.cfg_scale, Some(decoder.as_ref()))
                .context("ZONOS3: синтез не удался")?;
            let synth_secs = started.elapsed().as_secs_f32();
            match asr {
                Some(model) => {
                    // FastConformer обучен на 16 кГц; синтез — 44.1 кГц.
                    let asr_started = std::time::Instant::now();
                    let pcm16 = crate::tts_zonos::speaker::sinc_resample(
                        &out.audio,
                        SAMPLE_RATE,
                        16000,
                    );
                    let said = model.transcribe(&pcm16)?;
                    let asr_secs = asr_started.elapsed().as_secs_f32();
                    let sim = speech_similarity(text, said.text.trim());
                    if std::env::var_os("JARVIS_VOICE_DEBUG").is_some() {
                        eprintln!(
                            "🎤 attempt {attempt} seed {} synth {synth_secs:.1}s asr {asr_secs:.1}s sim {sim:.3} len {:.1}s: {}",
                            sp.seed,
                            out.audio.len() as f32 / SAMPLE_RATE as f32,
                            said.text.trim()
                        );
                    }
                    if best.as_ref().is_none_or(|(b, _)| sim > *b) {
                        best = Some((sim, out.audio));
                    }
                    if sim >= 0.88 {
                        break;
                    }
                }
                None => return Ok(out.audio),
            }
        }
        Ok(best.map(|(_, a)| a).expect("хотя бы одна попытка"))
    }
}

/// Ленивый FastConformer для самопроверки синтеза; None — модель не скачана.
/// Считаем на CPU: клип длиной 3–5 с на Metal тратит секунды на накладные
/// расходы мелких операций RNN-T, на CPU — десятые доли секунды.
fn voice_asr() -> Option<&'static crate::asr::model::FastConformer> {
    static ASR: std::sync::OnceLock<Option<crate::asr::model::FastConformer>> =
        std::sync::OnceLock::new();
    ASR.get_or_init(|| {
        crate::asr::model::FastConformer::load_on(
            &crate::paths::models_dir(),
            candle_core::Device::Cpu,
        )
        .ok()
    })
    .as_ref()
}

/// Сходство сказанного с текстом: слова, нормированные к нижнему регистру
/// без пунктуации (ё → е). Словесный Левенштейн, но замена слова на похожее
/// (≥ 0.75 по символам) считается совпадением: ASR путает «штатном» и
/// «статном», и пересинтез из-за одной буквы только добавляет секунды.
/// Порог 0.75 не пропускает заикания вроде «максимаксимальная» (0.73).
fn speech_similarity(expected: &str, said: &str) -> f32 {
    fn norm(s: &str) -> Vec<String> {
        s.to_lowercase()
            .replace('ё', "е")
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation() || "—–«»…".contains(c))
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    }
    let a = norm(expected);
    let b = norm(said);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    const WORD_MATCH: f32 = 0.75;
    let (n, m) = (a.len(), b.len());
    let mut d: Vec<f32> = (0..=m).map(|j| j as f32).collect();
    for i in 1..=n {
        let mut prev = d[0];
        d[0] = i as f32;
        for j in 1..=m {
            let cur = d[j];
            let same = a[i - 1] == b[j - 1]
                || crate::text::word_similarity(&a[i - 1], &b[j - 1]) >= WORD_MATCH;
            let sub = prev + if same { 0.0 } else { 1.0 };
            d[j] = (d[j] + 1.0).min(d[j - 1] + 1.0).min(sub);
            prev = cur;
        }
    }
    1.0 - d[m] / n.max(m) as f32
}

#[cfg(test)]
mod tests {
    use super::speech_similarity;

    #[test]
    fn identical_text_is_one() {
        let sim = speech_similarity("Привет, сэр. Все системы в норме.", "привет сэр все системы в норме");
        assert!((sim - 1.0).abs() < 1e-6, "{sim}");
    }

    #[test]
    fn minor_asr_error_passes() {
        let sim = speech_similarity(
            "Все системы работают в штатном режиме.",
            "Все системы работают в статном режиме.",
        );
        assert!(sim >= 0.88, "ASR-огрех не должен запускать пересинтез: {sim}");
    }

    #[test]
    fn real_garbling_fails() {
        let sim = speech_similarity(
            "Все системы работают в штатном режиме.",
            "Все системы работают в тост от отнажены.",
        );
        assert!(sim < 0.88, "искажение должно уходить в пересинтез: {sim}");
    }

    #[test]
    fn repeated_syllable_fails() {
        let sim = speech_similarity(
            "Курс установлен, скорость максимальная.",
            "Курс установлен, скорость максимаксимальная.",
        );
        assert!(sim < 0.88, "заикание должно уходить в пересинтез: {sim}");
    }

    #[test]
    fn truncation_fails() {
        let sim = speech_similarity(
            "Обнаружено входящее сообщение, сэр.",
            "Обнаружено входящее сообщение, с",
        );
        assert!(sim < 0.88, "обрыв фразы должен уходить в пересинтез: {sim}");
    }
}
