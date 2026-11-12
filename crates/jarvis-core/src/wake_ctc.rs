//! Wake word по CTC-постерам FastConformer: эталон — **текст**, а не облако аудио.
//!
//! Метрический детектор (`wake.rs`) сравнивает окно с облаком эмбеддингов,
//! набранным из синтетических голосов. Следствия: новое активационное слово
//! требует синтеза дикторов и записи сэмплов, а представление (mean-pool
//! кадров) теряет порядок звуков — «Джарвис» и «даже» сближаются.
//!
//! Здесь эталон — слово текстом. CTC-постеры кадров фонетичны: у разных
//! дикторов они сходятся к одному и тому же распределению по токенам,
//! поэтому совпадение не зависит от тембра. Слово задаётся текстом в
//! рантайме — любое активационное слово, без переобучения.
//!
//! Сопоставление идёт по ТЕКСТУ декода, а не по id токенов: модель
//! сегментирует услышанное слово своими кусками («компьютер» →
//! `▁ком|п|ью|тер`), а текстовый токенизатор — своими
//! (`▁Ко|м|п|ью|тер`). Сравнение по тексту переживает любую сегментацию.

use crate::stt::Stt;
use std::sync::{Arc, Mutex};

/// Лог-постеры CTC по кадрам окна: `frames × vocab`, раскладка построчно.
///
/// `blank` — последний индекс в словаре (CTC-blank в NeMo).
pub struct CtcPosteriors {
    pub frames: usize,
    pub vocab: usize,
    pub blank: usize,
    /// log-softmax по каждому кадру, построчно: `data[f * vocab + v]`.
    pub data: Vec<f32>,
}

impl CtcPosteriors {
    /// Лог-постер кадра `f` для токена `v`.
    pub fn logp(&self, f: usize, v: usize) -> f32 {
        self.data[f * self.vocab + v]
    }
}

/// Посчитать лог-постеры CTC для окна 16 кГц mono.
///
/// None — модель не отдала постеры (ошибка forward или пустой вход).
pub fn ctc_log_posteriors(stt: &Arc<Mutex<Stt>>, window: &[f32]) -> Option<CtcPosteriors> {
    let guard = stt.lock().ok()?;
    let model = guard.model();
    let (mel, length) = model.frontend.compute(window, &model.device).ok()?;
    let (enc, t_valid) = model.encoder.forward(&mel, length).ok()?;
    let (t, _d) = (enc.dim(1).ok()?, enc.dim(2).ok()?);
    let logits = model.ctc_head.logits(&enc).ok()?;
    let (bt, vocab) = (logits.dim(1).ok()?, logits.dim(2).ok()?);
    let flat: Vec<f32> = logits.reshape((bt * vocab,)).ok()?.to_vec1::<f32>().ok()?;
    let frames = t_valid.min(t).min(bt);
    let blank = vocab - 1;
    // log_softmax по каждому кадру — вручную, чтобы не зависеть от
    // поведения log_softmax на 3D-тензоре в разных бэкендах.
    let mut data = Vec::with_capacity(frames * vocab);
    for f in 0..frames {
        let row = &flat[f * vocab..(f + 1) * vocab];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        let lse = max + sum.ln();
        for &v in row {
            data.push(v - lse);
        }
    }
    Some(CtcPosteriors {
        frames,
        vocab,
        blank,
        data,
    })
}

/// Greedy-декод CTC: поток `(токен, лог-постер, кадр)` без blank'ов.
pub fn greedy_stream(post: &CtcPosteriors) -> Vec<(usize, f32, usize)> {
    let mut out = Vec::new();
    let mut prev = usize::MAX;
    for f in 0..post.frames {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for v in 0..post.vocab {
            let lp = post.logp(f, v);
            if lp > best_v {
                best_v = lp;
                best = v;
            }
        }
        if best != post.blank && best != prev {
            out.push((best, best_v, f));
        }
        prev = best;
    }
    out
}

/// Привести текст к сопоставимому виду: нижний регистр, только буквы и цифры.
pub fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Найденное вхождение слова: средний лог-постер на токен и кадровый размах.
pub struct WordMatch {
    /// Средний лог-постер по токенам вхождения (меньше — хуже).
    pub score: f32,
    /// Число кадров, которые заняло вхождение.
    pub span_frames: usize,
    /// Сколько токенов входит во вхождение.
    pub tokens: usize,
}

/// Найти вхождение слова в декоде CTC по тексту.
///
/// Возвращает None, если модель не услышала слово в этом окне. Счёт —
/// средний лог-постер по токенам вхождения: близко к 0 означает, что модель
/// была уверена в каждом токене (вероятность ≈ 1), сильно ниже — что
/// услышала, но неуверенно.
pub fn match_word_text(
    stt: &Arc<Mutex<Stt>>,
    post: &CtcPosteriors,
    word: &str,
) -> Option<WordMatch> {
    let needle = normalize(word);
    if needle.is_empty() || post.frames == 0 {
        return None;
    }
    let stream = greedy_stream(post);
    if stream.is_empty() {
        return None;
    }
    // Текст декода + карта «символ → кадр/лог-постер».
    let mut text = String::new();
    let mut chars: Vec<(usize, f32)> = Vec::new(); // (кадр, лог-постер) на символ
    {
        let guard = stt.lock().ok()?;
        let tokenizer = &guard.model().tokenizer;
        for (id, lp, frame) in &stream {
            let Some(piece) = tokenizer.piece(*id) else {
                continue;
            };
            for c in normalize(piece).chars() {
                text.push(c);
                chars.push((*frame, *lp));
            }
        }
    }
    if text.is_empty() {
        return None;
    }
    let hay: Vec<char> = text.chars().collect();
    let pat: Vec<char> = needle.chars().collect();
    // Ищем подстроку (буквы и цифры нормализованы, границы слов не важны).
    let mut start = 0usize;
    while start + pat.len() <= hay.len() {
        if hay[start..start + pat.len()] == pat[..] {
            let seg = &chars[start..start + pat.len()];
            // Среднее по токенам: у каждого кадра одна и та же вероятность
            // на все его символы, поэтому усредняем по уникальным кадрам.
            let mut seen: Vec<(usize, f32)> = Vec::new();
            for (frame, lp) in seg {
                if seen.last().map(|(f, _)| *f) != Some(*frame) {
                    seen.push((*frame, *lp));
                }
            }
            let score = seen.iter().map(|(_, lp)| *lp).sum::<f32>() / seen.len() as f32;
            let first = seen.first().map(|(f, _)| *f).unwrap_or(0);
            let last = seen.last().map(|(f, _)| *f).unwrap_or(0);
            return Some(WordMatch {
                score,
                span_frames: last.saturating_sub(first) + 1,
                tokens: seen.len(),
            });
        }
        start += 1;
    }
    None
}

/// Нормализовать слово-активацию (для сопоставления с декодом).
pub fn normalize_word(word: &str) -> String {
    normalize(word)
}

/// Токены слова в словаре CTC (SentencePiece Unigram) — для диагностики:
/// показывает, как модель может сегментировать это слово.
pub fn encode_word(stt: &Arc<Mutex<Stt>>, word: &str) -> Option<Vec<usize>> {
    let guard = stt.lock().ok()?;
    guard.model().tokenizer.encode(word).ok()
}

/// Параметры потокового CTC-детектора.
#[derive(Debug, Clone)]
pub struct CtcConfig {
    /// Длина окна, мс.
    pub window_ms: usize,
    /// Шаг между окнами, мс.
    pub hop_ms: usize,
    /// Пауза после срабатывания, мс.
    pub cooldown_ms: u64,
    /// Сколько последних окон участвуют в решении.
    pub smoothing: usize,
    /// Минимум окон со счётом ≥ порога для срабатывания.
    pub min_hits: usize,
    /// Абсолютный минимум RMS.
    pub vad_rms: f32,
    /// Порог относительно шумового пола.
    pub vad_ratio: f32,
    /// Сколько после конца речи ещё считаются окна, мс.
    pub hangover_ms: usize,
    /// Порог среднего лог-постера: 0 — модель уверена на 100%.
    pub threshold: f32,
}

impl Default for CtcConfig {
    fn default() -> Self {
        Self {
            window_ms: 1200,
            // Шаг 160 мс (а не 240): окно должно накрыть слово целиком
            // несколько раз подряд — иначе модель видит обрезанное слово
            // и декодит его неверно («Дервись» вместо «Джарвис»).
            hop_ms: 160,
            cooldown_ms: 2500,
            smoothing: 3,
            min_hits: 2,
            vad_rms: 0.004,
            vad_ratio: 3.0,
            hangover_ms: 1200,
            // Порог с большим запасом: на синтетическом корпусе худший
            // позитив даёт −0.08, а на шуме и негативах слово не
            // декодируется вовсе (счёт = −∞). Порог защищает только от
            // неуверенных совпадений; основную защиту даёт требование,
            // чтобы слово реально прозвучало в декоде.
            threshold: -0.5,
        }
    }
}

/// Диагностика последнего окна.
#[derive(Debug, Clone, Default)]
pub struct CtcDiag {
    pub rms: f32,
    pub noise_floor: f32,
    pub voiced: bool,
    pub hangover_left: usize,
    pub encoder_ran: bool,
    /// Что услышала модель (декод CTC).
    pub decoded: String,
    /// Счёт совпадения (средний лог-постер), None — слова не было.
    pub score: Option<f32>,
    /// Сколько последних окон набрали ≥ порога.
    pub hits: usize,
    /// Порог.
    pub threshold: f32,
}

/// Срабатывание детектора.
#[derive(Debug, Clone)]
pub struct CtcDetection {
    /// Счёт лучшего окна.
    pub score: f32,
    /// Что услышала модель в сработавшем окне.
    pub decoded: String,
}

/// Потоковый детектор wake word по CTC-постерам.
///
/// Логика окна повторяет проверенную метрическую (VAD с адаптивным шумовым
/// полом, hangover, hop по сэмплам, cooldown по аудио-времени), но решение
/// принимается не по сходству с облаком эмбеддингов, а по факту: модель
/// должна была услышать слово.
pub struct CtcWakeEngine {
    stt: Arc<Mutex<Stt>>,
    word: String,
    config: CtcConfig,
    buffer: std::collections::VecDeque<f32>,
    since_embed: usize,
    since_detection: usize,
    hangover_left: usize,
    noise_floor: f32,
    /// Окна-счёты: (позиция в аудио-времени, счёт).
    recent: std::collections::VecDeque<(usize, f32)>,
    last_score: Option<f32>,
    last_diag: Option<CtcDiag>,
    total_samples: usize,
    samples_per_window: usize,
    samples_per_hop: usize,
    cooldown_samples: usize,
    hangover_samples: usize,
}

impl CtcWakeEngine {
    /// Создать детектор для слова `word` (любое активационное слово).
    pub fn new(stt: Arc<Mutex<Stt>>, word: &str, config: CtcConfig) -> Self {
        let samples_per_window = config.window_ms * 16_000 / 1000;
        let samples_per_hop = (config.hop_ms * 16_000 / 1000).max(1);
        let cooldown_samples = config.cooldown_ms as usize * 16_000 / 1000;
        let hangover_samples = config.hangover_ms * 16_000 / 1000;
        Self {
            stt,
            word: normalize(word),
            config,
            buffer: std::collections::VecDeque::with_capacity(samples_per_window + samples_per_hop),
            since_embed: 0,
            since_detection: 0,
            hangover_left: 0,
            noise_floor: 0.0,
            recent: std::collections::VecDeque::new(),
            last_score: None,
            last_diag: None,
            total_samples: 0,
            samples_per_window,
            samples_per_hop,
            cooldown_samples,
            hangover_samples,
        }
    }

    /// Слово-активация.
    pub fn word(&self) -> &str {
        &self.word
    }

    /// Счёт последнего окна (даже без совпадения).
    pub fn last_score(&self) -> Option<f32> {
        self.last_score
    }

    /// Диагностика последнего окна.
    pub fn last_diag(&self) -> Option<CtcDiag> {
        self.last_diag.clone()
    }

    /// Сброс потокового состояния.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.recent.clear();
        self.since_embed = 0;
        self.since_detection = 0;
        self.hangover_left = 0;
        self.noise_floor = 0.0;
        self.last_score = None;
        self.last_diag = None;
    }

    /// Подать кусок потока 16 кГц mono. Возвращает срабатывание либо None.
    pub fn process(&mut self, chunk: &[f32]) -> Option<CtcDetection> {
        let mut detection = None;
        let mut offset = 0;
        while offset < chunk.len() {
            let take = (chunk.len() - offset).min(self.samples_per_hop);
            if let Some(d) = self.process_sub(&chunk[offset..offset + take]) {
                detection = Some(d);
            }
            offset += take;
        }
        detection
    }

    fn process_sub(&mut self, sub: &[f32]) -> Option<CtcDetection> {
        let mut diag = CtcDiag {
            noise_floor: self.noise_floor,
            hangover_left: self.hangover_left,
            threshold: self.config.threshold,
            ..Default::default()
        };

        self.buffer.extend(sub.iter().copied());
        while self.buffer.len() > self.samples_per_window {
            self.buffer.pop_front();
        }
        self.since_embed += sub.len();
        self.since_detection += sub.len();
        self.total_samples += sub.len();

        // Шумовой пол: быстро вниз, медленно вверх.
        let rms = (sub.iter().map(|x| x * x).sum::<f32>() / sub.len().max(1) as f32).sqrt();
        let voiced = rms > self.config.vad_rms && rms > self.noise_floor * self.config.vad_ratio;
        if rms < self.noise_floor {
            self.noise_floor = rms;
        } else {
            let dt = sub.len() as f32 / 16_000.0;
            self.noise_floor += 0.02 * dt * (rms - self.noise_floor);
        }
        if voiced {
            self.hangover_left = self.hangover_samples;
        } else if self.hangover_left > 0 {
            self.hangover_left = self.hangover_left.saturating_sub(sub.len());
        } else {
            // Тишина: окна-счёты протухают — иначе старый пик выстрелит на
            // первом же постороннем звуке.
            self.recent.clear();
        }

        diag.rms = rms;
        diag.noise_floor = self.noise_floor;
        diag.voiced = voiced;
        diag.hangover_left = self.hangover_left;

        let window_full = self.buffer.len() == self.samples_per_window;
        let hop_ready = self.since_embed >= self.samples_per_hop;
        let encoder_ran = window_full && hop_ready && (voiced || self.hangover_left > 0);
        if !encoder_ran {
            self.last_diag = Some(diag);
            return None;
        }
        self.since_embed = 0;
        diag.encoder_ran = true;

        let window: Vec<f32> = self.buffer.iter().copied().collect();
        let post = match ctc_log_posteriors(&self.stt, &window) {
            Some(p) => p,
            None => {
                self.last_diag = Some(diag);
                return None;
            }
        };
        let stream = greedy_stream(&post);
        {
            let Ok(guard) = self.stt.lock() else {
                self.last_diag = Some(diag);
                return None;
            };
            let ids: Vec<usize> = stream.iter().map(|(id, _, _)| *id).collect();
            diag.decoded = guard.model().tokenizer.decode(&ids);
        }
        let m = match_word_text(&self.stt, &post, &self.word);
        let score = m.as_ref().map(|m| m.score);
        diag.score = score;

        if let Some(s) = score {
            self.last_score = Some(s);
            self.recent.push_back((self.total_samples, s));
        }
        let span = self.samples_per_window;
        self.recent
            .retain(|(at, _)| self.total_samples.saturating_sub(*at) <= span);
        while self.recent.len() > self.config.smoothing.max(1) {
            self.recent.pop_front();
        }
        let hits = self
            .recent
            .iter()
            .filter(|(_, s)| *s >= self.config.threshold)
            .count();
        diag.hits = hits;

        if hits < self.config.min_hits {
            self.last_diag = Some(diag);
            return None;
        }
        if self.since_detection < self.cooldown_samples {
            self.last_diag = Some(diag);
            return None;
        }
        self.since_detection = 0;
        // Счёт лучшего окна берём ДО очистки очереди.
        let best = self
            .recent
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let best = if best.is_finite() {
            best
        } else {
            score.unwrap_or_default()
        };
        self.recent.clear();
        let decoded = diag.decoded.clone();
        self.last_diag = Some(diag);
        Some(CtcDetection {
            score: best,
            decoded,
        })
    }
}
