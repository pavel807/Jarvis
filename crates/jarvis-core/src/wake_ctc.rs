//! Wake word по CTC-постерам FastConformer: эталон — **слово (или фраза) текстом**,
//! а не облако аудио.
//!
//! Метрический детектор (`wake.rs`) сравнивает окно с облаком эмбеддингов,
//! набранным из синтетических голосов. Следствия: новое активационное слово
//! требует синтеза дикторов и записи сэмплов, а представление (mean-pool
//! кадров) теряет порядок звуков — «Джарвис» и «даже» сближаются.
//!
//! Здесь эталон — фраза текстом. CTC-постеры кадров фонетичны: у разных
//! дикторов они сходятся к одному и тому же распределению по токенам,
//! поэтому совпадение не зависит от тембра. Фраза задаётся текстом в
//! рантайме — любая, без переобучения.
//!
//! Сопоставление идёт по ТЕКСТУ декода, а не по id токенов: модель
//! сегментирует услышанное слово своими кусками («компьютер» →
//! `▁ком|п|ью|тер`), а текстовый токенизатор — своими
//! (`▁Ко|м|п|ью|тер`). Сравнение по тексту переживает любую сегментацию.
//!
//! Сопоставление нечёткое (выравнивание Селлерса): живая речь декодируется
//! с фонетическими ошибками («Джарвис» → «Джавис», «Джалис»), дословный
//! поиск терял половину произнесений. Ошибки стоят штрафа в единицах
//! лог-постера, поэтому счёт остаётся на той же шкале, что и раньше.
//!
//! Горячий путь (окно раз в ~160 мс) почти не аллоцирует:
//! - `log_softmax` и `argmax` считаются на стороне модели (`candle-nn`), в
//!   Rust приходит только строка argmax-токенов и их лог-постеры — вместо
//!   матрицы «кадры × словарь»;
//! - переиспользуются буферы [`CtcScratch`]: поток декода, текст, карта
//!   символов и список id для декодирования;
//! - нормализация кусков идёт без промежуточных `String`.

use crate::stt::Stt;
use std::sync::{Arc, Mutex};

/// Что CTC услышала в окне: argmax-токен каждого кадра и его лог-постер.
///
/// Полная матрица постеров (кадры × словарь) не нужна: для решения хватает
/// argmax и его вероятности, а она считается на стороне модели.
#[derive(Default, Clone)]
pub struct CtcFrames {
    /// Число кадров.
    pub frames: usize,
    /// Размер словаря вместе с blank.
    pub vocab: usize,
    /// blank — последний индекс словаря (CTC-blank в NeMo).
    pub blank: usize,
    /// argmax-токен кадра.
    pub ids: Vec<usize>,
    /// лог-постер argmax-токена кадра.
    pub logp: Vec<f32>,
}

/// Буферы, переиспользуемые между окнами: в горячем пути аллокаций нет.
#[derive(Default)]
pub struct CtcScratch {
    /// Результат декода модели: argmax-токены кадров.
    pub frames: CtcFrames,
    /// Поток greedy-декода: (токен, лог-постер, кадр).
    pub stream: Vec<(usize, f32, usize)>,
    /// Нормализованный текст декода.
    pub text: String,
    /// Карта «символ → (кадр, лог-постер)».
    pub chars: Vec<(usize, f32)>,
    /// id для декодирования текста.
    pub decode_ids: Vec<usize>,
    /// Символы декода (для выравнивания).
    pub hay: Vec<char>,
    /// Символы фразы (для выравнивания).
    pub pat: Vec<char>,
    /// Два слоя таблицы выравнивания (цены).
    pub dp_a: Vec<f32>,
    pub dp_b: Vec<f32>,
    /// Два слоя таблицы выравнивания (начала вхождений).
    pub st_a: Vec<usize>,
    pub st_b: Vec<usize>,
}

/// Посчитать argmax-токены и их лог-постеры для окна 16 кГц mono.
///
/// None — модель не отдала логиты (ошибка forward или пустой вход).
pub fn ctc_frames(
    stt: &Arc<Mutex<Stt>>,
    window: &[f32],
    scratch: &mut CtcScratch,
) -> Option<()> {
    use candle_core::D;
    let guard = stt.lock().ok()?;
    let model = guard.model();
    let (mel, length) = model.frontend.compute(window, &model.device).ok()?;
    let (enc, t_valid) = model.encoder.forward(&mel, length).ok()?;
    let (t, _d) = (enc.dim(1).ok()?, enc.dim(2).ok()?);
    let logits = model.ctc_head.logits(&enc).ok()?;
    let (bt, vocab) = (logits.dim(1).ok()?, logits.dim(2).ok()?);
    // log-softmax и argmax — на стороне модели: в Rust тянулась бы вся
    // матрица «кадры × словарь» (сотни килобайт на каждое окно).
    let post = candle_nn::ops::log_softmax(&logits, D::Minus1).ok()?;
    let values = post.max(D::Minus1).ok()?;
    let indices = post.argmax(D::Minus1).ok()?;
    let logp: Vec<f32> = values.reshape((bt,)).ok()?.to_vec1::<f32>().ok()?;
    let ids: Vec<u32> = indices.reshape((bt,)).ok()?.to_vec1::<u32>().ok()?;
    let frames = t_valid.min(t).min(bt).min(logp.len()).min(ids.len());
    scratch.frames.frames = frames;
    scratch.frames.vocab = vocab;
    scratch.frames.blank = vocab.saturating_sub(1);
    scratch.frames.ids.clear();
    scratch.frames.ids.extend(ids.iter().take(frames).map(|v| *v as usize));
    scratch.frames.logp.clear();
    scratch.frames.logp.extend(logp.iter().take(frames).copied());
    Some(())
}

/// Greedy-декод CTC: поток `(токен, лог-постер, кадр)` без blank'ов.
///
/// Пишет в переиспользуемый буфер `out`.
pub fn greedy_stream(frames: &CtcFrames, out: &mut Vec<(usize, f32, usize)>) {
    out.clear();
    let mut prev = usize::MAX;
    for f in 0..frames.frames {
        let id = frames.ids[f];
        if id != frames.blank && id != prev {
            out.push((id, frames.logp[f], f));
        }
        prev = id;
    }
}

/// Привести текст к сопоставимому виду: нижний регистр, только буквы и цифры.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    push_normalized(&mut out, text);
    out
}

/// Дописать в `out` нормализованные символы `text` — без промежуточного `String`.
///
/// «ё» приравнивается к «е»: декод модели и фраза пользователя могут
/// использовать любой вариант, а по звуку это один и тот же гласный.
fn push_normalized(out: &mut String, text: &str) {
    for c in text.chars() {
        if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                out.push(if lc == 'ё' { 'е' } else { lc });
            }
        }
    }
}

/// Поднять уровень окна до целевого RMS (модель обучена на речи с RMS ~0.09).
///
/// Масштабирует только вверх: громкую речь не трогаем — иначе изменились бы
/// измерения на обучающем корпусе. Усиление ограничено `max_gain`, чтобы шум
/// не раздувался до уровня речи.
pub fn normalize_level(src: &[f32], dst: &mut Vec<f32>, target_rms: f32, max_gain: f32) {
    dst.clear();
    dst.extend_from_slice(src);
    if target_rms <= 0.0 || dst.is_empty() {
        return;
    }
    let rms = (dst.iter().map(|x| x * x).sum::<f32>() / dst.len() as f32).sqrt();
    if rms <= 0.0 || rms >= target_rms {
        return;
    }
    let gain = (target_rms / rms).min(max_gain);
    if gain <= 1.0 {
        return;
    }
    for x in dst.iter_mut() {
        *x *= gain;
    }
}

/// Найденное вхождение слова: средний лог-постер на символ и кадровый размах.
pub struct WordMatch {
    /// Средний лог-постер по символам фразы (меньше — хуже).
    pub score: f32,
    /// Число кадров, которые заняло вхождение.
    pub span_frames: usize,
    /// Сколько уникальных кадров входит во вхождение.
    pub tokens: usize,
}

/// Штрафы нечёткого сопоставления в единицах лог-постера (за ошибку, не за
/// символ; прибавляются к положительной цене выравнивания). Реальный голос
/// почти всегда декодируется с 1–2 фонетическими ошибками («Джарвис» →
/// «Джавис», «Джалис»): дословный поиск их терял. Штрафы подобраны так, что
/// совпадение с одной ошибкой теряет ~0.05–0.1 на символ — много меньше
/// разрыва до мусорных выравниваний (см. порог).
const SUB_PENALTY: f32 = 0.35;
const DEL_PENALTY: f32 = 0.5;
const INS_PENALTY: f32 = 0.15;

/// Найти вхождение фразы в декоде CTC по тексту, нечётко.
///
/// Выравнивание Селлерса: фраза обязана войти целиком, а декод вокруг и
/// между её символами — любой (CTC вставляет мусор между звуками).
/// Цена совпадения символа — минус его лог-постер (≈ 0.01 при уверенности),
/// замена — постер + [`SUB_PENALTY`], пропуск символа фразы —
/// [`DEL_PENALTY`], мусор между символами — [`INS_PENALTY`].
///
/// Счёт = минус минимальная цена / длина фразы: близко к 0 — модель была
/// уверена в каждом символе, заметно ниже — услышала с ошибками или
/// неуверенно. None — декода нет вовсе.
pub fn match_word_text(
    stt: &Arc<Mutex<Stt>>,
    scratch: &mut CtcScratch,
    word: &str,
) -> Option<WordMatch> {
    let needle = normalize(word);
    if needle.is_empty() || scratch.frames.frames == 0 {
        return None;
    }
    greedy_stream(&scratch.frames, &mut scratch.stream);
    if scratch.stream.is_empty() {
        return None;
    }
    // Текст декода + карта «символ → кадр/лог-постер».
    scratch.text.clear();
    scratch.chars.clear();
    {
        let guard = stt.lock().ok()?;
        let tokenizer = &guard.model().tokenizer;
        for (id, lp, frame) in &scratch.stream {
            let Some(piece) = tokenizer.piece(*id) else {
                continue;
            };
            let before = scratch.text.len();
            push_normalized(&mut scratch.text, piece);
            for _ in before..scratch.text.len() {
                scratch.chars.push((*frame, *lp));
            }
        }
    }
    if scratch.text.is_empty() {
        return None;
    }
    // Символы декода и фразы — в переиспользуемые буферы.
    scratch.hay.clear();
    scratch.hay.extend(scratch.text.chars());
    scratch.pat.clear();
    scratch.pat.extend(needle.chars());
    let Some((cost, start, end)) = fuzzy_align(
        &scratch.hay,
        &scratch.pat,
        &scratch.chars,
        &mut scratch.dp_a,
        &mut scratch.dp_b,
        &mut scratch.st_a,
        &mut scratch.st_b,
    ) else {
        return None;
    };
    // Среднее по уникальным кадрам выравнивания: символы одного кадра делят
    // один постер, а мусор между символами не должен размывать счёт.
    let seg = &scratch.chars[start..end];
    let mut seen: Vec<(usize, f32)> = Vec::new();
    for (frame, lp) in seg {
        if seen.last().map(|(f, _)| *f) != Some(*frame) {
            seen.push((*frame, *lp));
        }
    }
    let score = -cost / scratch.pat.len() as f32;
    let first = seen.first().map(|(f, _)| *f).unwrap_or(0);
    let last = seen.last().map(|(f, _)| *f).unwrap_or(0);
    Some(WordMatch {
        score,
        span_frames: last.saturating_sub(first) + 1,
        tokens: seen.len(),
    })
}

/// Нечёткое выравнивание фразы `pat` на декоде `hay` — алгоритм Селлерса
/// с бюджетом правок: фраза обязана войти целиком, декод вокруг и между её
/// символами — любой.
///
/// Бюджет правок: сколько символов фразы может быть услышано с ошибкой
/// (замена, пропуск) или разделено мусором. Без бюджета нечёткость
/// сопоставляет фразе вообще чужие слова: «анфиса» выравнивалась на
/// «джарвис» двумя пропусками и двумя заменами. Бюджет = треть длины,
/// короткие фразы (до 5 символов) — только дословно. Право на совпадение
/// получает только выравнивание, уложившееся в бюджет: случайно рассыпанных
/// по декоду символов для этого не хватает.
///
/// Возвращает (цена, начало, конец) лучшего куска декода; None — если ни
/// одно выравнивание не уложилось в бюджет. `dp_*`/`st_*` — переиспользуемые
/// буферы таблицы.
fn fuzzy_align(
    hay: &[char],
    pat: &[char],
    chars: &[(usize, f32)],
    dp_a: &mut Vec<f32>,
    dp_b: &mut Vec<f32>,
    st_a: &mut Vec<usize>,
    st_b: &mut Vec<usize>,
) -> Option<(f32, usize, usize)> {
    let (n, m) = (pat.len(), hay.len());
    // Короткие фразы (до 7 символов) — только дословно: одна правка на
    // 5–6 символах — это до 20% слова, и «робот» ловил «работ» из обычной
    // речи со счётом −0.07. Дословный поиск короткое слово тоже найдёт в
    // мусоре редко — совпадение по 5+ символам подряд само по себе специфично.
    let max_edits = if n < 7 { 0 } else { (n / 3).clamp(1, 3) };
    let layer = (max_edits + 1) * (m + 1);
    dp_a.clear();
    dp_a.resize(layer, f32::INFINITY);
    st_a.clear();
    st_a.resize(layer, 0);
    dp_b.clear();
    dp_b.resize(layer, f32::INFINITY);
    st_b.clear();
    st_b.resize(layer, 0);
    // Слой i = 0: пустая фраза совпадает с любым куском декода бесплатно,
    // вхождение может начинаться где угодно.
    for e in 0..=max_edits {
        let base = e * (m + 1);
        for j in 0..=m {
            dp_a[base + j] = 0.0;
            st_a[base + j] = j;
        }
    }
    // Слои чередуются: слой i-1 в одном буфере, слой i пишется в другой.
    let mut prev_in_a = true;
    for i in 1..=n {
        let (prev_d, prev_s, cur_d, cur_s): (&[f32], &[usize], &mut [f32], &mut [usize]) =
            if prev_in_a {
                (dp_a.as_slice(), st_a.as_slice(), dp_b, st_b)
            } else {
                (dp_b.as_slice(), st_b.as_slice(), dp_a, st_a)
            };
        cur_d.fill(f32::INFINITY);
        cur_s.fill(0);
        for e in 0..=max_edits {
            let base = e * (m + 1);
            // Слой e-1 в предыдущей строке; при e = 0 чтений оттуда нет.
            let pbase1 = e.saturating_sub(1) * (m + 1);
            // Слой e-1 в текущей строке (мусор между символами): заполнен
            // раньше в этом же проходе.
            let cbase = e.saturating_sub(1) * (m + 1);
            // j = 0: префикс фразы выровнен только пропусками символов.
            cur_d[base] = if e >= 1 {
                prev_d[pbase1] + DEL_PENALTY
            } else {
                f32::INFINITY
            };
            cur_s[base] = if e >= 1 { prev_s[pbase1] } else { 0 };
            for j in 1..=m {
                let (_, lp) = chars[j - 1];
                let matched = hay[j - 1] == pat[i - 1];
                let mut best = f32::INFINITY;
                let mut bstart = 0usize;
                // Пропуск символа фразы.
                if e >= 1 {
                    let c = prev_d[pbase1 + j] + DEL_PENALTY;
                    if c < best {
                        best = c;
                        bstart = prev_s[pbase1 + j];
                    }
                    // Мусор между символами фразы.
                    let c = cur_d[cbase + j - 1] + INS_PENALTY;
                    if c < best {
                        best = c;
                        bstart = cur_s[cbase + j - 1];
                    }
                }
                // Потребление символа декода: совпадение (бесплатно от правок)
                // или замена (расходует одну правку).
                let c = if matched {
                    prev_d[base + j - 1] - lp
                } else if e >= 1 {
                    prev_d[pbase1 + j - 1] - lp + SUB_PENALTY
                } else {
                    f32::INFINITY
                };
                let s = if matched {
                    prev_s[base + j - 1]
                } else {
                    prev_s[pbase1 + j - 1]
                };
                if c < best {
                    best = c;
                    bstart = s;
                }
                cur_d[base + j] = best;
                cur_s[base + j] = bstart;
            }
        }
        prev_in_a = !prev_in_a;
    }
    // Фраза вошла целиком (в бюджет правок), кусок может кончаться где угодно.
    let (final_d, final_s) = if prev_in_a {
        (dp_a.as_slice(), st_a.as_slice())
    } else {
        (dp_b.as_slice(), st_b.as_slice())
    };
    let mut best_cost = f32::INFINITY;
    let mut best_j = 0usize;
    let mut best_start = 0usize;
    for e in 0..=max_edits {
        let base = e * (m + 1);
        for j in 0..=m {
            if final_d[base + j] < best_cost {
                best_cost = final_d[base + j];
                best_j = j;
                best_start = final_s[base + j];
            }
        }
    }
    if best_j <= best_start || !best_cost.is_finite() {
        return None;
    }
    Some((best_cost, best_start, best_j))
}

/// Нормализовать фразу-активацию (для сопоставления с декодом).
pub fn normalize_word(word: &str) -> String {
    normalize(word)
}

/// Токены фразы в словаре CTC (SentencePiece Unigram) — для диагностики:
/// показывает, как модель может сегментировать эту фразу.
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
    /// Целевой RMS окна для модели: ниже этого модель не декодирует слова.
    /// 0 — не нормализовать.
    pub norm_target_rms: f32,
    /// Максимальное усиление при нормализации (защита от усиления шума).
    pub norm_max_gain: f32,
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
            // Абсолютный порог VAD низкий: гарнитура/ноутбук дают речь в
            // 20–30 раз тише обучающих записей (RMS 0.09). Отсекает шум
            // относительный порог (vad_ratio) от адаптивного шумового пола.
            vad_rms: 0.0006,
            vad_ratio: 3.0,
            hangover_ms: 1200,
            // Порог с большим запасом: на синтетическом корпусе худший
            // позитив даёт −0.08, а на шуме и негативах фраза не
            // декодируется вовсе (счёт = −∞). Порог защищает только от
            // неуверенных совпадений; основную защиту даёт требование,
            // чтобы фраза реально прозвучала в декоде.
            threshold: -0.5,
            // Модель обучена на речи с RMS ~0.09. Микрофон может дать в 30
            // раз тише — тогда CTC не декодирует ничего, и слово не слышно
            // никогда. Окно поднимается до целевого уровня (только вверх:
            // громкую речь не трогаем, лишние искажения не нужны).
            norm_target_rms: 0.08,
            norm_max_gain: 30.0,
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
    /// Счёт совпадения (средний лог-постер), None — фразы не было.
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
/// должна была услышать фразу.
pub struct CtcWakeEngine {
    stt: Arc<Mutex<Stt>>,
    /// Нормализованная фраза-активация (сравнивается с декодом).
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
    /// Переиспользуемые буферы горячего пути.
    scratch: CtcScratch,
    /// Окно с нормализованным уровнем (копия: кольцевой буфер не трогаем).
    window: Vec<f32>,
}

impl CtcWakeEngine {
    /// Создать детектор для фразы `word` (любая активационная фраза).
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
            scratch: CtcScratch::default(),
            window: Vec::with_capacity(samples_per_window),
        }
    }

    /// Фраза-активация.
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

        // Окно копируется в переиспользуемый буфер и нормализуется по уровню:
        // кольцевой буфер не трогаем (иначе масштабирование накапливалось бы).
        let src = self.buffer.make_contiguous();
        normalize_level(
            src,
            &mut self.window,
            self.config.norm_target_rms,
            self.config.norm_max_gain,
        );
        if ctc_frames(&self.stt, &self.window, &mut self.scratch).is_none() {
            self.last_diag = Some(diag);
            return None;
        }
        // Поток ТЕКУЩЕГО окна: раньше он заполнялся только внутри
        // match_word_text, и декод в диагностике отставал на одно окно.
        greedy_stream(&self.scratch.frames, &mut self.scratch.stream);
        {
            let Ok(guard) = self.stt.lock() else {
                self.last_diag = Some(diag);
                return None;
            };
            // Декод для журнала: id переиспользуем, текст — для диагностики.
            self.scratch
                .decode_ids
                .clear();
            self.scratch
                .decode_ids
                .extend(self.scratch.stream.iter().map(|(id, _, _)| *id));
            diag.decoded = guard
                .model()
                .tokenizer
                .decode(&self.scratch.decode_ids);
        }
        let m = match_word_text(&self.stt, &mut self.scratch, &self.word);
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
        let best = if best.is_finite() { best } else { score.unwrap_or_default() };
        self.recent.clear();
        let decoded = diag.decoded.clone();
        self.last_diag = Some(diag);
        Some(CtcDetection { score: best, decoded })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Декод: каждый символ — свой кадр с постером `lp` (уверенный = -0.02).
    fn chars_of(text: &str, lp: f32) -> Vec<(usize, f32)> {
        text.chars().enumerate().map(|(i, _)| (i, lp)).collect()
    }

    fn align(hay: &str, pat: &str, lp: f32) -> Option<f32> {
        let chars = chars_of(hay, lp);
        let hay: Vec<char> = hay.chars().collect();
        // Боевой путь нормализует фразу до выравнивания (match_word_text).
        let pat: Vec<char> = normalize(pat).chars().collect();
        fuzzy_align(
            &hay,
            &pat,
            &chars,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .map(|(cost, _, _)| -cost / pat.len() as f32)
    }

    #[test]
    fn дословное_совпадение_уверенно() {
        // Джарвис внутри длинного декода: счёт = постер символов.
        let s = align("вот джарвис слушает", "джарвис", -0.02).expect("должно совпасть");
        assert!((s + 0.02).abs() < 1e-4, "счёт {s}");
    }

    #[test]
    fn одна_замена_как_джалис() {
        // Реальный промах калибровки: в→л (замена) — счёт заметно хуже
        // дословного, но в пределах боевого порога -0.5.
        let s = align("джалис", "джарвис", -0.1).expect("замена в бюджете");
        // 5 совпадений (д ж а и с), замена л→р (0.1 + 0.35), пропуск в (0.5).
        let expected = -(5.0 * 0.1 + 0.45 + 0.5) / 7.0;
        assert!((s - expected).abs() < 1e-4, "счёт {s} ≠ {expected}");
        assert!(s > -0.5, "счёт {s} должен проходить порог");
    }

    #[test]
    fn чужая_фраза_анфиса_отклонена() {
        // «анфиса» не содержит р и в: минимум 4 правки на 7 символов —
        // за бюджетом (2), как в калибровке на корпусе-приманке.
        assert_eq!(align("анфиса", "джарвис", -0.02), None);
    }

    #[test]
    fn рассыпанные_символы_не_совпадают() {
        // Символы фразы, разбросанные по посторонней речи, бюджет не набирают.
        assert_eq!(align("джаз нос ветка исло", "джарвис", -0.02), None);
    }

    #[test]
    fn короткая_фраза_только_дословно() {
        // До 7 символов бюджет правок нулевой: одна замена уже отклонение.
        assert!(align("дом", "дом", -0.02).is_some());
        assert_eq!(align("дым", "дом", -0.02), None);
        // Реальный случай калибровки: «робот» из речи ловил «работ» с
        // нечёткостью в одну замену — теперь короткие слова строги.
        assert!(align("робот", "робот", -0.02).is_some());
        assert_eq!(align("работ", "робот", -0.02), None);
    }

    #[test]
    fn отсутствующая_фраза_нет_совпадения() {
        assert_eq!(align("полная тишина без слов", "джарвис", -0.02), None);
        assert_eq!(align("", "джарвис", -0.02), None);
    }

    #[test]
    fn ё_приравнивается_к_е() {
        // Фраза с «ё» совпадает с декодом через «е» и наоборот.
        assert_eq!(normalize("ёжик"), "ежик");
        let s = align("ежик", "ёжик", -0.02).expect("ё → е");
        assert!((s + 0.02).abs() < 1e-4, "счёт {s}");
    }
}
