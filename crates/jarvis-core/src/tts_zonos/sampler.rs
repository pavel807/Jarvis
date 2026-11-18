//! Сэмплер ZONOS3: повтор, температура, top-k/top-p/min-p, мультиномиал.
//!
//! Порядок операций (эталон `servers/ggml/src/zonos3-sampler.cpp` и
//! `servers/web/src/sampler.ts`):
//!
//! 1. маска: `audio_pad_id` не выбирается никогда, спец-иды (EOA и выше) —
//!    только `stop_id`, и то не раньше `min_frames + codebook`;
//! 2. штраф за повтор по последним `rep_window` кадрам книг 0..rep_codebooks;
//! 3. greedy либо температура;
//! 4. top-k с сохранением ничьих (кандидатов может быть больше k);
//! 5. softmax по кандидатам;
//! 6. top-p по массе *до* токена (первый токен остаётся всегда);
//! 7. min-p относительно максимальной вероятности;
//! 8. мультиномиал по MT19937 (libc++ generate_canonical<double,53>).

/// Параметры сэмплирования.
#[derive(Debug, Clone)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub rep_penalty: f32,
    pub rep_window: usize,
    pub rep_codebooks: usize,
    pub seed: u32,
    pub greedy: bool,
}

impl Default for Sampling {
    /// Дефолты реестра генерации ZONOS3.
    fn default() -> Self {
        Self {
            temperature: 0.75,
            top_k: 80,
            top_p: 0.69,
            min_p: 0.0,
            rep_penalty: 2.4,
            rep_window: 96,
            rep_codebooks: 6,
            seed: 0,
            greedy: false,
        }
    }
}

/// MT19937 (32-бит) — как в эталоне.
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = 1812433253u32
                .wrapping_mul(prev)
                .wrapping_add(i as u32);
        }
        Self { mt, idx: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[i] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// libc++ `generate_canonical<double, 53>`: два розыгрыша, первый — младшее слово.
    fn canonical53(&mut self) -> f64 {
        let r1 = self.next() as f64;
        let r2 = self.next() as f64;
        (r1 + r2 * 4_294_967_296.0) / 18_446_744_073_709_551_616.0
    }

    /// `uniform_real_distribution<double>(0, s2)`.
    fn uniform(&mut self, s2: f64) -> f64 {
        s2 * self.canonical53()
    }
}

/// Сэмплер кодовых книг.
pub struct Sampler {
    pub sp: Sampling,
    rng: Mt19937,
    /// История последних токенов по книгам (для штрафа за повтор).
    hist: Vec<Vec<u32>>,
    /// Сколько кадров сгенерировано.
    pub generated: usize,
    /// Минимальное число кадров до появления стоп-токена.
    pub min_frames: usize,
    codebook_size: usize,
    eoa_id: u32,
    stop_id: u32,
    /// Смещение стоп-логита (LACI: усиление стоп-токена).
    pub stop_bias: f32,
}

impl Sampler {
    /// Паддинг в сэмплере не нужен: маска запрещает всё от `codebook_size`
    /// и выше, а пад-строки в поток добавляет промпт/движок.
    pub fn new(sp: Sampling, n_codebooks: usize, codebook_size: usize, eoa_id: u32) -> Self {
        Self {
            rng: Mt19937::new(sp.seed),
            hist: vec![Vec::new(); n_codebooks],
            generated: 0,
            min_frames: 0,
            codebook_size,
            eoa_id,
            stop_id: eoa_id,
            stop_bias: 0.0,
            sp,
        }
    }

    /// Запомнить сгенерированный кадр.
    pub fn accept(&mut self, frame: &[u32]) {
        for (cb, &token) in frame.iter().enumerate() {
            if cb >= self.hist.len() {
                break;
            }
            let h = &mut self.hist[cb];
            h.push(token);
            while h.len() > self.sp.rep_window {
                h.remove(0);
            }
        }
        self.generated += 1;
    }

    /// Выбрать токен книги `cb` из логитов `[cb*vocab .. (cb+1)*vocab)`.
    pub fn sample(&mut self, logits: &[f32], cb: usize) -> u32 {
        let vocab = logits.len();
        let mut logit = logits.to_vec();

        // 1. Маска: паддинг и спец-иды (кроме стоп-токена).
        for i in self.codebook_size..vocab {
            let allowed = i as u32 == self.stop_id
                && (i as u32 != self.eoa_id || self.eoa_id == self.stop_id)
                && self.generated >= self.min_frames + cb;
            if !allowed {
                logit[i] = f32::NEG_INFINITY;
            }
        }
        if self.stop_bias != 0.0 && (self.stop_id as usize) < vocab {
            let i = self.stop_id as usize;
            logit[i] += self.stop_bias;
        }

        // 2. Штраф за повтор: каждый уникальный код окна — один раз
        // (seen-маркировка, как в zonos3-sampler.cpp).
        if self.sp.rep_penalty != 1.0 && cb < self.sp.rep_codebooks {
            let pen = self.sp.rep_penalty;
            let n = self.codebook_size as usize;
            let mut seen = vec![false; n];
            for &t in &self.hist[cb] {
                let i = t as usize;
                if i < n && !seen[i] {
                    seen[i] = true;
                    logit[i] = if logit[i] > 0.0 {
                        logit[i] / pen
                    } else {
                        logit[i] * pen
                    };
                }
            }
        }

        // 3. Greedy.
        if self.sp.greedy {
            let mut best = 0usize;
            let mut bv = logit[0];
            for i in 1..vocab {
                if logit[i] > bv {
                    bv = logit[i];
                    best = i;
                }
            }
            return best as u32;
        }

        // 4. Температура.
        if self.sp.temperature > 0.0 {
            let temp = self.sp.temperature;
            for l in logit.iter_mut() {
                *l /= temp;
            }
        }

        // 5. top-k с сохранением ничьих.
        let mut idx: Vec<usize> = Vec::with_capacity(vocab);
        if self.sp.top_k > 0 && self.sp.top_k < vocab {
            let kth = kth_largest(&logit, self.sp.top_k);
            for i in 0..vocab {
                if logit[i] >= kth {
                    idx.push(i);
                }
            }
        } else {
            idx.extend(0..vocab);
        }
        if idx.is_empty() {
            return 0;
        }

        // 6. softmax по кандидатам.
        let n = idx.len();
        let mut mx = f32::NEG_INFINITY;
        for &i in &idx {
            if logit[i] > mx {
                mx = logit[i];
            }
        }
        let mut p = vec![0f32; n];
        let mut sum = 0f32;
        for (j, &i) in idx.iter().enumerate() {
            p[j] = (logit[i] - mx).exp();
            sum += p[j];
        }
        for v in p.iter_mut() {
            *v /= sum;
        }

        // Порядок: по убыванию вероятности, при равенстве — по индексу.
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| {
            p[b].partial_cmp(&p[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        // 7. top-p по массе до токена.
        let mut keep = vec![true; n];
        if self.sp.top_p > 0.0 && self.sp.top_p < 1.0 {
            let tp = self.sp.top_p;
            let mut before = 0f32;
            for &r in &ord {
                if before > tp {
                    keep[r] = false;
                }
                before += p[r];
            }
        }
        // 8. min-p.
        if self.sp.min_p > 0.0 {
            let thr = self.sp.min_p * p[ord[0]];
            for j in 0..n {
                if p[j] < thr {
                    keep[j] = false;
                }
            }
        }

        let mut kept_mass = 0f32;
        for j in 0..n {
            if !keep[j] {
                p[j] = 0.0;
            }
            kept_mass += p[j];
        }
        if kept_mass <= 0.0 {
            return idx[ord[0]] as u32;
        }

        // 9. Мультиномиал.
        let r = self.rng.uniform(kept_mass as f64) as f32;
        let mut acc = 0f32;
        for j in 0..n {
            acc += p[j];
            if acc >= r {
                return idx[j] as u32;
            }
        }
        idx[ord[0]] as u32
    }
}

/// k-е по величине значение (1 ≤ k ≤ len). Ничьи сохраняются.
fn kth_largest(values: &[f32], k: usize) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() - k]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_reference() {
        // Первые числа MT19937 для seed 5489 (стандартная инициализация).
        let mut rng = Mt19937::new(5489);
        assert_eq!(rng.next(), 3499211612);
        assert_eq!(rng.next(), 581869302);
    }

    #[test]
    fn greedy_picks_max() {
        let sp = Sampling {
            greedy: true,
            ..Sampling::default()
        };
        let mut s = Sampler::new(sp, 10, 4096, 4096);
        let mut logits = vec![0f32; 4098];
        logits[7] = 5.0;
        logits[100] = 3.0;
        assert_eq!(s.sample(&logits, 0), 7);
    }

    #[test]
    fn pad_is_never_sampled() {
        let sp = Sampling {
            greedy: true,
            ..Sampling::default()
        };
        let mut s = Sampler::new(sp, 10, 4096, 4096);
        let mut logits = vec![0f32; 4098];
        logits[4097] = 100.0; // audio_pad
        logits[5] = 1.0;
        assert_eq!(s.sample(&logits, 0), 5);
    }
}
