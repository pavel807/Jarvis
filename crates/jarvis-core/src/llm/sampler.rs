//! Сэмплинг токенов: repeat penalty + temperature/top-p.
//!
//! Чистые функции над срезом f32 — легко тестируются и не зависят от candle.

/// Штраф за повторение (семантика HF): каждый **уникальный** токен из
/// последнего `window` контекста делится на `penalty`, если его логит
/// положителен, и умножается иначе.
pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32], window: usize) {
    if penalty == 1.0 || context.is_empty() {
        return;
    }

    let start = context.len().saturating_sub(window);
    let mut seen = std::collections::HashSet::with_capacity(context.len() - start);
    for &tok in &context[start..] {
        if seen.insert(tok) {
            if let Some(l) = logits.get_mut(tok as usize) {
                *l = if *l > 0.0 {
                    *l / penalty
                } else {
                    *l * penalty
                };
            }
        }
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Выбор токена: temperature = 0 — жадно; иначе top-k + top-p с температурой.
///
/// Для больших словарей (Qwen3: 151k) полная сортировка логитов стоит
/// миллисекунды на каждый токен, поэтому сначала отбираем top-k за O(n)
/// (`select_nth_unstable`), и только в нём работаем с softmax/top-p.
/// Масса хвоста за пределами top-k при рабочих температурах пренебрежимо мала.
pub fn sample_top_p(
    logits: &[f32],
    temperature: f64,
    top_p: f64,
    rng: &mut impl rand::Rng,
) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    if logits.is_empty() {
        return 0;
    }

    const TOP_K: usize = 50;

    let n = logits.len();
    let k = TOP_K.min(n);
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let (top, _, _) = idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize])
    });
    top.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let top = &idx[..k];

    // softmax с температурой (в f64 для стабильности), только по top-k.
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut probs: Vec<f64> = Vec::with_capacity(k);
    let mut sum = 0.0;
    for &i in top {
        let p = ((logits[i as usize] as f64 - max) / temperature).exp();
        probs.push(p);
        sum += p;
    }
    if sum <= 0.0 {
        return argmax(logits);
    }

    // Топ-p: оставляем минимальный префикс top-k с накопленной вероятностью
    // >= top_p (доли считаются внутри top-k), остальное отбрасываем.
    let mut cumulative = 0.0;
    let mut cutoff = k;
    for (rank, &p) in probs.iter().enumerate() {
        cumulative += p / sum;
        if cumulative >= top_p {
            cutoff = rank + 1;
            break;
        }
    }
    let kept_sum: f64 = probs[..cutoff].iter().sum();
    if kept_sum <= 0.0 || cutoff == 0 {
        return argmax(logits);
    }

    let pick = rng.gen_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (rank, &p) in probs[..cutoff].iter().enumerate() {
        cumulative += p / kept_sum;
        if pick < cumulative {
            return top[rank];
        }
    }

    top[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> impl rand::Rng {
        // Детерминированный ГПСЧ для воспроизводимых тестов.
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn greedy_picks_max() {
        let logits = [0.1, 5.0, 2.0];
        assert_eq!(sample_top_p(&logits, 0.0, 0.9, &mut rng()), 1);
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn penalty_reduces_positive_logit() {
        let mut logits = vec![0.0; 8];
        logits[3] = 4.0;
        let context = [3u32, 3, 3];
        apply_repeat_penalty(&mut logits, 2.0, &context, 10);
        assert!((logits[3] - 2.0).abs() < 1e-6);
        assert_eq!(logits[0], 0.0);
    }

    #[test]
    fn penalty_boosts_negative_logit() {
        let mut logits = vec![0.0; 8];
        logits[3] = -4.0;
        apply_repeat_penalty(&mut logits, 2.0, &[3], 10);
        assert!((logits[3] + 8.0).abs() < 1e-6);
    }

    #[test]
    fn top_p_zero_is_greedy() {
        let logits = [0.1, 5.0, 2.0];
        assert_eq!(sample_top_p(&logits, 0.9, 1e-9, &mut rng()), 1);
    }

    #[test]
    fn top_p_one_is_stochastic_but_valid() {
        let logits = [0.1, 5.0, 2.0];
        let mut r = rng();
        for _ in 0..50 {
            let tok = sample_top_p(&logits, 0.9, 1.0, &mut r);
            assert!(tok < 3);
        }
    }

    #[test]
    fn penalty_respects_window() {
        let mut logits = vec![0.0; 8];
        logits[0] = 4.0;
        // Токен 0 в контексте старше окна — штраф не применяется.
        let context = [0u32, 1, 2, 3];
        apply_repeat_penalty(&mut logits, 2.0, &context, 2);
        assert!((logits[0] - 4.0).abs() < 1e-6);
    }
}
