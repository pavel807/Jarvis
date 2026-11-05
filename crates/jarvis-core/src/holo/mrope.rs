//! Holo-3.1/Qwen3.5 трёхосевые MRoPE-позиции: раскладка [t,h,w] по токенам.
//!
//! Для текста все три оси равны — применение совпадает с обычным RoPE.
//! Гибридный декодер применяет частичный интерлив-ротор сам
//! ([`super::hybrid::partial_interleaved_mrope`]); здесь — только подготовка
//! позиций и проверка секций.

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MropeConfig {
    pub interleaved: bool,
    /// Секции частот для осей времени, высоты и ширины.
    pub sections: [usize; 3],
}

impl MropeConfig {
    pub fn validate(&self, head_dim: usize) -> Result<()> {
        if !self.interleaved {
            bail!("Holo-3.1/Qwen3.5 требуют interleaved MRoPE");
        }
        if self.sections.iter().sum::<usize>() * 2 != head_dim {
            bail!("MRoPE sections {:?} не покрывают head_dim={head_dim}", self.sections);
        }
        Ok(())
    }
}

/// Создаёт `[time, height, width]`-позиции для последовательности с одним
/// непрерывным диапазоном image-токенов. Сетка — в до-merge патч-единицах.
pub fn position_ids(
    sequence_len: usize,
    image_span: (usize, usize),
    grid_thw: [u32; 3],
    merge_size: usize,
) -> Result<Vec<[u32; 3]>> {
    let (start, end) = image_span;
    if start >= end || end > sequence_len {
        bail!("Некорректный image token span {image_span:?} для последовательности {sequence_len}");
    }
    if merge_size == 0 {
        bail!("merge_size не может быть нулём");
    }
    let [t, h, w] = grid_thw;
    if h as usize % merge_size != 0 || w as usize % merge_size != 0 {
        bail!("Image grid {:?} не делится на spatial merge size {merge_size}", grid_thw);
    }
    let mh = h as usize / merge_size;
    let mw = w as usize / merge_size;
    let expected = t as usize * mh * mw;
    if end - start != expected {
        bail!("Image span содержит {} токенов, grid требует {expected}", end - start);
    }

    let mut ids = vec![[0u32; 3]; sequence_len];
    for (index, item) in ids.iter_mut().enumerate().take(start) {
        let p = index as u32;
        *item = [p, p, p];
    }

    let mut max_axis = 0u32;
    let mut cursor = start;
    for ti in 0..t as usize {
        for hi in 0..mh {
            for wi in 0..mw {
                let pos = [
                    start as u32 + ti as u32,
                    start as u32 + hi as u32,
                    start as u32 + wi as u32,
                ];
                max_axis = max_axis.max(pos[0]).max(pos[1]).max(pos[2]);
                ids[cursor] = pos;
                cursor += 1;
            }
        }
    }

    // Текстовые позиции продолжаются после максимума пространственных осей,
    // а не после числа визуальных токенов.
    let mut next = max_axis.saturating_add(1);
    for item in ids.iter_mut().skip(end) {
        *item = [next, next, next];
        next = next.saturating_add(1);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holo_sections_cover_rotary_half() {
        // Holo/Qwen3.5: head_dim 256, partial 0.25 → rotary 64,
        // половина ротора 32 = сумма секций.
        MropeConfig { interleaved: true, sections: [11, 11, 10] }.validate(64).unwrap();
    }

    #[test]
    fn image_positions_are_temporal_height_width_and_text_resumes_after_grid() {
        let ids = position_ids(9, (2, 6), [1, 4, 4], 2).unwrap();
        assert_eq!(&ids[2..6], &[[2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3]]);
        assert_eq!(ids[6], [4, 4, 4]);
        assert_eq!(ids[7], [5, 5, 5]);
    }

    #[test]
    fn mismatched_image_span_is_rejected() {
        assert!(position_ids(10, (1, 3), [1, 4, 4], 2).is_err());
    }
}
