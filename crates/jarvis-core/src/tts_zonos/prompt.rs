//! Сборка промпта ZONOS3.
//!
//! Порядок строк (эталон `servers/web/src/prompt.ts` и
//! `servers/ggml/docs/INTERNALS.md`):
//!
//! ```text
//! [ строка голоса (1) | префикс опорных кодов (N+9) ]
//! [ громкость ][ f_max ][ темп ]
//! BOS текст… EOS
//! [ кадры тишины ]
//! ```
//!
//! В каждой строке 11 колонок: 10 аудиокниг и одна текстовая. Все
//! не-текстовые колонки заполнены `audio_pad_id`, текстовая — `text_vocab`
//! (строка паддинга текста). Эмбеддинг строки голоса и строк условий
//! заменяется в бэкбоне.

use super::config::ZonosConfig;
use anyhow::{Context, Result, bail};

/// Максимум кадров тишины в префилле (реестр генерации).
pub const SILENCE_PREFILL_MAX: usize = 30;
/// Максимум кадров в опорном префиксе (60 секунд при 15 Гц).
pub const REFERENCE_MAX_FRAMES: usize = 900;

/// Готовый промпт.
pub struct Prompt {
    /// Строки по 11 колонок (10 аудиокниг + текст).
    pub ids: Vec<Vec<u32>>,
    /// Строка, которую занимает эмбеддинг голоса.
    pub speaker_row: Option<usize>,
    /// Строки условий: (номер строки, бакет).
    pub cond_rows: Vec<(usize, usize)>,
    /// Сколько кадров тишины добавлено в конец.
    pub prefill_frames: usize,
}

/// Параметры сборки промпта.
pub struct PromptSpec<'a> {
    /// Текстовые id: `[BOS] + payload + [EOS]`.
    pub text_ids: &'a [u32],
    /// Эмбеддинг голоса (192 измерения) — если есть, строка голоса будет.
    pub speaker: Option<&'a [f32]>,
    /// Бакеты условий: громкость, f_max, темп.
    pub buckets: [usize; 3],
    /// Число кадров тишины в префилле.
    pub silence_prefill: usize,
    /// Опорные коды (референс-клонирование): [кадры * 10].
    pub reference_codes: Option<&'a [u32]>,
}

/// Собрать промпт по правилам чекпоинта.
pub fn build_prompt(cfg: &ZonosConfig, spec: &PromptSpec<'_>) -> Result<Prompt> {
    let ncb = cfg.n_codebooks;
    let width = ncb + 1;
    let audio_pad = cfg.audio_pad_id;
    let text_pad = cfg.text_pad_id;

    if spec.speaker.is_some() && spec.reference_codes.is_some() {
        bail!("Эмбеддинг голоса и опорный префикс взаимно исключаются");
    }
    let ref_frames = spec
        .reference_codes
        .map(|c| c.len() / ncb)
        .unwrap_or(0);
    if spec.reference_codes.is_some() && ref_frames == 0 {
        bail!("Пустой опорный префикс");
    }
    if ref_frames > REFERENCE_MAX_FRAMES {
        bail!("Опорный префикс длиннее {REFERENCE_MAX_FRAMES} кадров");
    }
    if spec.silence_prefill > SILENCE_PREFILL_MAX {
        bail!("Кадров тишины должно быть 0..{SILENCE_PREFILL_MAX}");
    }
    for (i, bucket) in spec.buckets.iter().enumerate() {
        let n = cfg.condition_bucket_counts[i];
        if *bucket > n {
            bail!(
                "Бакет условия «{}» = {bucket} вне диапазона 0..{n} ({n} — «неизвестно»)",
                cfg.condition_names[i]
            );
        }
    }

    let reference_rows = if ref_frames > 0 { ref_frames + ncb - 1 } else { 0 };
    let speaker_rows = if spec.speaker.is_some() { 1 } else { reference_rows };
    let rows = speaker_rows + cfg.condition_bucket_counts.len() + spec.text_ids.len() + spec.silence_prefill;
    if rows == 0 {
        bail!("Пустой промпт");
    }
    if rows > cfg.max_seqlen {
        bail!("Промпт длиннее контекста {} позиций", cfg.max_seqlen);
    }

    let mut ids = vec![vec![audio_pad; width]; rows];
    for row in &mut ids {
        row[ncb] = text_pad;
    }

    // Префикс опорных кодов: строка t, колонка c = кадр t−c.
    if let Some(codes) = spec.reference_codes {
        for t in 0..reference_rows {
            for cb in 0..ncb {
                let f = t as i64 - cb as i64;
                if f >= 0 && (f as usize) < ref_frames {
                    let code = codes[f as usize * ncb + cb];
                    if code as usize >= cfg.codebook_size {
                        bail!("В опорном префиксе необычный код {code}");
                    }
                    ids[t][cb] = code;
                }
            }
        }
    }

    // Текстовые строки.
    let text_start = speaker_rows + cfg.condition_bucket_counts.len();
    for (i, id) in spec.text_ids.iter().enumerate() {
        ids[text_start + i][ncb] = *id;
    }

    // Кадры тишины со сдвигом (shear): строка k, колонка c = silence[c], если k ≥ c.
    let silence_start = text_start + spec.text_ids.len();
    for k in 0..spec.silence_prefill {
        for cb in 0..ncb {
            if k >= cb {
                let code = cfg
                    .silence_frame
                    .get(cb)
                    .copied()
                    .with_context(|| format!("В чекпоинте нет кода тишины {cb}"))?;
                ids[silence_start + k][cb] = code;
            }
        }
    }

    // Строки условий идут сразу после строки голоса/префикса.
    let cond_row0 = speaker_rows;
    let cond_rows: Vec<(usize, usize)> = spec
        .buckets
        .iter()
        .enumerate()
        .map(|(i, bucket)| (cond_row0 + i, *bucket))
        .collect();

    Ok(Prompt {
        ids,
        speaker_row: spec.speaker.map(|_| 0),
        cond_rows,
        prefill_frames: spec.silence_prefill,
    })
}

/// Бакет условия по значению: `bisect_right` по границам таблицы.
pub fn bucket_for(cfg: &ZonosConfig, table: usize, value: f32) -> usize {
    if !value.is_finite() {
        return cfg.unknown_bucket(table);
    }
    cfg.bucket_for(table, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ZonosConfig {
        ZonosConfig {
            n_layers: 11,
            dim: 1024,
            head_dim: 32,
            n_heads: 16,
            n_kv_heads: 4,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_seqlen: 32768,
            logit_softcap: 15.0,
            n_codebooks: 10,
            codebook_size: 4096,
            audio_vocab: 4098,
            eoa_id: 4096,
            audio_pad_id: 4097,
            text_vocab: 8196,
            text_extra_tokens: 0,
            gdn_heads: 4,
            gdn_head_dim: 128,
            gdn_kernel: 4,
            gdn_period: 4,
            n_experts: 64,
            moe_topk: 2,
            moe_hidden: 1536,
            moe_routed_dim: 256,
            ffn_hidden: 2048,
            speaker_dim: 192,
            condition_bucket_counts: vec![10, 10, 30],
            condition_edges: vec![vec![], vec![], vec![]],
            condition_names: vec![
                "loudness".into(),
                "fmax".into(),
                "speaking_rate".into(),
            ],
            condition_units: vec![],
            silence_frame: vec![1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337],
            text_offset: 4,
            bos_id: 2,
            eos_id: 3,
            text_pad_id: 8196,
            tokenizer_json: String::new(),
            text_tokenizer_spec: String::new(),
            silence_prefill_frames: 3,
        }
    }

    #[test]
    fn prompt_layout() {
        let c = cfg();
        let text_ids = vec![2u32, 100, 3];
        let spec = PromptSpec {
            text_ids: &text_ids,
            speaker: None,
            buckets: [10, 10, 30],
            silence_prefill: 3,
            reference_codes: None,
        };
        let p = build_prompt(&c, &spec).unwrap();
        // 3 условия + 3 текстовых + 3 тишины.
        assert_eq!(p.n_rows_of(), 9);
        assert_eq!(p.ids[0][10], 8196);
        assert_eq!(p.ids[3][10], 2);
        assert_eq!(p.ids[5][10], 3);
        // Первая строка тишины: только колонка 0 заполнена кодом тишины.
        assert_eq!(p.ids[6][0], 1918);
        assert_eq!(p.ids[6][1], 4097);
        assert_eq!(p.ids[7][0], 1918);
        assert_eq!(p.ids[7][1], 746);
    }

    impl Prompt {
        fn n_rows_of(&self) -> usize {
            self.ids.len()
        }
    }
}
