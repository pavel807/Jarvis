//! Checkpoint weight-map verification.
//!
//! Превращает [`HybridConfig`] в точный набор `(имя, форма, dtype)` тензоров,
//! который обязан содержать safetensors-чекпойнт Holo-3.1-4B, и сверяет с
//! реальным заголовком файлов. Отсутствующий, лишний, не той формы или не
//! того dtype тензор сообщается, а не грузится молча — это ворота
//! корректности до первого перемещения тензоров.

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::Path;

/// Все шарда весов каталога (`model*.safetensors`, без `.part`) в стабильном
/// порядке. Чекпойнт 4B шардирован на два файла.
pub fn weight_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Не удалось прочитать {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("model") && name.ends_with(".safetensors")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("В {} нет safetensors-файлов весов", dir.display());
    }
    Ok(files)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// What a header verification produced.
#[derive(Debug, Default)]
pub struct VerifyReport {
    pub tensors: usize,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub mismatches: Vec<String>,
}

impl VerifyReport {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.mismatches.is_empty()
    }
}

/// Reads just the safetensors header (8-byte little-endian length + JSON)
/// without loading any tensor bytes. Works on the full `.safetensors` file or
/// a file that contains only the header region.
pub fn read_safetensors_header(path: &Path) -> Result<BTreeMap<String, TensorInfo>> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Не удалось открыть {}", path.display()))?;
    use std::io::Read;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .context("Файл короче 8 байт — это не safetensors")?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    // Header is normally a few tens of KB; refuse anything absurd to avoid a
    // malformed file tricking us into a huge allocation.
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        bail!("Неправдоподобная длина заголовка safetensors: {header_len}");
    }
    let mut json = vec![0u8; header_len];
    file.read_exact(&mut json)
        .context("Обрезанный заголовок safetensors")?;
    parse_header_json(&json)
}

pub fn parse_header_json(json: &[u8]) -> Result<BTreeMap<String, TensorInfo>> {
    let value: serde_json::Value =
        serde_json::from_slice(json).context("Заголовок safetensors не является JSON")?;
    let obj = value.as_object().context("Заголовок safetensors не объект")?;
    let mut out = BTreeMap::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let shape = meta
            .get("shape")
            .and_then(|s| s.as_array())
            .with_context(|| format!("Тензор {name} без shape"))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .collect();
        let dtype = meta
            .get("dtype")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        out.insert(name.clone(), TensorInfo { shape, dtype });
    }
    Ok(out)
}

// =========================================================================
// Гибридная семья (Qwen3.5 / Holo-3.1-4B)
// =========================================================================

/// Ожидаемая карта весов гибридного чекпойнта (`qwen3_5`): DeltaNet-слои с
/// 32 V-головами, полные слои с gate, визионная башня без DeepStack.
pub fn expected_weights_hybrid(cfg: &super::hybrid::HybridConfig) -> BTreeMap<String, TensorInfo> {
    let mut m: BTreeMap<String, TensorInfo> = BTreeMap::new();
    let t = &cfg.text;
    let v = &cfg.vision;
    let hidden = t.hidden_size;
    let q_dim = t.num_attention_heads * t.head_dim;
    let kv_dim = t.num_key_value_heads * t.head_dim;
    let key_dim = t.linear_num_key_heads * t.linear_key_head_dim;
    let value_dim = t.linear_num_value_heads * t.linear_value_head_dim;
    let inter = t.intermediate_size;
    let dtype = "BF16".to_string();

    let put = |m: &mut BTreeMap<String, TensorInfo>, name: String, shape: Vec<usize>| {
        m.insert(name, TensorInfo { shape, dtype: dtype.clone() });
    };

    put(&mut m, "model.language_model.embed_tokens.weight".into(), vec![t.vocab_size, hidden]);
    put(&mut m, "model.language_model.norm.weight".into(), vec![hidden]);
    for i in 0..t.num_hidden_layers {
        let p = |s: &str| format!("model.language_model.layers.{i}.{s}");
        put(&mut m, p("input_layernorm.weight"), vec![hidden]);
        put(&mut m, p("post_attention_layernorm.weight"), vec![hidden]);
        put(&mut m, p("mlp.gate_proj.weight"), vec![inter, hidden]);
        put(&mut m, p("mlp.up_proj.weight"), vec![inter, hidden]);
        put(&mut m, p("mlp.down_proj.weight"), vec![hidden, inter]);
        match t.layer_types[i] {
            super::hybrid::HybridLayerType::LinearAttention => {
                put(&mut m, p("linear_attn.conv1d.weight"), vec![key_dim * 2 + value_dim, 1, t.conv_kernel_size]);
                put(&mut m, p("linear_attn.A_log"), vec![t.linear_num_value_heads]);
                put(&mut m, p("linear_attn.dt_bias"), vec![t.linear_num_value_heads]);
                put(&mut m, p("linear_attn.in_proj_qkv.weight"), vec![key_dim * 2 + value_dim, hidden]);
                put(&mut m, p("linear_attn.in_proj_z.weight"), vec![value_dim, hidden]);
                put(&mut m, p("linear_attn.in_proj_b.weight"), vec![t.linear_num_value_heads, hidden]);
                put(&mut m, p("linear_attn.in_proj_a.weight"), vec![t.linear_num_value_heads, hidden]);
                put(&mut m, p("linear_attn.norm.weight"), vec![t.linear_value_head_dim]);
                put(&mut m, p("linear_attn.out_proj.weight"), vec![hidden, value_dim]);
            }
            super::hybrid::HybridLayerType::FullAttention => {
                let hd = t.head_dim;
                put(&mut m, p("self_attn.q_proj.weight"), vec![t.num_attention_heads * hd * 2, hidden]);
                put(&mut m, p("self_attn.k_proj.weight"), vec![kv_dim, hidden]);
                put(&mut m, p("self_attn.v_proj.weight"), vec![kv_dim, hidden]);
                put(&mut m, p("self_attn.o_proj.weight"), vec![hidden, q_dim]);
                put(&mut m, p("self_attn.q_norm.weight"), vec![hd]);
                put(&mut m, p("self_attn.k_norm.weight"), vec![hd]);
            }
        }
    }

    // Визионная башня qwen3_5_vl (без DeepStack).
    let vs = v.hidden_size;
    let vi = v.intermediate_size;
    let merge_dim = vs * v.spatial_merge_size * v.spatial_merge_size;
    put(&mut m, "model.visual.patch_embed.proj.weight".into(), vec![vs, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size]);
    put(&mut m, "model.visual.patch_embed.proj.bias".into(), vec![vs]);
    put(&mut m, "model.visual.pos_embed.weight".into(), vec![v.num_position_embeddings, vs]);
    for i in 0..v.depth {
        let p = move |s: &str| format!("model.visual.blocks.{i}.{s}");
        put(&mut m, p("norm1.weight"), vec![vs]);
        put(&mut m, p("norm1.bias"), vec![vs]);
        put(&mut m, p("norm2.weight"), vec![vs]);
        put(&mut m, p("norm2.bias"), vec![vs]);
        put(&mut m, p("attn.qkv.weight"), vec![vs * 3, vs]);
        put(&mut m, p("attn.qkv.bias"), vec![vs * 3]);
        put(&mut m, p("attn.proj.weight"), vec![vs, vs]);
        put(&mut m, p("attn.proj.bias"), vec![vs]);
        put(&mut m, p("mlp.linear_fc1.weight"), vec![vi, vs]);
        put(&mut m, p("mlp.linear_fc1.bias"), vec![vi]);
        put(&mut m, p("mlp.linear_fc2.weight"), vec![vs, vi]);
        put(&mut m, p("mlp.linear_fc2.bias"), vec![vs]);
    }
    put(&mut m, "model.visual.merger.norm.weight".into(), vec![vs]);
    put(&mut m, "model.visual.merger.norm.bias".into(), vec![vs]);
    put(&mut m, "model.visual.merger.linear_fc1.weight".into(), vec![merge_dim, merge_dim]);
    put(&mut m, "model.visual.merger.linear_fc1.bias".into(), vec![merge_dim]);
    put(&mut m, "model.visual.merger.linear_fc2.weight".into(), vec![v.out_hidden_size, merge_dim]);
    put(&mut m, "model.visual.merger.linear_fc2.bias".into(), vec![v.out_hidden_size]);

    m
}

/// Верифицирует гибридный чекпойнт (шарды) против конфигурации.
pub fn verify_checkpoint_files_hybrid(
    cfg: &super::hybrid::HybridConfig,
    files: &[std::path::PathBuf],
) -> Result<VerifyReport> {
    let mut merged: BTreeMap<String, TensorInfo> = BTreeMap::new();
    for f in files {
        for (name, info) in read_safetensors_header(f)? {
            if merged.insert(name.clone(), info).is_some() {
                bail!("Тензор {name} встречается в нескольких шардах");
            }
        }
    }
    let expected = expected_weights_hybrid(cfg);
    let mut report = VerifyReport { tensors: merged.len(), ..Default::default() };
    for (name, exp) in &expected {
        match merged.get(name) {
            None => report.missing.push(name.clone()),
            Some(got) => {
                if &got.shape != &exp.shape || got.dtype != exp.dtype {
                    report.mismatches.push(format!(
                        "{name}: ожидалось {:?}/{} получено {:?}/{}",
                        exp.shape, exp.dtype, got.shape, got.dtype
                    ));
                }
            }
        }
    }
    for name in merged.keys() {
        // lm_head.weight и video-тензоры допустимы сверх обязательной карты.
        if !expected.contains_key(name) && name != "lm_head.weight" {
            report.unexpected.push(name.clone());
        }
    }
    if !report.ok() {
        bail!(
            "Чекпоинт Holo-3.1-4B не соответствует конфигурации: {} отсутствует, {} лишнего, {} несовпадений. Примеры: {:?} / {:?} / {:?}",
            report.missing.len(),
            report.unexpected.len(),
            report.mismatches.len(),
            report.missing.first(),
            report.unexpected.first(),
            report.mismatches.first(),
        );
    }
    Ok(report)
}
