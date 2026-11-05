//! Holo-3.1 vision tower on candle.
//!
//! Implements the `model.visual.*` subtree exactly as laid out by the verified
//! checkpoint header:
//!
//! - `patch_embed.proj` — a `[1024, 3, 2, 16, 16]` convolution whose input is
//!   already patch-extracted by [`super::processor`], so it reduces to a linear
//!   map over each `[C*T*P*P]` patch vector.
//! - `pos_embed` — a learned `[2304, 1024]` table on a 48×48 reference grid,
//!   bilinearly interpolated to the image's `(grid_h, grid_w)` and tiled over the
//!   temporal axis.
//! - `blocks.*` — 24 pre-norm ViT blocks (LayerNorm → 2-D-rotary bidirectional
//!   attention → LayerNorm → GELU-tanh MLP).
//! - `merger` — LayerNorm + 2×2 spatial-merge concat + fc1/GELU + fc2 to the
//!   text hidden size (2048).
//! - `deepstack_merger_list.*` — three merged-token projections taken from the
//!   intermediate blocks listed in `vision_config.deepstack_visual_indexes`.
//!
//! The tensor names, shapes and dtypes are pinned by `checkpoint::verify_*`.
//! The numeric *parity of the vision rotary and pos-embed interpolation against
//! the upstream transformers implementation is a separate verification step
//! that requires the full checkpoint plus a reference run (see STATUS doc).

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct VisionParams {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub out_hidden_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub eps: f64,
    /// Reference grid side for the learned pos-embed table (sqrt of 2304 = 48).
    pub pos_grid_side: usize,
    pub rope_theta: f32,
    pub deepstack_indexes: Vec<usize>,
}

impl VisionParams {
    /// Из «сырых» полей — общий конструктор для плотной и гибридной семей.
    pub fn from_raw(
        depth: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        out_hidden_size: usize,
        spatial_merge_size: usize,
        num_position_embeddings: usize,
        deepstack_indexes: Vec<usize>,
    ) -> Result<Self> {
        if hidden_size % num_heads != 0 {
            bail!("vision hidden_size {hidden_size} не делится на num_heads {num_heads}");
        }
        let pos_side = (num_position_embeddings as f64).sqrt().round() as usize;
        if pos_side * pos_side != num_position_embeddings {
            bail!("num_position_embeddings {num_position_embeddings} не является полным квадратом");
        }
        Ok(Self {
            depth,
            hidden_size,
            intermediate_size,
            num_heads,
            head_dim: hidden_size / num_heads,
            out_hidden_size,
            spatial_merge_size,
            num_position_embeddings,
            eps: 1e-6,
            pos_grid_side: pos_side,
            rope_theta: 10_000.0,
            deepstack_indexes,
        })
    }
}

/// One loaded linear (stored `[in, out]`).
struct Lin {
    w: Tensor,
    b: Option<Tensor>,
}
impl Lin {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.matmul(&self.w)?;
        match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?))?),
            None => Ok(y),
        }
    }
}

/// LayerNorm with weight + bias (vision norms are full LayerNorm, not RMS).
fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let centered = xf.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = var.affine(1.0, eps)?.sqrt()?;
    let normed = centered.broadcast_div(&denom)?;
    let out = normed
        .broadcast_mul(&w.to_dtype(DType::F32)?)?
        .broadcast_add(&b.to_dtype(DType::F32)?)?;
    out.to_dtype(dtype)
}

/// Numerically stable tanh via exponentials: `1 - 2/(exp(2x) + 1)`.
/// candle-nn 0.11 does not expose a tensor `tanh`, so it is built here.
fn tanh(x: &Tensor) -> candle_core::Result<Tensor> {
    let two = x.affine(2.0, 0.0)?;
    let denom = two.exp()?.affine(1.0, 1.0)?; // exp(2x) + 1
    Tensor::ones(x.dims(), DType::F32, x.device())?.broadcast_sub(&denom.recip()?.affine(2.0, 0.0)?)
}

/// Tanh-approximation GELU (`hidden_act: gelu_pytorch_tanh`):
/// `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`, computed in f32.
fn gelu_tanh(x: &Tensor) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let k0 = (2.0f64 / std::f64::consts::PI).sqrt();
    // inner = k0 * (x + 0.044715 x^3) = k0 * x * (1 + 0.044715 x^2)
    let poly = xf.sqr()?.affine(0.044715, 1.0)?; // 1 + 0.044715 x^2
    let t = xf.broadcast_mul(&poly)?.affine(k0, 0.0)?; // k0 * x * poly
    let one_plus = tanh(&t)?.affine(1.0, 1.0)?; // 1 + tanh
    let out = xf.broadcast_mul(&one_plus)?.affine(0.5, 0.0)?;
    out.to_dtype(dtype)
}

struct VisionBlock {
    norm1_w: Tensor,
    norm1_b: Tensor,
    qkv: Lin,
    proj: Lin,
    norm2_w: Tensor,
    norm2_b: Tensor,
    fc1: Lin,
    fc2: Lin,
    n_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl VisionBlock {
    fn forward(
        &self,
        x: &Tensor, // (seq, hidden)
        cos: &Tensor, // (seq, head_dim/2)
        sin: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let seq = x.dim(0)?;
        let hidden = x.dim(1)?;
        let normed = layer_norm(x, &self.norm1_w, &self.norm1_b, self.eps)?;
        // QKV → (seq, 3, heads, head_dim)
        let qkv = self.qkv.forward(&normed)?.reshape((seq, 3, self.n_heads, self.head_dim))?;
        let q = qkv.narrow(1, 0, 1)?.contiguous()?.squeeze(1)?; // (seq, nh, hd)
        let k = qkv.narrow(1, 1, 1)?.contiguous()?.squeeze(1)?;
        let v = qkv.narrow(1, 2, 1)?.contiguous()?.squeeze(1)?;

        // 2-D vision rotary over the head-dim pairs (rotate-half form).
        // cos/sin are (seq, head_dim/2); broadcast over the head axis.
        let apply = |t: &Tensor| -> candle_core::Result<Tensor> {
            let half = self.head_dim / 2;
            let a = t.narrow(2, 0, half)?; // (seq, nh, half)
            let b = t.narrow(2, half, half)?;
            let c = cos.unsqueeze(1)?.to_dtype(t.dtype())?; // (seq, 1, half)
            let s = sin.unsqueeze(1)?.to_dtype(t.dtype())?;
            let o1 = a.broadcast_mul(&c)?.broadcast_sub(&b.broadcast_mul(&s)?)?;
            let o2 = b.broadcast_mul(&c)?.broadcast_add(&a.broadcast_mul(&s)?)?;
            Tensor::cat(&[&o1, &o2], 2)
        };
        let q = apply(&q)?;
        let k = apply(&k)?;

        // (seq, heads, hd) → (heads, seq, hd)
        let q = q.transpose(0, 1)?.contiguous()?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = (self.head_dim as f64).sqrt().recip();
        // Bidirectional (no causal mask) vision attention, чанками по запросам:
        // полный скор-тензор на скриншоте — гигабайты (16 голов × патчи²),
        // Metal падает по памяти. Softmax независим по строкам, поэтому
        // чанки запросов математически идентичны полному проходу.
        const QUERY_CHUNK: usize = 512;
        let mut out_parts = Vec::new();
        let n_chunks = seq.div_ceil(QUERY_CHUNK);
        for c in 0..n_chunks {
            let start = c * QUERY_CHUNK;
            let len = QUERY_CHUNK.min(seq - start);
            let q_chunk = q.narrow(1, start, len)?.contiguous()?;
            let scores = q_chunk
                .matmul(&k.transpose(1, 2)?)?
                .affine(scale, 0.0)?
                .to_dtype(DType::F32)?;
            let attn = candle_nn::ops::softmax(&scores, D::Minus1)?.to_dtype(x.dtype())?;
            let part = attn.matmul(&v)?; // (heads, len, hd)
            out_parts.push(part.transpose(0, 1)?.contiguous()?);
        }
        let out = Tensor::cat(&out_parts, 0)?
            .reshape((seq, hidden))?; // (seq, nh*hd)
        let out = self.proj.forward(&out)?;
        let x = (x + out)?;

        let normed = layer_norm(&x, &self.norm2_w, &self.norm2_b, self.eps)?;
        let hidden_act = self.fc1.forward(&normed)?;
        let act = gelu_tanh(&hidden_act)?;
        let ff = self.fc2.forward(&act)?;
        x + ff
    }
}

/// The loaded Holo-3.1 vision tower.
pub struct Qwen3VlVision {
    params: VisionParams,
    patch_w: Tensor, // (hidden, patch_dim)
    patch_b: Tensor, // (hidden)
    pos_embed: Tensor, // (num_pos, hidden)
    blocks: Vec<VisionBlock>,
    merger_norm_w: Tensor,
    merger_norm_b: Tensor,
    merger_fc1: Lin,
    merger_fc2: Lin,
    deepstack: Vec<DeepStackMerger>,
    device: Device,
}

struct DeepStackMerger {
    norm_w: Tensor,
    norm_b: Tensor,
    fc1: Lin,
    fc2: Lin,
}

impl Qwen3VlVision {
    pub fn params(&self) -> &VisionParams {
        &self.params
    }

    /// Runs the vision tower. `pixel_values` is `(patch_count, patch_dim)` in
    /// the processor's patch order, `grid_thw` is `[t, h, w]` in patch units.
    /// Returns merged tokens `(merged, out_hidden)` plus one DeepStack feature
    /// tensor per deepstack index.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: [u32; 3]) -> Result<(Tensor, Vec<Tensor>)> {
        let p = &self.params;
        let [t, gh, gw] = grid_thw;
        let patch_count = (t as usize) * (gh as usize) * (gw as usize);
        if pixel_values.dim(0)? != patch_count {
            bail!("pixel_values строк {} не совпадает с grid_thw {:?} ({patch_count})", pixel_values.dim(0)?, grid_thw);
        }
        // Patch embed: linear over flattened patches.
        let mut x = pixel_values.matmul(&self.patch_w)?;
        x = x.broadcast_add(&self.patch_b.reshape((1, p.hidden_size))?)?; // (patch_count, hidden)

        // Interpolated 2-D positional embedding, tiled over the temporal axis.
        let pos = self.interpolated_pos_embed(gh as usize, gw as usize)?; // (gh*gw, hidden)
        let pos = pos
            .unsqueeze(0)?
            .broadcast_as((t as usize, gh as usize * gw as usize, p.hidden_size))?
            .contiguous()?
            .reshape((patch_count, p.hidden_size))?;
        x = (x + pos)?;

        // Vision rope cos/sin for the (row, col) grid.
        let (cos, sin) = self.vision_rope(gh as usize, gw as usize, t as usize)?;

        let mut deepstack_raw: Vec<Tensor> = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &cos, &sin)?;
            // Коммит command buffer каждого блока: без него все операции
            // башни копятся в одном буфере и Metal падает по памяти на
            // больших скриншотах.
            self.device
                .synchronize()
                .map_err(|e| anyhow::anyhow!("vision sync: {e}"))?;
            if p.deepstack_indexes.contains(&i) {
                deepstack_raw.push(x.clone());
            }
        }

        // Final merger produces the visual tokens consumed by the decoder.
        let merged = self.run_merger(&x, gh as usize, gw as usize, t as usize, p.spatial_merge_size)?;

        // DeepStack features.
        let mut deepstack_feats = Vec::with_capacity(deepstack_raw.len());
        for (j, raw) in deepstack_raw.iter().enumerate() {
            let merger = self.deepstack.get(j).context("нет deepstack-мерджера для индекса")?;
            let feat = self.run_deepstack(merger, raw, gh as usize, gw as usize, t as usize)?;
            deepstack_feats.push(feat);
        }

        Ok((merged, deepstack_feats))
    }

    /// Bilinearly interpolates the `[pos_grid_side^2, hidden]` table to the
    /// image's `(h, w)` grid and reorders the result to the *merged patch*
    /// order used by the checkpoint: `(h/merge, w/merge, merge, merge)`.
    ///
    /// Upstream does `view(t, h/merge, merge, w/merge, merge, -1).permute(0,1,3,2,4,5)`
    /// before adding the table to the patch embeddings; skipping that permute
    /// shifts every positional vector after the first merge block.
    fn interpolated_pos_embed(&self, h: usize, w: usize) -> Result<Tensor> {
        let p = &self.params;
        let side = p.pos_grid_side;
        let merge = p.spatial_merge_size;
        if h % merge != 0 || w % merge != 0 {
            bail!("grid ({h},{w}) не делится на spatial_merge {merge}");
        }
        let table = self.pos_embed.reshape((side, side, p.hidden_size))?;
        // Continuous coordinates matching HF's `linspace(0, side-1, n)`.
        let axis_coords = |n: usize| -> Vec<f64> {
            if n <= 1 {
                vec![0.0]
            } else {
                let step = (side - 1) as f64 / (n - 1) as f64;
                (0..n).map(|i| (i as f64 * step).clamp(0.0, (side - 1) as f64)).collect()
            }
        };
        let ys = axis_coords(h);
        let xs = axis_coords(w);
        let tv = table.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let bilinear = |iy: usize, ix: usize| -> Vec<f32> {
            let fy = ys[iy];
            let fx = xs[ix];
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(side - 1);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(side - 1);
            let dy = (fy - y0 as f64) as f32;
            let dx = (fx - x0 as f64) as f32;
            let mut row = vec![0f32; p.hidden_size];
            for c in 0..p.hidden_size {
                let a = tv[y0][x0][c];
                let b = tv[y0][x1][c];
                let cc = tv[y1][x0][c];
                let d = tv[y1][x1][c];
                let top = a + (b - a) * dx;
                let bot = cc + (d - cc) * dx;
                row[c] = top + (bot - top) * dy;
            }
            row
        };
        let mut out = Vec::with_capacity(h * w * p.hidden_size);
        for block_y in 0..(h / merge) {
            for block_x in 0..(w / merge) {
                for inner_y in 0..merge {
                    for inner_x in 0..merge {
                        let iy = block_y * merge + inner_y;
                        let ix = block_x * merge + inner_x;
                        out.extend_from_slice(&bilinear(iy, ix));
                    }
                }
            }
        }
        Ok(Tensor::from_vec(out, (h * w, p.hidden_size), &self.device)?
            .to_dtype(self.pos_embed.dtype())?)
    }

    /// Builds vision rope cos/sin of width `head_dim/2` per position in the
    /// merged patch order, matching `Holo3.1VisionModel.rot_pos_emb`:
    /// the first `head_dim/4` entries rotate by the row index and the second
    /// `head_dim/4` by the column, both with the *same* frequency table
    /// `1 / theta^(i / (head_dim/4))`.
    fn vision_rope(&self, gh: usize, gw: usize, t: usize) -> Result<(Tensor, Tensor)> {
        let p = &self.params;
        let merge = p.spatial_merge_size;
        if gh % merge != 0 || gw % merge != 0 {
            bail!("grid ({gh},{gw}) не делится на spatial_merge {merge}");
        }
        let half = p.head_dim / 2; // 64 for UI-Venus
        let freq_count = half / 2; // 32
        // Holo3.1VisionRotaryEmbedding(head_dim // 2): arange(0, dim, 2)/dim.
        let inv: Vec<f32> = (0..freq_count)
            .map(|i| 1.0f32 / p.rope_theta.powf(i as f32 / freq_count as f32))
            .collect();
        let mut cos_rows = Vec::with_capacity(t * gh * gw * half);
        let mut sin_rows = Vec::with_capacity(t * gh * gw * half);
        for _ti in 0..t {
            for block_y in 0..(gh / merge) {
                for block_x in 0..(gw / merge) {
                    for inner_y in 0..merge {
                        for inner_x in 0..merge {
                            let r = (block_y * merge + inner_y) as f32;
                            let c = (block_x * merge + inner_x) as f32;
                            for i in 0..freq_count {
                                let (sr, cr) = (r * inv[i]).sin_cos();
                                cos_rows.push(cr);
                                sin_rows.push(sr);
                            }
                            for i in 0..freq_count {
                                let (sc, cc) = (c * inv[i]).sin_cos();
                                cos_rows.push(cc);
                                sin_rows.push(sc);
                            }
                        }
                    }
                }
            }
        }
        let cos = Tensor::from_vec(cos_rows, (t * gh * gw, half), &self.device)?;
        let sin = Tensor::from_vec(sin_rows, (t * gh * gw, half), &self.device)?;
        Ok((cos, sin))
    }

    fn run_merger(&self, x: &Tensor, gh: usize, gw: usize, t: usize, merge: usize) -> Result<Tensor> {
        let p = &self.params;
        if gh % merge != 0 || gw % merge != 0 {
            bail!("grid ({gh},{gw}) не делится на spatial_merge {merge}");
        }
        let mh = gh / merge;
        let mw = gw / merge;
        // LayerNorm before merge (merger.norm has hidden_size width).
        let normed = layer_norm(x, &self.merger_norm_w, &self.merger_norm_b, p.eps)?;
        let merged = spatial_merge_2x2(&normed, t, gh, gw, merge)?; // (t*mh*mw, 4*hidden)
        let fc1 = self.merger_fc1.forward(&merged)?;
        let act = gelu_tanh(&fc1)?;
        let out = self.merger_fc2.forward(&act)?; // (merged, out_hidden)
        let _ = mh;
        let _ = mw;
        Ok(out)
    }

    fn run_deepstack(&self, m: &DeepStackMerger, x: &Tensor, gh: usize, gw: usize, t: usize) -> Result<Tensor> {
        let p = &self.params;
        let merged = spatial_merge_2x2(x, t, gh, gw, p.spatial_merge_size)?;
        let normed = layer_norm(&merged, &m.norm_w, &m.norm_b, p.eps)?;
        let fc1 = m.fc1.forward(&normed)?;
        let act = gelu_tanh(&fc1)?;
        Ok(m.fc2.forward(&act)?)
    }

    /// Загрузка для гибридной семьи Qwen3.5 (`qwen3_5_vision`): та же башня,
    /// но конфиг парсится из [`super::hybrid::HybridConfig`].
    pub fn load_hybrid(
        dir: &std::path::Path,
        cfg: &super::hybrid::HybridConfig,
        device: &Device,
    ) -> Result<Self> {
        let weights = super::checkpoint::weight_files(dir)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights, cfg.weight_dtype(), &Device::Cpu)
                .context("Не удалось открыть safetensors-веса vision")?
        };
        let v = &cfg.vision;
        let params = VisionParams::from_raw(
            v.depth,
            v.hidden_size,
            v.intermediate_size,
            v.num_heads,
            v.out_hidden_size,
            v.spatial_merge_size,
            v.num_position_embeddings,
            Vec::new(),
        )?;
        Self::build(&vb, params, v.in_channels, v.temporal_patch_size, v.patch_size, device)
    }

    fn build(
        vb: &VarBuilder,
        params: VisionParams,
        in_channels: usize,
        temporal_patch_size: usize,
        patch_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let hidden = params.hidden_size;
        let patch_dim = in_channels * temporal_patch_size * patch_size * patch_size;
        // Conv3d weight хранится 5-D `[out, C, T, P, P]`; порядок in-компонент
        // (C,T,P,P) совпадает с порядком патч-вектора процессора, поэтому
        // линейное отображение — это flatten + transpose (reshape одного лишь
        // веса перемешал бы веса).
        let patch_w = vb.get(
            (hidden, in_channels, temporal_patch_size, patch_size, patch_size),
            "model.visual.patch_embed.proj.weight",
        )?
        .reshape((hidden, patch_dim))?
        .t()?
        .contiguous()?
        .to_device(device)?;
        let patch_b = vb.get(hidden, "model.visual.patch_embed.proj.bias")?.to_device(device)?;
        let pos_embed = vb
            .get((params.num_position_embeddings, hidden), "model.visual.pos_embed.weight")?
            .to_device(device)?;

        let lin = |prefix: String, out: usize, inn: usize| -> Result<Lin> {
            let w = vb.get((out, inn), &format!("{prefix}.weight"))?.t()?.contiguous()?.to_device(device)?;
            let b = vb.get(out, &format!("{prefix}.bias"))?.to_device(device)?;
            Ok(Lin { w: w.reshape((inn, out))?.to_device(device)?, b: Some(b.to_device(device)?) })
        };
        let inter = params.intermediate_size;
        let merge_dim = hidden * params.spatial_merge_size * params.spatial_merge_size;

        let mut blocks = Vec::with_capacity(params.depth);
        for i in 0..params.depth {
            let pfx = |s: &str| format!("model.visual.blocks.{i}.{s}");
            blocks.push(VisionBlock {
                norm1_w: vb.get(hidden, &pfx("norm1.weight"))?.to_device(device)?,
                norm1_b: vb.get(hidden, &pfx("norm1.bias"))?.to_device(device)?,
                qkv: lin(pfx("attn.qkv"), hidden * 3, hidden)?,
                proj: lin(pfx("attn.proj"), hidden, hidden)?,
                norm2_w: vb.get(hidden, &pfx("norm2.weight"))?.to_device(device)?,
                norm2_b: vb.get(hidden, &pfx("norm2.bias"))?.to_device(device)?,
                fc1: lin(pfx("mlp.linear_fc1"), inter, hidden)?,
                fc2: lin(pfx("mlp.linear_fc2"), hidden, inter)?,
                n_heads: params.num_heads,
                head_dim: params.head_dim,
                eps: params.eps,
            });
        }
        let merger_fc1 = lin("model.visual.merger.linear_fc1".into(), merge_dim, merge_dim)?;
        let merger_fc2 = lin("model.visual.merger.linear_fc2".into(), params.out_hidden_size, merge_dim)?;

        let mut deepstack = Vec::new();
        for j in 0..params.deepstack_indexes.len() {
            let pfx = |s: &str| format!("model.visual.deepstack_merger_list.{j}.{s}");
            deepstack.push(DeepStackMerger {
                norm_w: vb.get(merge_dim, &pfx("norm.weight"))?.to_device(device)?,
                norm_b: vb.get(merge_dim, &pfx("norm.bias"))?.to_device(device)?,
                fc1: lin(pfx("linear_fc1"), merge_dim, merge_dim)?,
                fc2: lin(pfx("linear_fc2"), params.out_hidden_size, merge_dim)?,
            });
        }

        Ok(Self {
            params,
            patch_w,
            patch_b,
            pos_embed,
            blocks,
            merger_norm_w: vb.get(hidden, "model.visual.merger.norm.weight")?.to_device(device)?,
            merger_norm_b: vb.get(hidden, "model.visual.merger.norm.bias")?.to_device(device)?,
            merger_fc1,
            merger_fc2,
            deepstack,
            device: device.clone(),
        })
    }
}

/// Reorders `(t*gh*gw, C)` patches into merged `(t*mh*mw, C*merge*merge)` where
/// the 2×2 block neighbours are concatenated, matching the Qwen patch layout.
pub fn spatial_merge_2x2(x: &Tensor, t: usize, gh: usize, gw: usize, merge: usize) -> Result<Tensor> {
    if gh % merge != 0 || gw % merge != 0 {
        bail!("grid ({gh},{gw}) не делится на merge {merge}");
    }
    let c = x.dim(1)?;
    let mh = gh / merge;
    let mw = gw / merge;
    let data = x.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(t * mh * mw);
    for ti in 0..t {
        for by in 0..mh {
            for bx in 0..mw {
                let mut row: Vec<f32> = Vec::with_capacity(c * merge * merge);
                for iy in 0..merge {
                    for ix in 0..merge {
                        let patch_index = ti * gh * gw + (by * merge + iy) * gw + (bx * merge + ix);
                        row.extend_from_slice(&data[patch_index]);
                    }
                }
                out.push(row);
            }
        }
    }
    let flat: Vec<f32> = out.into_iter().flatten().collect();
    Ok(Tensor::from_vec(flat, (t * mh * mw, c * merge * merge), x.device())?.to_dtype(x.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tiny_cfg() -> VisionParams {
        VisionParams {
            depth: 2,
            hidden_size: 16,
            intermediate_size: 32,
            num_heads: 4,
            head_dim: 4,
            out_hidden_size: 32,
            spatial_merge_size: 2,
            num_position_embeddings: 4, // 2x2 reference grid
            eps: 1e-6,
            pos_grid_side: 2,
            rope_theta: 10_000.0,
            deepstack_indexes: vec![1],
        }
    }

    fn random_vision(p: VisionParams) -> Qwen3VlVision {
        let dev = Device::Cpu;
        let hidden = p.hidden_size;
        let patch_dim = 3 * 2 * 4 * 4; // tiny patch geometry for the test image
        let rnd = |shape: &[usize]| Tensor::rand(-0.3f32, 0.3f32, shape, &dev).unwrap();
        let merge_dim = hidden * p.spatial_merge_size * p.spatial_merge_size;
        let inter = p.intermediate_size;
        let blocks = (0..p.depth)
            .map(|_| VisionBlock {
                norm1_w: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
                norm1_b: Tensor::zeros(hidden, DType::F32, &dev).unwrap(),
                norm2_w: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
                norm2_b: Tensor::zeros(hidden, DType::F32, &dev).unwrap(),
                qkv: Lin { w: rnd(&[hidden, hidden * 3]), b: Some(rnd(&[hidden * 3])) },
                proj: Lin { w: rnd(&[hidden, hidden]), b: Some(rnd(&[hidden])) },
                fc1: Lin { w: rnd(&[hidden, inter]), b: Some(rnd(&[inter])) },
                fc2: Lin { w: rnd(&[inter, hidden]), b: Some(rnd(&[hidden])) },
                n_heads: p.num_heads,
                head_dim: p.head_dim,
                eps: p.eps,
            })
            .collect();
        let deepstack = p
            .deepstack_indexes
            .iter()
            .map(|_| DeepStackMerger {
                norm_w: Tensor::ones(merge_dim, DType::F32, &dev).unwrap(),
                norm_b: Tensor::zeros(merge_dim, DType::F32, &dev).unwrap(),
                fc1: Lin { w: rnd(&[merge_dim, merge_dim]), b: Some(rnd(&[merge_dim])) },
                fc2: Lin { w: rnd(&[merge_dim, p.out_hidden_size]), b: Some(rnd(&[p.out_hidden_size])) },
            })
            .collect();
        Qwen3VlVision {
            params: p.clone(),
            patch_w: rnd(&[patch_dim, hidden]),
            patch_b: rnd(&[hidden]),
            pos_embed: rnd(&[p.num_position_embeddings, hidden]),
            blocks,
            merger_norm_w: Tensor::ones(hidden, DType::F32, &dev).unwrap(),
            merger_norm_b: Tensor::zeros(hidden, DType::F32, &dev).unwrap(),
            merger_fc1: Lin { w: rnd(&[merge_dim, merge_dim]), b: Some(rnd(&[merge_dim])) },
            merger_fc2: Lin { w: rnd(&[merge_dim, p.out_hidden_size]), b: Some(rnd(&[p.out_hidden_size])) },
            deepstack,
            device: dev,
        }
    }

    #[test]
    fn vision_forward_shapes_and_determinism() {
        let v = random_vision(tiny_cfg());
        let (t, gh, gw) = (1usize, 4usize, 4usize);
        let patch_dim = 3 * 2 * 4 * 4;
        let px = Tensor::rand(-1f32, 1f32, (t * gh * gw, patch_dim), &Device::Cpu).unwrap();
        let (merged, deep) = v.forward(&px, [t as u32, gh as u32, gw as u32]).unwrap();
        // merged tokens = (gh/merge)*(gw/merge)
        assert_eq!(merged.dims(), &[2 * 2, v.params().out_hidden_size]);
        assert_eq!(deep.len(), 1);
        assert_eq!(deep[0].dims(), &[2 * 2, v.params().out_hidden_size]);
        // Determinism: same input → same output.
        let (m2, _) = v.forward(&px, [t as u32, gh as u32, gw as u32]).unwrap();
        let diff = (&merged - &m2).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-5, "vision нестабилна: {diff}");
        let finite = merged.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|x| x.is_finite());
        assert!(finite, "vision вернула NaN/Inf");
    }

    #[test]
    fn spatial_merge_reorders_neighbours() {
        // 2x2 grid, merge 2 → one merged token concatenating 4 patches.
        let dev = Device::Cpu;
        let x = Tensor::from_vec(
            vec![
                1.0f32, 1.0, // patch (0,0)
                2.0, 2.0, // patch (0,1)
                3.0, 3.0, // patch (1,0)
                4.0, 4.0, // patch (1,1)
            ],
            (4, 2),
            &dev,
        )
        .unwrap();
        let merged = spatial_merge_2x2(&x, 1, 2, 2, 2).unwrap();
        assert_eq!(merged.dims(), &[1, 8]);
        assert_eq!(
            merged.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]]
        );
    }
}
