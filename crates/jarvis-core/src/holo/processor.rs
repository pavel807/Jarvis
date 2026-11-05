//! Rust image preprocessing and multimodal prompt assembly for UI-Venus/Holo-3.1.
//! The implementation follows the checkpoint's processor metadata and Holo-3.1
//! patch ordering. It deliberately does not pretend to execute the decoder.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, imageops::FilterType};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Deserialize)]
struct RawProcessorConfig {
    #[serde(default = "default_true")]
    do_resize: bool,
    #[serde(default = "default_true")]
    do_rescale: bool,
    #[serde(default = "default_true")]
    do_normalize: bool,
    #[serde(default = "default_rescale")]
    rescale_factor: f32,
    #[serde(default = "default_mean")]
    image_mean: Vec<f32>,
    #[serde(default = "default_std")]
    image_std: Vec<f32>,
    #[serde(default = "default_patch")]
    patch_size: usize,
    #[serde(default = "default_temporal_patch")]
    temporal_patch_size: usize,
    #[serde(default = "default_merge")]
    merge_size: usize,
    #[serde(default)]
    size: Option<SizeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct SizeConfig {
    shortest_edge: Option<u64>,
    longest_edge: Option<u64>,
}

fn default_true() -> bool { true }
fn default_rescale() -> f32 { 1.0 / 255.0 }
fn default_mean() -> Vec<f32> { vec![0.5, 0.5, 0.5] }
fn default_std() -> Vec<f32> { vec![0.5, 0.5, 0.5] }
fn default_patch() -> usize { 16 }
fn default_temporal_patch() -> usize { 2 }
fn default_merge() -> usize { 2 }

#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// Flattened patch tensor in Holo-3.1 order: [patch_count, C*T*P*P].
    pub pixel_values: Vec<f32>,
    /// (temporal, patch_height, patch_width), for a single still image T=1.
    pub grid_thw: [u32; 3],
    pub resized_width: u32,
    pub resized_height: u32,
    pub image_token_count: usize,
    patch_dim: usize,
}

impl ProcessedImage {
    pub fn pixel_values_tensor(&self, device: &Device, dtype: DType) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            self.pixel_values.clone(),
            (self.grid_thw[1] as usize * self.grid_thw[2] as usize, self.patch_dim),
            device,
        )?.to_dtype(dtype)?)
    }

    pub fn grid_tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_vec(self.grid_thw.to_vec(), (1, 3), device)?)
    }
}

#[derive(Debug, Clone)]
pub struct PreparedPrompt {
    pub input_ids: Vec<u32>,
    /// Half-open token span containing all image placeholder tokens.
    pub image_span: (usize, usize),
    /// Per-token `[time, height, width]` position IDs for interleaved MRoPE.
    pub mrope_position_ids: Vec<[u32; 3]>,
    pub image: ProcessedImage,
}

#[derive(Debug, Clone)]
pub struct UiVenusProcessor {
    cfg: RawProcessorConfig,
    image_token_id: u32,
}

impl UiVenusProcessor {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let cfg_path = model_dir.join("preprocessor_config.json");
        let mut cfg: RawProcessorConfig = serde_json::from_slice(
            &std::fs::read(&cfg_path).with_context(|| format!("Не найден {}", cfg_path.display()))?
        ).context("Некорректный preprocessor_config.json UI-Venus")?;
        let model_cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("config.json"))?
        ).context("Некорректный config.json UI-Venus")?;
        let image_token_id = model_cfg.get("image_token_id").and_then(|v| v.as_u64())
            .context("В config.json отсутствует image_token_id")? as u32;
        if cfg.patch_size != 16 || cfg.temporal_patch_size != 2 || cfg.merge_size != 2 {
            bail!("Неподдерживаемая конфигурация патчей UI-Venus: patch={}, temporal={}, merge={}", cfg.patch_size, cfg.temporal_patch_size, cfg.merge_size);
        }
        if cfg.image_mean.len() != 3 || cfg.image_std.len() != 3 || cfg.image_std.iter().any(|v| *v == 0.0) {
            bail!("image_mean/image_std должны содержать три корректных RGB-компонента");
        }
        // Потолок пикселей: полный скриншот ретины (5–6 МП) даёт гигабайты
        // на тензор vision-внимания — Metal падает по памяти. Кап на точное
        // разрешение обучения Holo (1440×900 ≈ 1,31 МП): в распределении
        // модели, текст читаем. Переопределяется JARVIS_HOLO_MAX_PIXELS.
        let cap: u64 = std::env::var("JARVIS_HOLO_MAX_PIXELS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_310_720) as u64; // 1440×900 — разрешение обучения Holo
        if let Some(size) = &mut cfg.size
            && let Some(longest) = size.longest_edge
            && longest > cap
        {
            size.longest_edge = Some(cap);
        }
        Ok(Self { cfg, image_token_id })
    }

    pub fn process_image(&self, path: &Path) -> Result<ProcessedImage> {
        let image = image::open(path).with_context(|| format!("Не удалось прочитать изображение {}", path.display()))?;
        self.process_dynamic_image(image)
    }

    pub fn process_dynamic_image(&self, image: DynamicImage) -> Result<ProcessedImage> {
        let original_w = image.width() as usize;
        let original_h = image.height() as usize;
        if original_w == 0 || original_h == 0 { bail!("Пустое изображение"); }
        let factor = self.cfg.patch_size * self.cfg.merge_size;
        let min_pixels = self.cfg.size.as_ref().and_then(|s| s.shortest_edge).unwrap_or(65_536) as usize;
        let max_pixels = self.cfg.size.as_ref().and_then(|s| s.longest_edge).unwrap_or(16_777_216) as usize;
        let (target_h, target_w) = if self.cfg.do_resize {
            smart_resize(original_h, original_w, factor, min_pixels, max_pixels)?
        } else {
            if original_h % factor != 0 || original_w % factor != 0 {
                bail!("do_resize=false, но размеры изображения должны делиться на {factor}");
            }
            (original_h, original_w)
        };
        let resized = image.resize_exact(target_w as u32, target_h as u32, FilterType::CatmullRom).to_rgb8();
        let grid_h = target_h / self.cfg.patch_size;
        let grid_w = target_w / self.cfg.patch_size;
        let temporal = self.cfg.temporal_patch_size;
        let patch = self.cfg.patch_size;
        let merge = self.cfg.merge_size;
        let patch_dim = 3 * temporal * patch * patch;
        let patch_count = grid_h * grid_w;
        let mut values = Vec::with_capacity(patch_count * patch_dim);

        // Holo-3.1 patch order: (T, H/merge, W/merge, merge, merge, C, temporal, P, P).
        // A still image is duplicated along the temporal patch axis.
        for _grid_t in 0..1 {
            for block_y in 0..(grid_h / merge) {
                for block_x in 0..(grid_w / merge) {
                    for inner_y in 0..merge {
                        for inner_x in 0..merge {
                            for channel in 0..3 {
                                for _t in 0..temporal {
                                    for py in 0..patch {
                                        for px in 0..patch {
                                            let x = (block_x * merge + inner_x) * patch + px;
                                            let y = (block_y * merge + inner_y) * patch + py;
                                            let raw = resized.get_pixel(x as u32, y as u32)[channel] as f32;
                                            let mut value = if self.cfg.do_rescale { raw * self.cfg.rescale_factor } else { raw };
                                            if self.cfg.do_normalize {
                                                value = (value - self.cfg.image_mean[channel]) / self.cfg.image_std[channel];
                                            }
                                            values.push(value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let token_count = patch_count / (merge * merge);
        if values.len() != patch_count * patch_dim {
            bail!("Внутренняя ошибка patchify: {} значений вместо {}", values.len(), patch_count * patch_dim);
        }
        Ok(ProcessedImage {
            pixel_values: values,
            grid_thw: [1, grid_h as u32, grid_w as u32],
            resized_width: target_w as u32,
            resized_height: target_h as u32,
            image_token_count: token_count,
            patch_dim,
        })
    }

    /// Builds the Holo-3.1 chat layout with expanded image_pad tokens for the
    /// GUI-agent loop: system + блок GUI-инструментов, user-ход «задача +
    /// история шагов + скриншот», assistant-префикс с открытым `<think>`.
    pub fn prepare_prompt(&self, tokenizer: &Tokenizer, image: ProcessedImage, task: &str) -> Result<PreparedPrompt> {
        self.prepare_prompt_with_history(tokenizer, image, task, &[])
    }

    /// Same as [`Self::prepare_prompt`] but includes the executed action log
    /// («Step N: …»), чтобы модель видела, что уже сделано, и могла
    /// самоисправляться по свежему скриншоту.
    pub fn prepare_prompt_with_history(
        &self,
        tokenizer: &Tokenizer,
        image: ProcessedImage,
        task: &str,
        history: &[String],
    ) -> Result<PreparedPrompt> {
        let tools = super::action::tool_definitions();
        let mut system = format!(
            "{}\n\n# Tools\n\nYou may call one or more functions to assist with the user task.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
            super::action::system_prompt(),
        );
        for tool in tools.as_array().context("tool_definitions не массив")? {
            system.push('\n');
            system.push_str(&tool.to_string());
        }
        system.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );

        let mut user = task.to_string();
        if !history.is_empty() {
            user.push_str("\n\nPreviously executed actions:");
            for (i, line) in history.iter().enumerate() {
                user.push_str(&format!("\nStep {i}: {line}"));
            }
        }

        let image_tokens = "<|image_pad|>".repeat(image.image_token_count);
        // Закрытый think-блок: агентный шаг отвечает сразу действием.
        // Открытый `<think>` уводил модель в многословные рассуждения,
        // которые съедали лимит токенов до вызова инструмента.
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n{user}\n<|vision_start|>{image_tokens}<|vision_end|><|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("Tokenizer Holo-3.1: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        let mut positions = ids.iter().enumerate().filter_map(|(i, id)| (*id == self.image_token_id).then_some(i));
        let first = positions.next().context("Tokenizer не распознал <|image_pad|> как image_token_id")?;
        let mut end = first + 1;
        for idx in positions {
            if idx != end { bail!("image_pad токены оказались не непрерывным диапазоном"); }
            end += 1;
        }
        if end - first != image.image_token_count {
            bail!("Tokenizer создал {} image_pad токенов, ожидалось {}", end - first, image.image_token_count);
        }
        let mrope_position_ids = super::mrope::position_ids(
            ids.len(),
            (first, end),
            image.grid_thw,
            self.cfg.merge_size,
        )?;
        Ok(PreparedPrompt {
            input_ids: ids,
            image_span: (first, end),
            mrope_position_ids,
            image,
        })
    }
}

fn smart_resize(height: usize, width: usize, factor: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 || min_pixels == 0 || max_pixels < min_pixels {
        bail!("Некорректные параметры smart_resize");
    }
    let aspect = height.max(width) as f64 / height.min(width) as f64;
    if aspect > 200.0 { bail!("Соотношение сторон изображения слишком велико: {aspect:.1}"); }
    let round_to = |v: f64| ((v / factor as f64).round() as usize).max(1) * factor;
    let mut h = round_to(height as f64);
    let mut w = round_to(width as f64);
    if h * w > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h = (((height as f64 / beta / factor as f64).floor() as usize).max(1)) * factor;
        w = (((width as f64 / beta / factor as f64).floor() as usize).max(1)) * factor;
    } else if h * w < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h = (((height as f64 * beta / factor as f64).ceil() as usize).max(1)) * factor;
        w = (((width as f64 * beta / factor as f64).ceil() as usize).max(1)) * factor;
    }
    Ok((h, w))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smart_resize_keeps_factor_and_bounds() {
        let (h, w) = smart_resize(1080, 1920, 32, 65_536, 1_048_576).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h * w <= 1_048_576);
        assert!(h * w >= 65_536);
    }

    #[test]
    fn still_image_patchify_has_qwen3_vl_dimensions() {
        let cfg = RawProcessorConfig {
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            rescale_factor: 1.0 / 255.0,
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            size: Some(SizeConfig { shortest_edge: Some(4096), longest_edge: Some(16384) }),
        };
        let processor = UiVenusProcessor { cfg, image_token_id: 151655 };
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 64, image::Rgb([127, 127, 127])));
        let processed = processor.process_dynamic_image(image).unwrap();
        assert_eq!(processed.grid_thw, [1, 4, 4]);
        assert_eq!(processed.image_token_count, 4);
        assert_eq!(processed.pixel_values.len(), 16 * 3 * 2 * 16 * 16);
    }
}
