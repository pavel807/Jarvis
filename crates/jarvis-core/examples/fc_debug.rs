//! Пошаговая отладка блока 0 энкодера против ONNX-снимка (attn_ref.json).
//! Диагностический стенд верификации FastConformer.
//!
//! Запуск: cargo run --release --example fc_debug -p jarvis-core

use anyhow::Result;
use jarvis_core::asr::model::FastConformer;
use serde_json::Value;

/// Рекурсивно сплющивает JSON-массив чисел.
fn flatten(v: &Value, out: &mut Vec<f32>) {
    match v {
        Value::Array(items) => {
            for it in items {
                flatten(it, out);
            }
        }
        Value::Number(n) => out.push(n.as_f64().unwrap_or(0.0) as f32),
        _ => {}
    }
}

fn tensor_flat(t: &candle_core::Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

fn main() -> Result<()> {
    let model = FastConformer::load(&jarvis_core::paths::models_dir())?;
    let fx: Value =
        serde_json::from_str(&std::fs::read_to_string(format!(
            "build/fastconformer/fixtures/{}.json",
            std::env::args().nth(1).unwrap_or_else(|| "cmd1".into())
        ))?)?;
    let fixture = std::env::args().nth(1).unwrap_or_else(|| "cmd1".into());
    let mut stages: Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "build/fastconformer/stages_{}.json",
        fixture
    ))?)?;
    if fixture == "cmd1" {
        let extra: Value = serde_json::from_str(
            &std::fs::read_to_string("build/fastconformer/ff1_ref.json")?,
        )?;
        if let (Some(a), Some(b)) = (stages.as_object_mut(), extra.as_object()) {
            for (k, v) in b {
                a.insert(k.clone(), v.clone());
            }
        }
    }

    let audio: Vec<f32> = fx["audio"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let (mel, _length) = model.frontend.compute(&audio, &model.device)?;

    let pre = model.encoder.subsampling_forward(&mel)?;
    // NeMo scale_input: слои получают x * sqrt(d_model).
    let pre = pre.affine((model.cfg.encoder.d_model as f64).sqrt(), 0.0)?;
    let table = jarvis_core::asr::attention::build_pos_table(
        model.cfg.encoder.pos_emb_max_len,
        model.cfg.encoder.d_model,
        &model.device,
    )?;
    let t = pre.dim(1)?;
    let pos = jarvis_core::asr::attention::slice_pos_emb(
        &table,
        model.cfg.encoder.pos_emb_max_len,
        t,
    )?;

    // Маски как в энкодере (cmd2: T''=19, valid=18).
    let t_valid = fx["enc_valid"].as_u64().unwrap() as usize;
    let masks = if t > t_valid {
        let idx = candle_core::Tensor::arange(0u32, t as u32, &model.device)?;
        let bound = candle_core::Tensor::new(t_valid as u32, &model.device)?
            .broadcast_as(idx.shape())?;
        let keep = idx.lt(&bound)?.to_dtype(candle_core::DType::F32)?;
        let invalid = keep.affine(-1.0, 1.0)?.affine(-10000.0, 0.0)?;
        let add = invalid
            .reshape((1, 1, 1, t))?
            .broadcast_add(&invalid.reshape((1, 1, t, 1))?)?;
        let row_zero = keep.reshape((1, 1, t, 1))?;
        Some((add, row_zero))
    } else {
        None
    };
    let block = &model.encoder.blocks[0];
    let trace = block.forward_trace(&pre, &pos, masks.as_ref().map(|(a, b)| (a, b)))?;

    // --- Проба (а): совпадают ли НАБОРЫ значений bd_raw (перестановка?) ---
    {
        let mut ref_flat = Vec::new();
        flatten(
            &stages["l0.self_attn_MatMul"],
            &mut ref_flat,
        );
        let my_flat = tensor_flat(&trace[0].1)?;
        let mut r2 = ref_flat.clone();
        let mut m2 = my_flat.clone();
        r2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        m2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d: f32 = r2
            .iter()
            .zip(m2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let ref_max = ref_flat.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!(
            "bd_raw: ref max|{ref_max:.1} | сортированные Δ={d:.4e} (0 = та же совокупность значений, перестановка)"
        );
    }

    // --- Проба (б): прямое сравнение входов matmul bd ---
    let mut qk: Value =
        serde_json::from_str(&std::fs::read_to_string("build/fastconformer/qk_ref.json")?)?;
    {
        let ff1: Value =
            serde_json::from_str(&std::fs::read_to_string("build/fastconformer/ff1_ref.json")?)?;
        if let (Some(a), Some(b)) = (qk.as_object_mut(), ff1.as_object()) {
            for (k, v) in b {
                a.insert(k.clone(), v.clone());
            }
        }
    }
    // --- Проба (в): dw-свёртка+BN вручную на CPU против Metal и ONNX ---
    {
        use candle_core::{DType, Device, Tensor};
        use candle_nn::VarBuilder;
        // ГЛАУ выход (мой, Metal) — вход свёртки.
        let glu = trace
            .iter()
            .find(|(n, _)| n == "conv_glu")
            .map(|(_, t)| t.clone())
            .unwrap();
        let glu_cpu = glu.to_device(&Device::Cpu)?;
        let (b, c, tlen) = glu_cpu.dims3()?;
        // Веса слоя 0 из safetensors на CPU.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &["models/fastconformer/model.safetensors"],
                DType::F32,
                &Device::Cpu,
            )?
        };
        let conv_vb = vb.pp("encoder").pp("layers").pp("0").pp("conv");
        let w = conv_vb.get((c, 1, 9), "depthwise_conv.weight")?;
        let bias = conv_vb.get(c, "depthwise_conv.bias")?;
        let bn_w = vb
            .pp("encoder")
            .pp("layers")
            .pp("0")
            .pp("conv")
            .pp("batch_norm")
            .get(c, "weight")?;
        let bn_b = vb
            .pp("encoder")
            .pp("layers")
            .pp("0")
            .pp("conv")
            .pp("batch_norm")
            .get(c, "bias")?;
        let bn_rm = vb
            .pp("encoder")
            .pp("layers")
            .pp("0")
            .pp("conv")
            .pp("batch_norm")
            .get(c, "running_mean")?;
        let bn_rv = vb
            .pp("encoder")
            .pp("layers")
            .pp("0")
            .pp("conv")
            .pp("batch_norm")
            .get(c, "running_var")?;
        let wv = w.flatten_all()?.to_vec1::<f32>()?;
        let bv = bias.to_vec1::<f32>()?;
        let bw = bn_w.to_vec1::<f32>()?;
        let bb = bn_b.to_vec1::<f32>()?;
        let rm = bn_rm.to_vec1::<f32>()?;
        let rv = bn_rv.to_vec1::<f32>()?;
        let x = glu_cpu.to_vec3::<f32>()?;
        let pad = 4usize;
        // Ручная depthwise-свёртка + BN (fold-эквивалент ONNX).
        let mut manual = vec![0f32; b * c * tlen];
        for bi in 0..b {
            for ch in 0..c {
                let scale = bw[ch] / (rv[ch] + 1e-5).sqrt();
                let shift = bb[ch] - bw[ch] * rm[ch] / (rv[ch] + 1e-5).sqrt();
                for ti in 0..tlen {
                    let mut acc = 0f32;
                    for k in 0..9 {
                        let si = ti + k; // паддинг слева 4: индекс входа ti+k-4+4
                        let xi = if si >= pad && si < pad + tlen {
                            x[bi][ch][si - pad]
                        } else {
                            0.0
                        };
                        acc += xi * wv[ch * 9 + k];
                    }
                    manual[bi * c * tlen + ch * tlen + ti] = acc * scale + shift + bv[ch] * scale;
                }
            }
        }
        // Сравнение: мой Metal dw+BN (conv_out до pw2 недоступен — берём dw_raw
        // и применяем BN формулой? Нет: сравним dw_raw с ручной свёрткой без BN).
        let dw_raw: Vec<f32> = match trace.iter().find(|(n, _)| n == "conv_dw_raw") {
            Some((_, t)) => t
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            None => anyhow::bail!("нет conv_dw_raw в трассе"),
        };
        // Ручная свёртка БЕЗ BN для сравнения с dw_raw:
        let mut manual_raw = vec![0f32; b * c * tlen];
        for bi in 0..b {
            for ch in 0..c {
                for ti in 0..tlen {
                    let mut acc = 0f32;
                    for k in 0..9 {
                        let si = ti + k;
                        let xi = if si >= pad && si < pad + tlen {
                            x[bi][ch][si - pad]
                        } else {
                            0.0
                        };
                        acc += xi * wv[ch * 9 + k];
                    }
                    manual_raw[bi * c * tlen + ch * tlen + ti] = acc + bv[ch];
                }
            }
        }
        let d1: f32 = dw_raw
            .iter()
            .zip(manual_raw.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "dw_raw (Metal) vs ручная CPU: Δ={d1:.4e} (0 = Metal depthwise conv корректен)"
        );
        // ONNX dw conv (свёрнутый BN) vs ручная dw+BN:
        let onnx_conv: Vec<f32> = {
            let mut v = Vec::new();
            flatten(&stages["l0.conv_pointwise_conv2_Conv"], &mut v);
            v // это pw2 out — не годится; нужен dw Conv — см. ниже
        };
        let _ = onnx_conv;
    }

    // Строки softmax для расхождения cmd2: (голова, запрос) = (1, 17) и (0, 0).
    {
        let t = trace
            .iter()
            .find(|(n, _)| n == "a.attn")
            .map(|(_, t)| t.clone())
            .unwrap();
        let my_flat = tensor_flat(&t)?;
        let mut ref_flat = Vec::new();
        flatten(&stages["l0.self_attn_Softmax"], &mut ref_flat);
        let get = |v: &[f32], h: usize, i: usize| -> Vec<f32> {
            let base = h * 19 * 19 + i * 19;
            v[base..base + 19].to_vec()
        };
        for (h, i) in [(1usize, 17usize), (0, 0), (0, 17)] {
            println!(
                "softmax h={h} q={i}:\n  ref={:?}\n  my ={:?}",
                get(&ref_flat, h, i),
                get(&my_flat, h, i)
            );
        }
    }

    for (name, key) in [("ff1_res", "l0.Add"), ("q_v", "Add_1"), ("p_t", "Transpose_3")] {
        let Some(t) = trace.iter().find(|(n, _)| n == name).map(|(_, t)| t) else {
            continue;
        };
        let mut ref_flat = Vec::new();
        flatten(&qk[key], &mut ref_flat);
        let my_flat = tensor_flat(t)?;
        let n = ref_flat.len().min(my_flat.len());
        let mut max = 0f32;
        let mut idx = 0usize;
        for i in 0..n {
            let d = (ref_flat[i] - my_flat[i]).abs();
            if d > max {
                max = d;
                idx = i;
            }
        }
        println!(
            "{name}: Δ={max:.4e} @#{idx} (n={n}; ref[0..3]={:?} my[0..3]={:?})",
            &ref_flat[..3.min(ref_flat.len())],
            &my_flat[..3.min(my_flat.len())]
        );
    }
    for (name, tensor) in &trace {
        let key = match name.as_str() {
            "a.bd_raw" => "l0.MatMul",
            "a.bd_shifted" => "l0.Slice_3",
            "a.ac" => "l0.self_attn_MatMul_1",
            "a.scores" => "l0.self_attn_Div",
            "a.attn" => "l0.self_attn_Softmax",
            "a.attn_v" => "l0.self_attn_MatMul_2",
            "a.out" => "l0.self_attn_linear_out_Add",
            "attn_res" => "l0.Add_1",
            "conv_glu" => "l0.conv_Mul",
            "conv_out" => "l0.conv_pw2",
            "conv_res" => "l0.Add_2",
            _ => continue,
        };
        let Some(refv) = stages.get(key) else {
            println!("{name}: нет эталона {key}");
            continue;
        };
        let mut ref_flat = Vec::new();
        flatten(refv, &mut ref_flat);
        let my_flat = tensor_flat(tensor)?;
        let n = ref_flat.len().min(my_flat.len());
        let mut max = 0f32;
        let mut argmax = 0usize;
        for i in 0..n {
            let d = (ref_flat[i] - my_flat[i]).abs();
            if d > max {
                max = d;
                argmax = i;
            }
        }
        println!(
            "{name:14} {key:34} Δ={max:.4e} @#{argmax} (n={n} ref={} my={}) ref@Δ={:+.6} my@Δ={:+.6}",
            ref_flat.len(),
            my_flat.len(),
            ref_flat.get(argmax).copied().unwrap_or(f32::NAN),
            my_flat.get(argmax).copied().unwrap_or(f32::NAN)
        );
    }
    Ok(())
}
