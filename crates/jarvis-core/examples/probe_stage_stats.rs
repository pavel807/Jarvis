//! Постатейная статистика декодера на эталонных кодах.
use anyhow::Context;
use candle_core::{Device, Tensor};
use jarvis_core::tts_zonos::ZacDecoder;
use std::path::Path;

fn read_npy_codes(path: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
    let data = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let header = String::from_utf8_lossy(&data[10..10 + hlen]).to_string();
    let shape_part = header.split("'shape':").nth(1).context("нет shape")?;
    let parens = shape_part.split('(').nth(1).context("нет (shape)")?;
    let nums: Vec<usize> = parens
        .split(|c| c == ',' || c == ')')
        .filter_map(|tok| tok.trim().parse::<usize>().ok())
        .collect();
    let (t, books) = (nums[0], nums[1]);
    let off = 10 + hlen;
    let mut rows = Vec::with_capacity(t);
    for f in 0..t {
        let mut row = Vec::with_capacity(books);
        for b in 0..books {
            let i = off + (f * books + b) * 4;
            row.push(u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn stats(t: &Tensor, label: &str) -> anyhow::Result<()> {
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let n = v.len();
    let mean = v.iter().sum::<f32>() / n as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &v {
        mn = mn.min(x);
        mx = mx.max(x);
    }
    println!("{label:<28} n={n:<9} mean={mean:+.4} std={:.4} min={mn:+.4} max={mx:+.4}", var.sqrt());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let codes_path = args.get(1).cloned().unwrap_or("/tmp/zonos_ref/jett.codes.npy".into());
    let frames = read_npy_codes(Path::new(&codes_path))?;
    // Возьмём отрезок 40 кадров, где речь (по эталону — стационарно).
    let frames = &frames[40..80];

    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = jarvis_core::tts_zonos::GgufFile::open(&dir.join("zac-keep2.gguf"))?;
    // Размеры ключевых тензоров.
    for n in ["sem.table", "sem.bias", "res.0.table", "res.bias", "sem_head.0.dw.weight", "sem_head.out.weight", "stage.0.in.weight", "stage.0.out.weight", "stage.1.in.weight", "head.out.0.weight", "head.mixer.weight", "resample.kernel"] {
        if let Some(t) = f.tensors.get(n) {
            println!("{n:<24} ne={:?} type={}", t.dims, t.ggml_type);
        }
    }

    let dev = Device::Cpu;
    let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &dev)?;
    dec.dump_pipeline(&frames)?;
    Ok(())
}
