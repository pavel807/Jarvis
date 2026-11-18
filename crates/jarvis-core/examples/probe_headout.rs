use jarvis_core::tts_zonos::GgufFile;
fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("zac-keep2.gguf"))?;
    let dev = candle_core::Device::Cpu;
    for i in 0..2 {
        let b = f.tensor_f32(&format!("head.out.{i}.bias"), &dev)?.to_vec1::<f32>()?;
        let w = f.tensor_f32(&format!("head.out.{i}.weight"), &dev)?.to_vec2::<f32>()?;
        let n = b.len();
        println!("head.out.{i}: bias len={n}, weight {:?}x{:?}", w.len(), w[0].len());
        // Профиль bias по частотам (логмаг-часть = первые n/2).
        let bins = n / 2;
        let step = bins / 12;
        print!("  bias logmag: ");
        for k in 0..12 {
            let seg = &b[k * step..(k + 1) * step.max(1)];
            let m = seg.iter().sum::<f32>() / seg.len() as f32;
            print!("{m:+.2} ");
        }
        println!();
        print!("  bias phase : ");
        for k in 0..12 {
            let seg = &b[bins + k * step..bins + (k + 1) * step.max(1)];
            let m = seg.iter().sum::<f32>() / seg.len() as f32;
            print!("{m:+.2} ");
        }
        println!();
        // std веса
        let mut s = 0.0; let mut s2 = 0.0; let n_all = 0;
        let _ = n_all;
        let flat: Vec<f32> = w.iter().flatten().copied().collect();
        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        let std = (flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt();
        println!("  weight std={std:.4}");
    }
    Ok(())
}
