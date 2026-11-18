use candle_core::Tensor;
use jarvis_core::tts_zonos::GgufFile;
fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    for n in ["speaker_proj.weight", "speaker_proj.bias"] {
        let t = &f.tensors[n];
        println!("{n}: ne={:?} type={}", t.dims, t.ggml_type);
    }
    let w = f.tensor_f32("speaker_proj.weight", &candle_core::Device::Cpu)?;
    let b = f.tensor_f32("speaker_proj.bias", &candle_core::Device::Cpu)?;
    let wv = w.to_vec2::<f32>()?;
    println!("candle shape: {:?}", w.shape().dims());
    let mut stats = |name: &str, v: &[f32]| {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        println!("{name}: mean={mean:.4} std={std:.4} max_abs={:.4}", v.iter().fold(0f32, |a, &x| a.max(x.abs())));
    };
    let flat: Vec<f32> = wv.iter().flatten().copied().collect();
    stats("weight", &flat);
    stats("bias", &b.to_vec1::<f32>()?);
    // Спроецируем jarvis-эмбеддинг и посмотрим масштаб.
    let data = std::fs::read("/Users/pavellabushev/Library/Application Support/Jarvis/models/zonos3/voices/jarvis.embedding.npy")?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let off = 10 + hlen;
    let v: Vec<f32> = data[off..off + 192 * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let vt = Tensor::from_vec(v.clone(), (1, 192), &candle_core::Device::Cpu)?;
    let p = vt.matmul(&w.t()?)?.broadcast_add(&b.unsqueeze(0)?)?;
    let pv = p.to_vec2::<f32>()?;
    stats("projected", &pv[0]);
    println!("projected первые 8: {:?}", &pv[0][..8.min(pv[0].len())]);
    Ok(())
}
