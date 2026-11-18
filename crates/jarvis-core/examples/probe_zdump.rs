//! Дамп активаций и весов семантической цепочки + stage0 для python-пересчёта.
use candle_core::{Device, Tensor};
use jarvis_core::tts_zonos::GgufFile;
use std::path::Path;

fn read_npy_codes(path: &Path) -> anyhow::Result<Vec<Vec<u32>>> {
    let data = std::fs::read(path)?;
    let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
    let off = 10 + hlen;
    let mut rows = Vec::with_capacity(162);
    for f in 0..162 {
        let mut row = Vec::with_capacity(10);
        for b in 0..10 {
            let i = off + (f * 10 + b) * 4;
            row.push(u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn dump(t: &Tensor, name: &str) -> anyhow::Result<()> {
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("/tmp/zcheck/{name}.bin"), bytes)?;
    let dims = t.shape().dims();
    std::fs::write(format!("/tmp/zcheck/{name}.shape"), format!("{dims:?}"))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let frames = &read_npy_codes(Path::new("/tmp/zonos_ref/jett.codes.npy"))?[40..42];
    println!("кадры: {frames:?}");
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("zac-keep2.gguf"))?;
    let dev = Device::Cpu;
    let t32 = |n: &str| f.tensor_f32(n, &dev);

    let nf = frames.len();
    let idx = |cb: usize| -> anyhow::Result<Tensor> {
        let ids: Vec<u32> = frames.iter().map(|fr| fr[cb]).collect();
        Ok(Tensor::from_vec(ids, &[nf], &dev)?)
    };
    let sem = t32("sem.table")?
        .index_select(&idx(0)?, 0)?
        .t()?
        .contiguous()?
        .broadcast_add(&t32("sem.bias")?.unsqueeze(1)?)?;
    dump(&sem, "sem_in")?;

    // Цепочка sem_head.
    let mut x = sem.clone();
    for b in 0..4 {
        let p = format!("sem_head.{b}");
        dump(&t32(&format!("{p}.dw.weight"))?, &format!("s{b}_dw"))?;
        dump(&t32(&format!("{p}.dw.bias"))?, &format!("s{b}_dwb"))?;
        dump(&t32(&format!("{p}.norm.weight"))?, &format!("s{b}_nw"))?;
        dump(&t32(&format!("{p}.norm.bias"))?, &format!("s{b}_nb"))?;
        dump(&t32(&format!("{p}.pw1.weight"))?, &format!("s{b}_pw1"))?;
        dump(&t32(&format!("{p}.pw1.bias"))?, &format!("s{b}_pw1b"))?;
        dump(&t32(&format!("{p}.pw2.weight"))?, &format!("s{b}_pw2"))?;
        dump(&t32(&format!("{p}.pw2.bias"))?, &format!("s{b}_pw2b"))?;
        // Один блок вручную (та же логика, что в codec.rs).
        let (_c, t) = x.dims2()?;
        let k = x.dim(0)? * 0 + 7;
        let dw = t32(&format!("{p}.dw.weight"))?;
        let xp = x.pad_with_zeros(1, k - 1, 0)?;
        let mut h: Option<Tensor> = None;
        for j in 0..k {
            let shifted = xp.narrow(1, j, t)?;
            let col = dw.narrow(1, j, 1)?;
            let term = shifted.broadcast_mul(&col)?;
            h = Some(match h {
                None => term,
                Some(prev) => prev.add(&term)?,
            });
        }
        let h = h.unwrap().broadcast_add(&t32(&format!("{p}.dw.bias"))?.unsqueeze(1)?)?;
        let ht = h.transpose(0, 1)?.contiguous()?;
        let hn = candle_nn::ops::layer_norm(&ht, &t32(&format!("{p}.norm.weight"))?, &t32(&format!("{p}.norm.bias"))?, 1e-6)?;
        let h = hn.transpose(0, 1)?.contiguous()?;
        let h = t32(&format!("{p}.pw1.weight"))?.matmul(&h)?.broadcast_add(&t32(&format!("{p}.pw1.bias"))?.unsqueeze(1)?)?;
        let h = h.gelu_erf()?;
        let h = t32(&format!("{p}.pw2.weight"))?.matmul(&h)?.broadcast_add(&t32(&format!("{p}.pw2.bias"))?.unsqueeze(1)?)?;
        x = x.add(&h)?;
        dump(&x, &format!("after{b}"))?;
    }
    let z0 = t32("sem_head.out.weight")?.matmul(&x)?.broadcast_add(&t32("sem_head.out.bias")?.unsqueeze(1)?)?;
    dump(&t32("sem_head.out.weight")?, "semout_w")?;
    dump(&t32("sem_head.out.bias")?, "semout_b")?;
    dump(&z0, "z0")?;
    // Остаточные книги — только выбранные строки.
    let mut z = z0.clone();
    for cb in 0..9 {
        let part = t32(&format!("res.{cb}.table"))?.index_select(&idx(cb + 1)?, 0)?.t()?.contiguous()?;
        dump(&part, &format!("r{cb}_rows"))?;
        z = z.broadcast_add(&part)?;
    }
    z = z.broadcast_add(&t32("res.bias")?.unsqueeze(1)?)?;
    dump(&z, "z")?;
    dump(&t32("res.bias")?, "res_bias")?;

    // stage0 вход и веса слоёв.
    dump(&t32("stage.0.in.weight")?, "st0_in")?;
    let xa = t32("stage.0.in.weight")?.matmul(&z)?;
    dump(&xa, "st0_afterin")?;
    for l in 0..2 {
        let p = format!("stage.0.l{l}");
        for w in ["norm1.weight", "norm1.bias", "qkv.weight", "wo.weight", "attn_scale", "norm2.weight", "norm2.bias", "ffn1.weight", "ffn2.weight", "ffn_scale"] {
            let wn = w.replace(".", "_");
            dump(&t32(&format!("{p}.{w}"))?, &format!("st0_l{l}_{wn}"))?;
        }
    }
    dump(&t32("stage.0.out.weight")?, "st0_out")?;
    // Прогон stage0 слой за слоем (та же логика, что в codec.rs moss_layer).
    let mut sx = xa;
    for l in 0..2 {
        let p = format!("stage.0.l{l}");
        let g = |n: &str| t32(&format!("{p}.{n}")).unwrap();
        let (d, t) = sx.dims2().unwrap();
        let (nh, hd) = (20usize, 64usize);
        let h = layer_norm2(&sx, &g("norm1.weight"), &g("norm1.bias"), 1e-5).unwrap();
        let qkv = g("qkv.weight").matmul(&h).unwrap();
        let q = qkv.narrow(0, 0, d).unwrap();
        let k = qkv.narrow(0, d, d).unwrap();
        let v = qkv.narrow(0, 2 * d, d).unwrap();
        let rope = |z: &Tensor| -> Tensor {
            let zt = z.transpose(0, 1).unwrap().contiguous().unwrap().reshape(&[t, nh, hd]).unwrap().contiguous().unwrap();
            let r = rope_apply2(&zt, 0, 64, &dev).unwrap();
            r.reshape(&[t, nh * hd]).unwrap().transpose(0, 1).unwrap().contiguous().unwrap()
        };
        let (q, k) = (rope(&q), rope(&k));
        let split = |z: &Tensor| z.reshape(&[nh, hd, t]).unwrap().contiguous().unwrap();
        let (qh, kh, vh) = (split(&q), split(&k), split(&v));
        let qb = qh.transpose(1, 2).unwrap().contiguous().unwrap();
        let kb = kh.transpose(1, 2).unwrap().contiguous().unwrap();
        let vb = vh.transpose(1, 2).unwrap().contiguous().unwrap();
        let scores = qb.matmul(&kb.transpose(1, 2).unwrap()).unwrap().affine(1.0 / (hd as f64).sqrt(), 0.0).unwrap();
        // Каузальная маска T×T.
        let mut mask = vec![0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                if j > i { mask[i * t + j] = f32::NEG_INFINITY; }
            }
        }
        let mask = Tensor::from_vec(mask, &[t, t], &dev).unwrap();
        let probs = candle_nn::ops::softmax(&scores.broadcast_add(&mask.unsqueeze(0).unwrap()).unwrap(), 2).unwrap();
        let ob = probs.matmul(&vb).unwrap();
        let ob = ob.transpose(0, 1).unwrap().contiguous().unwrap().reshape(&[t, d]).unwrap().transpose(0, 1).unwrap().contiguous().unwrap();
        let o = g("wo.weight").matmul(&ob).unwrap().broadcast_mul(&g("attn_scale").unsqueeze(1).unwrap()).unwrap();
        let sx1 = sx.add(&o).unwrap();
        let h2 = layer_norm2(&sx1, &g("norm2.weight"), &g("norm2.bias"), 1e-5).unwrap();
        let f1 = g("ffn1.weight").matmul(&h2).unwrap().gelu_erf().unwrap();
        let f2 = g("ffn2.weight").matmul(&f1).unwrap().broadcast_mul(&g("ffn_scale").unsqueeze(1).unwrap()).unwrap();
        sx = sx1.add(&f2).unwrap();
        dump(&sx, &format!("st0_after_l{l}"))?;
    }
    let sout = t32("stage.0.out.weight")?.matmul(&sx)?;
    dump(&sout, "st0_after_out")?;
    println!("дамп готов: /tmp/zcheck/");
    Ok(())
}

fn layer_norm2(x: &Tensor, w: &Tensor, b: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
    let (c, t) = x.dims2()?;
    let xt = x.transpose(0, 1)?.contiguous()?;
    let out = candle_nn::ops::layer_norm(&xt, w, b, eps)?;
    Ok(out.transpose(0, 1)?.reshape(&[c, t])?.contiguous()?)
}

fn rope_apply2(x: &Tensor, pos0: usize, hd: usize, dev: &Device) -> anyhow::Result<Tensor> {
    let (t, nh, _) = x.dims3()?;
    let pairs = hd / 2;
    let x4 = x.reshape(&[t, nh, pairs, 2])?.contiguous()?;
    let a = x4.narrow(3, 0, 1)?.squeeze(3)?;
    let b = x4.narrow(3, 1, 1)?.squeeze(3)?;
    let mut cos = vec![0.0f32; t * pairs];
    let mut sin = vec![0.0f32; t * pairs];
    for p in 0..t {
        for i in 0..pairs {
            let ang = (pos0 + p) as f32 * 10000f32.powf(-2.0 * i as f32 / hd as f32);
            cos[p * pairs + i] = ang.cos();
            sin[p * pairs + i] = ang.sin();
        }
    }
    let cos = Tensor::from_vec(cos, &[t, 1, pairs], dev)?;
    let sin = Tensor::from_vec(sin, &[t, 1, pairs], dev)?;
    let ra = (a.broadcast_mul(&cos)? - b.broadcast_mul(&sin)?)?;
    let rb = (a.broadcast_mul(&sin)? + b.broadcast_mul(&cos)?)?;
    Ok(Tensor::cat(&[ra.unsqueeze(3)?, rb.unsqueeze(3)?], 3)?.reshape(&[t, nh, hd])?)
}
