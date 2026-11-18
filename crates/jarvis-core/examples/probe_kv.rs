// Печать ключевых KV из backbone-ptq.gguf.
use jarvis_core::tts_zonos::GgufFile;

fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut keys: Vec<_> = f.kv.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let v = &f.kv[&k];
        let s = match v {
            jarvis_core::tts_zonos::Value::Array(a) => {
                let items: Vec<String> = a.iter().map(|x| format!("{x:?}")).collect();
                if items.len() > 24 {
                    format!("[{} … {}]", items[..12].join(", "), items[items.len() - 4..].join(", "))
                } else {
                    format!("[{}]", items.join(", "))
                }
            }
            other => format!("{other:?}"),
        };
        println!("{k} = {s}");
    }
    dump_stages()?;
    dump_dtypes()
}

fn dump_dtypes() -> anyhow::Result<()> {
    use jarvis_core::tts_zonos::GgufFile;
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("zac-keep2.gguf"))?;
    let mut counts: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut names_by_type: std::collections::BTreeMap<u32, Vec<String>> = Default::default();
    for (name, t) in f.tensors.iter() {
        *counts.entry(t.ggml_type).or_default() += 1;
        names_by_type.entry(t.ggml_type).or_default().push(name.clone());
    }
    println!("zac file_type: {:?}", f.kv.get("general.file_type"));
    for (dt, n) in &counts {
        println!("dtype {dt}: {n} тензоров, напр. {:?}", names_by_type[dt].iter().take(3).cloned().collect::<Vec<_>>());
    }
    Ok(())
}
// Хвост: стадии декодера из zac-keep2.gguf.
fn dump_stages() -> anyhow::Result<()> {
    use jarvis_core::tts_zonos::ZacDecoder;
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;
    println!("n_fft={:?} hop={} fir={} headroom={}", dec.cfg.n_fft, dec.cfg.head_hop, dec.cfg.fir_kernel, dec.cfg.headroom_db);
    for (i, s) in dec.stages.iter().enumerate() {
        println!(
            "stage {i}: in={} out={:?} d={} heads={} layers={} n_ff={} context={} patch={}",
            s.hp.in_dim, s.hp.out_dim, s.hp.d_model, s.hp.n_head, s.hp.n_layer, s.hp.n_ff, s.hp.context, s.hp.patch
        );
    }
    Ok(())
}
