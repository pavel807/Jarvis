use jarvis_core::tts_zonos::GgufFile;
fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("spk-encoder.gguf"))?;
    let mut keys: Vec<_> = f.kv.keys().cloned().collect();
    keys.sort();
    for k in keys { println!("KV {k} = {:?}", f.kv[&k]); }
    let mut names: Vec<_> = f.tensors.keys().cloned().collect();
    names.sort();
    for n in &names {
        let t = &f.tensors[n];
        println!("T {n}: ne={:?} type={}", t.dims, t.ggml_type);
    }
    Ok(())
}
