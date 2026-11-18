use jarvis_core::tts_zonos::GgufFile;
fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("zac-keep2.gguf"))?;
    let mut keys: Vec<_> = f.kv.keys().cloned().collect();
    keys.sort();
    for k in keys {
        println!("{k} = {:?}", f.kv[&k]);
    }
    Ok(())
}
