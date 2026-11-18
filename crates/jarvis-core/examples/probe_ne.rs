use jarvis_core::tts_zonos::GgufFile;
fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("spk-encoder.gguf"))?;
    for n in ["fe.head.conv.weight", "b0.pos.dw.weight"] {
        if let Some(t) = f.tensors.get(n) {
            println!("{n}: ne={:?}", t.dims);
        }
    }
    let mut names: Vec<_> = f.tensors.keys().cloned().collect();
    names.sort();
    for n in names.iter().filter(|n| n.contains("pos.dw.weight")).take(3) {
        println!("{n}: ne={:?}", f.tensors[n].dims);
    }
    Ok(())
}
