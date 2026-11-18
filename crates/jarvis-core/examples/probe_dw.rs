// Дамп размеров dw/pw-тензоров zac-декодера и бэкбона.
use jarvis_core::tts_zonos::GgufFile;

fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    for fname in ["zac-keep2.gguf", "backbone-ptq.gguf"] {
        let f = GgufFile::open(&dir.join(fname))?;
        println!("=== {fname} ===");
        let mut names: Vec<_> = f.tensors.keys().cloned().collect();
        names.sort();
        for n in &names {
            if n.contains("dw") || n.contains("conv") {
                let t = &f.tensors[n];
                println!("{n}: ne={:?} type={}", t.dims, t.ggml_type);
            }
        }
        // и первый stage-тензор каждого вида, чтобы понять нейминг
        for n in names.iter().take(40) {
            println!("  have: {n}");
        }
    }
    Ok(())
}
