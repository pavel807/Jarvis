//! Probe: сколько BF16-памяти реально держит Metal на этой машине.
use candle_core::{DType, Device, Tensor};
fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    let mut bufs = Vec::new();
    let step_mb = 512usize;
    for i in 0..64 {
        let elems = step_mb * 1_000_000 / 2;
        match Tensor::zeros((elems,), DType::BF16, &dev) {
            Ok(t) => bufs.push(t),
            Err(e) => {
                println!("FAIL после {} МБ: {e}", i * step_mb);
                return Ok(());
            }
        }
        print!("{} МБ ок; ", (i + 1) * step_mb);
        use std::io::Write;
        std::io::stdout().flush()?;
        // периодический sync, чтобы аллокации реально коммитились
        if (i + 1) % 4 == 0 { dev.synchronize()?; }
    }
    println!("\n32 ГБ выделились без ошибки — лимит выше");
    Ok(())
}
