//! Проверка окна Ханна и ядра ресемплера из GGUF против их математических формул.
use jarvis_core::tts_zonos::GgufFile;

fn main() -> anyhow::Result<()> {
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let f = GgufFile::open(&dir.join("zac-keep2.gguf"))?;
    let dev = candle_core::Device::Cpu;
    for (i, n) in [736usize, 2208].iter().enumerate() {
        let w = f.tensor_f32(&format!("head.window.{i}"), &dev)?.to_vec1::<f32>()?;
        println!("window.{i}: len={}", w.len());
        // Периодическое окно Ханна: w[j] = 0.5·(1 − cos(2πj/n)).
        let mut max_err = 0f32;
        for (j, &v) in w.iter().enumerate() {
            let want = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / *n as f32).cos());
            max_err = max_err.max((v - want).abs());
        }
        println!("  max|w − hann| = {max_err:.6}  (первые: {:?} … {:?})", &w[..4], &w[w.len()-3..]);
    }
    // Ядро ресемплера: 735 фаз × 770 отводов. Сумма по фазе ≈ 736/735 (коэффициент усиления).
    let k = f.tensor_f32("resample.kernel", &dev)?.to_vec1::<f32>()?;
    println!("kernel len={}", k.len());
    for phase in [0usize, 100, 367, 734] {
        let row = &k[phase * 770..(phase + 1) * 770];
        let sum: f32 = row.iter().sum();
        let (mx, mn) = row.iter().fold((f32::MIN, f32::MAX), |(a, b), &v| (a.max(v), b.min(v)));
        println!("  фаза {phase}: сумма={sum:.6} max={mx:.4} min={mn:.4}");
    }
    Ok(())
}
