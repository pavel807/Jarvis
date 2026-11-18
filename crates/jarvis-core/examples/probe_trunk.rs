// Смотрим logmag-спектры головы: тишина vs речь.
use jarvis_core::tts_zonos::{Sampling, ZacDecoder};

fn main() -> anyhow::Result<()> {
    jarvis_core::init_tuning();
    let dir = jarvis_core::model_fetch::zonos3_dir();
    let gguf = jarvis_core::tts_zonos::GgufFile::open(&dir.join("backbone-ptq.gguf"))?;
    let mut engine = jarvis_core::tts_zonos::ZonosEngine::load(&gguf, candle_core::Device::Cpu)?;
    let sp = Sampling { greedy: true, ..Sampling::default() };
    let frames = engine.synthesize_codes("Привет! Как дела?", None, [10, 10, 30], &sp)?;
    let silence = [1918u32, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337];
    let sil: Vec<Vec<u32>> = (0..frames.len()).map(|_| silence.to_vec()).collect();
    let dec = ZacDecoder::open(&dir.join("zac-keep2.gguf"), &candle_core::Device::Cpu)?;

    for (name, fr) in [("тишина", &sil), ("речь", &frames)] {
        let specs = dec.debug_spectra(fr)?;
        for (bi, s) in specs.iter().enumerate() {
            let n = dec.cfg.n_fft[bi];
            let bins = n / 2 + 1;
            let per = n + 2;
            // Статистика по кадрам: средний logmag по бинам (низ/высок).
            let mut lo = 0f32; let mut hi = 0f32; let mut ph = 0f32;
            let nf = fr.len();
            for t in 0..nf {
                let f = &s[t * per..(t + 1) * per];
                for k in 0..bins / 2 { lo += f[k]; }
                for k in bins / 2..bins { hi += f[k]; }
                for k in 0..bins { ph += f[bins + k].abs(); }
            }
            let nrm = (nf * bins) as f32;
            println!("{name} ветка {bi}: logmag низ {:#.3} высок {:#.3}, |phase| ср {:#.3}", lo / nrm, hi / nrm, ph / nrm);
        }
    }
    Ok(())
}
