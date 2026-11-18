// Спектральный портрет WAV: средняя энергия по полосам (512-точечное FFT).
use rustfft::{num_complex::Complex, FftPlanner};

fn bands(path: &str) -> anyhow::Result<()> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f32> = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / (1 << (spec.bits_per_sample - 1)) as f32)
        .collect();
    let n = 512;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let frames = raw.len() / n;
    let mut acc = vec![0f32; n / 2];
    for f in 0..frames {
        let mut buf: Vec<Complex<f32>> = raw[f * n..(f + 1) * n]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();
        fft.process(&mut buf);
        for (i, b) in buf.iter().take(n / 2).enumerate() {
            acc[i] += b.norm_sqr();
        }
    }
    // Полосы: 0-250, 250-500, 500-1000, 1000-2000, 2000-4000, 4000-8000, 8000+ Гц.
    let edges = [250usize, 500, 1000, 2000, 4000, 8000];
    let per_bin = spec.sample_rate as f32 / n as f32;
    let mut out = Vec::new();
    let mut lo = 0usize;
    let mut total = 0f32;
    for &hi in edges.iter().chain(std::iter::once(&22050usize)) {
        let a: f32 = acc[lo..(hi as f32 / per_bin) as usize].iter().sum();
        out.push(a);
        total += a;
        lo = (hi as f32 / per_bin) as usize;
    }
    // Доли энергии по полосам в процентах.
    let pct: Vec<String> = out.iter().map(|v| format!("{:.0}%", 100.0 * v / total)).collect();
    println!("{path}: [0-250 250-500 0.5-1к 1-2к 2-4к 4-8к 8к+] = {}", pct.join(" "));
    Ok(())
}

fn main() -> anyhow::Result<()> {
    for p in std::env::args().skip(1) {
        bands(&p)?;
    }
    Ok(())
}
