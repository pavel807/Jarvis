//! Замер скорости matmul в candle на этой машине (диагностика BLAS).
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() {
    let dev = Device::Cpu;
    let a = Tensor::randn(0f32, 1f32, (512, 2048), &dev).unwrap();
    let b = Tensor::randn(0f32, 1f32, (2048, 512), &dev).unwrap();
    let _ = a.matmul(&b).unwrap(); // прогрев
    let started = Instant::now();
    let n = 20;
    for _ in 0..n {
        let _ = a.matmul(&b).unwrap();
    }
    let dt = started.elapsed().as_secs_f64() / n as f64;
    let flops = 2.0 * 512.0 * 2048.0 * 512.0;
    println!("sgemm 512x2048x512: {:.1} мс → {:.1} GFLOP/s", dt * 1e3, flops / dt / 1e9);

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("ядер: {threads}");
}
