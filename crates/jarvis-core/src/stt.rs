//! STT: GigaAM v3 (ONNX, int8) — загрузка модели и распознавание.

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use std::fs;
use std::path::{Path, PathBuf};
use transcribe_rs::onnx::gigaam::GigaAMModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{SpeechModel, TranscribeOptions};

const QUANTIZATION: Quantization = Quantization::Int8;

pub fn init_env() {
    // Ограничиваем потоки ORT, чтобы не душить систему (только до создания сессий).
    unsafe {
        std::env::set_var("ORT_INTRA_OP_NUM_THREADS", "4");
        std::env::set_var("ORT_INTER_OP_NUM_THREADS", "2");
    }
}

fn ensure_model_downloaded(model_dir: &Path, quantization: &Quantization) -> Result<()> {
    let vocab_file = model_dir.join("vocab.txt");

    let model_file = match quantization {
        Quantization::Int8 => model_dir.join("model.int8.onnx"),
        _ => model_dir.join("model.onnx"),
    };

    if model_file.exists() && vocab_file.exists() {
        return Ok(());
    }

    println!("📥 Загрузка GigaAM v3 в {:?}...", model_dir);
    fs::create_dir_all(model_dir)?;

    let client = HFClientSync::new()?;
    let repo = client.model("istupakov".to_string(), "gigaam-v3-onnx".to_string());

    if !vocab_file.exists() {
        let downloaded_vocab = repo.download_file().filename("v3_vocab.txt").send()?;
        fs::copy(&downloaded_vocab, &vocab_file)?;
    }

    if !model_file.exists() {
        let filename = match quantization {
            Quantization::Int8 => "v3_ctc.int8.onnx",
            _ => "v3_ctc.onnx",
        };
        let downloaded_model = repo.download_file().filename(filename).send()?;
        fs::copy(&downloaded_model, &model_file)?;
    }

    println!("✅ Модель GigaAM v3 успешно загружена!");
    Ok(())
}

pub struct Stt {
    model: GigaAMModel,
}

impl Stt {
    /// Скачивает модель при необходимости и загружает её в память.
    pub fn load(models_dir: &PathBuf) -> Result<Self> {
        let dir = models_dir.join("gigaam-v3");
        ensure_model_downloaded(&dir, &QUANTIZATION)?;

        let start = std::time::Instant::now();
        let model = GigaAMModel::load(&dir, &QUANTIZATION)
            .context("Не удалось загрузить GigaAM v3")?;
        println!("✅ STT-модель загружена за {:.2?}", start.elapsed());

        Ok(Self { model })
    }

    /// Распознавание готового буфера 16 кГц mono f32 (без доп. паддинга).
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let options = TranscribeOptions::default();
        let res = self.model.transcribe(samples, &options)?;
        Ok(res.text.trim().to_string())
    }
}
