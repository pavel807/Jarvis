//! Небольшие нейросети на candle, портированные из ONNX-моделей ударителя.

pub mod distilbert;
pub mod roformer;

use anyhow::{Result, bail};
use std::path::Path;
use std::process::Command;

/// Проверяет, что safetensors на месте. Если его нет — пытается собрать
/// веса из ONNX скриптом `scripts/export_stress_models.py` (нужен python3
/// с onnx/numpy/torch/safetensors). В готовой сборке файлы уже лежат рядом,
/// так что шаг нужен только при первом запуске из исходников.
pub fn ensure_safetensors(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    let script = root.join("scripts/export_stress_models.py");
    if !script.exists() {
        bail!(
            "Нет {path:?} и нет {:?} — соберите веса: python3 scripts/export_stress_models.py",
            script
        );
    }
    println!(
        "🔧 Собираю веса ударителя из ONNX ({})...",
        script.display()
    );
    let status = Command::new("python3")
        .arg(&script)
        .current_dir(&root)
        .status();
    match status {
        Ok(s) if s.success() && path.exists() => Ok(()),
        Ok(s) => bail!("Скрипт экспорта весов завершился с кодом {s}; ожидался {path:?}"),
        Err(e) => bail!(
            "Не удалось запустить python3 для экспорта весов ({e}); \
             соберите вручную: python3 scripts/export_stress_models.py"
        ),
    }
}