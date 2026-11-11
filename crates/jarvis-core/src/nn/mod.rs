//! Небольшие нейросети на candle, портированные из ONNX-моделей ударителя.

pub mod distilbert;
pub mod roformer;

use anyhow::{Result, bail, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Путь к uv: сначала PATH, затем стандартное место установщика
/// (~/.local/bin — у GUI-приложений PATH часто без него). None — uv нет.
fn find_uv() -> Option<PathBuf> {
    if let Ok(output) = Command::new("uv").arg("--version").output()
        && output.status.success()
    {
        return Some(PathBuf::from("uv"));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let local = home.join(".local/bin/uv");
    if local.is_file() {
        return Some(local);
    }
    None
}

/// Ставит uv при первом запуске, если его нет в системе — официальный
/// установщик в ~/.local/bin, один раз, ~20 МБ. Нужен для скриптов
/// подготовки моделей: они объявляют зависимости в заголовке, и uv сам
/// их ставит в изолированное окружение (python-систему не трогаем).
fn ensure_uv() -> Result<PathBuf> {
    if let Some(uv) = find_uv() {
        return Ok(uv);
    }
    println!("🔧 Устанавливаю uv (менеджер Python-окружений, ~/.local/bin)…");
    let status = Command::new("sh")
        .arg("-c")
        .arg("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .status()
        .map_err(|e| anyhow!("не удалось запустить установщик uv: {e}"))?;
    if !status.success() {
        bail!("Установщик uv завершился с кодом {status}");
    }
    find_uv().ok_or_else(|| anyhow!("uv установился, но не найден в ~/.local/bin — перезапусти приложение (PATH обновится)"))
}

/// Проверяет, что safetensors на месте. Если его нет — собирает веса из
/// ONNX скриптом `scripts/export_stress_models.py` через uv: зависимости
/// из заголовка скрипта ставятся сами (torch — CPU-сборка), uv при
/// отсутствии ставится на месте. В готовой сборке файлы уже лежат рядом,
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
            "Нет {path:?} и нет {:?} — соберите веса: uv run scripts/export_stress_models.py",
            script
        );
    }
    let uv = ensure_uv()?;
    println!(
        "🔧 Собираю веса ударителя из ONNX ({})...",
        script.display()
    );
    // Скрипт пишет в ту же папку моделей, куда смотрит приложение.
    let status = Command::new(&uv)
        .arg("run")
        .arg("--quiet")
        .arg(&script)
        .env("JARVIS_MODELS_DIR", crate::paths::models_dir())
        .current_dir(&root)
        .status();
    match status {
        Ok(s) if s.success() && path.exists() => Ok(()),
        Ok(s) => bail!("Скрипт экспорта весов завершился с кодом {s}; ожидался {path:?}"),
        Err(e) => bail!(
            "Не удалось запустить uv ({e}); соберите веса вручную: \
             uv run scripts/export_stress_models.py"
        ),
    }
}