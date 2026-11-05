//! Загрузка моделей, у которых нет внешнего источника (HF/Mirai).
//!
//! silero (TTS, сконвертирован вручную) и Qwen3.5-0.8B (бэкенд candle) лежат
//! в отдельном релизе GitHub — по умолчанию `pavel807/jarvis-models`, тег
//! `models-v1`. Базовый URL можно переопределить переменной окружения
//! `JARVIS_MODELS_BASE_URL`.
//!
//! Архивы — обычные `tar.gz`, распаковываются в каталог `models/`, поэтому
//! внутри должны лежать пути относительно него (`silero/…`, `qwen3.5-0.8b/…`).

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::time::{Duration, Instant};

const DEFAULT_BASE_URL: &str =
    "https://github.com/pavel807/jarvis-models/releases/download/models-v1";

/// Как часто сообщать о прогрессе при скачивании, мс.
const PROGRESS_STEP: Duration = Duration::from_millis(500);

/// Базовый URL релиза с моделями (без имени ассета).
pub fn base_url() -> String {
    std::env::var("JARVIS_MODELS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
}

/// Гарантирует, что файлы Silero v5 есть в `models/silero/`, иначе качает
/// `silero.tar.gz` из релиза.
pub fn ensure_silero() -> Result<()> {
    ensure_silero_with_progress(&mut |_| {})
}

/// Как [`ensure_silero`], но с отчётом о прогрессе (0..=100).
pub fn ensure_silero_with_progress(progress: &mut dyn FnMut(u8)) -> Result<()> {
    let dir = crate::paths::models_dir().join("silero");
    if silero_ready(&dir) {
        progress(100);
        return Ok(());
    }
    ensure_archive("silero.tar.gz", progress)?;
    if silero_ready(&dir) {
        Ok(())
    } else {
        bail!(
            "После загрузки silero в {:?} нет v5_cis_base.safetensors/alphabet.json",
            dir
        )
    }
}

/// Гарантирует наличие файлов Qwen3.5-0.8B в `dir` (config, tokenizer, веса),
/// иначе качает `qwen3.5-0.8b.tar.gz` из релиза.
pub fn ensure_qwen35(dir: &Path) -> Result<()> {
    let mut noop = |_pct: u8| {};
    ensure_qwen35_with_progress(dir, &mut noop)
}

/// То же, что [`ensure_qwen35`], но с отчётом о прогрессе (0..100).
pub fn ensure_qwen35_with_progress(dir: &Path, progress: &mut dyn FnMut(u8)) -> Result<()> {
    if qwen35_ready(dir) {
        progress(100);
        return Ok(());
    }
    ensure_archive("qwen3.5-0.8b.tar.gz", progress)?;
    if qwen35_ready(dir) {
        Ok(())
    } else {
        bail!(
            "После загрузки Qwen3.5 в {:?} не хватает файлов модели",
            dir
        )
    }
}

/// Гарантирует файлы FastConformer-Hybrid Large RU в `models/fastconformer/`,
/// иначе качает `fastconformer.tar.gz` (~460 МБ: веса, конвертированные из
/// `.nemo` чекпоинта NVIDIA, config, токенизатор) из релиза.
pub fn ensure_fastconformer() -> Result<()> {
    let dir = crate::paths::models_dir().join("fastconformer");
    if fastconformer_ready(&dir) {
        return Ok(());
    }
    ensure_archive("fastconformer.tar.gz", &mut |_| {})?;
    if fastconformer_ready(&dir) {
        Ok(())
    } else {
        bail!(
            "После загрузки FastConformer в {:?} нет model.safetensors/config.json/tokenizer.vocab",
            dir
        )
    }
}

/// Размер файлов FastConformer, если модель на месте; иначе `None`.
pub fn fastconformer_status() -> Option<u64> {
    let dir = crate::paths::models_dir().join("fastconformer");
    fastconformer_ready(&dir).then(|| dir_size(&dir))
}

fn fastconformer_ready(dir: &Path) -> bool {
    dir.join("model.safetensors").is_file()
        && dir.join("config.json").is_file()
        && dir.join("tokenizer.vocab").is_file()
}

/// Размер файлов Qwen3.5 в `dir`, если модель на месте; иначе `None`.
pub fn qwen35_status(dir: &Path) -> Option<u64> {
    qwen35_ready(dir).then(|| dir_size(dir))
}

/// Удаляет каталог модели (для повторной загрузки).
pub fn qwen35_remove(dir: &Path) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)
            .with_context(|| format!("Не удалось удалить {}", dir.display()))?;
    }
    Ok(())
}

fn silero_ready(dir: &Path) -> bool {
    dir.join("v5_cis_base.safetensors").is_file() && dir.join("alphabet.json").is_file()
}

fn qwen35_ready(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() || !dir.join("tokenizer.json").is_file() {
        return false;
    }
    has_safetensors(dir)
}

fn has_safetensors(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|e| {
        let name = e.file_name();
        let name = name.to_string_lossy();
        name.starts_with("model.safetensors") && name.ends_with(".safetensors")
    })
}

/// Скачивает `asset` из релиза и распаковывает его в `models/`.
fn ensure_archive(asset: &str, progress: &mut dyn FnMut(u8)) -> Result<()> {
    let models = crate::paths::models_dir();
    std::fs::create_dir_all(&models)
        .with_context(|| format!("Не удалось создать {}", models.display()))?;

    let url = format!("{}/{}", base_url(), asset);
    let tmp = std::env::temp_dir().join(format!("jarvis-{asset}"));
    let expected = remote_size(&url);

    println!("📥 Загрузка модели: {url}");
    let started = Instant::now();
    download(&url, &tmp, expected, progress)?;
    extract(&tmp, &models)?;
    let _ = std::fs::remove_file(&tmp);
    println!("✅ Модель получена за {:.1?}", started.elapsed());
    Ok(())
}

/// Качает `url` в `dest`, дёргая `progress(pct)` по мере роста файла.
/// `expected` — ожидаемый размер в байтах (0, если неизвестен).
fn download(url: &str, dest: &Path, expected: u64, progress: &mut dyn FnMut(u8)) -> Result<()> {
    let mut child = std::process::Command::new(curl_bin())
        .args(["-fL", "--retry", "3", "--connect-timeout", "20", "-o"])
        .arg(dest)
        .arg(url)
        .spawn()
        .context("Не удалось запустить curl")?;

    let mut last = 0u8;
    loop {
        match child.try_wait().context("curl: ошибка ожидания")? {
            Some(status) => {
                if !status.success() {
                    bail!("curl завершился с кодом {status}");
                }
                break;
            }
            None => {
                if expected > 0 {
                    let cur = std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0);
                    let pct = ((cur.min(expected) as f64 / expected as f64) * 100.0) as u8;
                    if pct != last {
                        last = pct;
                        progress(pct.min(99));
                    }
                }
                std::thread::sleep(PROGRESS_STEP);
            }
        }
    }
    progress(100);
    Ok(())
}

/// Размер ассета в релизе через `Content-Length` (0, если не удалось узнать).
fn remote_size(url: &str) -> u64 {
    let out = std::process::Command::new(curl_bin())
        .args(["-sIL", "--connect-timeout", "20"])
        .arg(url)
        .output();
    let Ok(out) = out else {
        return 0;
    };
    parse_content_length(&String::from_utf8_lossy(&out.stdout))
}

/// Последний `Content-Length` из заголовков (после редиректов — итоговый).
fn parse_content_length(headers: &str) -> u64 {
    headers
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("content-length") {
                value.trim().parse::<u64>().ok()
            } else {
                None
            }
        })
        .last()
        .unwrap_or(0)
}

fn extract(archive: &Path, dest: &Path) -> Result<()> {
    let status = std::process::Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .status()
        .context("Не удалось запустить tar")?;
    if !status.success() {
        bail!("tar завершился с кодом {status}");
    }
    Ok(())
}

/// Суммарный размер файлов в каталоге (рекурсивно).
fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

/// Первый существующий curl, иначе полагаемся на PATH.
fn curl_bin() -> &'static str {
    for p in [
        "/usr/bin/curl",
        "/opt/homebrew/bin/curl",
        "/usr/local/bin/curl",
    ] {
        if Path::new(p).exists() {
            return p;
        }
    }
    "curl"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_length_takes_last_header() {
        let headers = "HTTP/2 302\r\ncontent-length: 10\r\nHTTP/2 200\r\nContent-Length: 42\r\n";
        assert_eq!(parse_content_length(headers), 42);
    }

    #[test]
    fn content_length_missing_is_zero() {
        assert_eq!(parse_content_length("HTTP/2 200\r\n"), 0);
    }
}