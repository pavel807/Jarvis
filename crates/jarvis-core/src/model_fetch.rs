//! Загрузка моделей, у которых нет внешнего источника (HF/Mirai).
//!
//! Qwen3.5-0.8B (бэкенд candle) и FastConformer (распознавание) лежат
//! в отдельном релизе GitHub — по умолчанию `pavel807/jarvis-models`, тег
//! `models-v1`. Базовый URL можно переопределить переменной окружения
//! `JARVIS_MODELS_BASE_URL`.
//!
//! Архивы — обычные `tar.gz`, распаковываются в каталог `models/`, поэтому
//! внутри должны лежать пути относительно него (`qwen3.5-0.8b/…`).
//! ZONOS3 качается напрямую из Hugging Face (см. `zonos3.rs`/`model_fetch`
//! ниже по файлу).

use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Запрос пользователя отменить текущую загрузку моделей.
///
/// Флаг проверяется потоковым загрузчиком по мере роста файла — отмена
/// срабатывает внутри одного файла (архивы/веса по 460+ МБ), а не только
/// между файлами. Сбрасывается при старте новой загрузки.
pub static DOWNLOAD_CANCEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Отменить текущую загрузку моделей: частичные файлы удаляются.
pub fn cancel_downloads() {
    DOWNLOAD_CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Идёт ли отмена (для вызывающих потоков — чтобы отличить отмену от ошибки).
pub fn downloads_cancelled() -> bool {
    DOWNLOAD_CANCEL.load(std::sync::atomic::Ordering::SeqCst)
}

fn ensure_not_cancelled() -> Result<()> {
    if downloads_cancelled() {
        bail!("Загрузка отменена");
    }
    Ok(())
}

const DEFAULT_BASE_URL: &str =
    "https://github.com/pavel807/jarvis-models/releases/download/models-v1";

/// Базовый URL релиза с моделями (без имени ассета).
pub fn base_url() -> String {
    std::env::var("JARVIS_MODELS_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
}

/// Докачать файл вакворда из релиза (jarvis_model_2.rpw, jarvis_model.jref):
/// маленький файл, прогресс не важен. Ошибка — честная (ассет может быть
/// ещё не выложен), вызывающий решает сам.
pub fn ensure_wakeword(name: &str) -> Result<()> {
    let dest = crate::paths::wakeword_model_file(name);
    if dest.exists() {
        return Ok(());
    }
    let dir = dest.parent().context("нет родителя у пути модели")?;
    std::fs::create_dir_all(dir).with_context(|| format!("не создать {}", dir.display()))?;
    let url = format!("{}/{}", base_url(), name);
    // Ассет может быть не выложен (опциональная модель) — это не ошибка:
    // движок умеет жить без него (метрический эталон или обучение в приложении).
    let expected = remote_size(&url);
    if expected == 0 {
        println!("ℹ️ В релизе нет «{name}» — пропускаю докачку");
        return Ok(());
    }
    println!("📥 Загрузка модели вакворда: {url}");
    download(&url, &dest, expected, &mut |_| {})
        .with_context(|| format!("не удалось скачать {name}"))
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

/// Гарантирует файлы FastConformer-Hybrid Large RU в `models/fastconformer/`.
///
/// Источник один: архив `fastconformer.tar.gz` (~460 МБ) из GitHub-релиза —
/// модель распознавания речи встроена в программу, настройка не нужна.
pub fn ensure_fastconformer() -> Result<()> {
    let mut noop = |_pct: u8| {};
    ensure_fastconformer_with_progress(&mut noop)
}

/// Как [`ensure_fastconformer`], но с отчётом о прогрессе (0..=100).
pub fn ensure_fastconformer_with_progress(progress: &mut dyn FnMut(u8)) -> Result<()> {
    let dir = crate::paths::models_dir().join("fastconformer");
    if fastconformer_ready(&dir) {
        progress(100);
        return Ok(());
    }
    ensure_archive("fastconformer.tar.gz", progress)?;
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

/// Удаляет каталог модели по имени — те, что приложение само скачивает.
/// Пользовательские каталоги LLM (положенные в models/ руками) не трогаем.
pub fn model_dir_remove(name: &str) -> Result<()> {
    let dir = match name {
        "qwen3.5-0.8b" | "fastconformer" => crate::paths::models_dir().join(name),
        // ZONOS3 лежит в отдельном каталоге zonos3.
        "zonos" => zonos3_dir(),
        other => bail!(
            "Модель «{other}» не удаляется из приложения: каталог пользовательский"
        ),
    };
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("Не удалось удалить {}", dir.display()))?;
    }
    Ok(())
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

pub fn qwen35_ready(dir: &Path) -> bool {
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
    DOWNLOAD_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
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

/// Агент загрузок: таймауты на соединение, но без глобального — файлы по
/// 460 МБ на медленном канале живут дольше любого разумного лимита.
fn agent_download() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        ureq::Agent::config_builder()
            .timeout_resolve(Some(Duration::from_secs(3)))
            .timeout_connect(Some(Duration::from_secs(20)))
            .max_idle_connections_per_host(2)
            .build()
            .new_agent()
    })
}

/// Размер ассета в релизе (Content-Length) и признак его существования.
/// 0 — ассета нет (не выложен) или размер неизвестен: опциональный файл
/// пропускается без ошибки.
fn remote_size(url: &str) -> u64 {
    match agent_download().head(url).call() {
        Ok(response) => {
            if response.status().as_u16() != 200 {
                return 0;
            }
            content_length(&response).unwrap_or(0)
        }
        Err(_) => 0,
    }
}

/// Последний Content-Length из заголовков ответа (после редиректов — итог).
fn content_length(response: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    let mut length = None;
    for (name, value) in response.headers() {
        if name.as_str().eq_ignore_ascii_case("content-length")
            && let Ok(text) = value.to_str()
            && let Ok(parsed) = text.trim().parse::<u64>()
        {
            length = Some(parsed);
        }
    }
    length
}

/// Каталог с весами ZONOS3-Mini (GGUF с HuggingFace).
pub fn zonos3_dir() -> PathBuf {
    crate::paths::models_dir().join("zonos3")
}

/// Имена файлов ZONOS3, которые нужны движку. Декодер — компактный
/// `zac-keep2` (дефолт web/ggml-серверов: 2 слоя на стадию), полный
/// `zac.gguf` не нужен: тот же звук, но заметно тяжелее.
pub const ZONOS3_FILES: &[(&str, &str)] = &[
    ("backbone-ptq.gguf", "gguf/backbone-ptq.gguf"),
    ("zac-keep2.gguf", "gguf/zac-keep2.gguf"),
    ("spk-encoder.gguf", "gguf/spk-encoder.gguf"),
];

/// Гарантирует, что веса ZONOS3 на месте, иначе качает их с HuggingFace.
///
/// Файлы крупные (бэкбон ~660 МБ, компактный декодер и спикер-энкодер —
/// сотни МБ и ~45 МБ), поэтому скачивание идёт с отчётом о прогрессе и
/// только по отсутствующим.
pub fn ensure_zonos3_with_progress(progress: &mut dyn FnMut(u8)) -> Result<()> {
    let dir = zonos3_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("не создать {}", dir.display()))?;
    DOWNLOAD_CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
    let total = ZONOS3_FILES.len();
    for (i, (local, remote)) in ZONOS3_FILES.iter().enumerate() {
        ensure_not_cancelled()?;
        let dest = dir.join(local);
        if !dest.exists() {
            println!("📥 Загрузка ZONOS3: {local}...");
            // Сквозной прогресс: база за скачанные файлы + доля текущего файла.
            let base = i * 100 / total;
            let share = 100 / total;
            let mut file_progress = |pct: u8| {
                progress((base + (pct.min(99) as usize) * share / 100) as u8);
            };
            hf_file("Zyphra", "ZONOS3-Mini", remote, &dest, &mut file_progress)
                .with_context(|| format!("не удалось скачать {remote}"))?;
        }
        progress(((i + 1) * 100 / total) as u8);
    }
    Ok(())
}

/// Гарантирует, что веса ZONOS3 на месте.
pub fn ensure_zonos3() -> Result<()> {
    ensure_zonos3_with_progress(&mut |_| {})
}

/// Готовы ли веса ZONOS3.
pub fn zonos3_ready() -> bool {
    let dir = zonos3_dir();
    ZONOS3_FILES.iter().all(|(local, _)| dir.join(local).exists())
}

/// Файл `filename` из репозитория HuggingFace `owner/name` в `dest`.
///
/// Прямые resolve-URL через собственный потоковый загрузчик: он даёт
/// отмену по флагу [`DOWNLOAD_CANCEL`] и плавный прогресс внутри одного
/// файла (веса ZONOS3 — до 663 МБ), чего у блокирующего hf-hub нет.
pub(crate) fn hf_file(owner: &str, name: &str, filename: &str, dest: &Path, progress: &mut dyn FnMut(u8)) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    println!("📥 Загрузка {filename}...");
    let url = format!("https://huggingface.co/{owner}/{name}/resolve/main/{filename}");
    download(&url, dest, 0, progress).with_context(|| format!("Не удалось скачать {filename}"))
}

/// Качает `url` в `dest`, дёргая `progress(pct)` по мере роста файла.
/// HTTP-загрузка в процессе: ureq проходит редиректы GitHub/HF, размер берём
/// из Content-Length ответа. Внешние curl/tar не нужны ни на одной ОС —
/// на iOS их в сэндбоксе просто нет.
///
/// Пишет во временный `dest.part` и переименовывает по завершении: отмена
/// или обрыв не оставляют поломанный файл под настоящим именем.
fn download(url: &str, dest: &Path, _expected: u64, progress: &mut dyn FnMut(u8)) -> Result<()> {
    let response = agent_download()
        .get(url)
        .call()
        .with_context(|| format!("запрос {url} не удался"))?;
    if response.status().as_u16() != 200 {
        bail!("{} вернул HTTP {}", url, response.status());
    }
    let expected = content_length(&response).unwrap_or(0);

    let tmp = dest.with_extension(format!(
        "part{}",
        std::process::id()
    ));
    let mut reader = response.into_body().into_reader();
    let mut file = std::fs::File::create(&tmp)
        .with_context(|| format!("не создать файл {}", tmp.display()))?;

    let cancel_msg = "Загрузка отменена";
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    let mut last = 0u8;
    loop {
        ensure_not_cancelled()?;
        let n = reader
            .read(&mut buf)
            .map_err(|e| {
                let _ = std::fs::remove_file(&tmp);
                if downloads_cancelled() {
                    anyhow::Error::msg(cancel_msg)
                } else {
                    anyhow::Error::new(e).context("обрыв при скачивании")
                }
            })?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|e| anyhow::Error::new(e).context("запись файла модели"))?;
        total += n as u64;
        if expected > 0 {
            let pct = ((total.min(expected) as f64 / expected as f64) * 100.0) as u8;
            if pct != last {
                last = pct;
                progress(pct.min(99));
            }
        }
    }
    std::io::Write::flush(&mut file).ok();
    drop(file);
    if downloads_cancelled() {
        let _ = std::fs::remove_file(&tmp);
        bail!("{cancel_msg}");
    }
    if expected > 0 && total < expected {
        let _ = std::fs::remove_file(&tmp);
        bail!("скачано {total} из {expected} байт — файл оборвался");
    }
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("не переименовать {}", tmp.display()))?;
    progress(100);
    Ok(())
}

/// Распаковка tar.gz в процессе: flate2 разворачивает поток, tar раскладывает
/// файлы. Внешние бинарники не нужны — в iOS-сэндбоксе их и не бывает.
fn extract(archive: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("не открыть {}", archive.display()))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut ar = tar::Archive::new(gz);
    ar.set_preserve_permissions(true);
    ar.unpack(dest)
        .with_context(|| format!("распаковка в {} не удалась", dest.display()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_packs_tar_gz_back() {
        // Круг: каталог → tar.gz (системного tar нет, пакуем тем же tar-крейтом)
        // → extract() → содержимое на месте.
        let dir = tempfile::tempdir().expect("временный каталог");
        std::fs::write(dir.path().join("qwen3.5-0.8b_marker.txt"), "внутри").unwrap();
        let gz_path = dir.path().join("test.tar.gz");
        let gz = flate2::write::GzEncoder::new(
            std::fs::File::create(&gz_path).unwrap(),
            flate2::Compression::default(),
        );
        {
            let mut builder = tar::Builder::new(gz);
            builder
                .append_dir_all("models", dir.path())
                .expect("архив собирается");
            builder.into_inner().expect("gz закрывается");
        }
        let out = dir.path().join("out");
        std::fs::create_dir(&out).unwrap();
        extract(&gz_path, &out).expect("распаковка успешна");
        let marker = out.join("models/qwen3.5-0.8b_marker.txt");
        assert_eq!(
            std::fs::read_to_string(marker).unwrap(),
            "внутри",
            "файл после круга распаковки на месте"
        );
    }
}
