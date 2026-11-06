//! Native desktop screenshot capture for the UI-Venus perception loop.
//! No Python, browser automation server, or C/C++ runtime is used.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct CapturedScreenshot {
    pub path: PathBuf,
    /// Размер снимка в физических пикселях (на retina — 2× логического).
    pub width: u32,
    pub height: u32,
    /// Рабочий стол в логических точках (координатная система мыши);
    /// 0 — определить не удалось (тогда масштаб 1:1).
    pub logical_width: u32,
    pub logical_height: u32,
}

/// Логические границы рабочего стола macOS (точки, не пиксели).
/// Кэшируется: разрешение не меняется между шагами задачи.
fn logical_desktop_size() -> Option<(u32, u32)> {
    static CACHE: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            let output = Command::new("osascript")
                .args(["-e", "tell application \"Finder\" to get bounds of window of desktop"])
                .output()
                .ok()?;
            let text = String::from_utf8_lossy(&output.stdout);
            // Формат: "0, 0, 1512, 982" → (left, top, right, bottom).
            let nums: Vec<u32> = text
                .split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect();
            if nums.len() == 4 && nums[2] > nums[0] && nums[3] > nums[1] {
                Some((nums[2] - nums[0], nums[3] - nums[1]))
            } else {
                None
            }
        }
        #[cfg(not(target_os = "macos"))]
        None
    })
}

/// Captures the primary desktop to a PNG and validates that the file is readable.
pub fn capture_desktop() -> Result<CapturedScreenshot> {
    let dir = super::repository::model_dir().join("captures");
    std::fs::create_dir_all(&dir).with_context(|| format!("Не удалось создать {}", dir.display()))?;
    let id = NEXT_CAPTURE.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("screen-{}-{id}.png", std::process::id()));
    capture_to(&path)?;
    let decoded = image::ImageReader::open(&path)
        .with_context(|| format!("Не удалось открыть скриншот {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .context("Скриншот создан, но PNG не удалось декодировать")?;
    let (width, height) = (decoded.width(), decoded.height());
    if width == 0 || height == 0 {
        bail!("Операционная система вернула скриншот нулевого размера");
    }
    let (logical_width, logical_height) = logical_desktop_size().unwrap_or((0, 0));
    Ok(CapturedScreenshot { path, width, height, logical_width, logical_height })
}

fn capture_to(path: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("/usr/sbin/screencapture")
            .args(["-x", "-t", "png"])
            .arg(path)
            .output()
            .context("Не удалось запустить macOS screencapture")?;
        return check_output("screencapture", output);
    }
    #[cfg(target_os = "linux")]
    {
        let candidates: &[(&str, &[&str])] = &[
            ("grim", &[]),
            ("scrot", &[]),
            ("gnome-screenshot", &["-f"]),
        ];
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        let ordered: Vec<(&str, &[&str])> = if wayland {
            vec![candidates[0], candidates[1], candidates[2]]
        } else {
            vec![candidates[1], candidates[2], candidates[0]]
        };
        for (bin, prefix) in ordered {
            if !binary_exists(bin) { continue; }
            let mut cmd = Command::new(bin);
            cmd.args(prefix).arg(path);
            let output = cmd.output().with_context(|| format!("Не удалось запустить {bin}"))?;
            return check_output(bin, output);
        }
        bail!("Не найден инструмент снимка экрана. Установи grim (Wayland), scrot или gnome-screenshot (X11)")
    }
    #[cfg(target_os = "windows")]
    {
        let encoded = path.to_string_lossy().as_bytes().iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let script = format!(r#"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms
$hex='{encoded}';$bytes=New-Object byte[] ($hex.Length/2)
for($i=0;$i -lt $bytes.Length;$i++){{$bytes[$i]=[Convert]::ToByte($hex.Substring($i*2,2),16)}}
$p=[Text.Encoding]::UTF8.GetString($bytes)
$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp=New-Object System.Drawing.Bitmap($b.Width,$b.Height)
$g=[System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size)
$bmp.Save($p,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
"OK"
"#);
        let output = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &script])
            .output()
            .context("Не удалось запустить PowerShell для захвата экрана")?;
        return check_output("PowerShell screenshot", output);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    bail!("Захват экрана не поддерживается на этой платформе")
}

fn check_output(bin: &str, output: std::process::Output) -> Result<()> {
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!("{bin}: {}", if stderr.trim().is_empty() { stdout.trim() } else { stderr.trim() })
    }
}

#[cfg(target_os = "linux")]
fn binary_exists(bin: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|p| p.join(bin).is_file())
    })
}
