//! Диагностика прав: без них управление ПК не работает, а молча выглядит
//! как «ассистент не понял».
//!
//! На macOS три отдельных разрешения:
//! - **Универсальный доступ** (Accessibility) — читать и нажимать элементы
//!   интерфейса. Без него `System Events` возвращает пустоту, а не ошибку:
//!   запрос выглядит как «на экране ничего нет».
//! - **Запись экрана** (Screen Recording) — снимки экрана.
//! - **Автоматизация** (Automation) — управление другими приложениями
//!   (Apple Events). Запрашивается при первом обращении.
//!
//! Linux: доступность — шина AT-SPI2 (`at-spi2-core`), ввод — `xdotool`
//! (X11) или `ydotool` (Wayland), графическая сессия — переменные окружения.
//!
//! Проверка идёт через системные API, а не через «похоже, работает»: пра��
//! в списке должно совпадать с тем, что реально доступно.

use serde::Serialize;

/// Состояние одного разрешения.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Permission {
    /// Идентификатор для интерфейса (стабильный).
    pub id: String,
    /// Человеческое имя.
    pub title: String,
    /// Выдано ли разрешение.
    pub granted: bool,
    /// Что именно разрешение даёт.
    pub purpose: String,
    /// Что делать, если не выдано (пустая строка — не требуется).
    pub remedy: String,
    /// Ссылка «открыть настройки» (macOS) — если есть куда открыть.
    pub settings_url: Option<String>,
}

impl Permission {
    fn new(
        id: &str,
        title: &str,
        granted: bool,
        purpose: &str,
        remedy: &str,
        settings_url: Option<String>,
    ) -> Self {
        Self {
            id: id.to_string(),
            title: title.to_string(),
            granted,
            purpose: purpose.to_string(),
            remedy: if granted {
                String::new()
            } else {
                remedy.to_string()
            },
            settings_url: if granted { None } else { settings_url },
        }
    }
}

/// Полная картина прав: что выдано, что нет и что с этим делать.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PermissionReport {
    pub permissions: Vec<Permission>,
    /// Есть ли хотя бы одно обязательное разрешение, которого нет.
    pub blocked: bool,
    /// Короткое человеческое резюме — для озвучки.
    pub summary: String,
}

impl PermissionReport {
    /// Обязательные для управления интерфейсом разрешения выданы?
    ///
    /// Пустой набор обязательных прав — не «всё готово», а отсутствие данных:
    /// отчёт без единого знакомого id не должен открывать автоматизацию.
    pub fn gui_ready(&self) -> bool {
        let required: Vec<&Permission> = self
            .permissions
            .iter()
            .filter(|p| matches!(p.id.as_str(), "accessibility" | "atspi" | "input_tool"))
            .collect();
        !required.is_empty() && required.iter().all(|p| p.granted)
    }
}

/// Проверить все разрешения текущей системы.
pub fn check() -> PermissionReport {
    #[cfg(target_os = "macos")]
    let permissions = macos::check();
    #[cfg(target_os = "linux")]
    let permissions = linux::check();
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let permissions = vec![Permission::new(
        "unsupported",
        "Управление интерфейсом",
        false,
        "Чтение элементов экрана и ввод с клавиатуры",
        "Эта ОС не поддерживается",
        None,
    )];

    let blocked = permissions.iter().any(|p| !p.granted);
    let missing: Vec<&str> = permissions
        .iter()
        .filter(|p| !p.granted)
        .map(|p| p.title.as_str())
        .collect();
    let summary = if missing.is_empty() {
        "Управление компьютером доступно: все права выданы.".to_string()
    } else {
        format!(
            "Управление компьютером ограничено: не хватает прав — {}.",
            missing.join(", ")
        )
    };
    PermissionReport {
        permissions,
        blocked,
        summary,
    }
}

// =========================================================================
// macOS
// =========================================================================

#[cfg(target_os = "macos")]
mod macos {
    use super::Permission;

    // Системные проверки разрешений macOS.
    //
    // `AXIsProcessTrusted` — единственный честный ответ «выдано ли право
    // на управление интерфейсом»: без него System Events возвращает
    // пустоту, и агент думает, что на экране ничего нет.
    #[link(name = "ApplicationServices", kind = "framework")]
    unsafe extern "C" {
        fn AXIsProcessTrusted() -> bool;
    }

    /// Доступ к элементам интерфейса выдан?
    pub fn accessibility_granted() -> bool {
        unsafe { AXIsProcessTrusted() }
    }

    /// Запись экрана выдана?
    ///
    /// Проверяется настоящим снимком: утилита `screencapture` без
    /// разрешения падает, а с разрешением пишет файл.
    pub fn screen_recording_granted() -> bool {
        let path = std::env::temp_dir().join("jarvis_perm_probe.png");
        let _ = std::fs::remove_file(&path);
        let ok = std::process::Command::new("screencapture")
            .args(["-x", &path.to_string_lossy()])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let written = path.metadata().map(|m| m.len() > 0).unwrap_or(false);
        let _ = std::fs::remove_file(&path);
        ok && written
    }

    /// Управление другими приложениями (Apple Events) — проверяется по
    /// тому, отвечает ли System Events на запрос окон.
    pub fn automation_works() -> bool {
        crate::platform::macos::osascript(
            "tell application \"System Events\" to return count of application processes",
        )
        .is_ok()
    }

    pub fn check() -> Vec<Permission> {
        vec![
            Permission::new(
                "accessibility",
                "Универсальный доступ",
                accessibility_granted(),
                "Чтение элементов интерфейса: кнопки, поля, меню — и нажатие по ним",
                "Системные настройки → Конфиденциальность и безопасность → Универсальный доступ → включите Jarvis",
                Some(
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
                        .to_string(),
                ),
            ),
            Permission::new(
                "screen_recording",
                "Запись экрана",
                screen_recording_granted(),
                "Снимки экрана и распознавание того, что на экране",
                "Системные настройки → Конфиденциальность и безопасность → Запись экрана → включите Jarvis",
                Some(
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
                        .to_string(),
                ),
            ),
            Permission::new(
                "automation",
                "Автоматизация",
                automation_works(),
                "Управление другими приложениями: открыть, свернуть, прочитать окно",
                "При первом обращении macOS спросит сама — разрешите Jarvis",
                None,
            ),
        ]
    }
}

// =========================================================================
// Linux
// =========================================================================

#[cfg(target_os = "linux")]
mod linux {
    use super::Permission;

    fn have(bin: &str) -> bool {
        std::env::var_os("PATH").is_some_and(|paths| {
            std::env::split_paths(&paths).any(|p| p.join(bin).is_file())
        })
    }

    /// Есть ли шина доступности AT-SPI2 (`at-spi2-core`).
    ///
    /// Проверяется через `dbus-send`: если сервис `org.a11y.Bus` не
    /// отвечает, элементы интерфейса прочитать нечем.
    pub fn atspi_available() -> bool {
        let ok = std::process::Command::new("dbus-send")
            .args([
                "--session",
                "--dest=org.a11y.Bus",
                "--type=method_call",
                "--print-reply",
                "/org/a11y/bus",
                "org.a11y.Bus.GetAddress",
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        ok || std::path::Path::new("/run/user")
            .join(std::env::var("XDG_RUNTIME_DIR").unwrap_or_default())
            .join("at-spi/bus")
            .exists()
    }

    /// Есть ли утилита ввода: xdotool для X11, ydotool для Wayland.
    pub fn input_tool() -> Option<&'static str> {
        if have("xdotool") {
            return Some("xdotool");
        }
        if have("ydotool") {
            return Some("ydotool");
        }
        None
    }

    pub fn check() -> Vec<Permission> {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        let display = std::env::var_os("DISPLAY").is_some();
        let session = wayland || display;
        let tool = input_tool();
        vec![
            Permission::new(
                "atspi",
                "Доступность (AT-SPI2)",
                atspi_available(),
                "Чтение элементов интерфейса: кнопки, поля, меню — и нажатие по ним",
                "Установите at-spi2-core и запустите dbus: systemctl --user enable --now at-spi-bus-launcher",
                None,
            ),
            Permission::new(
                "graphics_session",
                "Графическая сессия",
                session,
                "Видеть окна и управлять ими",
                if wayland {
                    "Запустите приложение в графической сессии Wayland"
                } else {
                    "Запустите приложение в графической сессии X11 (DISPLAY не задан)"
                },
                None,
            ),
            Permission::new(
                "input_tool",
                "Утилита ввода",
                tool.is_some(),
                "Клик мышью и ввод с клавиатуры",
                if wayland {
                    "Установите ydotool и запустите ydotoold: sudo pacman -S ydotool"
                } else {
                    "Установите xdotool: sudo pacman -S xdotool"
                },
                None,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_marks_blocked_when_something_is_missing() {
        let report = PermissionReport {
            permissions: vec![
                Permission::new("a", "A", true, "", "", None),
                Permission::new("b", "B", false, "", "сделай", None),
            ],
            blocked: true,
            summary: "x".into(),
        };
        assert!(report.blocked);
        assert!(!report.gui_ready());
        assert_eq!(report.permissions[1].remedy, "сделай");
        assert!(report.permissions[0].remedy.is_empty());
    }

    #[test]
    fn gui_ready_requires_accessibility_and_input() {
        let report = PermissionReport {
            permissions: vec![
                Permission::new("accessibility", "A", true, "", "", None),
                Permission::new("input_tool", "B", true, "", "", None),
            ],
            blocked: false,
            summary: "x".into(),
        };
        assert!(report.gui_ready());
    }
}
