//! doctor — наблюдаемость платформы (§9): что умеет бэкенд, какие права
//! нужны, что не поддержано и что с этим делать. Каждое «не умею» называет
//! платформу, причину и следующий шаг.

use std::fmt::Write as _;

/// Одна строка отчёта о возможности: поддержка, причина деградации,
/// следующий шаг для пользователя.
#[derive(Debug, Clone)]
pub struct CapabilityLine {
    pub feature: String,
    pub support: String,
    pub note: String,
}

/// Полный отчёт о возможностях платформы (§12.3 — «фича × платформа ×
/// поддержка × деградация», заполненный по `Capabilities`).
#[derive(Debug, Default)]
pub struct CapabilityReport {
    pub backend: String,
    pub platform: String,
    pub os_version: String,
    pub lines: Vec<CapabilityLine>,
    /// Честный список того, что не заработало и почему.
    pub problems: Vec<String>,
}

/// Отчёт о текущей платформе: xa11y описывает бэкенд и фичи, окружение
/// проверяем сами (сессия, AT-SPI, uinput на Wayland).
pub fn report() -> CapabilityReport {
    let mut report = CapabilityReport::default();

    match xa11y::Capabilities::current() {
        Ok(caps) => {
            report.backend = caps.backend.to_string();
            report.platform = format!("{:?}", caps.platform);
            report.os_version = caps.os_version.clone().unwrap_or_else(|| "—".into());
            for cap in caps.entries() {
                report.lines.push(CapabilityLine {
                    feature: cap.name.to_string(),
                    support: format!("{:?}", cap.support),
                    note: cap
                        .note
                        .clone()
                        .unwrap_or_else(|| "—".to_string()),
                });
            }
        }
        Err(e) => {
            report
                .problems
                .push(format!("Возможности платформы не прочитаны: {e}"));
        }
    }

    check_environment(&mut report);
    report
}

/// Проверки окружения: сессия, AT-SPI шина, права на uinput (Wayland).
fn check_environment(report: &mut CapabilityReport) {
    let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
    match session.as_str() {
        "x11" => report.lines.push(CapabilityLine {
            feature: "сессия X11".into(),
            support: "есть".into(),
            note: "AT-SPI2 + XTest".into(),
        }),
        "wayland" => {
            report.lines.push(CapabilityLine {
                feature: "сессия Wayland".into(),
                support: "есть".into(),
                note: "AT-SPI2 + виртуальный evdev через /dev/uinput".into(),
            });
            check_uinput(report);
        }
        "" => report
            .problems
            .push("XDG_SESSION_TYPE не задан — тип сессии определить нельзя; запуск под графической сессией?".into()),
        other => report.lines.push(CapabilityLine {
            feature: format!("сессия {other}"),
            support: "неизвестно".into(),
            note: "нестандартная сессия — проверки ввода пропущены".into(),
        }),
    }
}

/// Wayland: единственная реальная настройка — запись в /dev/uinput.
/// Без неё ввод невозможен; doctor обязан это назвать с командой починки.
fn check_uinput(report: &mut CapabilityReport) {
    match std::fs::OpenOptions::new().write(true).open("/dev/uinput") {
        Ok(_) => report.lines.push(CapabilityLine {
            feature: "права на /dev/uinput".into(),
            support: "есть".into(),
            note: "ввод уровня ОС на Wayland будет работать".into(),
        }),
        Err(e) => {
            report.lines.push(CapabilityLine {
                feature: "права на /dev/uinput".into(),
                support: "нет".into(),
                note: format!("ошибка записи: {e}"),
            });
            report.problems.push(
                "Нет записи в /dev/uinput — синтетический ввод на Wayland невозможен. \
Следующий шаг: добавьте пользователя в группу input (sudo usermod -aG input $USER, \
затем перелогиньтесь) или поставьте udev-правило на /dev/uinput."
                    .into(),
            );
        }
    }
}

impl CapabilityReport {
    /// Человекочитаемый отчёт.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "autocore doctor");
        let _ = writeln!(
            out,
            "Бэкенд: {} · Платформа: {} · ОС: {}",
            self.backend, self.platform, self.os_version
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "ФЕЧА × ПОДДЕРЖКА × ДЕГРАДАЦИЯ");
        for l in &self.lines {
            let _ = writeln!(out, "  {:38} {:10} {}", l.feature, l.support, l.note);
        }
        if !self.problems.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "ПРОБЛЕМЫ (что не заработало и что делать):");
            for p in &self.problems {
                let _ = writeln!(out, "  ⚠️ {p}");
            }
        } else {
            let _ = writeln!(out);
            let _ = writeln!(out, "Проблем не найдено.");
        }
        out
    }

    /// Дым-тест на реальной машине: перечислить приложения и снять
    /// bounded capture корня первого — чтения, никаких действий.
    pub fn smoke(&mut self) -> String {
        match smoke_run() {
            Ok(lines) => {
                self.lines.push(CapabilityLine {
                    feature: "дым-тест (список приложений + capture)".into(),
                    support: "есть".into(),
                    note: lines.clone(),
                });
                format!("Дым-тест: {lines}")
            }
            Err(e) => {
                self.problems.push(format!("Дым-тест не прошёл: {e}"));
                format!("Дым-тест: не прошёл — {e}")
            }
        }
    }
}

/// Сам дым-тест: список приложений, capture корня первого из них.
fn smoke_run() -> Result<String, String> {
    use xa11y::AppExt as _;
    let apps = xa11y::App::list()
        .map_err(|e| format!("список приложений не читается: {e}"))?;
    if apps.is_empty() {
        return Err(
            "приложений с accessibility-деревом не видно: на Wayland/Hyprland нужны \
GTK/Qt-приложения и поднятая шина AT-SPI (проверьте at-spi-bus-launcher)"
                .into(),
        );
    }
    let first = apps.first().expect("список не пуст");
    let root = first.as_element();
    let (shadow, capture) = crate::capture::capture_shadow(&root, &crate::capture::default_options())
        .map_err(|e| format!("capture корня «{}» не удался: {e}", first.name))?;
    Ok(format!(
        "приложений {}, корень «{}»: {} узлов в тени, чтений {}",
        apps.len(),
        first.name,
        shadow.nodes.len(),
        capture.stats.children_reads
    ))
}
