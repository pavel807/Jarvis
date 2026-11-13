//! Зонд управления ПК: проверяет, что реально доступно на этой машине.
//!
//! Ничего не меняет: только чтения (дерево Accessibility, список окон,
//! каталог приложений, снимок экрана в файл). Ничего не кликает и не печатает
//! в чужие окна.
//!
//! Запуск:
//!   cargo run --release -p jarvis-core --example pc_probe
//!
//! Вывод — по одному блоку на возможность:
//!   ДА/НЕТ + причина. НЕТ означает не «сломано», а «нужно разрешение или
//!   нет утилиты»: именно эти строки показывают, где реально дыра.

use jarvis_core::platform;

fn main() {
    println!("=== Зонд управления ПК ===");
    println!("ОС: {}\n", tools_os());

    // 0. Права: без них всё остальное ниже — иллюзия.
    let report = jarvis_core::permissions::check();
    let mut lines = vec![format!("Итог: {}", report.summary)];
    for p in &report.permissions {
        lines.push(format!(
            "  [{}] {} — {}{}",
            if p.granted { "да" } else { "НЕТ" },
            p.title,
            p.purpose,
            if p.remedy.is_empty() {
                String::new()
            } else {
                format!("\n        Что делать: {}", p.remedy)
            }
        ));
    }
    print_block("Права доступа", &lines.join("\n"));

    // 1. Дерево Accessibility активного приложения — основа «клик по имени».
    print_block(
        "Чтение элементов экрана (ui_read)",
        &match jarvis_core::auto::describe_screen() {
            Ok(text) => {
                let head: String = text.chars().take(900).collect();
                format!("ДА\n{head}")
            }
            Err(e) => format!("НЕТ — {e}"),
        },
    );

    // 1a. Глубокий обход: сколько элементов на самом деле видно.
    #[cfg(target_os = "macos")]
    print_block(
        "Глубокий обход дерева (счётчик элементов)",
        &match jarvis_core::auto::ax::collect(None) {
            Ok(snapshot) => {
                let mut by_role: Vec<(String, usize)> = Vec::new();
                for el in &snapshot.elements {
                    match by_role.iter_mut().find(|(r, _)| *r == el.role) {
                        Some((_, n)) => *n += 1,
                        None => by_role.push((el.role.clone(), 1)),
                    }
                }
                let list: Vec<String> = by_role
                    .iter()
                    .map(|(r, n)| format!("{r}: {n}"))
                    .collect();
                format!(
                    "ДА — всего {} элементов в окне «{}»\n{}",
                    snapshot.elements.len(),
                    snapshot.window,
                    list.join(", ")
                )
            }
            Err(e) => format!("НЕТ — {e}"),
        },
    );

    // 2. Список окон — что вообще открыто.
    let windows = platform::window_control("list", None);
    print_block(
        "Список окон (window list)",
        &if windows.starts_with("Не могу") || windows.contains("нет графической сессии") {
            format!("НЕТ — {windows}")
        } else {
            format!("ДА\n{windows}")
        },
    );

    // 3. Активное окно.
    print_block(
        "Активное окно",
        &match platform::active_window() {
            Some((app, title)) => format!("ДА\n{app} — {title}"),
            None => "НЕТ — не прочиталось".into(),
        },
    );

    // 4. Каталог установленных приложений — из него модель берёт open_app.
    let catalog = platform::app_catalog();
    print_block(
        "Каталог приложений (open_app)",
        &if catalog.is_empty() {
            "НЕТ — каталог пуст".into()
        } else {
            format!("ДА ({} шт.)\n{}", catalog.len(), catalog.join(", "))
        },
    );

    // 5. Снимок экрана — будущий вход для зрительной модели.
    print_block(
        "Снимок экрана (screenshot)",
        &match platform::screenshot() {
            Some(path) => format!("ДА\n{path}"),
            None => "НЕТ — снимок не сохранился".into(),
        },
    );

    // 6. Утилита ввода: без неё нельзя ни печатать, ни кликать.
    print_block(
        "Утилита ввода (клик/печать)",
        &input_tool_status(),
    );

    // 7. Буфер обмена — часть наблюдения.
    print_block(
        "Буфер обмена",
        &match platform::clipboard_read() {
            Some(text) => {
                let head: String = text.chars().take(80).collect();
                format!("ДА\n{head}")
            }
            None => "НЕТ — не прочитался".into(),
        },
    );
}

fn print_block(title: &str, body: &str) {
    println!("--- {title} ---\n{body}\n");
}

fn tools_os() -> &'static str {
    jarvis_core::tools::os_name()
}

/// Какой ввод доступен на этой ОС: утилита для кликов/печати.
fn input_tool_status() -> String {
    #[cfg(target_os = "macos")]
    {
        let found = ["/opt/homebrew/bin/cliclick", "/usr/local/bin/cliclick"]
            .iter()
            .any(|p| std::path::Path::new(p).is_file());
        if found {
            "ДА — cliclick".into()
        } else {
            "НЕТ — нет cliclick (brew install cliclick)".into()
        }
    }
    #[cfg(target_os = "linux")]
    {
        let x = which("xdotool");
        let y = which("ydotool");
        match (x, y) {
            (Some(_), _) => "ДА — xdotool".into(),
            (_, Some(_)) => "ДА — ydotool".into(),
            _ => "НЕТ — нет xdotool/ydotool".into(),
        }
    }
    #[cfg(target_os = "windows")]
    {
        "ДА — Win32 SendInput".into()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "НЕТ — ОС не поддерживается".into()
    }
}

fn which(bin: &str) -> Option<String> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(bin))
            .find(|p| p.is_file())
            .map(|p| p.to_string_lossy().to_string())
    })
}
