//! E2E-батарея реальных инструментов (не сухой прогон): файлы, задачи,
//! shell, буфер обмена, громкость, веб-поиск — с проверкой фактических
//! последствий. Side effects изолируются: JARVIS_HOME — временный каталог,
//! буфер обмена сохраняется и восстанавливается, громкость ставится в
//! текущее значение. Скриншоты/открытие браузера не запускаются.
//!
//! Запуск: cargo run --release --example agent_e2e -p jarvis-core

use anyhow::Result;
use jarvis_core::platform;
use jarvis_core::tools::{self, ToolCall, ToolOutcome};
use std::sync::mpsc::{channel, Sender};

struct Ctx {
    audio_tx: Sender<jarvis_core::tts::AudioMessage>,
    events_tx: Sender<jarvis_core::events::JarvisEvent>,
}

impl Ctx {
    fn new() -> Self {
        let (audio_tx, _) = channel();
        let (events_tx, _) = channel();
        Self { audio_tx, events_tx }
    }

    fn exec(&self, call: ToolCall) -> ToolOutcome {
        tools::execute(call, &self.audio_tx, &self.events_tx)
    }
}

struct Report {
    passed: usize,
    failed: Vec<String>,
}

impl Report {
    fn new() -> Self {
        Self { passed: 0, failed: Vec::new() }
    }

    fn check(&mut self, name: &str, ok: bool, detail: &str) {
        if ok {
            self.passed += 1;
            println!("✅ {name}");
        } else {
            println!("❌ {name}: {detail}");
            self.failed.push(format!("{name}: {detail}"));
        }
    }
}

fn main() -> Result<()> {
    // Песочница: задачи и настройки идут во временный каталог.
    let sandbox = tempfile::tempdir()?;
    // edition 2024: set_var небезопасен — процесс однопоточный до этого места.
    unsafe { std::env::set_var("JARVIS_HOME", sandbox.path()) };
    println!("🧪 JARVIS_HOME = {}", sandbox.path().display());

    let ctx = Ctx::new();
    let mut report = Report::new();

    // ---------- 1. system_info — реальные сведения ----------
    let out = ctx.exec(ToolCall::SystemInfo);
    report.check(
        "system_info вернул непустые сведения",
        out.ok && !out.detail.trim().is_empty(),
        &out.speech,
    );

    // ---------- 2. Файловый цикл: write → read → verify ----------
    let dir = sandbox.path().join("files");
    let f1 = dir.join("e2e_write.txt");
    let p1 = f1.to_string_lossy().to_string();
    let out = ctx.exec(ToolCall::FileWrite {
        path: p1.clone(),
        content: "первая строка\nвторая строка\n".into(),
    });
    let written_ok = out.ok;
    let content = std::fs::read_to_string(&f1).unwrap_or_default();
    report.check(
        "file_write + перечитывание",
        written_ok && content.contains("вторая строка"),
        &out.speech,
    );

    let out = ctx.exec(ToolCall::FileRead {
        path: p1.clone(),
        offset: Some(2),
        limit: Some(1),
    });
    report.check(
        "file_read с offset/limit",
        out.ok && out.detail.contains("вторая"),
        &out.detail,
    );

    // ---------- 3. file_append ----------
    let out = ctx.exec(ToolCall::FileAppend {
        path: p1.clone(),
        content: "третья строка".into(),
    });
    let content = std::fs::read_to_string(&f1).unwrap_or_default();
    report.check(
        "file_append + перечитывание хвоста",
        out.ok && content.trim_end().ends_with("третья строка"),
        &out.speech,
    );

    // ---------- 4. file_edit: замена фрагмента ----------
    let out = ctx.exec(ToolCall::FileEdit {
        path: p1.clone(),
        old: "вторая".into(),
        new: "ИЗМЕНЁННАЯ".into(),
        all: false,
    });
    let content = std::fs::read_to_string(&f1).unwrap_or_default();
    report.check(
        "file_edit замена + отсутствие старого",
        out.ok && content.contains("ИЗМЕНЁННАЯ") && !content.contains("вторая строка\n"),
        &out.speech,
    );

    // file_edit несуществующего фрагмента — честный отказ
    let out = ctx.exec(ToolCall::FileEdit {
        path: p1.clone(),
        old: "такого нет".into(),
        new: "x".into(),
        all: false,
    });
    report.check(
        "file_edit несуществующего — отказ",
        !out.ok && out.speech.contains("найден"),
        &out.speech,
    );

    // ---------- 5. file_list / file_search / file_glob ----------
    let out = ctx.exec(ToolCall::FileList {
        path: dir.to_string_lossy().to_string(),
    });
    report.check(
        "file_list видит файл",
        out.ok && out.detail.contains("e2e_write"),
        &out.detail,
    );

    let out = ctx.exec(ToolCall::FileSearch {
        query: "ИЗМЕНЁННАЯ".into(),
        path: Some(dir.to_string_lossy().to_string()),
        context: None,
    });
    report.check(
        "file_search находит строку",
        out.ok && out.detail.contains("ИЗМЕНЁННАЯ"),
        &out.detail,
    );

    let out = ctx.exec(ToolCall::FileGlob {
        pattern: "*.txt".into(),
        path: Some(dir.to_string_lossy().to_string()),
    });
    report.check(
        "file_glob по маске",
        out.ok && out.detail.contains("e2e_write"),
        &out.detail,
    );

    // ---------- 6. Задачник: add → list → done → remove ----------
    let out = ctx.exec(ToolCall::TaskAdd {
        text: "e2e-тестовая задача".into(),
    });
    report.check("task_add", out.ok, &out.speech);

    let out = ctx.exec(ToolCall::TaskList);
    report.check(
        "task_list видит задачу",
        out.ok && out.detail.contains("e2e-тестовая"),
        &out.detail,
    );

    let out = ctx.exec(ToolCall::TaskDone { id: 1 });
    report.check("task_done id=1", out.ok, &out.speech);

    let out = ctx.exec(ToolCall::TaskList);
    // Голосовой список перечисляет только ОТКРЫТЫЕ задачи; закрытые —
    // счётчиком «Всего задач: N, невыполненных: 0». Это по дизайну.
    report.check(
        "task_list: закрытая задача ушла из списка",
        out.ok && out.detail.contains("невыполненных: 0"),
        &out.detail,
    );

    // id без номера — контракт: задачника с id=1 уже нет (закрыта), молчаливый
    // дефолт удалён — parse вернёт Failed, но через execute напрямую id обязателен.
    let out = ctx.exec(ToolCall::TaskRemove { id: 1 });
    report.check("task_remove id=1", out.ok, &out.speech);

    // ---------- 7. shell: stdout, код возврата, вывод ошибки ----------
    let out = ctx.exec(ToolCall::Shell {
        cmd: "echo e2e_shell_ok".into(),
    });
    report.check(
        "shell echo → stdout в detail",
        out.ok && out.detail.contains("e2e_shell_ok"),
        &out.detail,
    );

    let out = ctx.exec(ToolCall::Shell {
        cmd: "false".into(),
    });
    report.check(
        "shell false → честный провал",
        !out.ok,
        &out.speech,
    );

    let out = ctx.exec(ToolCall::Shell {
        cmd: "echo ошибка_поток >&2; exit 3".into(),
    });
    report.check(
        "shell exit 3 + stderr → провал с причиной",
        !out.ok && (out.detail.contains("ошибка_поток") || out.detail.contains("3")),
        &out.detail,
    );

    // ---------- 8. Буфер обмена: сохранить → тест → восстановить ----------
    let saved = platform::clipboard_read().unwrap_or_default();
    let out = ctx.exec(ToolCall::ClipboardWrite {
        text: "e2e_clipboard_маркер".into(),
    });
    let wrote = out.ok;
    let read = platform::clipboard_read().unwrap_or_default();
    let roundtrip = wrote && read.contains("e2e_clipboard_маркер");
    if !saved.is_empty() {
        let _ = platform::clipboard_write(&saved);
    }
    report.check("clipboard write→read roundtrip", roundtrip, &read);

    // ---------- 9. Громкость: readback без слышимых изменений ----------
    let current = platform::get_volume().unwrap_or(50);
    let out = ctx.exec(ToolCall::SetVolume {
        percent: current.clamp(10, 90),
    });
    let after = platform::get_volume();
    let vol_ok = match after {
        Some(v) => (i16::from(v) - i16::from(current.clamp(10, 90))).abs() <= 2,
        None => out.ok, // платформа без readback — верим верификации execute
    };
    report.check(
        "set_volume: значение подтверждено readback",
        vol_ok,
        &format!("{} → {:?}", out.speech, after),
    );

    // ---------- 10. Веб-поиск: реальная сеть ----------
    let out = ctx.exec(ToolCall::WebSearch {
        query: "Rust programming language".into(),
    });
    report.check(
        "web_search: реальная выдача",
        out.ok && !out.detail.trim().is_empty(),
        &out.detail.chars().take(120).collect::<String>(),
    );

    // ---------- 11. Пустой web_search без контекста — честный отказ ----------
    // Ремонт «запрос = фраза пользователя» происходит на слое LLM
    // (run_action/chat), где фраза известна; прямой execute без контекста
    // отказываться обязан — иначе искали бы пустоту.
    let out = ctx.exec(ToolCall::WebSearch { query: "".into() });
    report.check(
        "web_search без запроса и контекста — честный отказ",
        !out.ok,
        &out.speech,
    );

    // ---------- Итог ----------
    println!("─────");
    println!(
        "E2E итог: {}/{} успешно",
        report.passed,
        report.passed + report.failed.len()
    );
    if !report.failed.is_empty() {
        println!("Провалы:");
        for f in &report.failed {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
    println!("🎉 Все e2e-проверки прошли");

    Ok(())
}