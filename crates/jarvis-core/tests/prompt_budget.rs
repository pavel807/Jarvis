//! Бюджет системных промптов в токенах на реальном токенизаторе модели.
//!
//! Отдельный интеграционный тест намеренно: он сам выставляет `JARVIS_HOME`
//! на корень репозитория, поэтому не зависит от того, откуда запущен тест.
//! Размер промпта — это время до первого звука, и раньше action-промпт со
//! всеми 32 инструментами занимал 1662 токена против 250 токенов ответа.
//!
//! Потолки держим с запасом ~15% на будущие правки.

use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("корень репозитория")
        .to_path_buf()
}

/// Пути к данным в тестах не должны зависеть от текущего каталога: модели
/// лежат в корне репозитория, а не там, откуда запущен cargo.
fn use_repo_as_home() {
    if std::env::var_os("JARVIS_HOME").is_none() {
        // Однопоточная инициализация до первого обращения к путям.
        unsafe { std::env::set_var("JARVIS_HOME", repo_root()) };
    }
}

fn tokens(text: &str) -> Option<usize> {
    use_repo_as_home();
    jarvis_core::prompt_budget::count(text)
}

fn budget(name: &str, text: &str, limit: usize) {
    let Some(tokens) = tokens(text) else {
        panic!(
            "«{name}»: токенизатор не найден в {} — бюджет проверить нельзя, \
             а молча пропускать эту проверку нельзя тоже",
            repo_root().join("models").display()
        );
    };
    assert!(
        tokens <= limit,
        "«{name}»: {tokens} токенов при потолке {limit}. Промпт раздулся — \
         сократи формулировки или убери лишние инструменты."
    );
}

#[test]
fn chat_and_web_prompts_stay_small() {
    budget("chat", jarvis_core::llm::CHAT_SYSTEM_PROMPT, 160);
    budget("web", jarvis_core::llm::WEB_SYSTEM_PROMPT, 120);
}

// Состав action-промпта: ядро кода (шапка + инструменты + правила) ~880
// токенов — за его раздувание и следим; каталог пользовательских команд
// (commands.toml) — данные пользователя, сейчас ~500 токенов и растут с
// каждым новым файлом команд, поэтому ему свой запас. Потолки держим с
// запасом ~10% на будущие правки.
#[test]
fn action_prompt_fits_budget() {
    let core = jarvis_core::router::action_prompt();
    // Ядро инструментов (включая файловые операции) вместе с каталогом команд.
    budget("action с каталогом", &core, 1400);
}

#[test]
fn action_slices_stay_within_budget() {
    let with_slice = jarvis_core::router::action_prompt_for("сверни зум");
    budget("action со срезом", &with_slice, 1650);

    // Худший случай: просьба задевает сразу четыре среза.
    let worst = jarvis_core::router::action_prompt_for(
        "сверни зум, поставь таймер и отправь сообщение маме, потом включи музыку",
    );
    budget("action с четырьмя срезами", &worst, 1850);
}

/// Файловые инструменты — половина смысла агента: он должен уметь найти файл,
/// прочитать его по строкам и точечно поправить, а не перезаписывать целиком.
#[test]
fn file_tools_are_described_for_the_model() {
    let prompt = jarvis_core::router::action_prompt();
    for must in [
        "file_read",
        "file_write",
        "file_append",
        "file_edit",
        "file_search",
        "file_glob",
        "Рабочая папка",
    ] {
        assert!(prompt.contains(must), "в промпте нет «{must}»");
    }
}