//! AT-SPI2 — Linux-эквивалент macOS Accessibility.
//!
//! Тот же контракт, что у `ax.rs` (AppleScript/System Events): LLM читает
//! экран текстом (приложение, окна, кнопки, поля) и действует по именам
//! элементов — клик по кнопке «Продолжить», ввод в поле «Поиск», пункт
//! меню «Файл → Сохранить». Элементы отдаёт AT-SPI2-шина (at-spi2-core):
//! GTK и Qt выводят своё дерево через неё, поэтому работает и X11, и
//! Wayland. Нужен пакет `at-spi2-core` (шина org.a11y.Bus); для
//! Electron-приложений дерево включается ключом запуска
//! `--enable-accessibility` / `--force-renderer-accessibility`.
//!
//! Формат снимка экрана и семантика поиска (точное имя → вхождение)
//! повторяют `ax.rs`, чтобы промпты и поведение инструмента не зависели
//! от ОС.

use anyhow::{anyhow, Result};
use atspi::{
    proxy::{
        accessible::AccessibleProxy,
        action::ActionProxy,
        component::ComponentProxy,
        editable_text::EditableTextProxy,
        text::TextProxy,
    },
    AccessibilityConnection, CoordType, State,
};
use atspi::zbus;
use std::sync::OnceLock;

// Границы обхода: дерево GTK-приложения может быть огромным (веб-контент
// в Electron разворачивается в сотни узлов), без капов чтение экрана
// заняло бы секунды и раздуло промпт.
const MAX_DEPTH: usize = 30;
const MAX_NODES: usize = 2500;
const MAX_WINDOWS: usize = 3;
const MAX_FIELDS: usize = 8;
const MAX_LABELS: usize = 12;
const MENU_SETTLE_MS: u64 = 250;

/// Адрес узла на шине AT-SPI + прочитанные атрибуты.
#[derive(Clone)]
struct Node {
    dest: String,
    path: String,
    role: String,
    name: String,
}

impl Node {
    /// Адресация из ссылки, отданной get_children.
    fn from_ref(r: atspi::ObjectRefOwned) -> (String, String) {
        let inner = r.into_inner();
        (
            inner.name_as_str().unwrap_or_default().to_string(),
            inner.path_as_str().to_string(),
        )
    }

    /// Прочитать роль и имя узла за один заход.
    async fn probe(dest: String, path: String, conn: &zbus::Connection) -> zbus::Result<Node> {
        let proxy = accessible(conn, &dest, &path).await?;
        let role = proxy.get_role_name().await?;
        let name = proxy.name().await?;
        drop(proxy);
        Ok(Node { dest, path, role, name })
    }
}

/// Свой tokio-реактор: вызовы идут из обычных потоков движка, а zbus
/// требует асинхронный контекст. Один глобальный runtime — подключения
/// к шине живут внутри block_on и наружу не выносятся.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_io()
            .enable_time()
            .build()
            .expect("tokio runtime для AT-SPI")
    })
    .block_on(fut)
}

// =========================================================================
// Прокси интерфейсов AT-SPI: адресные сборки без кэша свойств
// =========================================================================

async fn accessible<'a>(
    conn: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> zbus::Result<AccessibleProxy<'a>> {
    AccessibleProxy::builder(conn)
        .destination(dest)?
        .path(path)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

async fn action<'a>(
    conn: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> zbus::Result<ActionProxy<'a>> {
    ActionProxy::builder(conn)
        .destination(dest)?
        .path(path)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

async fn text_iface<'a>(
    conn: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> zbus::Result<TextProxy<'a>> {
    TextProxy::builder(conn)
        .destination(dest)?
        .path(path)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

async fn component<'a>(
    conn: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> zbus::Result<ComponentProxy<'a>> {
    ComponentProxy::builder(conn)
        .destination(dest)?
        .path(path)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

async fn editable<'a>(
    conn: &'a zbus::Connection,
    dest: &'a str,
    path: &'a str,
) -> zbus::Result<EditableTextProxy<'a>> {
    EditableTextProxy::builder(conn)
        .destination(dest)?
        .path(path)?
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

// =========================================================================
// Словарь ролей: классы из общего словаря инструментов → роли AT-SPI
// =========================================================================

/// Роли AT-SPI, соответствующие классу из промпта (`button`, `text field`…).
/// Словарь инструментов говорит на языке macOS-классов — Linux переводит
/// их в свои имена ролей, чтобы один и тот же tool-вызов работал на обеих ОС.
pub fn atspi_roles(normalized: &str) -> &'static [&'static str] {
    match normalized {
        "button" => &["push button"],
        "checkbox" => &["check box"],
        "radio button" => &["radio button"],
        "pop up button" => &["combo box"],
        "text field" => &["entry", "text", "terminal"],
        "secure text field" => &["password text"],
        "static text" => &["label", "static"],
        "link" => &["link"],
        "slider" => &["slider"],
        "menu bar item" => &["menu item", "menu"],
        _ => &[],
    }
}

/// Совпадает ли имя элемента с просьбой: точное (регистр не важен — так
/// работает сравнение строк в AppleScript, отсюда паритет), затем вхождение.
/// `description` — запасное поле: у GTK-полей ввода имя часто пустое, а
/// описание держит подпись/плейсхолдер.
pub fn name_matches(candidate: &str, description: &str, want: &str) -> bool {
    let want = want.trim();
    if want.is_empty() {
        return false;
    }
    let eq = |s: &str| s.trim().eq_ignore_ascii_case(want);
    let contains = |s: &str| {
        let s = s.trim().to_lowercase();
        !s.is_empty() && s.contains(&want.to_lowercase())
    };
    eq(candidate) || eq(description) || contains(candidate) || contains(description)
}

// =========================================================================
// Подключение и обход дерева
// =========================================================================

/// Открыть AT-SPI-шину с человекопонятной ошибкой.
async fn connect() -> Result<AccessibilityConnection> {
    AccessibilityConnection::new().await.map_err(|e| {
        anyhow!(
            "AT-SPI2-шина недоступна: {e}. Установи at-spi2-core (шина org.a11y.Bus) и проверь, что приложение запущено в графической сессии"
        )
    })
}

/// Список приложений: дети корня реестра с ролью «application».
async fn applications(conn: &AccessibilityConnection) -> Result<Vec<Node>> {
    let root = conn
        .root_accessible_on_registry()
        .await
        .map_err(|e| anyhow!("не удалось получить корень AT-SPI: {e}"))?;
    let zconn = conn.connection().clone();
    let refs = root
        .get_children()
        .await
        .map_err(|e| anyhow!("не удалось перечислить приложения: {e}"))?;
    let mut apps = Vec::new();
    for r in refs {
        let (dest, path) = Node::from_ref(r);
        if let Ok(node) = Node::probe(dest, path, &zconn).await
            && node.role == "application"
        {
            apps.push(node);
        }
    }
    Ok(apps)
}

/// Дети-окна приложения: роли Frame/Window/Dialog/Alert.
async fn window_children(app: &Node, conn: &zbus::Connection) -> Result<Vec<Node>> {
    let proxy = accessible(conn, &app.dest, &app.path).await?;
    let refs = proxy.get_children().await.unwrap_or_default();
    let mut windows = Vec::new();
    for r in refs {
        let (dest, path) = Node::from_ref(r);
        if let Ok(node) = Node::probe(dest, path, conn).await
            && matches!(node.role.as_str(), "frame" | "window" | "dialog" | "alert")
        {
            windows.push(node);
        }
    }
    Ok(windows)
}

/// BFS от корня с капами: все узлы поддерева с их глубиной.
async fn walk(start: &Node, conn: &zbus::Connection) -> Result<Vec<(Node, usize)>> {
    let mut out = Vec::new();
    let mut queue: std::collections::VecDeque<(String, String, usize)> =
        std::collections::VecDeque::new();
    queue.push_back((start.dest.clone(), start.path.clone(), 0));
    while let Some((dest, path, depth)) = queue.pop_front() {
        if out.len() >= MAX_NODES || depth > MAX_DEPTH {
            continue;
        }
        let node = match Node::probe(dest, path, conn).await {
            Ok(n) => n,
            Err(_) => continue,
        };
        if let Ok(proxy) = accessible(conn, &node.dest, &node.path).await {
            for r in proxy.get_children().await.unwrap_or_default() {
                let (dest, path) = Node::from_ref(r);
                queue.push_back((dest, path, depth + 1));
            }
        }
        out.push((node, depth));
    }
    Ok(out)
}

/// Активное приложение: у одного из его окон стоит состояние Active.
/// Если явного Active нет — берём первое приложение с окнами (фокус может
/// не транслироваться в AT-SPI на части Wayland-стеков).
async fn active_app(conn: &AccessibilityConnection) -> Result<(Node, Option<Node>)> {
    let zconn = conn.connection().clone();
    let apps = applications(conn).await?;
    if apps.is_empty() {
        return Err(anyhow!(
            "AT-SPI не видит приложений: доступность (a11y) у рабочей среды выключена"
        ));
    }
    let mut fallback: Option<(Node, Option<Node>)> = None;
    for app in apps {
        let windows = window_children(&app, &zconn).await?;
        if windows.is_empty() {
            continue;
        }
        for win in &windows {
            let proxy = accessible(&zconn, &win.dest, &win.path).await?;
            if proxy
                .get_state()
                .await
                .map(|s| s.contains(State::Active))
                .unwrap_or(false)
            {
                return Ok((app, Some(win.clone())));
            }
        }
        if fallback.is_none() {
            fallback = Some((app.clone(), windows.into_iter().next()));
        }
    }
    fallback.ok_or_else(|| anyhow!("не найдено приложение с окнами"))
}

/// Разрешить просьбу в приложение AT-SPI: точное/вхождение по имени,
/// иначе — активное (паритет resolve_app из ax.rs).
async fn resolve_app(
    conn: &AccessibilityConnection,
    app: Option<&str>,
) -> Result<(Node, Option<Node>)> {
    if let Some(want) = app.map(str::trim).filter(|s| !s.is_empty()) {
        let zconn = conn.connection().clone();
        for candidate in applications(conn).await? {
            // process_matches знает русские алиасы и транслит: «наутилус» →
            // org.gnome.Nautilus — как resolve_app на macOS.
            if crate::platform::process_matches(&candidate.name, want) {
                let windows = window_children(&candidate, &zconn).await?;
                return Ok((candidate, windows.into_iter().next()));
            }
        }
        return Err(anyhow!(
            "приложение «{want}» не видно в AT-SPI: оно не запущено или не отдаёт дерево доступности"
        ));
    }
    active_app(conn).await
}

/// Окно-корень поиска: по номеру/имени, иначе первое (паритет window_clause).
async fn resolve_window(
    conn: &AccessibilityConnection,
    target: &Node,
    window: Option<&str>,
) -> Result<Node> {
    let zconn = conn.connection().clone();
    let windows = window_children(target, &zconn).await?;
    match window.map(str::trim).filter(|s| !s.is_empty()) {
        None => windows
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("у {} нет окон", target.name)),
        Some(w) if w.chars().all(|c| c.is_ascii_digit()) => windows
            .get(w.parse::<usize>().unwrap_or(1).saturating_sub(1))
            .cloned()
            .ok_or_else(|| anyhow!("окно №{w} не найдено у {}", target.name)),
        Some(w) => windows
            .into_iter()
            .find(|win| name_matches(&win.name, "", w))
            .ok_or_else(|| anyhow!("окно «{w}» не найдено у {}", target.name)),
    }
}

/// Значение текстового узла через интерфейс Text.
async fn read_text(node: &Node, conn: &zbus::Connection) -> Result<String> {
    let text = text_iface(conn, &node.dest, &node.path).await?;
    let count = text.character_count().await?;
    if count == 0 {
        return Ok(String::new());
    }
    Ok(text.get_text(0, count).await.unwrap_or_default())
}

/// Прочитать имя активного приложения (паритет `front_app` из ax.rs).
pub fn front_app() -> Result<String> {
    block_on(async {
        let conn = connect().await?;
        let (app, _) = active_app(&conn).await?;
        Ok(app.name)
    })
}

// =========================================================================
// Снимок экрана
// =========================================================================

/// Снимок экрана: активное приложение и элементы управления.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScreenDump {
    /// Имя активного приложения (может быть пустым).
    pub app: String,
    /// Текстовый отчёт, собранный из AT-SPI (уже человекочитаемый).
    pub report: String,
}

impl ScreenDump {
    /// Текст для пользователя и для модели.
    pub fn to_text(&self) -> String {
        if self.report.trim().is_empty() {
            format!(
                "Активное приложение: {}. Элементов не видно (окно может быть пустым или защищённым).",
                self.app
            )
        } else {
            self.report.trim().to_string()
        }
    }
}

/// Положить список имён в строку отчёта, как joinList в AppleScript.
fn join(names: &[String]) -> String {
    if names.is_empty() {
        "—".into()
    } else {
        names.join(", ")
    }
}

fn clip(s: &str, max: usize) -> String {
    let mut out: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        out.push('…');
    }
    out
}

/// Перечислить активное приложение, окна и элементы управления.
/// Формат повторяет READ_SCREEN_SCRIPT из ax.rs — секции «Кнопки/Флажки/
/// Списки/Поля/Надписи/Меню», чтобы промпт модели был одним на обе ОС.
pub fn read_screen() -> Result<ScreenDump> {
    block_on(async {
        let conn = connect().await?;
        let (app, _) = active_app(&conn).await?;
        let zconn = conn.connection().clone();
        let windows = window_children(&app, &zconn).await?;
        if windows.is_empty() {
            return Ok(ScreenDump {
                app: app.name.clone(),
                report: format!("Активное приложение: {} (окон нет)", app.name),
            });
        }
        let mut report = String::new();
        for (idx, win) in windows.iter().take(MAX_WINDOWS).enumerate() {
            report.push_str(&window_report(idx, win, &zconn).await);
        }
        Ok(ScreenDump {
            app: app.name.clone(),
            report,
        })
    })
}

/// Диагностика: отчёт по элементам названного приложения, даже неактивного
/// (пример atspi_probe). Возвращает то же текстовое дерево, что ui_read.
pub fn describe_app(app: &str) -> Result<String> {
    block_on(async {
        let conn = connect().await?;
        let (target, _) = resolve_app(&conn, Some(app)).await?;
        let zconn = conn.connection().clone();
        let windows = window_children(&target, &zconn).await?;
        if windows.is_empty() {
            return Ok(format!("{}: окон нет", target.name));
        }
        let mut report = String::new();
        for (idx, win) in windows.iter().take(MAX_WINDOWS).enumerate() {
            report.push_str(&window_report(idx, win, &zconn).await);
        }
        Ok(report)
    })
}

/// Отчёт по одному окну: секции элементов, как в READ_SCREEN_SCRIPT.
async fn window_report(idx: usize, win: &Node, conn: &zbus::Connection) -> String {
    let mut report = format!("Окно {}: «{}»\n", idx + 1, clip(&win.name, 80));
    let tree = match walk(win, conn).await {
        Ok(t) => t,
        Err(e) => return format!("{report}Обход не удался: {e:#}\n"),
    };

    let mut buttons = Vec::new();
    let mut checkboxes = Vec::new();
    let mut combos = Vec::new();
    let mut items: Vec<String> = Vec::new();
    let mut labels = Vec::new();
    let mut fields: Vec<(Node, String)> = Vec::new();
    let mut menus = Vec::new();
    for (node, _) in &tree {
        match node.role.as_str() {
            "push button" => buttons.push(clip(&node.name, 60)),
            "check box" => checkboxes.push(clip(&node.name, 60)),
            "combo box" => combos.push(clip(&node.name, 60)),
            // GTK-навигация: действия часто живут в list item, а не в кнопке.
            "list item" | "page tab" => {
                if !node.name.trim().is_empty() && items.len() < MAX_LABELS {
                    items.push(clip(&node.name, 80));
                }
            }
            "label" => {
                if !node.name.trim().is_empty() && labels.len() < MAX_LABELS {
                    labels.push(clip(&node.name, 80));
                }
            }
            "entry" | "text" | "password text" => {
                if fields.len() >= MAX_FIELDS {
                    continue;
                }
                // Читаем фактическое значение поля — так модель видит
                // не только список полей, но и что в них написано.
                let value = read_text(node, conn).await.unwrap_or_default();
                fields.push((node.clone(), value));
            }
            "menu bar" => {
                if let Ok(proxy) = accessible(conn, &node.dest, &node.path).await
                    && let Ok(items) = proxy.get_children().await
                {
                    for item in items {
                        let (dest, path) = Node::from_ref(item);
                        if let Ok(child) = Node::probe(dest, path, conn).await {
                            menus.push(clip(&child.name, 40));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    report.push_str(&format!("Кнопки: {}\n", join(&buttons)));
    report.push_str(&format!("Флажки: {}\n", join(&checkboxes)));
    report.push_str(&format!("Списки: {}\n", join(&combos)));
    if fields.is_empty() {
        report.push_str("Поля: —\n");
    } else {
        let rendered: Vec<String> = fields
            .iter()
            .map(|(n, v)| format!("поле «{}» = «{}»", clip(&n.name, 40), clip(v, 80)))
            .collect();
        report.push_str(&format!("Поля: {}\n", rendered.join(", ")));
    }
    report.push_str(&format!("Надписи: {}\n", join(&labels)));
    report.push_str(&format!("Пункты: {}\n", join(&items)));
    report.push_str(&format!("Меню: {}\n", join(&menus)));
    if buttons.is_empty()
        && checkboxes.is_empty()
        && combos.is_empty()
        && items.is_empty()
        && labels.is_empty()
        && fields.is_empty()
        && menus.is_empty()
    {
        // Диагностика пустого снимка: приложение отдаёт дерево, но без
        // распознанных элементов (или a11y у него выключен).
        let mut roles: Vec<&str> = tree.iter().map(|(n, _)| n.role.as_str()).collect();
        roles.sort_unstable();
        roles.dedup();
        let shown: Vec<&str> = roles.iter().take(10).copied().collect();
        report.push_str(&format!(
            "Элементов не видно (узлов в дереве: {}, роли: {})",
            tree.len(),
            shown.join(", ")
        ));
    }
    report
}

// =========================================================================
// Действия: клик, ввод, меню
// =========================================================================

/// Проверка отмены — тот же флаг, что у macOS-автоматизации.
fn ensure_not_cancelled() -> Result<()> {
    if crate::auto::is_cancelled() {
        return Err(anyhow!("автоматизация остановлена голосом"));
    }
    Ok(())
}

/// Клик по элементу окна по имени и роли. Проверка результата
/// (перечитывание экрана) — на вызывающей стороне через [`read_screen`].
pub fn click_element(
    app: Option<&str>,
    role: &str,
    name: &str,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let normalized = super::normalize_role(role);
    // Меню-пункты идут отдельным путём — паритет с ax.rs.
    if normalized == "menu bar item" {
        return menu_path(app, &[name]);
    }
    // Класс словаря → роли AT-SPI; незнакомая роль (модель повторила то,
    // что увидела в снимке: «list item», «page tab») идёт как есть.
    let mapped = atspi_roles(&normalized);
    let roles: Vec<&str> = if mapped.is_empty() && normalized.is_ascii() {
        vec![normalized.as_str()]
    } else {
        mapped.to_vec()
    };
    if roles.is_empty() {
        return Err(anyhow!(
            "роль «{normalized}» не поддерживается: используй одну из {}",
            super::ELEMENT_ROLES.join(", ")
        ));
    }
    block_on(async {
        let conn = connect().await?;
        let (target, _) = resolve_app(&conn, app).await?;
        let zconn = conn.connection().clone();
        let root = resolve_window(&conn, &target, window).await?;

        // Точное совпадение имени, затем вхождение — как name is →
        // description → title в click_script.
        let tree = walk(&root, &zconn).await?;
        let node = tree
            .iter()
            .find(|(n, _)| roles.contains(&n.role.as_str()) && name_matches(&n.name, "", name))
            .map(|(n, _)| n.clone())
            .ok_or_else(|| {
                anyhow!("{normalized} «{name}» не найден в приложении {}", target.name)
            })?;

        let detail = perform_click(&zconn, &node, name).await?;
        Ok(format!(
            "нажато: {normalized} «{name}» в приложении {} ({detail})",
            target.name
        ))
    })
}

/// Вписать значение в текстовое поле, вернуть фактическое значение поля
/// после записи (честный readback для верификации). Отправка Enter —
/// действием «activate» поля (GTK/Qt), иначе клавишей через desktop-слой.
pub fn set_field_value(
    app: Option<&str>,
    role: &str,
    name: &str,
    text: &str,
    submit: bool,
    window: Option<&str>,
) -> Result<String> {
    ensure_not_cancelled()?;
    let normalized = super::normalize_role(role);
    if normalized != "text field" && normalized != "secure text field" {
        return Err(anyhow!(
            "вписывать текст можно только в поле, получена роль «{normalized}»"
        ));
    }
    block_on(async {
        let conn = connect().await?;
        let (target, _) = resolve_app(&conn, app).await?;
        let zconn = conn.connection().clone();
        let root = resolve_window(&conn, &target, window).await?;

        let tree = walk(&root, &zconn).await?;
        let roles = atspi_roles(&normalized);
        let node = tree
            .iter()
            .find(|(n, _)| roles.contains(&n.role.as_str()) && name_matches(&n.name, "", name))
            .map(|(n, _)| n.clone())
            .ok_or_else(|| anyhow!("поле «{name}» не найдено в приложении {}", target.name))?;

        let editable = editable(&zconn, &node.dest, &node.path)
            .await
            .map_err(|e| anyhow!("поле «{name}» недоступно для записи: {e}"))?;
        let written = editable
            .set_text_contents(text)
            .await
            .map_err(|e| anyhow!("запись в поле не удалась: {e}"))?;
        if !written {
            return Err(anyhow!("поле «{name}» отклонило запись значения"));
        }
        if submit {
            submit_entry(&node, &zconn);
        }
        // Readback — паритет с type_script: возвращаем то, что реально
        // в поле, а не то, что просили вписать.
        let actual = read_text(&node, &zconn).await.unwrap_or_default();
        Ok(actual)
    })
}

/// Отправить ввод: сначала действие «activate» самого поля (работает без
/// xdotool/ydotool), затем клавиша Enter через desktop-слой.
fn submit_entry(node: &Node, conn: &zbus::Connection) {
    let activated = block_on(async {
        let action = action(conn, &node.dest, &node.path).await?;
        for (idx, a) in action.get_actions().await?.iter().enumerate() {
            if a.name.eq_ignore_ascii_case("activate")
                && action.do_action(idx as i32).await.unwrap_or(false)
            {
                return Ok(());
            }
        }
        Err(anyhow!("нет действия activate"))
    });
    if activated.is_ok() {
        return;
    }
    // Запасной путь — синтетическая клавиша. Ошибку не поднимаем: значение
    // уже в поле, вызывающий увидит это по readback.
    let _ = crate::platform::desktop::execute(
        &crate::platform::desktop::DesktopAction::PressKey {
            key: "enter".into(),
        },
    );
}

/// Клик по элементу: сначала его собственное действие click/press (GTK3,
/// Qt), затем синтетический клик в центр по координатам Component (GTK4
/// отдаёт свои имена действий — slot.reload, clipboard.copy — и без
/// координатного клика они нажимали бы не то).
async fn perform_click(conn: &zbus::Connection, node: &Node, name: &str) -> Result<String> {
    if let Ok(ap) = action(conn, &node.dest, &node.path).await {
        if let Ok(actions) = ap.get_actions().await {
            if let Some(idx) = actions
                .iter()
                .position(|a| a.name.eq_ignore_ascii_case("click") || a.name.eq_ignore_ascii_case("press"))
                && ap.do_action(idx as i32).await.unwrap_or(false)
            {
                return Ok("собственное действие элемента".into());
            }
        }
    }
    let cp = component(conn, &node.dest, &node.path)
        .await
        .map_err(|e| anyhow!("элемент «{name}» не поддерживает ни действий, ни координат: {e}"))?;
    let (x, y, w, h) = cp
        .get_extents(CoordType::Screen)
        .await
        .map_err(|e| anyhow!("не удалось получить координаты «{name}»: {e}"))?;
    if w <= 0 || h <= 0 {
        return Err(anyhow!("у «{name}» нулевой размер: кликнуть некуда"));
    }
    crate::platform::desktop::execute(&crate::platform::desktop::DesktopAction::Click {
        x: x + w / 2,
        y: y + h / 2,
        button: crate::platform::desktop::MouseButton::Left,
        count: 1,
    })
    .map(|_| "синтетический клик по координатам".to_string())
    .map_err(|e| anyhow!("{e}"))
}

/// Выбрать пункт меню: цепочка «Файл → Сохранить» — открыть каждый уровень
/// и нажать последний пункт (паритет menu_chain: 1–3 пункта). Подменю у GTK
/// наполняются лениво, поэтому перед следующим уровнем дерево перечитывается
/// из открытого пункта.
pub fn menu_path(app: Option<&str>, path: &[&str]) -> Result<String> {
    ensure_not_cancelled()?;
    if path.is_empty() || path.len() > 3 {
        return Err(anyhow!("путь меню: 1–3 пункта, получено {}", path.len()));
    }
    block_on(async {
        let conn = connect().await?;
        let (target, _) = resolve_app(&conn, app).await?;
        let zconn = conn.connection().clone();
        let root = resolve_window(&conn, &target, None).await?;

        let mut tree = walk(&root, &zconn).await?;
        let mut matched = String::new();
        for (depth, want) in path.iter().enumerate() {
            let want = want.trim();
            if want.is_empty() {
                return Err(anyhow!("пустой пункт меню в пути"));
            }
            let roles: &[&str] = if depth == 0 {
                &["menu item", "menu"]
            } else {
                &["menu item", "check menu item", "radio menu item"]
            };
            let node = tree
                .iter()
                .find(|(n, _)| roles.contains(&n.role.as_str()) && name_matches(&n.name, "", want))
                .map(|(n, _)| n.clone())
                .ok_or_else(|| {
                    anyhow!("пункт меню «{want}» не найден в приложении {}", target.name)
                })?;

            let action = action(&zconn, &node.dest, &node.path)
                .await
                .map_err(|e| anyhow!("пункт «{want}» не содержит действий: {e}"))?;
            let done = action
                .do_action(0)
                .await
                .map_err(|e| anyhow!("нажатие «{want}» не удалось: {e}"))?;
            if !done {
                return Err(anyhow!("приложение отклонило пункт меню «{want}»"));
            }
            matched = if matched.is_empty() {
                want.to_string()
            } else {
                format!("{matched} → {want}")
            };
            if depth + 1 < path.len() {
                tokio::time::sleep(std::time::Duration::from_millis(MENU_SETTLE_MS)).await;
                tree = walk(&node, &zconn).await?;
            }
        }
        Ok(format!(
            "выбран пункт меню {matched} в приложении {}",
            target.name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_translate_per_os_dictionary() {
        assert_eq!(atspi_roles("button"), &["push button"]);
        assert_eq!(atspi_roles("text field"), &["entry", "text", "terminal"]);
        assert_eq!(atspi_roles("secure text field"), &["password text"]);
        assert!(atspi_roles("неизвестная роль").is_empty());
    }

    #[test]
    fn name_matching_exact_then_contains() {
        assert!(name_matches("Продолжить", "", "Продолжить"));
        // Регистр не важен — как `is` в AppleScript.
        assert!(name_matches("Search", "", "search"));
        // Вхождение: модель иногда пишет имя неполностью.
        assert!(name_matches("Файл: Сохранить всё", "", "Сохранить"));
        // Описание — запасное поле для полей без имени.
        assert!(name_matches("", "Введите поисковый запрос", "поисковый"));
        assert!(!name_matches("Кнопка", "", "Совсем другое"));
        assert!(!name_matches("Кнопка", "", "  "));
    }
}
