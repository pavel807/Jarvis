//! Фейковый десктоп: синтетическое дерево Provider'а над памятью.
//!
//! Обязательное требование спецификации (§10): все алгоритмы тестируются
//! без реальной ОС — иначе CI не проверит три платформы. Дерево мутабельное
//! (приложение «обновилось», «открылся диалог», «строка исчезла»), действия
//! логируются, чтения можно считать.

use std::sync::{Arc, Mutex};
use xa11y::Error;
use xa11y::{Element, ElementData, Role, ShellSurfaceKind};

/// Узел синтетического дерева.
#[derive(Debug, Clone)]
struct FakeNode {
    role: Role,
    name: String,
    value: Option<String>,
    children: Vec<usize>,
    parent: Option<usize>,
    /// Узел «закрыт приложением»: присутствует, но дети/identity рвутся.
    removed: bool,
    focused: bool,
}

/// Поставщик синтетического дерева. Один инстанс = одно «приложение».
pub struct FakeProvider {
    nodes: Mutex<Vec<FakeNode>>,
    pid: u32,
    app_name: String,
    /// Лог действий: проверяем, что действие дошло до нужного узла.
    actions: Mutex<Vec<String>>,
    /// Счётчик чтений детей: бенч холодного и тёплого хода.
    children_reads: std::sync::atomic::AtomicUsize,
}

impl FakeProvider {
    fn node_from(&self, nodes: &std::sync::MutexGuard<'_, Vec<FakeNode>>, idx: usize) -> ElementData {
        let n = &nodes[idx];
        let mut d = ElementData::for_role(n.role);
        d.name = (!n.name.is_empty()).then(|| n.name.clone());
        d.value = n.value.clone();
        d.pid = Some(self.pid);
        d.stable_id = Some(format!("{}-{}", self.app_name, idx));
        d.handle = (idx + 1) as u64;
        d
    }

    fn idx_of(&self, element: &ElementData) -> Option<usize> {
        // handle назначает сам поставщик: idx+1.
        let h = element.handle;
        (h > 0).then(|| (h - 1) as usize)
    }

    fn log_action(&self, what: &str, element: &ElementData) {
        self.actions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("{what}@{}", element.stable_id.as_deref().unwrap_or("?")));
    }
}

/// Десктоп: точка входа для тестов и бенчей.
pub struct FakeDesktop {
    provider: Arc<FakeProvider>,
}

impl FakeDesktop {
    /// Простой экран: окно с тулбаром и `content_nodes` строками списка.
    pub fn new() -> Self {
        let mut d = Self::empty();
        d.build_simple_app("TestApp", 8);
        d
    }

    fn empty() -> Self {
        Self {
            provider: Arc::new(FakeProvider {
                nodes: Mutex::new(Vec::new()),
                pid: 4242,
                app_name: "TestApp".into(),
                actions: Mutex::new(Vec::new()),
                children_reads: std::sync::atomic::AtomicUsize::new(0),
            }),
        }
    }

    fn build_simple_app(&mut self, name: &str, list_rows: usize) {
        let p = Arc::get_mut(&mut self.provider).expect("провайдер ещё не расшарен");
        p.app_name = name.to_string();
        let mut nodes = p.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes.clear();
        let app = FakeNode {
            role: Role::Application,
            name: name.to_string(),
            value: None,
            children: vec![1],
            parent: None,
            removed: false,
            focused: false,
        };
        let window = FakeNode {
            role: Role::Window,
            name: format!("{name} — главное окно"),
            value: None,
            children: vec![2, 3, 4],
            parent: Some(0),
            removed: false,
            focused: true,
        };
        let toolbar = FakeNode {
            role: Role::Toolbar,
            name: "Панель инструментов".to_string(),
            value: None,
            children: vec![5, 6],
            parent: Some(1),
            removed: false,
            focused: false,
        };
        let mut list_children = Vec::new();
        for i in 0..list_rows {
            list_children.push(7 + i);
        }
        let list = FakeNode {
            role: Role::List,
            name: "Записи".to_string(),
            value: None,
            children: list_children.clone(),
            parent: Some(1),
            removed: false,
            focused: false,
        };
        let status = FakeNode {
            role: Role::StaticText,
            name: format!("Всего {list_rows} записей"),
            value: None,
            children: vec![],
            parent: Some(1),
            removed: false,
            focused: false,
        };
        let back = FakeNode {
            role: Role::Button,
            name: "Назад".to_string(),
            value: None,
            children: vec![],
            parent: Some(2),
            removed: false,
            focused: false,
        };
        let save = FakeNode {
            role: Role::Button,
            name: "Сохранить".to_string(),
            value: None,
            children: vec![],
            parent: Some(2),
            removed: false,
            focused: false,
        };
        nodes.push(app);
        nodes.push(window);
        nodes.push(toolbar);
        nodes.push(list);
        nodes.push(status);
        nodes.push(back);
        nodes.push(save);
        for i in 0..list_rows {
            nodes.push(FakeNode {
                role: Role::ListItem,
                name: format!("Запись {i} — текст строки списка"),
                value: None,
                children: vec![],
                parent: Some(3),
                removed: false,
                focused: false,
            });
        }
    }

    /// Открыть ещё одно «приложение» с `list_rows` строками.
    pub fn open_app(&mut self, name: &str, list_rows: usize) -> Element {
        self.build_simple_app(name, list_rows);
        self.root_element()
    }

    /// Плотное приложение: ветвление `branch`, глубина `depth` — дерево
    /// на сотни и тысячи узлов для бенчей и тестов бюджета.
    pub fn open_app_dense(&mut self, name: &str, branch: usize, depth: usize) -> Element {
        let p = Arc::get_mut(&mut self.provider).expect("провайдер ещё не расшарен");
        p.app_name = name.to_string();
        let mut nodes = p.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes.clear();
        // Breadth-first: корень, уровни по branch детей до depth.
        nodes.push(FakeNode {
            role: Role::Application,
            name: name.to_string(),
            value: None,
            children: Vec::new(),
            parent: None,
            removed: false,
            focused: false,
        });
        let mut level: Vec<usize> = vec![0];
        for d in 1..=depth {
            let mut next_level = Vec::new();
            for parent in level.drain(..) {
                let role = if d == depth { Role::StaticText } else { Role::Group };
                for _ in 0..branch {
                    let idx = nodes.len();
                    nodes.push(FakeNode {
                        role,
                        name: format!("{name}-узел {idx}"),
                        value: None,
                        children: Vec::new(),
                        parent: Some(parent),
                        removed: false,
                        focused: false,
                    });
                    nodes[parent].children.push(idx);
                    next_level.push(idx);
                }
            }
            level = next_level;
        }
        drop(nodes);
        self.root_element()
    }

    fn root_element(&self) -> Element {
        let nodes = self
            .provider
            .nodes
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let data = self.provider.node_from(&nodes, 0);
        Element::new(data, Arc::clone(&self.provider) as Arc<dyn xa11y::Provider>)
    }

    /// Элемент по имени узла (для мутаций из тестов).
    pub fn element_named(&self, name: &str) -> Option<Element> {
        let nodes = self.provider.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let idx = nodes.iter().position(|n| n.name == name)?;
        let data = self.provider.node_from(&nodes, idx);
        Some(Element::new(
            data,
            Arc::clone(&self.provider) as Arc<dyn xa11y::Provider>,
        ))
    }

    /// Приложение «обновилось»: у узла другое имя (элемент протух семантически).
    pub fn rename(&self, old: &str, new: &str) -> bool {
        let mut nodes = self.provider.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let Some(idx) = nodes.iter().position(|n| n.name == old) else {
            return false;
        };
        nodes[idx].name = new.to_string();
        true
    }

    /// Приложение уничтожило узел.
    pub fn remove(&self, name: &str) -> bool {
        let mut nodes = self.provider.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let Some(idx) = nodes.iter().position(|n| n.name == name) else {
            return false;
        };
        nodes[idx].removed = true;
        true
    }

    /// Открылся модальный диалог: добавлен поддерево к окну.
    pub fn open_dialog(&self, title: &str, buttons: &[&str]) {
        let mut nodes = self.provider.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let window = nodes.iter().position(|n| n.role == Role::Window).expect("окно есть");
        let dialog_idx = nodes.len();
        nodes.push(FakeNode {
            role: Role::Dialog,
            name: title.to_string(),
            value: None,
            children: Vec::new(),
            parent: Some(window),
            removed: false,
            focused: true,
        });
        let mut children = Vec::new();
        for b in buttons {
            let idx = nodes.len();
            nodes.push(FakeNode {
                role: Role::Button,
                name: (*b).to_string(),
                value: None,
                children: vec![],
                parent: Some(dialog_idx),
                removed: false,
                focused: false,
            });
            children.push(idx);
        }
        nodes[dialog_idx].children = children;
        nodes[window].children.push(dialog_idx);
    }

    /// Лог действий (что и куда дошло).
    pub fn actions(&self) -> Vec<String> {
        self.provider
            .actions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Сколько раз читались дети: метрика тёплого и холодного хода.
    pub fn children_reads(&self) -> usize {
        self.provider
            .children_reads
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn provider(&self) -> Arc<dyn xa11y::Provider> {
        Arc::clone(&self.provider) as Arc<dyn xa11y::Provider>
    }
}

impl Default for FakeDesktop {
    fn default() -> Self {
        Self::new()
    }
}

impl xa11y::Provider for FakeProvider {
    fn get_children(&self, element: Option<&ElementData>) -> Result<Vec<ElementData>, Error> {
        self.children_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let idx = match element {
            None => return Ok(vec![self.node_from(&nodes, 0)]),
            Some(e) => self.idx_of(e),
        };
        let Some(idx) = idx else {
            return Err(Error::invalid_action_data("фейковый узел вне дерева"));
        };
        if nodes[idx].removed {
            return Err(Error::stale(
                xa11y::StaleReason::Removed,
                format!("узел {} закрыт приложением", nodes[idx].name),
            ));
        }
        Ok(nodes[idx]
            .children
            .iter()
            .filter(|c| !nodes[**c].removed)
            .map(|c| self.node_from(&nodes, *c))
            .collect())
    }

    fn get_parent(&self, element: &ElementData) -> Result<Option<ElementData>, Error> {
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        let Some(idx) = self.idx_of(element) else {
            return Err(Error::invalid_action_data("фейковый узел вне дерева"));
        };
        Ok(nodes[idx].parent.map(|p| self.node_from(&nodes, p)))
    }

    fn list_apps(&self) -> Result<Vec<ElementData>, Error> {
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        Ok(vec![self.node_from(&nodes, 0)])
    }

    fn app_by_pid(&self, pid: u32) -> Result<ElementData, Error> {
        if pid == self.pid {
            let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
            Ok(self.node_from(&nodes, 0))
        } else {
            Err(Error::invalid_action_data("в фейке одно приложение"))
        }
    }

    fn focused_app(&self) -> Result<ElementData, Error> {
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        Ok(self.node_from(&nodes, 0))
    }

    fn list_shell_surfaces(&self) -> Result<Vec<(ShellSurfaceKind, ElementData)>, Error> {
        // У фейка нет оболочки: честная пустота, не выдумка.
        Ok(Vec::new())
    }

    fn press(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("press", element);
        Ok(())
    }

    fn focus(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("focus", element);
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = self.idx_of(element) {
            for n in nodes.iter_mut() {
                n.focused = false;
            }
            nodes[i].focused = true;
        }
        Ok(())
    }

    fn blur(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("blur", element);
        Ok(())
    }

    fn toggle(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("toggle", element);
        Ok(())
    }

    fn select(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("select", element);
        Ok(())
    }

    fn expand(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("expand", element);
        Ok(())
    }

    fn collapse(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("collapse", element);
        Ok(())
    }

    fn show_menu(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("show_menu", element);
        Ok(())
    }

    fn increment(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("increment", element);
        Ok(())
    }

    fn decrement(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("decrement", element);
        Ok(())
    }

    fn scroll_into_view(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("scroll_into_view", element);
        Ok(())
    }

    fn set_value(&self, element: &ElementData, value: &str) -> Result<(), Error> {
        self.log_action(&format!("set_value={value}"), element);
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = self.idx_of(element) {
            nodes[i].value = Some(value.to_string());
            nodes[i].name = value.to_string();
        }
        Ok(())
    }

    fn set_numeric_value(&self, element: &ElementData, value: f64) -> Result<(), Error> {
        self.log_action(&format!("set_numeric_value={value}"), element);
        Ok(())
    }

    fn type_text(&self, element: &ElementData, text: &str) -> Result<(), Error> {
        self.log_action(&format!("type_text={text}"), element);
        Ok(())
    }

    fn set_text_selection(&self, element: &ElementData, _start: u32, _end: u32) -> Result<(), Error> {
        self.log_action("set_text_selection", element);
        Ok(())
    }

    fn perform_action(&self, element: &ElementData, action: &str) -> Result<(), Error> {
        self.log_action(&format!("perform_action={action}"), element);
        Ok(())
    }

    fn activate(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("activate", element);
        Ok(())
    }

    fn minimize(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("minimize", element);
        Ok(())
    }

    fn maximize(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("maximize", element);
        Ok(())
    }

    fn enter_fullscreen(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("enter_fullscreen", element);
        Ok(())
    }

    fn restore(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("restore", element);
        Ok(())
    }

    fn close(&self, element: &ElementData) -> Result<(), Error> {
        self.log_action("close", element);
        Ok(())
    }

    fn move_to(&self, element: &ElementData, _x: i32, _y: i32) -> Result<(), Error> {
        self.log_action("move_to", element);
        Ok(())
    }

    fn resize_to(&self, element: &ElementData, _width: u32, _height: u32) -> Result<(), Error> {
        self.log_action("resize_to", element);
        Ok(())
    }

    fn table_cell_at(
        &self,
        _table: &ElementData,
        _row: u32,
        _column: u32,
    ) -> Result<Option<ElementData>, Error> {
        Err(Error::Unsupported {
            feature: "table_cell_at в фейке".into(),
        })
    }

    fn subscribe(&self, _element: &ElementData) -> Result<xa11y::Subscription, Error> {
        // Этап 1: событийная доставка в фейке не поднята — честный отказ,
        // тесты водят патчи напрямую (см. est::tests).
        Err(Error::Unsupported {
            feature: "события в фейке".into(),
        })
    }

    fn identity(&self, element: &ElementData) -> Result<xa11y::identity::NodeIdentity, Error> {
        // handle назначает сам поставщик (idx+1): identity без замка.
        let key = format!("{}-{}", self.app_name, element.handle - 1);
        Ok(xa11y::identity::NodeIdentity::from_key(
            "fake",
            self.pid.to_string(),
            key,
        ))
    }

    fn capabilities(&self) -> Result<xa11y::Capabilities, Error> {
        // Честный ответ о фейке: бэкенд «fake».
        Err(Error::Unsupported {
            feature: "capabilities в фейке".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_tree_walks_and_actions_log() {
        let desktop = FakeDesktop::new();
        let root = desktop.element_named("TestApp").expect("корень есть");
        let capture = root.capture(&crate::capture::default_options()).expect("capture");
        assert!(capture.nodes.len() >= 10, "узлы читаются");

        // Действие доходит и логируется.
        let save = desktop.element_named("Сохранить").expect("кнопка есть");
        save.press().expect("нажатие прошло");
        assert!(desktop.actions().iter().any(|a| a.starts_with("press@")), "{:?}", desktop.actions());
    }

    #[test]
    fn mutations_change_subsequent_captures() {
        let desktop = FakeDesktop::new();
        let before = desktop
            .element_named("TestApp")
            .expect("корень")
            .capture(&crate::capture::default_options())
            .expect("capture");
        let names_before: Vec<_> = before
            .nodes
            .iter()
            .filter_map(|n| n.element.name.clone())
            .collect();
        assert!(names_before.iter().any(|n| n.contains("Запись 0")));

        // Строка исчезла, диалог открылся.
        assert!(desktop.remove("Запись 0 — текст строки списка"));
        desktop.open_dialog("Удалить запись?", &["Удалить", "Отмена"]);
        let after = desktop
            .element_named("TestApp")
            .expect("корень")
            .capture(&crate::capture::default_options())
            .expect("capture");
        let names_after: Vec<_> = after
            .nodes
            .iter()
            .filter_map(|n| n.element.name.clone())
            .collect();
        assert!(!names_after.iter().any(|n| n.contains("Запись 0")));
        assert!(names_after.iter().any(|n| n == "Удалить запись?"));
    }

    #[test]
    fn dense_tree_reaches_thousand_nodes() {
        let mut desktop = FakeDesktop::new();
        let root = desktop.open_app_dense("Плотный", 6, 5);
        // 6+36+216+1296+7776 = 9330 узлов — плотное приложение с запасом.
        let mut wide = crate::capture::default_options();
        wide.max_depth = None;
        wide.max_nodes = 100_000;
        let capture = root.capture(&wide).expect("capture плотного дерева");
        assert!(capture.nodes.len() >= 1000, "получили {}", capture.nodes.len());
    }
}
