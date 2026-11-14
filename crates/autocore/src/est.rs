//! A2. EST — Event-Sourced Tree: теневое дерево в памяти, патчи только по
//! нативным событиям. В steady state ни одного повторного обхода дерева:
//! первый заход — bounded capture, дальше дерево живёт патчами.
//!
//! Предохранители (обязательные по спецификации):
//! - подписка может терять события (`Subscription::dropped`) — при потере
//!   всё поддерево инвалидируется, а не «угадывается»;
//! - дыры в покрытии событий закрываются `dirty_set`: затронутые узлы
//!   дочитываются ПЕРЕД действием, а не по таймеру;
//! - патчи коалесцируют в окне ~16 мс — шторм анимации не съедает CPU.

use crate::simhash::SimHash;
use std::collections::{HashMap, HashSet};
use xa11y::Role;

/// Идентификатор узла в теневом дереве — SMI (см. [`crate::smi`]).
pub type Smi = u64;

/// Один узел теневого дерева: то, что нужно для поиска и действий.
#[derive(Debug, Clone)]
pub struct ShadowNode {
    pub smi: Smi,
    pub role: Role,
    /// Нормализованное имя («» когда имени нет).
    pub name: String,
    pub depth: usize,
    pub parent: Option<Smi>,
    pub children: Vec<Smi>,
    /// Сколько детей у нативного узла, включая не дочитанных.
    pub children_count: Option<usize>,
    pub children_truncated: bool,
}

/// Теневое дерево.
#[derive(Debug, Default)]
pub struct Est {
    nodes: HashMap<Smi, ShadowNode>,
    roots: Vec<Smi>,
    /// Узлы, чьи поддеревья надо перечитать перед следующим действием.
    dirty: HashSet<Smi>,
    /// Инкрементальный SimHash: вклад каждого узла уже учтён.
    sim: SimHash,
    /// Прошлый отпечаток для сравнения «экран изменился?».
    last_gate: Option<crate::simhash::Fingerprint>,
    /// Число коллизий SMI, разрешённых дискриминатором (логируются).
    collisions: u64,
}

/// Патч по нативному событию.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Patch {
    /// Поддерево перечитать лениво, перед следующим действием.
    SubtreeDirty(Smi),
    /// Имя (или метка) узла изменилось.
    Name(Smi, String),
    /// Булев флаг состояния изменился (enabled, checked, …).
    State(Smi, xa11y::StateFlag, bool),
    /// Структура детей узла изменилась.
    Children(Smi),
    /// Узел исчез.
    Gone(Smi),
}

/// Очередь патчей с коалесцированием: патчи, пришедшие в одном окне
/// ~16 мс, схлопываются — «Name A → Name A» и «Children A» трижды
/// превращаются в один грязный узел.
#[derive(Debug, Default)]
pub struct PatchQueue {
    pending: Vec<Patch>,
    window_opened_at: Option<std::time::Instant>,
}

/// Окно коалесцирования: один кадр дисплея.
pub const COALESCE_WINDOW: std::time::Duration = std::time::Duration::from_millis(16);

impl PatchQueue {
    pub fn push(&mut self, patch: Patch) {
        let now = std::time::Instant::now();
        match self.window_opened_at {
            Some(t) if now.duration_since(t) <= COALESCE_WINDOW => {}
            _ => self.window_opened_at = Some(now),
        }
        self.pending.push(patch);
    }

    /// Забрать схлопнутые патчи: очередь опустошается, повторные патчи на
    /// тот же узел сводятся к одному.
    pub fn drain(&mut self) -> Vec<Patch> {
        let mut out: Vec<Patch> = Vec::new();
        let mut seen_dirty: HashSet<Smi> = HashSet::new();
        let mut seen_children: HashSet<Smi> = HashSet::new();
        let mut names: HashMap<Smi, String> = HashMap::new();
        let mut gone: HashSet<Smi> = HashSet::new();
        let mut states: HashMap<(Smi, xa11y::StateFlag), bool> = HashMap::new();
        for p in std::mem::take(&mut self.pending) {
            match p {
                Patch::SubtreeDirty(s) => {
                    if seen_dirty.insert(s) {
                        out.push(Patch::SubtreeDirty(s));
                    }
                }
                Patch::Children(s) => {
                    if seen_children.insert(s) {
                        out.push(Patch::Children(s));
                    }
                }
                Patch::Name(s, name) => {
                    // Последнее имя в окне — единственное, что увидит модель.
                    names.insert(s, name);
                }
                Patch::State(s, flag, val) => {
                    states.insert((s, flag), val);
                }
                Patch::Gone(s) => {
                    gone.insert(s);
                }
            }
        }
        // Имена и состояния выдают по последнему значению на узел.
        for (s, name) in names {
            out.push(Patch::Name(s, name));
        }
        for ((s, flag), val) in states {
            out.push(Patch::State(s, flag, val));
        }
        // Gone сильнее любого другого патча узла: узла больше нет.
        out.retain(|p| match p {
            Patch::SubtreeDirty(s) | Patch::Children(s) | Patch::Name(s, _) | Patch::State(s, _, _)
                if gone.contains(s) =>
            {
                false
            }
            _ => true,
        });
        for s in gone {
            out.push(Patch::Gone(s));
        }
        out
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// Потеря событий подпиской: буфер переполнился — инвалидируем всё,
/// гадать нельзя.
pub fn invalidation_on_dropped_events(est: &mut Est) {
    let roots: Vec<Smi> = est.roots.clone();
    for r in roots {
        est.apply(Patch::SubtreeDirty(r));
    }
}

/// Маппинг нативных событий в патчи: ровно те виды, что даёт
/// `EventKind` (см. спецификацию §2).
pub fn patch_for_event(kind: xa11y::EventKind, target: Smi) -> Option<Patch> {
    match kind {
        xa11y::EventKind::StructureChanged => Some(Patch::Children(target)),
        xa11y::EventKind::NameChanged => None, // имя приходит в событии, патч соберёт водитель
        xa11y::EventKind::ValueChanged => Some(Patch::SubtreeDirty(target)),
        xa11y::EventKind::StateChanged { .. } => Some(Patch::SubtreeDirty(target)),
        xa11y::EventKind::WindowOpened | xa11y::EventKind::WindowClosed => {
            Some(Patch::Children(target))
        }
        xa11y::EventKind::FocusChanged => None, // фокус не меняет дерево
        _ => None,
    }
}

impl Est {
    pub fn new() -> Self {
        Self::default()
    }

    /// Установить теневое дерево из capture (первый заход). Перезаписывает
    /// всё: инкрементальный SimHash пересчитывается добавлением узлов.
    pub fn install(&mut self, shadow: Shadow) {
        self.nodes.clear();
        self.roots = shadow.roots.clone();
        self.dirty.clear();
        self.sim = SimHash::new();
        for (smi, node) in shadow.nodes {
            self.sim
                .add_node(&format!("{:?}", node.role), &node.name);
            self.nodes.insert(smi, node);
        }
    }

    /// Число узлов в тени.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, smi: Smi) -> Option<&ShadowNode> {
        self.nodes.get(&smi)
    }

    pub fn roots(&self) -> &[Smi] {
        &self.roots
    }

    /// Узлы, требующие перечитывания перед действием.
    pub fn dirty(&self) -> &HashSet<Smi> {
        &self.dirty
    }

    /// Число разрешённых коллизий SMI.
    pub fn collisions(&self) -> u64 {
        self.collisions
    }

    /// Применить патч события. Возвращает true, если патч изменил дерево.
    pub fn apply(&mut self, patch: Patch) -> bool {
        match patch {
            Patch::SubtreeDirty(s) => self.dirty.insert(s),
            Patch::Children(s) => self.dirty.insert(s),
            Patch::Name(s, name) => {
                let Some(node) = self.nodes.get_mut(&s) else {
                    return false;
                };
                if node.name == name {
                    return false;
                }
                // Вклад старых шинглов снят, новых добавлен.
                self.sim
                    .remove_node(&format!("{:?}", node.role), &node.name);
                self.sim.add_node(&format!("{:?}", node.role), &name);
                node.name = name;
                true
            }
            Patch::State(_s, _flag, _val) => {
                // Состояния в тени этапа 1 не хранятся: патч отмечает узел
                // как требующий перечитывания перед действием.
                self.dirty.insert(_s)
            }
            Patch::Gone(s) => {
                let Some(node) = self.nodes.remove(&s) else {
                    return false;
                };
                self.sim
                    .remove_node(&format!("{:?}", node.role), &node.name);
                // Дети Gone-узла тоже исчезают (нативное дерево рвётся
                // целиком; дочитывание при перечитывании родителей вернёт
                // выживших под новыми SMI).
                let mut stack = node.children.clone();
                while let Some(c) = stack.pop() {
                    if let Some(child) = self.nodes.remove(&c) {
                        self.sim
                            .remove_node(&format!("{:?}", child.role), &child.name);
                        stack.extend(child.children);
                    }
                }
                if let Some(parent) = node.parent
                    && let Some(p) = self.nodes.get_mut(&parent)
                {
                    p.children.retain(|c| *c != s);
                    p.children_truncated = true;
                    self.dirty.insert(parent);
                } else {
                    self.roots.retain(|r| *r != s);
                }
                true
            }
        }
    }

    /// «Экран изменился?» — дистанция Хэмминга между текущим отпечатком
    /// и прошлой проверкой. None — первой проверки ещё не было.
    pub fn gate(&mut self) -> Option<u32> {
        let fp = self.sim.fingerprint();
        let d = self.last_gate.map(|last| crate::simhash::hamming(last, fp));
        self.last_gate = Some(fp);
        d
    }
}

/// Теневое дерево, готовое к установке в Est.
#[derive(Debug, Default)]
pub struct Shadow {
    pub nodes: HashMap<Smi, ShadowNode>,
    pub roots: Vec<Smi>,
}


#[cfg(test)]
mod tests {
    use super::*;

    fn nd(smi: u64, role: Role, name: &str, parent: Option<u64>, children: Vec<u64>) -> ShadowNode {
        ShadowNode {
            smi,
            role,
            name: name.to_string(),
            depth: 0,
            parent,
            children_count: Some(children.len()),
            children,
            children_truncated: false,
        }
    }

    fn tree_with_two_rows() -> Shadow {
        let mut nodes = HashMap::new();
        // window(1) → list(2) → row(3), row(4)
        nodes.insert(1, nd(1, Role::Window, "Главное", None, vec![2]));
        nodes.insert(2, nd(2, Role::List, "Список", Some(1), vec![3, 4]));
        nodes.insert(3, nd(3, Role::TableRow, "Строка А", Some(2), vec![]));
        nodes.insert(4, nd(4, Role::TableRow, "Строка Б", Some(2), vec![]));
        Shadow {
            nodes,
            roots: vec![1],
        }
    }

    #[test]
    fn install_counts_all_nodes() {
        let mut est = Est::new();
        est.install(tree_with_two_rows());
        assert_eq!(est.len(), 4);
        assert!(est.dirty().is_empty());
    }

    #[test]
    fn patch_queue_coalesces_names_and_children() {
        let mut q = PatchQueue::default();
        q.push(Patch::Name(3, "первый".into()));
        q.push(Patch::Name(3, "второй".into()));
        q.push(Patch::Children(2));
        q.push(Patch::Children(2));
        q.push(Patch::Gone(3));
        let drained = q.drain();
        // Gone глушит прочие патчи узла; Children(2) схлопнулся в один.
        let children = drained
            .iter()
            .filter(|p| matches!(p, Patch::Children(2)))
            .count();
        assert_eq!(children, 1);
        assert!(drained.iter().any(|p| matches!(p, Patch::Gone(3))));
        assert!(!drained.iter().any(|p| matches!(p, Patch::Name(_, _))));
    }

    #[test]
    fn name_patch_updates_simhash_incrementally() {
        let mut est = Est::new();
        est.install(tree_with_two_rows());
        let before = est.sim.fingerprint();
        assert!(est.apply(Patch::Name(3, "Строка А2".into())));
        let after = est.sim.fingerprint();
        assert_ne!(before, after);
        // Повтор того же имени — не изменение.
        assert!(!est.apply(Patch::Name(3, "Строка А2".into())));
        assert_eq!(after, est.sim.fingerprint());
    }

    #[test]
    fn gone_removes_subtree_and_marks_parent_dirty() {
        let mut est = Est::new();
        est.install(tree_with_two_rows());
        assert!(est.apply(Patch::Gone(3)));
        assert!(est.node(3).is_none());
        let parent = est.node(2).expect("родитель жив");
        assert!(!parent.children.contains(&3));
        assert!(est.dirty().contains(&2));
    }

    #[test]
    fn dropped_events_invalidate_everything() {
        let mut est = Est::new();
        est.install(tree_with_two_rows());
        assert!(est.dirty().is_empty());
        invalidation_on_dropped_events(&mut est);
        // Корень в грязных: перед действием всё перечитается, ничего
        // не «угадывается».
        assert!(est.dirty().contains(&1));
    }

    #[test]
    fn gate_sees_change_after_label_patch() {
        let mut est = Est::new();
        est.install(tree_with_two_rows());
        // Первая проверка — база.
        assert!(est.gate().is_none());
        assert_eq!(est.gate(), Some(0));
        est.apply(Patch::Name(4, "Совсем другая строка".into()));
        let d = est.gate().expect("изменение видно");
        assert!(d > 0, "смена имени обязана менять отпечаток");
    }
}
