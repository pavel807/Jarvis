//! Bounded capture: единственное место, где autocore читает нативное
//! дерево. Всё остальное живёт в тени ([`crate::est`]) и патчах.
//!
//! Бюджеты задают жёсткие границы чтения: глубина, число узлов, байты
//! текста, время. Что обрезано — честно видно в `Capture::truncated` и
//! `children_truncated`, ничего не домысливается.

use crate::est::{Shadow, ShadowNode, Smi};
use crate::smi::{self, PathSeg};
use std::collections::HashMap;
use xa11y::Element;

/// Бюджеты первого захода: достаточно, чтобы увидеть экран, мало — чтобы
/// не утонуть в плотном приложении.
pub fn default_options() -> xa11y::capture::CaptureOptions {
    // CaptureOptions #[non_exhaustive]: конструкция только через Default.
    let mut o = xa11y::capture::CaptureOptions::default();
    o.max_depth = Some(12);
    o.max_nodes = 600;
    o.max_text_bytes = 4096;
    o.text = false;
    o.relations = false;
    o.formatting = false;
    o.timeout = Some(std::time::Duration::from_secs(2));
    o
}

/// Бюджеты точечного перечитывания поддерева (тёплый путь EST).
pub fn narrow_options() -> xa11y::capture::CaptureOptions {
    let mut o = xa11y::capture::CaptureOptions::default();
    o.max_depth = Some(4);
    o.max_nodes = 32;
    o.max_text_bytes = 512;
    o.text = false;
    o.relations = false;
    o.formatting = false;
    o.timeout = Some(std::time::Duration::from_millis(500));
    o
}

/// Считать поддерево и превратить в тень со SMI-идентификаторами.
/// Возвращает тень и сырой capture — со статистикой чтений, по которой
/// бенч сравнивает холодный и тёплый ход.
pub fn capture_shadow(
    root: &Element,
    options: &xa11y::capture::CaptureOptions,
) -> xa11y::Result<(Shadow, xa11y::capture::Capture)> {
    let capture = root.capture(options)?;
    let shadow = shadow_from(&capture);
    Ok((shadow, capture))
}

/// Перегнать готовый capture в тень: каждому узлу — SMI его структурного
/// пути и нормализованного имени. Коллизии (два одинаковых элемента списка
/// с одним именем) разрешаются дискриминатором и считаются: рост числа
/// коллизий — сигнал, что имена перестают различать элементы.
pub fn shadow_from(capture: &xa11y::capture::Capture) -> Shadow {
    let mut shadow = Shadow::default();
    let mut index_to_smi: HashMap<usize, Smi> = HashMap::with_capacity(capture.nodes.len());
    let mut seen: HashMap<Smi, (xa11y::Role, String)> = HashMap::new();

    // Первый проход: SMI каждого узла и запись в тень.
    for node in &capture.nodes {
        let chain = chain_for(capture, node.index);
        let raw_name = node.element.name.clone().unwrap_or_default();
        let name = smi::normalize_name(&raw_name);
        let mut hash = smi::smi(&chain, &name);
        // Полный дубль (два одинаковых элемента списка дают одинаковую
        // цепочку: позиция повторяющейся роли в хэш не входит) — честная
        // коллизия, дискриминатор разъезжает идентификаторы.
        let mut disc = 0u32;
        while seen
            .get(&hash)
            .map(|(r, n)| *r == node.role && *n == name)
            .unwrap_or(false)
        {
            disc += 1;
            hash = smi::smi_with_disc(&chain, &name, disc);
        }
        seen.insert(hash, (node.role, name.clone()));
        index_to_smi.insert(node.index, hash);
        shadow.nodes.insert(
            hash,
            ShadowNode {
                smi: hash,
                role: node.role,
                name,
                depth: node.depth,
                parent: None,
                children: Vec::new(),
                children_count: node.children_count,
                children_truncated: node.children_truncated,
            },
        );
    }

    // Второй проход: дети и родители по SMI.
    for node in &capture.nodes {
        let own = index_to_smi[&node.index];
        let parent_smi = node.parent.map(|p| index_to_smi[&p]);
        let entry = shadow
            .nodes
            .get_mut(&own)
            .expect("узел вставлен первым проходом");
        entry.parent = parent_smi;
        entry.children = node
            .children
            .iter()
            .filter_map(|c| index_to_smi.get(c).copied())
            .collect();
        if parent_smi.is_none() {
            shadow.roots.push(own);
        }
    }

    shadow
}

/// Цепочка сегментов пути от корня к узлу: роль + позиция среди соседей.
/// Позиция берётся из списка детей родителя; повторяющиеся роли помечаются.
fn chain_for(capture: &xa11y::capture::Capture, index: usize) -> Vec<PathSeg> {
    let mut chain = Vec::new();
    let mut idx = Some(index);
    while let Some(i) = idx {
        let node = &capture.nodes[i];
        let sibling_idx = node
            .parent
            .map(|p| {
                capture.nodes[p]
                    .children
                    .iter()
                    .position(|c| *c == i)
                    .unwrap_or(0) as u16
            })
            .unwrap_or(0);
        chain.push(PathSeg {
            role: node.role,
            sibling_idx,
            repeating: smi::is_repeating_role(node.role),
        });
        idx = node.parent;
    }
    chain.reverse();
    chain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::FakeDesktop;
    use std::sync::Arc;

    #[test]
    fn capture_builds_shadow_with_smis() {
        let mut desktop = FakeDesktop::new();
        let app = desktop.open_app("Блокнот", 4);
        let (shadow, capture) =
            capture_shadow(&app, &default_options()).expect("capture по фейковому дереву");
        assert!(!shadow.roots.is_empty());
        assert_eq!(shadow.nodes.len(), capture.nodes.len());
        // Все SMI уникальны: узлы дерева различимы.
        assert_eq!(shadow.nodes.len(), shadow.nodes.keys().collect::<Vec<_>>().len());
        // Родители указывают на живые узлы.
        for node in shadow.nodes.values() {
            if let Some(p) = node.parent {
                assert!(shadow.nodes.contains_key(&p), "родитель {p} жив");
            }
        }
    }

    #[test]
    fn identical_tree_gives_identical_smis() {
        let mut desktop = FakeDesktop::new();
        let first = desktop.open_app("Терминал", 3);
        let (a, _) = capture_shadow(&first, &default_options()).expect("первый capture");
        let (b, _) = capture_shadow(&first, &default_options()).expect("второй capture");
        let mut a_keys: Vec<_> = a.nodes.keys().copied().collect();
        let mut b_keys: Vec<_> = b.nodes.keys().copied().collect();
        a_keys.sort_unstable();
        b_keys.sort_unstable();
        assert_eq!(a_keys, b_keys, "тот же экран — те же идентификаторы");
    }

    #[test]
    fn budgets_hold() {
        let mut desktop = FakeDesktop::new();
        // Плотное дерево: 6 ветвей × глубина 7 — много сотен узлов.
        let app = desktop.open_app_dense("Плотное", 6, 7);
        let mut opts = default_options();
        opts.max_nodes = 50;
        let (shadow, capture) = capture_shadow(&app, &opts).expect("обрезанный capture");
        assert!(shadow.nodes.len() <= 50);
        assert!(capture.nodes.len() <= 50);
        // Обрезание честно помечено, а не спрятано.
        assert!(capture.truncated.by_nodes || capture.truncated.by_depth);
    }
}

#[cfg(test)]
mod warm_cold_tests {
    use super::*;
    use crate::est::Est;
    use crate::fake::FakeDesktop;

    /// Числовой критерий Этапа 1 (§11): тёплый ход на порядок дешевле
    /// холодного по числу чтений бэкенда.
    #[test]
    fn warm_hops_are_an_order_cheaper_than_cold() {
        let mut desktop = FakeDesktop::new();
        let root = desktop.open_app_dense("Плотный", 6, 4);

        // Холодный ход: полный bounded capture + установка тени.
        let reads_before = desktop.children_reads();
        let (shadow, capture) = capture_shadow(&root, &default_options()).expect("cold");
        let cold_reads = desktop.children_reads() - reads_before;
        let mut est = Est::new();
        est.install(shadow);
        assert_eq!(est.len(), capture.nodes.len());

        // Тёплый ход: одно событие меняет одну строку — перечитывается
        // узкое поддерево ГРЯЗНОГО узла, а не всё дерево. (На реальной
        // машине узел достаётся из Smi через Locator — этап 2; здесь
        // фейк даёт элемент напрямую.)
        desktop.rename("Плотный-узел 5", "Плотный-узел 5 — обновлено");
        let dirty = desktop
            .element_named("Плотный-узел 5 — обновлено")
            .expect("грязный узел жив");
        let reads_before = desktop.children_reads();
        let (_, narrow) = capture_shadow(&dirty, &narrow_options()).expect("warm");
        let warm_reads = desktop.children_reads() - reads_before;
        let _ = narrow;

        assert!(
            cold_reads >= 10 * warm_reads.max(1),
            "холодный {cold_reads} чтений против тёплого {warm_reads}: запас меньше 10×"
        );
        // И статистика capture согласна с провайдером.
        assert!(capture.stats.children_reads >= cold_reads.saturating_sub(1));
    }

    /// EST ловит изменение: тот же экран даёт gate 0, изменение метки —
    /// над порогом (§11: три теста SimHash в связке с деревом).
    #[test]
    fn est_gate_sees_label_change() {
        let mut desktop = FakeDesktop::new();
        let root = desktop.open_app("Почта", 3);
        let (shadow, _) = capture_shadow(&root, &default_options()).expect("capture");
        let mut est = Est::new();
        est.install(shadow);
        assert!(est.gate().is_none());
        assert_eq!(est.gate(), Some(0));

        desktop.rename("Запись 0 — текст строки списка", "Запись 0 — новый текст");
        let (shadow2, _) = capture_shadow(&root, &default_options()).expect("capture 2");
        est.install(shadow2);
        let d = est.gate().expect("второй замок");
        assert!(d > 0, "изменение метки обязано быть видно: {d}");
    }
}
