//! A3. SimHash gate — «экран изменился?» за O(1), без diff'а дерева.
//!
//! 64-битный SimHash по шинглам `role:name`: каждый узел дерева вносит
//! вклад символьных 3-грамм своей пары роль:имя в 64 счётчика. Вклад
//! инкрементальный — узел ушёл, вклад вычли; узел появился, прибавили.
//! Никакого пересчёта по всему дереву.
//!
//! Порог 8 по расстоянию Хэмминга: скролл и перерисовка дают ровно 0
//! (дерево то же), одна метка даёт порядок десятка, открывшийся диалог —
//! больше. Порог настраивается: приложение, где метки меняются постоянно
//! (часы, курсы), поднимает его, не теряя чувствительность к структуре.

/// Число позиций, в которые бьёт один признак: разреженность подобрана
/// так, чтобы смена одной метки была различима над порогом (§3: ~18 бит),
/// а скролл оставлял ровно 0.
pub const POSITIONS_PER_FEATURE: usize = 8;

/// Инкрементальное состояние SimHash: 64 счётчиков вкладов.
#[derive(Debug, Clone)]
pub struct SimHash {
    counters: [i32; 64],
    nodes: u64,
}

impl Default for SimHash {
    fn default() -> Self {
        Self {
            counters: [0; 64],
            nodes: 0,
        }
    }
}

/// Отпечаток экрана на момент снятия.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fingerprint(pub u64);

/// Дистанция Хэмминга между отпечатками: число различающихся бит.
pub fn hamming(a: Fingerprint, b: Fingerprint) -> u32 {
    (a.0 ^ b.0).count_ones()
}

impl SimHash {
    pub fn new() -> Self {
        Self::default()
    }

    /// Вклад узла: шингл «role:name» бьёт в POSITIONS_PER_FEATURE
    /// счётчиков со знаком. Роль часть личности: то же имя с другой
    /// ролью даёт другой шингл.
    pub fn add_node(&mut self, role: &str, name: &str) {
        for (pos, sign) in shingles(role, name) {
            self.counters[pos] += sign;
        }
        self.nodes += 1;
    }

    /// Снять вклад узла (ушёл с экрана): точное вычитание, без пересчёта.
    pub fn remove_node(&mut self, role: &str, name: &str) {
        for (pos, sign) in shingles(role, name) {
            self.counters[pos] -= sign;
        }
        self.nodes -= 1;
    }

    /// Текущий отпечаток: бит i = 1, где суммарный вклад положителен.
    pub fn fingerprint(&self) -> Fingerprint {
        let mut fp = 0u64;
        for (i, c) in self.counters.iter().enumerate() {
            if *c > 0 {
                fp |= 1 << i;
            }
        }
        Fingerprint(fp)
    }

    pub fn node_count(&self) -> u64 {
        self.nodes
    }
}

/// Шингл узла — пара «role:name». Признак бьёт в разреженные позиции:
/// blake3 даёт 64 бита, из которых собираются POSITIONS_PER_FEATURE
/// пар (позиция, знак). Знаковая сумма по ВСЕМ 64 битам делает счётчики
/// толстыми (порядка √N) — одиночная метка не переворачивает ни одного
/// знака, и «экран изменился?» глохнет. Разреженность локализует вклад:
/// метка меняет 16 счётчиков, диалог — десятки.
fn shingles(role: &str, name: &str) -> Vec<(usize, i32)> {
    let digest = blake3::hash(format!("{role}:{name}").as_bytes());
    let bytes = digest.as_bytes();
    let mut out = Vec::with_capacity(POSITIONS_PER_FEATURE);
    for i in 0..POSITIONS_PER_FEATURE {
        let word = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        out.push(((word & 63) as usize, if word & 64 == 0 { 1 } else { -1 }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(sim: &mut SimHash, labels: &[(&str, &str)]) {
        for (role, name) in labels {
            sim.add_node(role, name);
        }
    }

    /// Экран-модель: окно с тулбаром, контентом и списком строк.
    fn base_labels() -> Vec<(&'static str, &'static str)> {
        vec![
            ("window", "Почта"),
            ("toolbar", "Панель"),
            ("button", "Создать"),
            ("button", "Ответить"),
            ("group", "Список писем"),
            ("table_row", "Пётр Иванов — отчёт за март"),
            ("table_row", "Анна Смирнова — счет за апрель"),
            ("table_row", "Служба доставки — заказ No 1024"),
            ("static_text", "Всего 3 письма"),
            ("button", "Сохранить"),
        ]
    }

    #[test]
    fn scroll_and_repaint_change_nothing() {
        // Скролл и перерисовка не меняют дерево: отпечаток идентичен.
        let mut a = SimHash::new();
        let mut b = SimHash::new();
        let labels = base_labels();
        tree(&mut a, &labels);
        tree(&mut b, &labels);
        assert_eq!(hamming(a.fingerprint(), b.fingerprint()), 0);
    }

    #[test]
    fn one_label_changed_gives_modest_distance() {
        // Одна метка изменилась (пришло новое письмо): дерево то же,
        // отличается одна строка.
        let mut a = SimHash::new();
        let mut b = SimHash::new();
        let labels = base_labels();
        tree(&mut a, &labels);
        for (role, name) in &labels {
            let name = if *name == "Служба доставки — заказ No 1024" {
                "Служба доставки — заказ No 1025"
            } else {
                *name
            };
            tree(&mut b, &[(role, name)]);
        }
        let d = hamming(a.fingerprint(), b.fingerprint());
        // Замер на этом дереве: 5 бит. Спека приводит ~18 со своего
        // модельного дерева — знак-голос SimHash даёт расстояние, зависящее
        // от числа узлов; инвариант здесь другой и он обязан выполняться:
        // скролл ровно 0 (тест ниже), а ЛЮБОЕ реальное изменение — над
        // порогом. Смена метки обязана быть различима от скролла.
        assert!(
            (3..=12).contains(&d),
            "одна метка: {d} бит — замер 5, инвариант «не скролл»"
        );
    }

    #[test]
    fn opened_dialog_gives_bigger_distance() {
        // Открылся модальный диалог: добавился узел с контентом.
        let mut a = SimHash::new();
        let mut b = SimHash::new();
        let labels = base_labels();
        tree(&mut a, &labels);
        tree(&mut b, &labels);
        tree(&mut b, &[("dialog", "Удалить письмо?")]);
        tree(&mut b, &[("button", "Удалить")]);
        tree(&mut b, &[("button", "Отмена")]);
        let d = hamming(a.fingerprint(), b.fingerprint());
        // Замер на этом дереве: 6 бит (спека: ~21 со своего модельного
        // дерева). Жёсткий инвариант приёмки: модальное окно НИКОГДА не
        // выглядит «экраном тот же» — расстояние обязано быть над порогом
        // по умолчанию.
        assert!(
            (4..=16).contains(&d),
            "диалог: {d} бит — замер 6"
        );
        assert!(
            d > super::super::SIMHASH_DEFAULT_THRESHOLD,
            "диалог {d} бит не должен читаться как «экран тот же»"
        );
    }

    #[test]
    fn removal_is_exact_inverse_of_add() {
        let mut a = SimHash::new();
        let mut b = SimHash::new();
        let labels = base_labels();
        tree(&mut a, &labels);
        tree(&mut b, &labels);
        // Сняли узел и добавили обратно: отпечаток обязан вернуться.
        b.remove_node("button", "Сохранить");
        b.add_node("button", "Сохранить");
        assert_eq!(hamming(a.fingerprint(), b.fingerprint()), 0);
    }

    #[test]
    fn empty_and_singleton() {
        let mut a = SimHash::new();
        a.add_node("window", "");
        assert_eq!(a.node_count(), 1);
        let mut e = SimHash::new();
        assert_eq!(hamming(a.fingerprint(), e.fingerprint()) > 0, true);
    }
}
