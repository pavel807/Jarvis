//! A1. SMI — Structural Merkle Identity: устойчивый идентификатор элемента,
//! переживающий скролл, вставку строк и перезапуск приложения.
//!
//! `NodeIdentity` из xa11y живёт только в сессии и не сериализуется; SMI
//! закрывает эту дыру: это хэш структурного пути (роль + позиция среди
//! соседей) плюс нормализованное имя.
//!
//! Ключевое правило: для ролей-контейнеров повторяющихся списков позиция
//! среди соседей в хэш не входит — иначе вставка строки выше сдвигала бы
//! идентификаторы всех строк ниже, и выученные планы протухали бы на каждом
//! обновлении списка.

use unicode_normalization::UnicodeNormalization;

/// Один сегмент структурного пути от корня до элемента.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathSeg {
    pub role: xa11y::Role,
    /// Позиция среди соседей. В хэш попадает ТОЛЬКО для неповторяющихся
    /// ролей — см. модульную документацию.
    pub sibling_idx: u16,
    /// true — элемент внутри повторяющегося списка (row, list_item, …).
    pub repeating: bool,
}

/// Роли, у которых соседей считать нельзя: содержимое списка меняется
/// постоянно, а семантика строки не зависит от её номера.
pub fn is_repeating_role(role: xa11y::Role) -> bool {
    use xa11y::Role::*;
    matches!(
        role,
        TableRow | TableCell | ListItem | TreeItem | MenuItem | Tab
    )
}

/// Нормализация имени: NFC, trim, схлопывание пробелов. Одна и та же
/// надпись, пришедшая в разных формах юникода, даёт один хэш.
pub fn normalize_name(name: &str) -> String {
    let nfc: String = name.nfc().collect();
    let mut out = String::with_capacity(nfc.len());
    let mut prev_space = false;
    for c in nfc.chars() {
        let space = c.is_whitespace();
        if space {
            if !prev_space && !out.is_empty() {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
        prev_space = space;
    }
    out.trim_end().to_string()
}

/// blake3-64: полная лавина (два разных текста — независимые биты, без
/// этого SimHash не видит смену метки) и детерминированность между
/// запусками (без этого SMI протухает при каждом старте процесса).
pub fn hash64(bytes: &[u8]) -> u64 {
    let digest = blake3::hash(bytes);
    let mut out = [0u8; 8];
    out.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(out)
}

/// Посчитать SMI цепочки пути и имени элемента.
pub fn smi(chain: &[PathSeg], name: &str) -> u64 {
    smi_with_disc(chain, name, 0)
}

/// SMI с дискриминатором коллизий: при совпадении (smi, role, name)
/// следующая попытка поднимает дискриминатор, и это логируется (см. est).
pub fn smi_with_disc(chain: &[PathSeg], name: &str, disc: u32) -> u64 {
    // Кодируем путь в байты детерминированно: роль, позиция (только для
    // неповторяющихся ролей), флаг повтора, затем имя и дискриминатор.
    let mut bytes: Vec<u8> = Vec::with_capacity(16 * chain.len() + name.len() + 4);
    for seg in chain {
        bytes.extend_from_slice(format!("{:?}", seg.role).as_bytes());
        if !seg.repeating {
            bytes.extend_from_slice(&seg.sibling_idx.to_le_bytes());
        }
        bytes.push(seg.repeating as u8);
        bytes.push(0xff); // разделитель сегментов
    }
    bytes.push(0xfe); // конец пути, начало имени
    bytes.extend_from_slice(normalize_name(name).as_bytes());
    bytes.extend_from_slice(&disc.to_le_bytes());
    hash64(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xa11y::Role;

    fn seg(role: Role, idx: u16, repeating: bool) -> PathSeg {
        PathSeg {
            role,
            sibling_idx: idx,
            repeating,
        }
    }

    fn row_chain(row_pos: u16) -> Vec<PathSeg> {
        // window → list → row «Пётр Иванов» с rows_above строками выше.
        vec![
            seg(Role::Window, 0, false),
            seg(Role::List, 1, false),
            seg(Role::TableRow, row_pos, true),
        ]
    }

    #[test]
    fn inserting_rows_does_not_move_smi() {
        // Наивный хэш с позицией соседа сдвинул бы строку при вставке двух
        // строк выше; SMI повторяющейся роли от позиции не зависит.
        let before = smi(&row_chain(5), "Пётр Иванов");
        let after = smi(&row_chain(7), "Пётр Иванов");
        assert_eq!(before, after);
    }

    #[test]
    fn non_repeating_position_matters() {
        // Кнопки тулбара — не список: позиция среди соседей часть личности.
        let a = smi(&[seg(Role::Window, 0, false), seg(Role::Button, 1, false)], "Сохранить");
        let b = smi(&[seg(Role::Window, 0, false), seg(Role::Button, 2, false)], "Сохранить");
        assert_ne!(a, b);
    }

    #[test]
    fn name_forms_hash_alike() {
        // NFC vs разложенная форма, лишние пробелы — одно и то же имя.
        let nfc = "Привет мир";
        let decomposed: String = "Привет мир".nfd().collect();
        assert_eq!(
            smi(&row_chain(0), nfc),
            smi(&row_chain(0), &decomposed)
        );
        assert_eq!(
            smi(&row_chain(0), "  Привет    мир  "),
            smi(&row_chain(0), "Привет мир")
        );
    }

    #[test]
    fn different_names_differ() {
        let a = smi(&row_chain(0), "Пётр Иванов");
        let b = smi(&row_chain(0), "Пётр Сергеев");
        assert_ne!(a, b);
    }

    #[test]
    fn discriminator_breaks_collision() {
        let a = smi(&row_chain(0), "X");
        let b = smi_with_disc(&row_chain(0), "X", 1);
        assert_ne!(a, b);
    }
}
