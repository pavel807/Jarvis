//! Встроенная база данных Jarvis (`redb`).
//!
//! База — один файл рядом с остальными данными (`jarvis.redb`). Первая
//! таблица — шутки: они не должны зависеть от интернета и модели, а «ещё
//! шутку» не должна повторять предыдущую. Схема намеренно простая: ключ —
//! номер, значение — текст; отдельная таблица `meta` хранит служебные
//! значения (последняя показанная шутка).
//!
//! `redb` выбран потому, что это чистый Rust без C-сборки: база не требует
//! sqlite в системе и одинаково работает на macOS, Linux и Windows.

use crate::paths;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Шутки: номер → текст.
const JOKES: TableDefinition<u64, &str> = TableDefinition::new("jokes");
/// Служебные значения: имя → число.
const META: TableDefinition<&str, u64> = TableDefinition::new("meta");

const LAST_JOKE: &str = "last_joke_id";

/// Открытая база. Один экземпляр на процесс: redb не разрешает открывать
/// один и тот же файл дважды, а глобальный `Mutex` сериализует короткие
/// операции записи.
pub struct Store {
    db: Database,
}

impl Store {
    /// Открыть (или создать) базу и засеять шутки при первом запуске.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let db = Database::create(path)?;
        let store = Self { db };
        store.ensure_seed()?;
        Ok(store)
    }

    fn ensure_seed(&self) -> anyhow::Result<()> {
        let write = self.db.begin_write()?;
        {
            let mut jokes = write.open_table(JOKES)?;
            if jokes.iter()?.count() == 0 {
                for (index, text) in crate::jokes::SEED.iter().enumerate() {
                    jokes.insert((index as u64) + 1, *text)?;
                }
            }
            // Таблица создаётся даже в пустой базе: чтение `meta` не должно
            // падать с «table does not exist».
            let _meta = write.open_table(META)?;
        }
        write.commit()?;
        Ok(())
    }

    /// Сколько шуток в базе.
    pub fn joke_count(&self) -> anyhow::Result<usize> {
        let read = self.db.begin_read()?;
        let jokes = read.open_table(JOKES)?;
        Ok(jokes.iter()?.count())
    }

    /// Последняя показанная шутка (0 — ещё ни одной).
    pub fn last_joke(&self) -> anyhow::Result<u64> {
        let read = self.db.begin_read()?;
        let meta = read.open_table(META)?;
        Ok(meta.get(LAST_JOKE)?.map(|v| v.value()).unwrap_or(0))
    }

    /// Запомнить показанную шутку.
    pub fn set_last_joke(&self, id: u64) -> anyhow::Result<()> {
        let write = self.db.begin_write()?;
        {
            let mut meta = write.open_table(META)?;
            meta.insert(LAST_JOKE, id)?;
        }
        write.commit()?;
        Ok(())
    }

    /// Выбрать случайную шутку, не равную `avoid` (если есть альтернатива).
    pub fn pick_joke(&self, avoid: u64) -> anyhow::Result<Option<(u64, String)>> {
        let read = self.db.begin_read()?;
        let jokes = read.open_table(JOKES)?;
        let mut all: Vec<(u64, String)> = Vec::new();
        for entry in jokes.iter()? {
            let (id, text) = entry?;
            all.push((id.value(), text.value().to_string()));
        }
        if all.is_empty() {
            return Ok(None);
        }
        // Немедленный повтор запрещён, но если шутка в базе одна — честно
        // показываем её, а не молчим.
        let candidates: Vec<&(u64, String)> =
            all.iter().filter(|(id, _)| *id != avoid).collect();
        let pool = if candidates.is_empty() {
            all.iter().collect::<Vec<_>>()
        } else {
            candidates
        };
        use rand::Rng;
        let index = rand::thread_rng().gen_range(0..pool.len());
        let (id, text) = pool[index];
        Ok(Some((*id, text.clone())))
    }
}

/// Путь к базе: рядом с settings.toml и tasks.md.
pub fn db_path() -> std::path::PathBuf {
    paths::base_dir().join("jarvis.redb")
}

/// Глобальная база процесса. Ошибка открытия кэшируется: повторять её на
/// каждую шутку бессмысленно, пользователь должен услышать причину.
fn global_store() -> Result<&'static Mutex<Store>, String> {
    static STORE: OnceLock<Result<Mutex<Store>, String>> = OnceLock::new();
    match STORE.get_or_init(|| {
        Store::open(&db_path())
            .map(Mutex::new)
            .map_err(|e| format!("не удалось открыть базу данных: {e:#}"))
    }) {
        Ok(store) => Ok(store),
        Err(e) => Err(e.clone()),
    }
}

/// Взять случайную шутку из базы, не повторяя предыдущую.
pub fn next_joke() -> Result<String, String> {
    let store = global_store()?;
    let store = store.lock().unwrap_or_else(|e| e.into_inner());
    let last = store
        .last_joke()
        .map_err(|e| format!("не удалось прочитать базу: {e:#}"))?;
    let (id, text) = store
        .pick_joke(last)
        .map_err(|e| format!("не удалось выбрать шутку: {e:#}"))?
        .ok_or_else(|| "в базе пока нет ни одной шутки".to_string())?;
    store
        .set_last_joke(id)
        .map_err(|e| format!("не удалось сохранить выбор: {e:#}"))?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_store() -> (TempDir, Store) {
        let dir = tempfile::tempdir().expect("временный каталог");
        let store = Store::open(&dir.path().join("test.redb")).expect("база открывается");
        (dir, store)
    }

    #[test]
    fn seeds_jokes_on_first_open() {
        let (_dir, store) = temp_store();
        assert_eq!(store.joke_count().expect("счёт"), crate::jokes::SEED.len());
        assert_eq!(store.last_joke().expect("мета"), 0);
    }

    #[test]
    fn does_not_repeat_last_joke() {
        let (_dir, store) = temp_store();
        let (first, _) = store.pick_joke(0).expect("выбор").expect("шутка");
        store.set_last_joke(first).expect("запись");
        for _ in 0..20 {
            let (next, text) = store
                .pick_joke(store.last_joke().expect("мета"))
                .expect("выбор")
                .expect("шутка");
            assert_ne!(next, store.last_joke().expect("мета"));
            assert!(!text.trim().is_empty());
            store.set_last_joke(next).expect("запись");
        }
    }
}
