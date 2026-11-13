//! Policy Gate (§5): правила безопасности ПЕРЕД ML и перед действием.
//! Порядок обязателен: правила → политика → остальное. Модель никогда
//! не последний арбитр — и не первый тоже.
//!
//! Хардкод-deny не отключается никаким флагом: уничтожение данных,
//! права/защита, финансы, сообщения от имени пользователя, публикации,
//! массовые действия. Всё остальное, что необратимо, — Ask.

use std::fmt::Write as _;

/// Решение гейта.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Действовать без вопросов.
    Allow,
    /// Спросить подтверждение текстом.
    Ask { reason: &'static str },
    /// Отказать: так или иначе — уничтожение.
    Deny { reason: &'static str },
}

/// Настройки гейта. N файлов для массовых действий настраивается;
/// остальное — фиксированные правила.
#[derive(Debug, Clone)]
pub struct Policy {
    pub max_files_bulk: usize,
    pub dry_run: bool,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_files_bulk: 20,
            dry_run: false,
        }
    }
}

/// Что хочет сделать агент.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Чтение/фокус/навигация: обратимо и не покидает машину.
    Read,
    /// Действие в интерфейсе: нажать, ввести, развернуть.
    UiAction { target: String },
    /// Удаление файла/документа.
    Delete { path: String },
    /// Перезапись или сохранение поверх существующего содержимого.
    Overwrite { path: String },
    /// Финансовая операция (оплата, перевод, покупка).
    Finance,
    /// Отправка сообщения наружу от имени пользователя.
    SendOutward { channel: String },
    /// Публикация (пост, коммит в общий репозиторий).
    Publish,
    /// Изменение прав/защиты/автозагрузки/ключей.
    SecurityChange,
    /// Действие, затрагивающее много файлов сразу.
    Bulk { count: usize },
    /// Перезагрузка или выключение.
    PowerAction,
}

/// Классификация без учёта dry-run (внутренняя для сухого режима).
fn classify_strict(action: &Action) -> Verdict {
    classify(&Policy::default(), action)
}

/// Классификация действия: deny — не отключаемо, ask — по умолчанию для
/// необратимого.
pub fn classify(policy: &Policy, action: &Action) -> Verdict {
    // dry-run понижает только Allow: Deny-правила (финансы, защита)
    // не превращаются в вопрос ни при каком флаге.
    if policy.dry_run
        && matches!(classify_strict(action), Verdict::Allow)
        && !matches!(action, Action::Read)
    {
        return Verdict::Ask {
            reason: "dry-run: действие только логируется, но не исполняется",
        };
    }
    match action {
        Action::Read => Verdict::Allow,
        Action::UiAction { .. } => Verdict::Allow,
        Action::Delete { .. } => Verdict::Ask {
            reason: "удаление — обратимо только через корзину",
        },
        Action::Overwrite { .. } => Verdict::Ask {
            reason: "перезапись может потерять прежнее содержимое",
        },
        Action::Finance => Verdict::Deny {
            reason: "финансовые операции запрещены правилом, не настройкой",
        },
        Action::SendOutward { .. } => Verdict::Ask {
            reason: "отправка наружу от вашего имени",
        },
        Action::Publish => Verdict::Ask {
            reason: "публикация видна другим и кэшируется",
        },
        Action::SecurityChange => Verdict::Deny {
            reason: "изменение прав/защиты запрещено правилом",
        },
        Action::Bulk { count } => {
            if *count > policy.max_files_bulk {
                Verdict::Ask {
                    reason: "действие затрагивает много файлов",
                }
            } else {
                Verdict::Allow
            }
        }
        Action::PowerAction => Verdict::Ask {
            reason: "перезагрузка/выключение прервёт вашу работу",
        },
    }
}

/// Текстовое объяснение вердикта (§6: что не получилось, почему, что делать).
pub fn explain(verdict: &Verdict, action: &Action) -> String {
    match verdict {
        Verdict::Allow => format!("{action:?}: разрешено правилами"),
        Verdict::Ask { reason } => {
            let mut s = String::new();
            let _ = write!(s, "Требует подтверждения: {reason}.");
            let _ = write!(
                s,
                " Подтвердите текстом — «да, делай» — или откажите."
            );
            s
        }
        Verdict::Deny { reason } => {
            let mut s = String::new();
            let _ = write!(s, "Отказано: {reason}.");
            let _ = write!(
                s,
                " Правило безопасности не отключается флагом; делайте это вручную."
            );
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_pass_and_ui_actions_pass() {
        let p = Policy::default();
        assert_eq!(classify(&p, &Action::Read), Verdict::Allow);
        assert_eq!(
            classify(&p, &Action::UiAction { target: "кнопка Сохранить".into() }),
            Verdict::Allow
        );
    }

    #[test]
    fn hard_deny_cannot_be_disabled() {
        // Ни dry_run, ни настройки не включают финансовые операции.
        let mut p = Policy::default();
        p.dry_run = false;
        assert!(matches!(
            classify(&p, &Action::Finance),
            Verdict::Deny { .. }
        ));
        assert!(matches!(
            classify(&p, &Action::SecurityChange),
            Verdict::Deny { .. }
        ));
    }

    #[test]
    fn destructive_needs_confirmation() {
        let p = Policy::default();
        assert!(matches!(
            classify(&p, &Action::Delete { path: "~/отчёт.pdf".into() }),
            Verdict::Ask { .. }
        ));
        assert!(matches!(
            classify(&p, &Action::Overwrite { path: "~/отчёт.pdf".into() }),
            Verdict::Ask { .. }
        ));
        assert!(matches!(
            classify(&p, &Action::PowerAction),
            Verdict::Ask { .. }
        ));
    }

    #[test]
    fn bulk_respects_configured_limit() {
        let mut p = Policy::default();
        assert_eq!(
            classify(&p, &Action::Bulk { count: 5 }),
            Verdict::Allow
        );
        assert!(matches!(
            classify(&p, &Action::Bulk { count: 50 }),
            Verdict::Ask { .. }
        ));
        p.max_files_bulk = 100;
        assert_eq!(
            classify(&p, &Action::Bulk { count: 50 }),
            Verdict::Allow
        );
    }

    #[test]
    fn dry_run_turns_every_action_into_logged_ask() {
        let p = Policy {
            dry_run: true,
            ..Default::default()
        };
        // Чтение остаётся чтением.
        assert_eq!(classify(&p, &Action::Read), Verdict::Allow);
        // Действие — «логирую, не делаю».
        assert!(matches!(
            classify(&p, &Action::UiAction { target: "кнопка".into() }),
            Verdict::Ask { .. }
        ));
        // Deny остаётся Deny: dry-run не превращает запрет в вопрос.
        assert!(matches!(
            classify(&p, &Action::Finance),
            Verdict::Deny { .. }
        ));
    }

    #[test]
    fn explain_names_next_step() {
        let p = Policy::default();
        let v = classify(&p, &Action::Finance);
        let text = explain(&v, &Action::Finance);
        assert!(text.contains("Отказано"));
        assert!(text.contains("вручную"), "назван следующий шаг: {text}");
    }
}
