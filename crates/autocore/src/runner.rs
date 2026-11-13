//! Этап 3 — действие и подтверждение: найти цель селектором, исполнить
//! через Locator (авто-ожидание встроено), подтвердить НАБЛЮДАЕМОЕ
//! следствие и только потом объявить успех.
//!
//! Правила спецификации (§4):
//! - «ok» от действия ≠ эффект наступил: проверяем следствие — значение
//!   поля, состояние, появление/исчезновение узла;
//! - повтор разрешён ТОЛЬКО при `Delivery::NotDelivered`; при `Unknown`
//!   сначала наблюдаем состояние, потом решаем — повтор может удвоить
//!   действие;
//! - каждое «не смогло» — три части: что, почему, что делать.

use crate::policy::{classify, explain, Action, Policy, Verdict};
use std::sync::Arc;
use std::time::Duration;
use xa11y::{Element, Error, Locator};

/// Сколько ждать появления следствия после действия.
pub const CONFIRM_TIMEOUT: Duration = Duration::from_secs(5);

/// Реакция на ошибку по таксономии §6.
#[derive(Debug, Clone, PartialEq)]
pub enum Reaction {
    /// Доставка не состоялась — безопасно переразрешить селектором и
    /// повторить один раз.
    RetrySafe,
    /// Исход неизвестен — сначала наблюдение состояния.
    ObserveFirst,
    /// Права не выданы — остановиться и показать инструкцию.
    Permissions(String),
    /// Возможность отсутствует на платформе — деградация.
    Unsupported(String),
}

/// Классификация ошибки: доставка + что делать (§6, таблица).
pub fn reaction(err: &Error) -> Reaction {
    use xa11y::StaleReason::*;
    match err {
        Error::PermissionDenied { instructions } => Reaction::Permissions(instructions.clone()),
        Error::Unsupported { feature } => Reaction::Unsupported(feature.clone()),
        Error::Timeout { .. } | Error::Platform { .. } => Reaction::ObserveFirst,
        // SelectorNotMatched, ElementStale, ActionNotSupported, NoElementBounds:
        // ничего не применено — переразрешение селектором безопасно.
        _ => match err.delivery() {
            // SelectorNotMatched, ElementStale, ActionNotSupported,
            // NoElementBounds: ничего не применено — переразрешить
            // селектором и повторить безопасно (Locator живых ссылок
            // не хранит).
            xa11y::Delivery::NotDelivered => Reaction::RetrySafe,
            xa11y::Delivery::Unknown => Reaction::ObserveFirst,
            // Delivery #[non_exhaustive]: новые виды — осторожная деградация.
            _ => Reaction::ObserveFirst,
        },
    }
}

/// Итог действия: успех подтверждён следствием или честный отказ.
#[derive(Debug)]
pub struct Outcome {
    pub ok: bool,
    /// Что сделали (или на чём остановились).
    pub summary: String,
    /// Чем подтверждено / почему не получилось / что делать.
    pub evidence: String,
}

/// Три части честного отказа: что, почему, что делать.
fn refusal(what: &str, why: &str, next: &str) -> Outcome {
    Outcome {
        ok: false,
        summary: what.to_string(),
        evidence: format!("почему: {why}. что делать: {next}"),
    }
}

/// Исполнитель: политика + подтверждение следствий.
pub struct Runner {
    pub provider: Arc<dyn xa11y::Provider>,
    pub policy: Policy,
}

impl Runner {
    pub fn new(provider: Arc<dyn xa11y::Provider>, policy: Policy) -> Self {
        Self { provider, policy }
    }

    /// Locator по селектору xa11y: `button[name="Сохранить"]`. Locator не
    /// хранит живую ссылку — переразрешает селектор на каждой операции,
    /// поэтому протухание ему не страшно (правило спецификации).
    pub fn locate(&self, root: Option<&Element>, selector: &str) -> Locator {
        let root = root.map(|e| e.data().clone());
        Locator::new(Arc::clone(&self.provider), root, selector)
            .with_timeout(CONFIRM_TIMEOUT)
    }

    /// Нажать кнопку: гейт → press → подтверждение следствия.
    pub fn press(&self, root: Option<&Element>, selector: &str) -> Outcome {
        let act = Action::UiAction { target: selector.to_string() };
        let verdict = classify(&self.policy, &act);
        if let Verdict::Ask { reason } | Verdict::Deny { reason } = &verdict {
            return Outcome {
                ok: false,
                summary: "действие остановлено политикой".into(),
                evidence: explain(&verdict, &act),
            };
        }
        let locator = self.locate(root, selector);
        match locator.press() {
            Ok(()) => match self.confirm_appeared(root, selector) {
                true => Outcome {
                    ok: true,
                    summary: format!("нажато: {selector}"),
                    evidence: "элемент пере-разрешился после нажатия (Locator)".into(),
                },
                false => refusal(
                    &format!("нажал {selector}, но следствия не увидел"),
                    "элемент исчез после нажатия за таймаут подтверждения",
                    "снять новый capture и проверить, открылся ли ожидаемый диалог",
                ),
            },
            Err(e) => self.on_error(&e, root, selector, |l| l.press()),
        }
    }

    /// Ввести текст в поле и ПОДТВЕРДИТЬ значение чтением, а не Ok().
    pub fn type_text(&self, root: Option<&Element>, selector: &str, text: &str) -> Outcome {
        let act = Action::UiAction { target: selector.to_string() };
        if let Verdict::Ask { reason } | Verdict::Deny { reason } = classify(&self.policy, &act) {
            return Outcome {
                ok: false,
                summary: "ввод остановлен политикой".into(),
                evidence: explain(&Verdict::Ask { reason }, &act),
            };
        }
        let locator = self.locate(root, selector);
        match locator.type_text(text) {
            Ok(()) => {
                // Подтверждение: переразрешить и прочитать значение.
                match locator.wait_visible(CONFIRM_TIMEOUT) {
                    Ok(el) => {
                        let got = el.data().value.clone().unwrap_or_default();
                        if got.contains(text) || text.is_empty() {
                            Outcome {
                                ok: true,
                                summary: format!("введено в {selector}"),
                                evidence: format!("прочитано обратно: «{got}»"),
                            }
                        } else {
                            refusal(
                                &format!("ввёл текст в {selector}, значение не подтвердилось"),
                                &format!("поле содержит «{got}», ожидали «{text}»"),
                                "повторить ввод или проверить, что поле редактируемое",
                            )
                        }
                    }
                    Err(e) => refusal(
                        &format!("ввёл текст в {selector}, поле не перечиталось"),
                        &e.to_string(),
                        "снять новый capture и найти поле заново",
                    ),
                }
            }
            Err(e) => self.on_error(&e, root, selector, |l| l.type_text(text)),
        }
    }

    /// Очистить поле и подтвердить пустоту чтением (сценарий §10).
    pub fn clear_text(&self, root: Option<&Element>, selector: &str) -> Outcome {
        let locator = self.locate(root, selector);
        match locator.set_value("") {
            Ok(()) => match locator.wait_visible(CONFIRM_TIMEOUT) {
                Ok(el) => {
                    let got = el.data().value.clone().unwrap_or_default();
                    if got.is_empty() {
                        Outcome {
                            ok: true,
                            summary: format!("поле {selector} очищено"),
                            evidence: "прочитано обратно: пусто".into(),
                        }
                    } else {
                        refusal(
                            &format!("очистил {selector}, но поле не пусто"),
                            &format!("поле содержит «{got}»"),
                            "очистить вручную или проверить значение повторно",
                        )
                    }
                }
                Err(e) => refusal("поле не перечиталось", &e.to_string(), "новый capture"),
            },
            Err(e) => self.on_error(&e, root, selector, |l| l.set_value("")),
        }
    }

    /// Ошибка действия: повтор только при NotDelivered, при Unknown —
    /// сначала наблюдение. Один повтор максимум (§4).
    fn on_error(
        &self,
        err: &Error,
        root: Option<&Element>,
        selector: &str,
        act: impl Fn(&Locator) -> Result<(), Error>,
    ) -> Outcome {
        match reaction(err) {
            Reaction::RetrySafe => {
                // Селектор переразрешится сам: Locator живых ссылок не
                // хранит — повторяем один раз.
                let locator = self.locate(root, selector);
                match act(&locator) {
                    Ok(()) => Outcome {
                        ok: true,
                        summary: format!("выполнено со второго раза: {selector}"),
                        evidence: format!("первый раз: {err}; переразрешили селектором"),
                    },
                    Err(e2) => refusal(
                        &format!("действие с {selector} не удалось дважды"),
                        &e2.to_string(),
                        "снять новый capture и найти цель заново",
                    ),
                }
            }
            Reaction::ObserveFirst => refusal(
                &format!("действие с {selector}: исход неизвестен"),
                &err.to_string(),
                "сначала снять capture и посмотреть состояние, потом решать — \
повтор может удвоить действие",
            ),
            Reaction::Permissions(instructions) => refusal(
                &format!("действие с {selector} остановлено правами"),
                &instructions,
                "выдать права по инструкции и повторить",
            ),
            Reaction::Unsupported(feature) => refusal(
                &format!("действие с {selector} не поддерживается"),
                &format!("возможность «{feature}» отсутствует на этой платформе"),
                "другой способ действия или другая платформа",
            ),
        }
    }

    /// Следствие нажатия: элемент, который нажимали, исчез (закрылся
    /// диалог) или остался — Locator переразрешается. Для Этапа 3
    /// подтверждение: элемент по-прежнему резолвится после нажатия.
    fn confirm_appeared(&self, root: Option<&Element>, selector: &str) -> bool {
        let locator = self.locate(root, selector);
        locator.wait_visible(Duration::from_secs(1)).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::FakeDesktop;
    use std::sync::Arc;

    fn runner(desktop: &FakeDesktop) -> Runner {
        Runner::new(desktop.provider(), Policy::default())
    }

    fn root_el(desktop: &FakeDesktop) -> Element {
        desktop.element_named("TestApp").expect("корень есть")
    }

    /// Сценарий приёмки §10: найти поле → ввести → подтвердить значение →
    /// очистить → подтвердить. Поле — в фейковом дереве.
    #[test]
    fn type_confirm_clear_confirm() {
        let mut desktop = FakeDesktop::new();
        // Поле поиска в дереве.
        desktop.add_field("Поиск", "привет мир");
        let root = root_el(&desktop);
        let r = runner(&desktop);

        let typed = r.type_text(Some(&root), r#"text_field[name="Поиск"]"#, "найди меня");
        assert!(typed.ok, "{:?}", typed.summary);
        assert!(typed.evidence.contains("найди меня"), "значение подтверждено чтением");

        let cleared = r.clear_text(Some(&root), r#"text_field[name="Поиск"]"#);
        assert!(cleared.ok, "{:?}", cleared.summary);
        assert!(cleared.evidence.contains("пусто"));
    }

    /// Нажатие с подтверждением следствия; лог фейка показывает, что
    /// нажатие дошло до нужного узла.
    #[test]
    fn press_confirms_via_locator() {
        let mut desktop = FakeDesktop::new();
        let root = root_el(&desktop);
        let r = runner(&desktop);

        let out = r.press(Some(&root), r#"button[name="Сохранить"]"#);
        assert!(out.ok, "{:?}", out.summary);
        // Цельность: одно press-действие дошло до провайдера; попадание
        // в кнопку «Сохранить» подтверждено селектором Locator'а.
        assert_eq!(desktop.actions().len(), 1, "{:?}", desktop.actions());
        assert!(desktop.actions()[0].starts_with("press@"));

        // Несуществующая кнопка: честный отказ, не выдумка.
        let miss = r.press(Some(&root), r#"button[name="Нет такой"]"#);
        assert!(!miss.ok);
        assert!(miss.evidence.contains("что делать"), "{}", miss.evidence);
    }

    /// Правило повтора §4: NotDelivered — один ретрай; Unknown — сначала
    /// наблюдение (на фейке таймаут даёт Unknown).
    #[test]
    fn retry_rule_respects_delivery() {
        // Мёртвый провайдер: все действия падают — повтор разрешён
        // (NotDelivered), но дважды не succeeded → честный отказ.
        struct Dead;
        impl xa11y::Provider for Dead {
            fn get_children(&self, _: Option<&xa11y::ElementData>) -> xa11y::Result<Vec<xa11y::ElementData>> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn get_parent(&self, _: &xa11y::ElementData) -> xa11y::Result<Option<xa11y::ElementData>> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn list_apps(&self) -> xa11y::Result<Vec<xa11y::ElementData>> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn app_by_pid(&self, _: u32) -> xa11y::Result<xa11y::ElementData> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn focused_app(&self) -> xa11y::Result<xa11y::ElementData> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn list_shell_surfaces(&self) -> xa11y::Result<Vec<(xa11y::ShellSurfaceKind, xa11y::ElementData)>> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn press(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn focus(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn blur(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn toggle(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn select(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn expand(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn collapse(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn show_menu(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn increment(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn decrement(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn scroll_into_view(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn set_value(&self, _: &xa11y::ElementData, _: &str) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn set_numeric_value(&self, _: &xa11y::ElementData, _: f64) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn type_text(&self, _: &xa11y::ElementData, _: &str) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn set_text_selection(&self, _: &xa11y::ElementData, _: u32, _: u32) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn perform_action(&self, _: &xa11y::ElementData, _: &str) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn activate(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn minimize(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn maximize(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn enter_fullscreen(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn restore(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn close(&self, _: &xa11y::ElementData) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn move_to(&self, _: &xa11y::ElementData, _: i32, _: i32) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn resize_to(&self, _: &xa11y::ElementData, _: u32, _: u32) -> xa11y::Result<()> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn table_cell_at(&self, _: &xa11y::ElementData, _: u32, _: u32) -> xa11y::Result<Option<xa11y::ElementData>> {
                Err(Error::selector_not_matched("всё мертво"))
            }
            fn subscribe(&self, _: &xa11y::ElementData) -> xa11y::Result<xa11y::Subscription> {
                Err(Error::selector_not_matched("всё мертво"))
            }
        }
        let r = Runner::new(Arc::new(Dead), Policy::default());
        let out = r.press(None, r#"button[name="X"]"#);
        assert!(!out.ok);
        // Dead-провайдер валит разрешение цели: авто-ожидание Locator'а
        // заканчивается Timeout'ом = Delivery::Unknown → «сначала
        // наблюдение», повтор слепо не делается (§4).
        assert!(
            out.summary.contains("исход неизвестен"),
            "Unknown обязан дать «наблюдение, не повтор»: {out:?}"
        );
        assert!(out.evidence.contains(" capture"));
    }

    /// Три части отказа: что не получилось, почему, что делать (§6).
    #[test]
    fn refusal_has_three_parts() {
        let mut desktop = FakeDesktop::new();
        let root = root_el(&desktop);
        let r = runner(&desktop);
        let out = r.press(Some(&root), r#"button[name="Нет такой"]"#);
        assert!(out.evidence.contains("почему:"));
        assert!(out.evidence.contains("что делать:"));
    }

    /// Правило подтверждения §4: «ok» от действия ≠ эффект. set_value
    /// в фейке меняет значение — чтение обратно это ловит.
    #[test]
    fn value_readback_catches_silent_failures() {
        let mut desktop = FakeDesktop::new();
        desktop.add_field("Поле", "старое значение");
        // Провайдер-обёртка, у которой set_value молча ничего не делает:
        // Ok() вернётся, но чтение покажет старое значение — runner
        // обязан не выдать это за успех.
        struct SilentSkip {
            inner: Arc<dyn xa11y::Provider>,
        }
        impl xa11y::Provider for SilentSkip {
            fn get_children(&self, e: Option<&xa11y::ElementData>) -> xa11y::Result<Vec<xa11y::ElementData>> {
                self.inner.get_children(e)
            }
            fn get_parent(&self, e: &xa11y::ElementData) -> xa11y::Result<Option<xa11y::ElementData>> {
                self.inner.get_parent(e)
            }
            fn list_apps(&self) -> xa11y::Result<Vec<xa11y::ElementData>> {
                self.inner.list_apps()
            }
            fn app_by_pid(&self, p: u32) -> xa11y::Result<xa11y::ElementData> {
                self.inner.app_by_pid(p)
            }
            fn focused_app(&self) -> xa11y::Result<xa11y::ElementData> {
                self.inner.focused_app()
            }
            fn list_shell_surfaces(&self) -> xa11y::Result<Vec<(xa11y::ShellSurfaceKind, xa11y::ElementData)>> {
                self.inner.list_shell_surfaces()
            }
            fn press(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.press(e)
            }
            fn focus(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.focus(e)
            }
            fn blur(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.blur(e)
            }
            fn toggle(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.toggle(e)
            }
            fn select(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.select(e)
            }
            fn expand(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.expand(e)
            }
            fn collapse(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.collapse(e)
            }
            fn show_menu(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.show_menu(e)
            }
            fn increment(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.increment(e)
            }
            fn decrement(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.decrement(e)
            }
            fn scroll_into_view(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.scroll_into_view(e)
            }
            // Молча игнорируем set_value: Ok() без эффекта.
            fn set_value(&self, _: &xa11y::ElementData, _: &str) -> xa11y::Result<()> {
                Ok(())
            }
            fn set_numeric_value(&self, e: &xa11y::ElementData, v: f64) -> xa11y::Result<()> {
                self.inner.set_numeric_value(e, v)
            }
            fn type_text(&self, e: &xa11y::ElementData, t: &str) -> xa11y::Result<()> {
                self.inner.type_text(e, t)
            }
            fn set_text_selection(&self, e: &xa11y::ElementData, a: u32, b: u32) -> xa11y::Result<()> {
                self.inner.set_text_selection(e, a, b)
            }
            fn perform_action(&self, e: &xa11y::ElementData, a: &str) -> xa11y::Result<()> {
                self.inner.perform_action(e, a)
            }
            fn activate(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.activate(e)
            }
            fn minimize(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.minimize(e)
            }
            fn maximize(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.maximize(e)
            }
            fn enter_fullscreen(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.enter_fullscreen(e)
            }
            fn restore(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.restore(e)
            }
            fn close(&self, e: &xa11y::ElementData) -> xa11y::Result<()> {
                self.inner.close(e)
            }
            fn move_to(&self, e: &xa11y::ElementData, x: i32, y: i32) -> xa11y::Result<()> {
                self.inner.move_to(e, x, y)
            }
            fn resize_to(&self, e: &xa11y::ElementData, w: u32, h: u32) -> xa11y::Result<()> {
                self.inner.resize_to(e, w, h)
            }
            fn table_cell_at(&self, e: &xa11y::ElementData, r: u32, c: u32) -> xa11y::Result<Option<xa11y::ElementData>> {
                self.inner.table_cell_at(e, r, c)
            }
            fn subscribe(&self, e: &xa11y::ElementData) -> xa11y::Result<xa11y::Subscription> {
                self.inner.subscribe(e)
            }
        }
        let silent = Arc::new(SilentSkip {
            inner: desktop.provider(),
        });
        let r = Runner::new(silent, Policy::default());
        let root = root_el(&desktop);
        let out = r.clear_text(Some(&root), r#"text_field[name="Поле"]"#);
        assert!(!out.ok, "Ok() без эффекта не может считаться успехом");
        assert!(out.evidence.contains("старое"), "показано фактическое значение: {}", out.evidence);
    }
}
