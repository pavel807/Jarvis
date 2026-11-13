//! Прямой доступ к Accessibility API macOS.
//!
//! Почему не System Events: обращение к чужим окнам через него идёт от
//! имени посредника, и посредник может быть не допущен — тогда запрос
//! возвращает **пустоту**, а не ошибку: агент видит «на экране ничего нет» и
//! вежливо отвечает, что не смог. Прямой вызов AX API лишён этой неоднозначности:
//! спрашивает сам процесс приложения, ответ приходит тем же путём.
//!
//! Плюсы прямого дохода:
//! - в 10–30 раз быстрее: один IPC-запрос на свойство вместо Apple Event;
//! - дерево читается целиком, без выборочных `every button of window`;
//! - ошибка видна как ошибка (`AXError`), а не как пустой список.
//!
//! Без новых моделей и без сети: только системный API, который уже есть на
//! любой macOS.

#![allow(non_upper_case_globals)]

use core::ffi::{c_char, c_void};
use core::ptr;

use objc2::rc::{Retained, autoreleasepool};
use objc2_foundation::NSString;

use super::elements::{Element, ScreenSnapshot};

// =========================================================================
// Типы и функции C
// =========================================================================

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFArrayRef = *const c_void;
type AXUIElementRef = CFTypeRef;
type AXError = i32;

/// Кодировка UTF-8 в терминах CoreFoundation.
const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;

/// Ошибка AX: элемент не поддерживает действие.
const AX_ERROR_ILLEGAL_ARGUMENT: AXError = -25262;
const AX_ERROR_CANNOT_PERFORM: AXError = -25267;
const AX_ERROR_APPLICATION_NOT_ACTIVE: AXError = -25308;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
    fn AXUIElementCreateApplication(pid: i32) -> AXUIElementRef;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementPerformAction(
        element: AXUIElementRef,
        action: CFStringRef,
        arguments: CFTypeRef,
    ) -> AXError;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> AXError;
    fn AXUIElementSetMessagingTimeout(element: AXUIElementRef, timeout: f64) -> AXError;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
    fn CFStringGetTypeID() -> usize;
    fn CFArrayGetTypeID() -> usize;
    fn CFStringGetLength(string: CFStringRef) -> isize;
    fn CFStringGetCString(
        string: CFStringRef,
        buffer: *mut c_char,
        size: isize,
        encoding: u32,
    ) -> bool;
    fn CFArrayGetCount(array: CFArrayRef) -> isize;
    fn CFArrayGetValueAtIndex(array: CFArrayRef, index: isize) -> CFTypeRef;
    fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
    fn CFRelease(cf: CFTypeRef);
}

/// Доверяет ли система этому процессу управлять интерфейсом.
pub fn trusted() -> bool {
    unsafe { AXIsProcessTrusted() }
}

// =========================================================================
// Обёртки над CoreFoundation
// =========================================================================

/// Владение ссылкой на объект CoreFoundation: освобождается при drop.
struct Owned(CFTypeRef);

impl Drop for Owned {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CFRelease(self.0) };
        }
    }
}

/// Строка из NSString в виде CFStringRef.
///
/// NSString и CFString — один и тот же объект (toll-free bridging), поэтому
/// приводить тип безопасно. Строка живёт до конца вызова.
fn cf_str(text: &str) -> Retained<NSString> {
    NSString::from_str(text)
}

/// Сырой `CFStringRef` из `NSString` (toll-free bridging).
///
/// В objc2 0.6 `Retained::as_ptr` — ассоциированная функция, поэтому
/// вызывается как `Retained::as_ptr(&value)`, а не методом.
fn cf_string_ref(value: &Retained<NSString>) -> CFStringRef {
    Retained::as_ptr(value) as CFStringRef
}

fn cf_string_value(value: CFTypeRef) -> Option<String> {
    if value.is_null() || unsafe { CFGetTypeID(value) } != unsafe { CFStringGetTypeID() } {
        return None;
    }
    let len = unsafe { CFStringGetLength(value) };
    if len <= 0 {
        return Some(String::new());
    }
    let mut buffer = vec![0u8; len as usize + 4];
    let ok = unsafe {
        CFStringGetCString(
            value,
            buffer.as_mut_ptr() as *mut c_char,
            buffer.len() as isize,
            K_CF_STRING_ENCODING_UTF8,
        )
    };
    if !ok {
        return None;
    }
    let end = buffer.iter().position(|b| *b == 0).unwrap_or(buffer.len());
    buffer.truncate(end);
    String::from_utf8(buffer).ok()
}

/// Значение атрибута как строка: строка, описание или текст элемента.
fn attribute_string(element: AXUIElementRef, attribute: &str) -> Option<String> {
    let key = cf_str(attribute);
    let mut value: CFTypeRef = ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element, cf_string_ref(&key), &mut value)
    };
    if err != 0 || value.is_null() {
        return None;
    }
    let owned = Owned(value);
    cf_string_value(owned.0)
}

/// Дети элемента.
fn attribute_children(element: AXUIElementRef) -> Vec<AXUIElementRef> {
    let key = cf_str("AXChildren");
    let mut value: CFTypeRef = ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element, cf_string_ref(&key), &mut value)
    };
    if err != 0 || value.is_null() {
        return Vec::new();
    }
    let owned = Owned(value);
    if unsafe { CFGetTypeID(owned.0) } != unsafe { CFArrayGetTypeID() } {
        return Vec::new();
    }
    let count = unsafe { CFArrayGetCount(owned.0) };
    let mut out = Vec::with_capacity(count.max(0) as usize);
    for index in 0..count {
        let child = unsafe { CFArrayGetValueAtIndex(owned.0, index) };
        if child.is_null() {
            continue;
        }
        // Дети принадлежат массиву: удерживаем, чтобы элемент пережил снимок.
        out.push(unsafe { CFRetain(child) });
    }
    out
}

// =========================================================================
// Снимок экрана
// =========================================================================

/// Снимок экрана с живыми ссылками на элементы: по ним и выполняется действие.
pub struct NativeSnapshot {
    pub snapshot: ScreenSnapshot,
    /// Ссылки на элементы по их id из снимка.
    refs: Vec<AXUIElementRef>,
}

impl Drop for NativeSnapshot {
    fn drop(&mut self) {
        for element in self.refs.drain(..) {
            if !element.is_null() {
                unsafe { CFRelease(element) };
            }
        }
    }
}

/// Сколько узлов дерева обходить и на какую глубину.
const MAX_NODES: usize = 600;
const MAX_DEPTH: u32 = 10;
/// Сколько окон читать.
const MAX_WINDOWS: usize = 3;

/// Роли-контейнеры: у них нет имени, и читать его — лишний запрос.
fn is_container(role: &str) -> bool {
    matches!(
        role,
        "AXGroup"
            | "AXSplitGroup"
            | "AXSplitter"
            | "AXScrollArea"
            | "AXToolbar"
            | "AXWindow"
            | "AXTabGroup"
            | "AXApplication"
            | "AXRow"
            | "AXCell"
            | "AXTable"
            | "AXOutline"
            | "AXLayoutArea"
            | "AXCollectionView"
            | "AXBrowser"
            | "AXSheet"
            | "AXMenu"
            | "AXMenuBar"
            | "AXMenuBarItem"
            | "AXStaticText"
    )
}

/// Перевод роли AX в общий словарь ролей инструментов.
pub fn ax_role(raw: &str) -> &'static str {
    match raw {
        "AXButton" => "button",
        "AXTextField" | "AXTextArea" => "text field",
        "AXSecureTextField" => "secure text field",
        "AXCheckBox" => "checkbox",
        "AXRadioButton" => "radio button",
        "AXPopUpButton" | "AXComboBox" => "pop up button",
        "AXMenuBarItem" | "AXMenuItem" => "menu bar item",
        "AXLink" => "link",
        "AXSlider" | "AXIncrementor" => "slider",
        "AXStaticText" | "AXText" => "static text",
        "AXCell" | "AXRow" | "AXOutlineRow" => "list item",
        "AXTabButton" => "tab",
        _ => "",
    }
}

/// Собрать дерево элементов приложения по его pid.
pub fn collect(pid: i32) -> Result<NativeSnapshot, String> {
    if !trusted() {
        return Err(
            "macOS не дала доступ к элементам интерфейса. Системные настройки → \
             Конфиденциальность и безопасность → Универсальный доступ — включите Jarvis"
                .to_string(),
        );
    }
    let app = unsafe { AXUIElementCreateApplication(pid) };
    if app.is_null() {
        return Err("не удалось обратиться к приложению".to_string());
    }
    let app = Owned(app);
    unsafe { AXUIElementSetMessagingTimeout(app.0, 1.0) };

    let windows = attribute_children(app.0);
    let mut elements: Vec<Element> = Vec::new();
    let mut refs: Vec<AXUIElementRef> = Vec::new();
    let mut window_title = String::new();

    for (index, window) in windows.iter().take(MAX_WINDOWS).enumerate() {
        if window_title.is_empty() {
            window_title = attribute_string(*window, "AXTitle").unwrap_or_default();
        }
        walk(
            *window,
            1,
            index + 1,
            &mut elements,
            &mut refs,
            &mut 0usize,
        );
    }

    let snapshot = ScreenSnapshot::new(
        String::new(),
        window_title,
        elements,
    );
    Ok(NativeSnapshot { snapshot, refs })
}

/// Рекурсивный обход дерева элементов.
fn walk(
    element: AXUIElementRef,
    depth: u32,
    window: usize,
    elements: &mut Vec<Element>,
    refs: &mut Vec<AXUIElementRef>,
    counter: &mut usize,
) {
    if depth > MAX_DEPTH || *counter >= MAX_NODES {
        return;
    }
    *counter += 1;

    let raw_role = attribute_string(element, "AXRole").unwrap_or_default();
    if raw_role.is_empty() {
        return;
    }
    let role = ax_role(&raw_role);
    let mut name = String::new();
    let mut description = String::new();
    let mut value = String::new();
    if !is_container(&raw_role) {
        name = attribute_string(element, "AXTitle").unwrap_or_default();
        if name.is_empty() {
            name = attribute_string(element, "AXIdentifier").unwrap_or_default();
        }
        description = attribute_string(element, "AXDescription").unwrap_or_default();
        if raw_role == "AXTextField" || raw_role == "AXTextArea" || raw_role == "AXSecureTextField" {
            value = attribute_string(element, "AXValue").unwrap_or_default();
        }
    }
    if !role.is_empty() {
        elements.push(Element {
            id: elements.len(),
            role: role.to_string(),
            raw_role: raw_role.clone(),
            name,
            description,
            value,
            depth,
            window,
            app: String::new(),
        });
        refs.push(unsafe { CFRetain(element) });
    }

    for child in attribute_children(element) {
        let child = Owned(child);
        walk(child.0, depth + 1, window, elements, refs, counter);
    }
}

// =========================================================================
// Действия
// =========================================================================

/// Нажать элемент: действие AXPress, при отказе — установка фокуса.
pub fn press(snapshot: &NativeSnapshot, id: usize) -> Result<(), String> {
    let element = snapshot
        .refs
        .get(id)
        .copied()
        .ok_or_else(|| "элемент не найден в снимке".to_string())?;
    let action = cf_str("AXPress");
    let err = unsafe {
        AXUIElementPerformAction(element, cf_string_ref(&action), ptr::null())
    };
    match err {
        0 => Ok(()),
        AX_ERROR_ILLEGAL_ARGUMENT | AX_ERROR_CANNOT_PERFORM => {
            Err("элемент не поддерживает нажатие".to_string())
        }
        AX_ERROR_APPLICATION_NOT_ACTIVE => {
            Err("приложение не в переднем плане".to_string())
        }
        other => Err(format!("Accessibility вернул ошибку {other}")),
    }
}

/// Вписать значение в поле и вернуть фактическое значение (readback).
pub fn set_value(snapshot: &NativeSnapshot, id: usize, text: &str) -> Result<String, String> {
    let element = snapshot
        .refs
        .get(id)
        .copied()
        .ok_or_else(|| "элемент не найден в снимке".to_string())?;
    // Значение поля в AX — не строка, а число (позиция курсора) либо текст.
    // Поэтому значение ставится через действие, а не через атрибут: так работает
    // в большинстве приложений (NSComboBox, WebKit, Electron).
    let ok = set_value_attribute(element, text);
    if !ok {
        return Err("поле не приняло значение".to_string());
    }
    Ok(attribute_string(element, "AXValue").unwrap_or_default())
}

/// Установить AXValue поля. Числовые поля (позиция курсора) не трогаем.
fn set_value_attribute(element: AXUIElementRef, text: &str) -> bool {
    let key = cf_str("AXValue");
    let value = cf_str(text);
    let err = unsafe {
        AXUIElementSetAttributeValue(
            element,
            cf_string_ref(&key),
            cf_string_ref(&value) as CFTypeRef,
        )
    };
    err == 0
}

/// Найти pid процесса по имени приложения.
pub fn pid_of(app_name: &str) -> Option<i32> {
    let trimmed = app_name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let output = std::process::Command::new("/usr/bin/pgrep")
        .args(["-x", trimmed])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next()?.trim();
    first.parse::<i32>().ok()
}

/// Имя переднего приложения (без обращения к AX).
pub fn frontmost_app() -> Option<String> {
    let output = std::process::Command::new("/usr/bin/osascript")
        .args([
            "-e",
            "tell application \"System Events\" to return name of first application process whose frontmost is true and background only is false",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Снимок переднего приложения.
pub fn collect_frontmost() -> Result<NativeSnapshot, String> {
    let app = frontmost_app().ok_or_else(|| "переднее приложение не найдено".to_string())?;
    let pid = pid_of(&app).ok_or_else(|| format!("не удалось найти процесс {app}"))?;
    let mut snapshot = collect(pid)?;
    snapshot.snapshot.app = app;
    for element in snapshot.snapshot.elements.iter_mut() {
        element.app = snapshot.snapshot.app.clone();
    }
    Ok(snapshot)
}

/// Снимок указанного приложения.
pub fn collect_named(app_name: &str) -> Result<NativeSnapshot, String> {
    let resolved = crate::platform::macos::app_resolve(app_name);
    let pid = pid_of(&resolved).ok_or_else(|| format!("приложение «{resolved}» не запущено"))?;
    let mut snapshot = collect(pid)?;
    snapshot.snapshot.app = resolved.clone();
    for element in snapshot.snapshot.elements.iter_mut() {
        element.app = resolved.clone();
    }
    Ok(snapshot)
}

/// Выполнить обход в пуле автовыпуска Objective-C.
pub fn in_pool<T>(f: impl FnOnce() -> T) -> T {
    autoreleasepool(|_| f())
}
