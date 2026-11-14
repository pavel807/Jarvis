use crate::ast::{Expr, Stmt};
use std::collections::HashSet;
use std::fs;
use std::process::Command;
use tempfile::Builder;

/// Системная прелюдия для плагинов-автоматизаций: аргументы командной
/// строки, shell, файлы, уведомления. Все функции возвращают String —
/// кодоген Spark'а оборачивает любой вызов в `.to_string()`.
const PRELUDE: &str = include_str!("prelude.rs");

/// Сгенерировать Rust-код из AST без вызова rustc: движок Джарвиса
/// компилирует плагины сам, CLI Spark — через эту же генерацию.
/// Контекст генерации: известные коллекции (`for x in items`) и функции
/// (`def f():` + `f()` без аргументов — вызов, а не константа).
pub struct Ctx {
    vecs: HashSet<String>,
    fns: HashSet<String>,
}

pub fn compile_to_string(ast: Vec<Stmt>) -> String {

    // Прелюдия самодостаточна (свои use) — отдельный заголовок не нужен.
    let mut rust_code = String::new();

    let mut funcs = Vec::new();
    let mut classes = Vec::new();
    let mut main_stmts = Vec::new();

    for stmt in ast {
        match stmt {
            Stmt::Def { .. } => funcs.push(stmt),
            Stmt::Class { .. } => classes.push(stmt),
            Stmt::Import { .. } => main_stmts.push(stmt),
            _ => main_stmts.push(stmt),
        }
    }

    // Переменные-коллекции: `items = [...]` — чтобы `for x in items`
    // итерировал по ним, а не делал 0..items. Плюс имена функций: вызов
    // без аргументов — вызов, а не поле.
    let mut vecs = HashSet::new();
    let mut fns = HashSet::new();
    collect_vecs(&main_stmts, &mut vecs);
    for f in &funcs {
        if let Stmt::Def { name, body, .. } = f {
            fns.insert(name.clone());
            collect_vecs(body, &mut vecs);
        }
    }
    let ctx = Ctx { vecs, fns };

    rust_code.push_str(PRELUDE);
    rust_code.push('\n');

    for class in classes {
        rust_code.push_str(&gen_stmt(class, &ctx));
        rust_code.push('\n');
    }

    for func in funcs {
        rust_code.push_str(&gen_stmt(func, &ctx));
    }

    rust_code.push_str("fn main() {\n");
    for stmt in main_stmts {
        rust_code.push_str(&gen_stmt(stmt, &ctx));
    }
    rust_code.push_str("}\n");

    rust_code
}

/// Скомпилировать .sk-программу в бинарник: генерация + rustc.
/// Возвращает Err с выводом rustc, если сборка не удалась.
pub fn compile_to_binary(ast: Vec<Stmt>, output: &str, opt_level: i32) -> Result<(), String> {
    let rust_code = compile_to_string(ast);

    let tmp_dir = Builder::new().prefix("spark_build").tempdir().unwrap();
    let rs_path = tmp_dir.path().join("main.rs");

    fs::write(&rs_path, &rust_code).expect("Failed to write Rust file");

    let opt_args: Vec<&str> = match opt_level {
        1 => vec!["-C", "opt-level=1"],
        2 => vec!["-C", "opt-level=2"],
        3 => vec!["-C", "opt-level=3"],
        _ => vec!["-C", "opt-level=3"],
    };

    let out = Command::new("rustc")
        .args(["--edition", "2024"])
        .args(&opt_args)
        .arg(rs_path.to_str().unwrap())
        .arg("-o")
        .arg(output)
        .output();

    match out {
        Ok(o) if o.status.success() => {
            println!("✅ Spark скомпилировался в: {} (opt-level={})", output, opt_level);
            Ok(())
        }
        Ok(o) => Err(String::from_utf8_lossy(&o.stderr).into_owned()),
        Err(e) => Err(format!("rustc не запустился: {e}")),
    }
}

fn gen_stmt(stmt: Stmt, ctx: &Ctx) -> String {
    match stmt {
        Stmt::Assign { target, value } => {
            format!("    let {} = {};\n", target, gen_expr(value, ctx))
        }
        Stmt::AugAssign { target, op, value } => {
            format!("    {} {} = {};\n", target, op, gen_expr(value, ctx))
        }
        Stmt::Expr(expr) => {
            let e = gen_expr(expr, ctx);
            if e.starts_with("println!") {
                format!("    {};\n", e)
            } else if e.contains(" = ") {
                format!("    let {};\n", e)
            } else {
                // Голый вызов — как в Python: значение отбрасывается.
                // Раньше say(...) печатал своё «ok» в stdout плагина.
                format!("    let _ = {};\n", e)
            }
        }
        Stmt::Print(expr) => {
            format!("    println!(\"{{}}\", {});\n", gen_expr(expr, ctx))
        }
        Stmt::If { condition, then_branch, elif_branches, else_branch } => {
            let mut code = format!("    if {} {{\n", gen_expr(condition, ctx));
            for s in then_branch {
                code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
            }
            code.push_str("    }");
            for (cond, body) in elif_branches {
                code.push_str(&format!("\n    else if {} {{\n", gen_expr(cond, ctx)));
                for s in body {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }");
            }
            if let Some(else_stmts) = else_branch {
                code.push_str(" else {\n");
                for s in else_stmts {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }");
            }
            code.push('\n');
            code
        }
        Stmt::While { condition, body } => {
            let mut code = format!("    while {} {{\n", gen_expr(condition, ctx));
            for s in body {
                code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
            }
            code.push_str("    }\n");
            code
        }
        Stmt::For { var, iter, body, else_body } => {
            let iter_str = gen_expr(iter.clone(), ctx);
            // Коллекция для for: литерал массива/кортежа, переменная,
            // помеченная как коллекция (`items = [...]`), или выражение,
            // уже содержащее диапазон/итератор. Иначе — счётчик 0..N.
            let collection = matches!(iter, Expr::Array(_) | Expr::Tuple(_))
                || matches!(&iter, Expr::Ident(name) if ctx.vecs.contains(name))
                || iter_str.contains("0..")
                || iter_str.contains(".iter()")
                || iter_str.contains("range");
            let mut code = if collection {
                format!("    for {} in {} {{\n", var, iter_str)
            } else {
                format!("    for {} in 0..{} {{\n", var, iter_str)
            };
            for s in body {
                code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
            }
            code.push_str("    }");
            if let Some(else_stmts) = else_body {
                code.push_str(" else {\n");
                for s in else_stmts {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }");
            }
            code.push('\n');
            code
        }
        Stmt::Def { name, args, body } => {
            let params: Vec<String> = args.iter().map(|(s, _)| format!("{}: &str", s)).collect();
            let mut code = format!("fn {}({}) -> String {{\n", name, params.join(", "));
            for s in body {
                code.push_str(&gen_stmt(s, ctx));
            }
            code.push_str("}\n");
            code
        }
        Stmt::Class { name, bases: _, body } => {
            let mut struct_code = format!("struct {} {{\n", name);
            let mut impl_code = format!("impl {} {{\n", name);
            let mut has_new = false;
            
            for stmt in body {
                match stmt {
                    Stmt::Def { name: func_name, args, body: func_body } => {
                        if func_name == name {
                            let params: Vec<String> = args.iter().map(|(s, _)| s.clone()).collect();
                            impl_code.push_str(&format!("    pub fn new({}) -> Self {{\n", params.join(", ")));
                            for s in func_body {
                                impl_code.push_str(&gen_stmt(s, ctx));
                            }
                            impl_code.push_str("    }\n");
                            has_new = true;
                        } else {
                            let params: Vec<String> = args.iter().map(|(s, _)| s.clone()).collect();
                            impl_code.push_str(&format!("    pub fn {}({}) {{\n", func_name, params.join(", ")));
                            for s in func_body {
                                impl_code.push_str(&gen_stmt(s, ctx));
                            }
                            impl_code.push_str("    }\n");
                        }
                    }
                    Stmt::Assign { target, value: _ } => {
                        struct_code.push_str(&format!("    pub {}: i32,\n", target));
                    }
                    _ => {}
                }
            }
            
            if !has_new {
                impl_code.push_str(&format!("    pub fn new() -> Self {{\n",));
                impl_code.push_str(&format!("        {} {{ }}\n", name));
                impl_code.push_str("    }\n");
            }
            
            struct_code.push_str("}\n\n");
            impl_code.push_str("}\n");
            
            struct_code + &impl_code
        }
        Stmt::Try { body, except_branches, else_branch, finally_body } => {
            let mut code = String::new();
            code.push_str("    {\n");
            
            for s in body {
                code.push_str(&gen_stmt(s, ctx));
            }
            
            if !except_branches.is_empty() {
                for (exc_type, alias, exc_body) in except_branches {
                    let _type_str = exc_type.unwrap_or_else(|| "Box<dyn std::error::Error>".to_string());
                    let alias_str = alias.unwrap_or_else(|| "e".to_string());
                    code.push_str(&format!("    }}\n    Err({}) => {{\n", alias_str));
                    for s in exc_body {
                        code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                    }
                    code.push_str("    }\n");
                }
            }
            
            if let Some(else_stmts) = else_branch {
                code.push_str("    Ok(_) => {\n");
                for s in else_stmts {
                    code.push_str(&indent_stmt(&gen_stmt(s, ctx)));
                }
                code.push_str("    }\n");
            }
            
            if let Some(finally_stmts) = finally_body {
                for s in finally_stmts {
                    code.push_str(&gen_stmt(s, ctx));
                }
            }
            
            code.push_str("    }\n");
            code
        }
        Stmt::Raise(expr) => {
            if let Some(e) = expr {
                format!("    panic!(\"{{}}\", {});\n", gen_expr(e, ctx))
            } else {
                "    panic!(\"Raised exception\");\n".to_string()
            }
        }
        Stmt::Assert(cond, msg) => {
            if let Some(m) = msg {
                format!("    assert!({}, \"{{}}\", {});\n", gen_expr(cond, ctx), gen_expr(m, ctx))
            } else {
                format!("    assert!({});\n", gen_expr(cond, ctx))
            }
        }
        Stmt::Return(expr) => {
            if let Some(e) = expr {
                let expr_str = gen_expr(e, ctx);
                if !expr_str.contains('=') {
                    // Def'ы Spark возвращают String: &str/число приводим.
                    if expr_str.starts_with('"') || expr_str.ends_with(".to_string()")
                        || expr_str.starts_with("format!") || expr_str.starts_with("String::")
                    {
                        format!("    {}\n", expr_str)
                    } else {
                        format!("    return {}.to_string();\n", expr_str)
                    }
                } else {
                    format!("    let {};\n", expr_str)
                }
            } else {
                "    return String::new();\n".to_string()
            }
        }
        Stmt::Break => "    break;\n".to_string(),
        Stmt::Continue => "    continue;\n".to_string(),
        Stmt::Pass => "".to_string(),
        Stmt::Global(vars) => {
            format!("    // global {}\n", vars.join(", "))
        }
        Stmt::Nonlocal(vars) => {
            format!("    // nonlocal {}\n", vars.join(", "))
        }
        Stmt::Import { module, items, alias } => {
            if let Some(a) = alias {
                format!("    // import {} as {}\n", module, a)
            } else if let Some(items) = items {
                format!("    // from {} import {}\n", module, items.join(", "))
            } else {
                format!("    // import {}\n", module)
            }
        }
        Stmt::With { items, body } => {
            let mut code = String::new();
            for (expr, alias) in items {
                let var = alias.unwrap_or_else(|| "_".to_string());
                code.push_str(&format!("    let {} = {};\n", var, gen_expr(expr, ctx)));
            }
            for s in body {
                code.push_str(&gen_stmt(s, ctx));
            }
            code
        }
    }
}


/// Системные built-ins плагинов: имена в Spark → функции прелюдии.
fn builtin(fname: &str, args: &[String]) -> Option<String> {
    let call = |n: &str| {
        if args.is_empty() {
            format!("{n}().to_string()")
        } else {
            format!("{n}({}).to_string()", args.join(", "))
        }
    };
    match fname {
        "arg" | "аргумент" => Some(call("spark_arg")),
        "env" | "переменная" => Some(call("spark_env")),
        "run" | "выполнить" => Some(call("spark_run")),
        "launch" | "запустить" => Some(call("spark_launch")),
        "read_file" | "прочитать" => Some(call("spark_read")),
        "write_file" | "записать" => Some(call("spark_write")),
        "notify" | "уведомить" => Some(call("spark_notify")),
        "say" | "сказать" => Some(call("spark_say")),
        // ── Автоматизация ОС, аналог AutoHotkey (v0.3) ──
        "send_hotkey" | "сочетание" | "хоткей" => Some(call("spark_send_hotkey")),
        "type_text" | "напечатать" | "напечатать_текст" => Some(call("spark_type_text")),
        "mouse_click" | "клик" => Some(call("spark_mouse_click")),
        "mouse_right_click" | "правый_клик" => Some(call("spark_mouse_right_click")),
        "mouse_move" | "движение_мыши" => Some(call("spark_mouse_move")),
        "mouse_scroll" | "прокрутка" => Some(call("spark_mouse_scroll")),
        "window_list" | "окна" => Some(call("spark_window_list")),
        "window_focus" | "фокус" => Some(call("spark_window_focus")),
        "window_close" | "закрыть_окно" => Some(call("spark_window_close")),
        "window_minimize" | "свернуть_окно" => Some(call("spark_window_minimize")),
        "window_maximize" | "развернуть_окно" => Some(call("spark_window_maximize")),
        "window_move_desktop" | "перенести_окно" => Some(call("spark_window_move_desktop")),
        "desktop_switch" | "стол" => Some(call("spark_desktop_switch")),
        "desktop_next" | "следующий_стол" => Some(call("spark_desktop_next")),
        "desktop_prev" | "предыдущий_стол" => Some(call("spark_desktop_prev")),
        "volume_set" | "громкость" => Some(call("spark_volume_set")),
        "volume_get" => Some(call("spark_volume_get")),
        "volume_mute" | "без_звука" => Some(call("spark_volume_mute")),
        "brightness_set" | "яркость" => Some(call("spark_brightness_set")),
        "screenshot" | "скриншот" => Some(call("spark_screenshot")),
        "clip_set" | "в_буфер" => Some(call("spark_clip_set")),
        "clip_get" | "буфер" => Some(call("spark_clip_get")),
        "http_get" | "запрос" => Some(call("spark_http_get")),
        "app_open" | "открыть_приложение" => Some(call("spark_app_open")),
        "app_quit" | "закрыть_приложение" => Some(call("spark_app_quit")),
        "battery" | "батарея" => Some(call("spark_battery")),
        "screen_size" | "экран_размер" => Some(call("spark_screen_size")),
        "sleep_ms" | "пауза" => Some(call("spark_sleep_ms")),
        // str(x) — Python-привычка: привести что угодно к строке.
        "str" if !args.is_empty() => Some(format!("format!(\"{{}}\", {})", args[0])),
        // int(x) — аргументы плагина приходят строками, арифметика
        // требует числа. Нечисло → 0 (плагин сам решает, что с этим делать).
        "int" if !args.is_empty() => {
            Some(format!("({}.to_string().parse::<i64>().unwrap_or(0))", args[0]))
        }
        _ => None,
    }
}

/// Собрать имена переменных, присвоенным значением которых был массив,
/// кортеж или вызов run: по ним `for x in имя` итерирует коллекцию.
fn collect_vecs(stmts: &[Stmt], out: &mut HashSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign { target, value } => {
                if matches!(value, Expr::Array(_) | Expr::Tuple(_)) {
                    out.insert(target.clone());
                }
            }
            Stmt::If { then_branch, elif_branches, else_branch, .. } => {
                collect_vecs(then_branch, out);
                for (_, body) in elif_branches {
                    collect_vecs(body, out);
                }
                if let Some(b) = else_branch {
                    collect_vecs(b, out);
                }
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => collect_vecs(body, out),
            _ => {}
        }
    }
}

fn indent_stmt(stmt: &str) -> String {
    stmt.lines()
        .map(|line| format!("    {}", line))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn gen_expr(expr: Expr, ctx: &Ctx) -> String {
    match expr {
        Expr::Number(n) => n.to_string(),
        Expr::Float(f) => f.to_string(),
        Expr::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Expr::Ident(i) => i,
        Expr::Bool(b) => if b { "true" } else { "false" }.to_string(),
        Expr::None => "None".to_string(),
        Expr::Array(els) => {
            let parts: Vec<String> = els.into_iter().map(|e| gen_expr(e, ctx)).collect();
            format!("vec![{}]", parts.join(", "))
        }
        Expr::Tuple(els) => {
            let parts: Vec<String> = els.into_iter().map(|e| gen_expr(e, ctx)).collect();
            format!("({})", parts.join(", "))
        }
        Expr::Dict(pairs) => {
            let parts: Vec<String> = pairs.iter()
                .map(|(k, v)| format!("{} => {}", gen_expr(k.clone(), ctx), gen_expr(v.clone(), ctx)))
                .collect();
            format!("std::collections::HashMap::from([{}])", parts.join(", "))
        }
        Expr::Set(els) => {
            let parts: Vec<String> = els.into_iter().map(|e| gen_expr(e, ctx)).collect();
            format!("std::collections::HashSet::from([{}])", parts.join(", "))
        }
        Expr::BinaryOp(left, op, right) => {
            if op == "+" {
                let l = gen_expr(*left, ctx);
                let r = gen_expr(*right, ctx);
                // Строковый контекст: литерал или то, что уже String
                // (вызовы Spark-функций кодоген оборачивает в .to_string()).
                let l_str = l.starts_with('"') || l.ends_with(".to_string()")
                    || l.starts_with("format!(");
                let r_str = r.starts_with('"') || r.ends_with(".to_string()")
                    || r.starts_with("format!(");
                if l_str || r_str {
                    format!("format!(\"{{}}{{}}\", {}, {})", l, r)
                } else {
                    format!("{} + {}", l, r)
                }
            } else {
                // Питоновские логика и принадлежность: and/or/not/in.
                match op.as_str() {
                    "and" => format!("({} && {})", gen_expr(*left, ctx), gen_expr(*right, ctx)),
                    "or" => format!("({} || {})", gen_expr(*left, ctx), gen_expr(*right, ctx)),
                    "in" => format!(
                        "({}).contains(&{})",
                        gen_expr(*right, ctx),
                        gen_expr(*left, ctx)
                    ),
                    _ => format!("({} {} {})", gen_expr(*left, ctx), op, gen_expr(*right, ctx)),
                }
            }
        }
        Expr::UnaryOp(op, expr) => {
            let e = gen_expr(*expr, ctx);
            match op.as_str() {
                "not" => format!("(!{})", e),
                _ => format!("({}{})", op, e),
            }
        }
        Expr::Compare(left, op, right) => {
            format!("({} {} {})", gen_expr(*left, ctx), op, gen_expr(*right, ctx))
        }
        Expr::Index { array, index } => {
            format!("{}[{}]", gen_expr(*array, ctx), gen_expr(*index, ctx))
        }
        Expr::MethodCall { receiver, method, args } => {
            let recv = gen_expr(*receiver, ctx);
            let a: Vec<String> = args.iter().map(|x| gen_expr(x.clone(), ctx)).collect();
            // Питоновские строковые методы → Rust; неизвестные — как есть
            // (плагин может звать и rust-методы напрямую).
            match method.as_str() {
                "upper" => format!("{recv}.to_uppercase()"),
                "lower" => format!("{recv}.to_lowercase()"),
                "trim" | "strip" => format!("{recv}.trim().to_string()"),
                "lstrip" => format!("{recv}.trim_start().to_string()"),
                "rstrip" => format!("{recv}.trim_end().to_string()"),
                "replace" if a.len() == 2 => format!("{recv}.replace({}, {})", a[0], a[1]),
                "contains" if a.len() == 1 => format!("{recv}.contains({})", a[0]),
                "startswith" if a.len() == 1 => format!("{recv}.starts_with({})", a[0]),
                "endswith" if a.len() == 1 => format!("{recv}.ends_with({})", a[0]),
                "split" if a.len() == 1 => format!(
                    "{recv}.split({}).map(|s| s.to_string()).collect::<Vec<String>>()", a[0]),
                "join" if a.len() == 1 => format!("{}.join(&{})", a[0], recv),
                "len" => format!("{recv}.chars().count()"),
                "find" | "index" if a.len() == 1 => format!(
                    "{recv}.find({}).map(|i| i as i64).unwrap_or(-1)", a[0]),
                _ => format!("{}.{}({})", recv, method, a.join(", ")),
            }
        }
        Expr::Call { func, args } => {
            let fname = gen_expr(*func, ctx);
            let args_str: Vec<String> = args.iter().map(|a| gen_expr(a.clone(), ctx)).collect();
            
            if fname == "print" {
                if args_str.is_empty() {
                    "()".to_string()
                } else {
                    format!("println!(\"{{}}\", {})", args_str.join(", "))
                }
            } else if fname == "range" {
                if args_str.len() == 1 {
                    format!("0..{}", args_str[0])
                } else if args_str.len() == 2 {
                    format!("{}..{}", args_str[0], args_str[1])
                } else {
                    format!("0..1")
                }
            } else if let Some(mapped) = builtin(&fname, &args_str) {
                mapped
            } else if args_str.is_empty() {
                if ctx.fns.contains(&fname) {
                    format!("{}().to_string()", fname)
                } else {
                    format!("{}.to_string()", fname)
                }
            } else {
                format!("{}({}).to_string()", fname, args_str.join(", "))
            }
        }
        Expr::Lambda { args, body } => {
            let params: Vec<String> = args.iter().map(|s| s.clone()).collect();
            let body_str = gen_expr(*body, ctx);
            format!("|{}| {}", params.join(", "), body_str)
        }
        Expr::Ternary { cond, then, else_ } => {
            format!(
                "if {} {{ {} }} else {{ {} }}",
                gen_expr(*cond, ctx),
                gen_expr(*then, ctx),
                gen_expr(*else_, ctx)
            )
        }
        Expr::Input(_prompt) => {
            format!("{{ input() }}")
        }
    }
}