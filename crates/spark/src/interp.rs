//! Интерпретатор Spark: выполняет AST напрямую, без rustc.
//!
//! Зачем: плагин обязан работать на машине пользователя без установленного
//! Rust-тулчейна. Компиляция в бинарник (codegen) остаётся основным путём —
//! она быстрее в десятки раз и не перезапускается; интерпретатор —
//! автоматический fallback, когда rustc недоступен.
//!
//! Покрытие — то, на чём пишут плагины: присваивания, if/elif/else, for
//! (по списку, строке и range), while, def с возвратом и рекурсией,
//! break/continue, try/except, строковые методы (те же, что в кодогене),
//! все built-ins прелюдии, включая автоматизацию ОС. Классы, with, lambda
//! честно отклоняются: они редко нужны плагину, а ломаться молча нельзя.

use crate::ast::{Expr, Stmt};
use crate::prelude::*;
use std::collections::HashMap;

/// Значение Spark в рантайме.
#[derive(Debug, Clone)]
enum Val {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    None,
    List(Vec<Val>),
}

impl Val {
    /// Отображение для склейки строк и say: как Python str().
    fn display(&self) -> String {
        match self {
            Val::Str(s) => s.clone(),
            Val::Int(n) => n.to_string(),
            Val::Float(f) => {
                // Целые float без хвоста: 5.0 → «5», 5.5 → «5.5».
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{}", *f as i64)
                } else {
                    f.to_string()
                }
            }
            Val::Bool(b) => {
                if *b { "True" } else { "False" }.to_string()
            }
            Val::None => "None".to_string(),
            Val::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.display()).collect();
                format!("[{}]", parts.join(", "))
            }
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Val::Str(s) => !s.is_empty(),
            Val::Int(n) => *n != 0,
            Val::Float(f) => *f != 0.0,
            Val::Bool(b) => *b,
            Val::None => false,
            Val::List(items) => !items.is_empty(),
        }
    }

    fn as_int(&self) -> i64 {
        match self {
            Val::Int(n) => *n,
            Val::Float(f) => *f as i64,
            Val::Bool(b) => {
                if *b { 1 } else { 0 }
            }
            Val::Str(s) => s.trim().parse().unwrap_or(0),
            Val::None | Val::List(_) => 0,
        }
    }

    /// Ключ сравнения: значения равны, если совпадают строковые формы
    /// или числовые. Этого хватает плагинам (сравнение с "", числами).
    fn eq(&self, other: &Val) -> bool {
        match (self, other) {
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Int(a), Val::Int(b)) => a == b,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::None, Val::None) => true,
            _ => {
                if let (Some(a), Some(b)) = (self.as_num(), other.as_num()) {
                    return a == b;
                }
                self.display() == other.display()
            }
        }
    }

    fn as_num(&self) -> Option<f64> {
        match self {
            Val::Int(n) => Some(*n as f64),
            Val::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// Локальный поток управления: return/break/continue как в Python.
enum Flow {
    Normal,
    Return(Val),
    Break,
    Continue,
}

#[derive(Clone)]
struct FuncDef {
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Stmt>,
}

/// Интерпретатор одного запуска плагина.
struct Interp<'a> {
    args: &'a [String],
    scopes: Vec<HashMap<String, Val>>,
    funcs: HashMap<String, FuncDef>,
    /// Строки «SAY: …» — речь для озвучки Джарвисом.
    say: Vec<String>,
    /// Вывод print — остальной stdout плагина.
    prints: Vec<String>,
}

/// Итог интерпретации: строки для озвучки и остальной вывод.
pub struct Interpreted {
    pub say: Vec<String>,
    pub prints: Vec<String>,
}

/// Выполнить программу Spark без rustc.
pub fn interpret(ast: &[Stmt], args: &[String]) -> Result<Interpreted, String> {
    let mut interp = Interp {
        args,
        scopes: vec![HashMap::new()],
        funcs: HashMap::new(),
        say: Vec::new(),
        prints: Vec::new(),
    };
    match interp.exec_block(ast) {
        Ok(Flow::Normal) | Ok(Flow::Continue) => Ok(Interpreted {
            say: interp.say,
            prints: interp.prints,
        }),
        Ok(Flow::Break) => Err("break вне цикла".to_string()),
        Ok(Flow::Return(_)) => Err("return вне функции".to_string()),
        Err(e) => Err(e),
    }
}

impl<'a> Interp<'a> {
    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        for stmt in stmts {
            match self.exec(stmt)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<Flow, String> {
        match stmt {
            Stmt::Assign { target, value } => {
                let v = self.eval(value)?;
                self.assign(target, v);
                Ok(Flow::Normal)
            }
            Stmt::AugAssign { target, op, value } => {
                let cur = self.lookup(target)?;
                let rhs = self.eval(value)?;
                let v = self.binop(&cur, op, &rhs)?;
                self.assign(target, v);
                Ok(Flow::Normal)
            }
            Stmt::Expr(expr) => {
                self.eval(expr)?;
                Ok(Flow::Normal)
            }
            Stmt::Print(expr) => {
                let v = self.eval(expr)?;
                self.prints.push(v.display());
                Ok(Flow::Normal)
            }
            Stmt::If { condition, then_branch, elif_branches, else_branch } => {
                if self.eval(condition)?.truthy() {
                    return self.exec_block(then_branch);
                }
                for (cond, body) in elif_branches {
                    if self.eval(cond)?.truthy() {
                        return self.exec_block(body);
                    }
                }
                if let Some(body) = else_branch {
                    return self.exec_block(body);
                }
                Ok(Flow::Normal)
            }
            Stmt::While { condition, body } => {
                while self.eval(condition)?.truthy() {
                    match self.exec_block(body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::For { var, iter, body, else_body } => {
                let items = self.iterable(iter)?;
                let mut broke = false;
                for item in items {
                    self.assign(var, item);
                    match self.exec_block(body)? {
                        Flow::Break => {
                            broke = true;
                            break;
                        }
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                if !broke && let Some(eb) = else_body {
                    self.exec_block(eb)?;
                }
                Ok(Flow::Normal)
            }
            Stmt::Def { name, args, body } => {
                self.funcs.insert(
                    name.clone(),
                    FuncDef {
                        params: args.clone(),
                        body: body.clone(),
                    },
                );
                Ok(Flow::Normal)
            }
            Stmt::Return(expr) => {
                let v = match expr {
                    Some(e) => self.eval(e)?,
                    None => Val::None,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
            Stmt::Pass => Ok(Flow::Normal),
            Stmt::Try { body, except_branches, else_branch, finally_body } => {
                let outcome = self.exec_block(body);
                match outcome {
                    Ok(flow) => {
                        if let Some(eb) = else_branch {
                            self.exec_block(eb)?;
                        }
                        if let Flow::Return(_) = flow {
                            self.run_finally(finally_body)?;
                            return Ok(flow);
                        }
                        self.run_finally(finally_body)?;
                        Ok(flow)
                    }
                    Err(e) => {
                        // except по типу исключения не различаем: Spark-скрипты
                        // пишут «except:» — ловим всё. Именованная переменная
                        // ошибки не поддержана, честно отклоняем.
                        let mut matched = false;
                        for (kind, name, handler) in except_branches {
                            if name.is_some() {
                                return Err("as в except не поддержано интерпретатором".to_string());
                            }
                            if let Some(k) = kind {
                                let k = k.to_lowercase();
                                let known = ["exception", "error", "runtimeerror", "valueerror", "typeerror", "oserror", "ioerror"];
                                if !known.contains(&k.as_str()) {
                                    return Err(format!("неизвестный тип исключения: {k}"));
                                }
                            }
                            matched = true;
                            self.exec_block(handler)?;
                            break;
                        }
                        if !matched {
                            self.run_finally(finally_body)?;
                            return Err(e);
                        }
                        self.run_finally(finally_body)?;
                        Ok(Flow::Normal)
                    }
                }
            }
            Stmt::Global(_) => Ok(Flow::Normal), // один стек областей: global не нужен
            Stmt::Class { name, .. } => Err(format!(
                "классы («class {name}») интерпретатором не поддержаны — нужен rustc"
            )),
            Stmt::With { .. } => Err("with интерпретатором не поддержан — нужен rustc".to_string()),
            Stmt::Raise(expr) => {
                let msg = match expr {
                    Some(e) => self.eval(e)?.display(),
                    None => "исключение".to_string(),
                };
                Err(msg)
            }
            Stmt::Assert(cond, msg) => {
                if self.eval(cond)?.truthy() {
                    Ok(Flow::Normal)
                } else {
                    let why = match msg {
                        Some(m) => self.eval(m)?.display(),
                        None => "assertion failed".to_string(),
                    };
                    Err(why)
                }
            }
            Stmt::Nonlocal(_) => Ok(Flow::Normal),
            Stmt::Import { module, .. } => Err(format!(
                "импорт модуля «{module}» интерпретатором не поддержан — нужен rustc"
            )),
        }
    }

    fn run_finally(&mut self, finally_body: &Option<Vec<Stmt>>) -> Result<(), String> {
        if let Some(fb) = finally_body {
            self.exec_block(fb)?;
        }
        Ok(())
    }

    fn assign(&mut self, target: &str, v: Val) {
        // Записываем в ближайшую область видимости (в функции — локальную).
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(target.to_string(), v);
        }
    }

    fn lookup(&self, name: &str) -> Result<Val, String> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Ok(v.clone());
            }
        }
        Ok(Val::None)
    }

    /// Что можно итерировать: список, строка (по буквам), range().
    fn iterable(&mut self, expr: &Expr) -> Result<Vec<Val>, String> {
        match self.eval(expr)? {
            Val::List(items) => Ok(items),
            Val::Str(s) => Ok(s.chars().map(|c| Val::Str(c.to_string())).collect()),
            Val::None => Ok(Vec::new()),
            other => Err(format!("нельзя итерировать {other:?} — нужен список, строка или range()")),
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Val, String> {
        match expr {
            Expr::Number(n) => Ok(Val::Int(*n)),
            Expr::Float(f) => Ok(Val::Float(*f)),
            Expr::Str(s) => Ok(Val::Str(s.clone())),
            Expr::Bool(b) => Ok(Val::Bool(*b)),
            Expr::None => Ok(Val::None),
            Expr::Ident(name) => self.lookup(name),
            Expr::Array(els) => {
                let mut items = Vec::with_capacity(els.len());
                for e in els {
                    items.push(self.eval(e)?);
                }
                Ok(Val::List(items))
            }
            Expr::Tuple(els) => {
                let mut items = Vec::with_capacity(els.len());
                for e in els {
                    items.push(self.eval(e)?);
                }
                Ok(Val::List(items))
            }
            Expr::Dict(_) | Expr::Set(_) => {
                Err("словари и множества интерпретатором не поддержаны — нужен rustc".to_string())
            }
            Expr::BinaryOp(l, op, r) => {
                // and/or ленивые, как в Python.
                if op == "and" {
                    let lv = self.eval(l)?;
                    if !lv.truthy() {
                        return Ok(lv);
                    }
                    return self.eval(r);
                }
                if op == "or" {
                    let lv = self.eval(l)?;
                    if lv.truthy() {
                        return Ok(lv);
                    }
                    return self.eval(r);
                }
                let lv = self.eval(l)?;
                let rv = self.eval(r)?;
                self.binop(&lv, op, &rv)
            }
            Expr::UnaryOp(op, inner) => {
                let v = self.eval(inner)?;
                match op.as_str() {
                    "not" => Ok(Val::Bool(!v.truthy())),
                    "-" => match v {
                        Val::Int(n) => Ok(Val::Int(-n)),
                        Val::Float(f) => Ok(Val::Float(-f)),
                        other => Err(format!("нельзя сделать отрицательным: {other:?}")),
                    },
                    "+" => Ok(v),
                    other => Err(format!("неизвестный унарный оператор «{other}»")),
                }
            }
            Expr::Compare(l, op, r) => {
                let lv = self.eval(l)?;
                let rv = self.eval(r)?;
                Ok(Val::Bool(match op.as_str() {
                    "==" => lv.eq(&rv),
                    "!=" => !lv.eq(&rv),
                    ">" | "<" | ">=" | "<=" => {
                        let (a, b) = num_pair(&lv, &rv)?;
                        match op.as_str() {
                            ">" => a > b,
                            "<" => a < b,
                            ">=" => a >= b,
                            _ => a <= b,
                        }
                    }
                    "is" => lv.eq(&rv),
                    "in" => contains(&rv, &lv)?,
                    other => return Err(format!("неизвестное сравнение «{other}»")),
                }))
            }
            Expr::Index { array, index } => {
                let a = self.eval(array)?;
                let i = self.eval(index)?;
                match (a, i.as_int()) {
                    (Val::Str(s), i) => {
                        let idx = norm_index(i, s.chars().count())?;
                        Ok(s.chars().nth(idx).map(|c| Val::Str(c.to_string())).unwrap_or(Val::None))
                    }
                    (Val::List(items), i) => {
                        let idx = norm_index(i, items.len())?;
                        Ok(items.get(idx).cloned().unwrap_or(Val::None))
                    }
                    (a, _) => Err(format!("нельзя индексировать {a:?}")),
                }
            }
            Expr::MethodCall { receiver, method, args } => {
                let recv = self.eval(receiver)?;
                let mut argv = Vec::with_capacity(args.len());
                for a in args {
                    argv.push(self.eval(a)?);
                }
                self.method(recv, method, &argv)
            }
            Expr::Call { func, args } => {
                let name = match func.as_ref() {
                    Expr::Ident(n) => n.clone(),
                    other => {
                        // вызов результата выражения — не поддержан
                        return Err(format!("нельзя вызвать {other:?}"));
                    }
                };
                let mut argv = Vec::with_capacity(args.len());
                for a in args {
                    argv.push(self.eval(a)?);
                }
                self.call(&name, &argv)
            }
            Expr::Lambda { .. } => {
                Err("lambda интерпретатором не поддержана — нужен rustc".to_string())
            }
            Expr::Ternary { cond, then, else_ } => {
                if self.eval(cond)?.truthy() {
                    self.eval(then)
                } else {
                    self.eval(else_)
                }
            }
            Expr::Input(_) => Ok(Val::Str(input())),
        }
    }

    fn binop(&self, l: &Val, op: &str, r: &Val) -> Result<Val, String> {
        match op {
            "+" => match (l, r) {
                (Val::Str(a), b) => Ok(Val::Str(format!("{}{}", a, b.display()))),
                (Val::List(a), Val::List(b)) => {
                    let mut items = a.clone();
                    items.extend(b.clone());
                    Ok(Val::List(items))
                }
                _ => {
                    let (a, b) = num_pair(l, r)?;
                    if l.matches_float() || r.matches_float() {
                        Ok(Val::Float(a + b))
                    } else {
                        Ok(Val::Int(a as i64 + b as i64))
                    }
                }
            },
            "-" | "*" | "/" | "//" | "%" | "**" => {
                // Строки: "ab" * 2 → "abab", как в Python.
                if op == "*" && matches!(l, Val::Str(_)) && matches!(r, Val::Int(_)) {
                    let (s, n) = (match l { Val::Str(s) => s.clone(), _ => String::new() }, r.as_int());
                    return Ok(Val::Str(s.repeat(n.max(0) as usize)));
                }
                let (a, b) = num_pair(l, r)?;
                let float = l.matches_float() || r.matches_float();
                if (op == "/" || op == "//" || op == "%") && b == 0.0 {
                    return Err("деление на ноль".to_string());
                }
                match op {
                    "-" => Ok(if float { Val::Float(a - b) } else { Val::Int(a as i64 - b as i64) }),
                    "*" => Ok(if float { Val::Float(a * b) } else { Val::Int(a as i64 * b as i64) }),
                    "/" => Ok(Val::Float(a / b)),
                    "//" => Ok(if float { Val::Float((a / b).floor()) } else { Val::Int(a as i64 / b as i64) }),
                    "%" => Ok(if float { Val::Float(a % b) } else { Val::Int(a as i64 % b as i64) }),
                    "**" => Ok(if float { Val::Float(a.powf(b)) } else { Val::Int((a as i64).pow(b as u32)) }),
                    _ => unreachable!(),
                }
            }
            other => Err(format!("неизвестный оператор «{other}»")),
        }
    }

    fn method(&mut self, recv: Val, method: &str, args: &[Val]) -> Result<Val, String> {
        let s = recv.display();
        let arg0 = args.first().map(|v| v.display()).unwrap_or_default();
        match method {
            "upper" => Ok(Val::Str(s.to_uppercase())),
            "lower" => Ok(Val::Str(s.to_lowercase())),
            "trim" | "strip" => Ok(Val::Str(s.trim().to_string())),
            "lstrip" => Ok(Val::Str(s.trim_start().to_string())),
            "rstrip" => Ok(Val::Str(s.trim_end().to_string())),
            "replace" if args.len() == 2 => {
                Ok(Val::Str(s.replace(&arg0, &args[1].display())))
            }
            "contains" if args.len() == 1 => Ok(Val::Bool(s.contains(&arg0))),
            "startswith" if args.len() == 1 => Ok(Val::Bool(s.starts_with(&arg0))),
            "endswith" if args.len() == 1 => Ok(Val::Bool(s.ends_with(&arg0))),
            "split" if args.len() == 1 => Ok(Val::List(
                s.split(&arg0).map(|p| Val::Str(p.to_string())).collect(),
            )),
            "join" if args.len() == 1 => match args.first() {
                Some(Val::List(items)) => {
                    let parts: Vec<String> = items.iter().map(|v| v.display()).collect();
                    Ok(Val::Str(parts.join(&s)))
                }
                _ => Err("join ждёт список".to_string()),
            },
            "len" => Ok(Val::Int(s.chars().count() as i64)),
            "find" | "index" if args.len() == 1 => Ok(Val::Int(
                s.find(&arg0).map(|i| i as i64).unwrap_or(-1),
            )),
            _ => Err(format!(
                "метод «.{method}» интерпретатором не поддержан — нужен rustc"
            )),
        }
    }

    fn call(&mut self, name: &str, argv: &[Val]) -> Result<Val, String> {
        // Функции пользователя — прежде built-ins: плагин может определить
        // свою «состояние», это не должно ломать язык.
        if let Some(def) = self.funcs.get(name).cloned() {
            return self.call_user(name, &def, argv);
        }
        let s = |i: usize| argv.get(i).map(|v| v.display()).unwrap_or_default();
        let n = |i: usize| argv.get(i).map(|v| v.as_int()).unwrap_or(0);
        let need = |count: usize, what: &str| -> Result<(), String> {
            if argv.len() < count {
                return Err(format!("{what}: мало аргументов (нужно {count})"));
            }
            Ok(())
        };
        match name {
            "print" => {
                let line = argv.iter().map(|v| v.display()).collect::<Vec<_>>().join(" ");
                self.prints.push(line);
                Ok(Val::None)
            }
            "say" => {
                self.say.push(s(0));
                Ok(Val::None)
            }
            "arg" => {
                // Аргументы берём из переданных плагину, не из env::args:
                // интерпретатор работает внутри чужого процесса. У скомпи-
                // лированного бинарника argv[0] — сам бинарник (nth(1) —
                // первый аргумент), здесь списка argv[0] нет, поэтому
                // смещение на единицу меньше.
                let idx = n(0).max(1) as usize;
                Ok(Val::Str(
                    self.args.get(idx.saturating_sub(1)).cloned().unwrap_or_default(),
                ))
            }
            "env" => Ok(Val::Str(spark_env(s(0)))),
            "run" => Ok(Val::Str(spark_run(s(0)))),
            "launch" => Ok(Val::Str(spark_launch(s(0)))),
            "read_file" => Ok(Val::Str(spark_read(s(0)))),
            "write_file" => Ok(Val::Str(spark_write(s(0), s(1)))),
            "notify" => Ok(Val::Str(spark_notify(s(0), s(1)))),
            "input" => Ok(Val::Str(input())),
            "str" => Ok(Val::Str(argv.first().map(|v| v.display()).unwrap_or_default())),
            "int" => Ok(Val::Int(argv.first().map(|v| v.as_int()).unwrap_or(0))),
            "float" => Ok(Val::Float(
                argv.first().and_then(|v| v.as_num()).unwrap_or(0.0),
            )),
            "len" => match argv.first() {
                Some(Val::Str(s)) => Ok(Val::Int(s.chars().count() as i64)),
                Some(Val::List(items)) => Ok(Val::Int(items.len() as i64)),
                _ => Err("len: нужна строка или список".to_string()),
            },
            "range" => {
                let (from, to) = match argv.len() {
                    1 => (0, argv[0].as_int()),
                    _ => (argv[0].as_int(), argv[1].as_int()),
                };
                Ok(Val::List((from..to).map(Val::Int).collect()))
            }
            // ── Автоматизация ОС (та же прелюдия, что в скомпилированном плагине) ──
            "send_hotkey" | "сочетание" | "хоткей" => Ok(Val::Str(spark_send_hotkey(s(0)))),
            "type_text" | "напечатать" | "напечатать_текст" => Ok(Val::Str(spark_type_text(s(0)))),
            "mouse_click" | "клик" => Ok(Val::Str(spark_mouse_click(n(0), n(1)))),
            "mouse_right_click" | "правый_клик" => Ok(Val::Str(spark_mouse_right_click(n(0), n(1)))),
            "mouse_move" | "движение_мыши" => Ok(Val::Str(spark_mouse_move(n(0), n(1)))),
            "mouse_scroll" | "прокрутка" => Ok(Val::Str(spark_mouse_scroll(s(0), n(1)))),
            "window_list" | "окна" => Ok(Val::Str(spark_window_list())),
            "window_focus" | "фокус" => Ok(Val::Str(spark_window_focus(s(0)))),
            "window_close" | "закрыть_окно" => Ok(Val::Str(spark_window_close(s(0)))),
            "window_minimize" | "свернуть_окно" => Ok(Val::Str(spark_window_minimize(s(0)))),
            "window_maximize" | "развернуть_окно" => Ok(Val::Str(spark_window_maximize(s(0)))),
            "window_move_desktop" | "перенести_окно" => Ok(Val::Str(spark_window_move_desktop(s(0), n(1)))),
            "desktop_switch" | "стол" => Ok(Val::Str(spark_desktop_switch(n(0)))),
            "desktop_next" | "следующий_стол" => Ok(Val::Str(spark_desktop_next())),
            "desktop_prev" | "предыдущий_стол" => Ok(Val::Str(spark_desktop_prev())),
            "volume_set" | "громкость" => Ok(Val::Str(spark_volume_set(n(0)))),
            "volume_get" => Ok(Val::Str(spark_volume_get())),
            "volume_mute" | "без_звука" => Ok(Val::Str(spark_volume_mute())),
            "brightness_set" | "яркость" => Ok(Val::Str(spark_brightness_set(n(0)))),
            "screenshot" | "скриншот" => Ok(Val::Str(spark_screenshot(s(0)))),
            "clip_set" | "в_буфер" => Ok(Val::Str(spark_clip_set(s(0)))),
            "clip_get" | "буфер" => Ok(Val::Str(spark_clip_get())),
            "http_get" | "запрос" => Ok(Val::Str(spark_http_get(s(0)))),
            "app_open" | "открыть_приложение" => Ok(Val::Str(spark_app_open(s(0)))),
            "app_quit" | "закрыть_приложение" => Ok(Val::Str(spark_app_quit(s(0)))),
            "battery" | "батарея" => Ok(Val::Str(spark_battery())),
            "screen_size" | "экран_размер" => Ok(Val::Str(spark_screen_size())),
            "sleep_ms" | "пауза" => Ok(Val::Str(spark_sleep_ms(n(0)))),
            other => {
                need(0, other)?;
                Err(format!("функция «{other}» не найдена"))
            }
        }
    }

    /// Вызов пользовательской функции: новая область видимости, значения
    /// по умолчанию вычисляются в вызывающем контексте.
    fn call_user(&mut self, _name: &str, def: &FuncDef, argv: &[Val]) -> Result<Val, String> {
        let mut scope = HashMap::new();
        for (i, (param, default_expr)) in def.params.iter().enumerate() {
            let v = match argv.get(i) {
                Some(v) => v.clone(),
                None => match default_expr {
                    Some(e) => self.eval(e)?,
                    None => Val::None,
                },
            };
            scope.insert(param.clone(), v);
        }
        self.scopes.push(scope);
        let flow = self.exec_block(&def.body.clone());
        self.scopes.pop();
        match flow? {
            Flow::Return(v) => Ok(v),
            _ => Ok(Val::None),
        }
    }
}

impl Val {
    /// Есть ли в паре float: «5» + 0.5 должно дать 5.5, а не 5.
    fn matches_float(&self) -> bool {
        matches!(self, Val::Float(_))
    }
}

/// Числовая пара для арифметики и сравнений: строки в арифметике — ошибка
/// с понятным текстом (в Python то же самое).
fn num_pair(l: &Val, r: &Val) -> Result<(f64, f64), String> {
    match (l.as_num(), r.as_num()) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(format!(
            "арифметика «{}» и «{}» не сходится: нужны числа",
            l.display(),
            r.display()
        )),
    }
}

/// Отрицательный индекс (-1 — последний), выход за край — честная ошибка.
fn norm_index(i: i64, len: usize) -> Result<usize, String> {
    let len = len as i64;
    let idx = if i < 0 { len + i } else { i };
    if idx < 0 || idx >= len {
        return Err(format!("индекс {i} за границей (длина {len})"));
    }
    Ok(idx as usize)
}

/// «needle in haystack»: в строке — подстрока, в списке — элемент.
fn contains(haystack: &Val, needle: &Val) -> Result<bool, String> {
    match haystack {
        Val::Str(s) => Ok(s.contains(&needle.display())),
        Val::List(items) => Ok(items.iter().any(|v| v.eq(needle))),
        other => Err(format!("«in» работает со строкой или списком, не с {other:?}")),
    }
}
