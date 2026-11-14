//! Spark Lang — скриптовый язык с питон-like синтаксисом, вендор из
//! github.com/pavel807/Spark. Компилирует .sk в Rust и собирает rustc'ом.
//!
//! Джарвис использует библиотеку для плагинов-автоматизаций: исходник
//! плагина → AST → Rust-код → бинарник (кэш по хэшу исходника). Бинарник
//! `spark` CLI сохранён как есть.

pub mod ast;
pub mod codegen;
pub mod interp;
pub mod prelude;
pub mod lexer;
pub mod parser;

use logos::Logos;

/// Прогнать исходник через лексер и парсер. Ошибка лексера — честный
/// текст с номером строки, а не паника: плагин с опечаткой не должен
/// ронять ассистента.
pub fn parse_source(source: &str) -> Result<Vec<ast::Stmt>, String> {
    let mut tokens = Vec::new();
    for (res, span) in lexer::Token::lexer(source).spanned() {
        let line = source[..span.start].bytes().filter(|b| *b == b'\n').count() + 1;
        tokens.push(res.map_err(|_| format!("строка {line}: не понял символ"))?);
    }
    let tokens = inject_indents(source, tokens)?;
    let mut parser = parser::Parser::new(tokens);
    Ok(parser.parse())
}

/// Отступы в токены: лексер пропускает пробелы, поэтому границы блоков для
/// него невидимы — раньше блок `for`/`while` «глотал» все строки после себя
/// до первой строки, начинающейся с def/for/while/if. По первому токену
/// каждой строки вычисляем отступ и вставляем Indent/Dedent, как в Python.
/// Пустые строки и строки-комментарии отступ не меняют.
pub(crate) fn inject_indents(
    source: &str,
    tokens: Vec<lexer::Token>,
) -> Result<Vec<lexer::Token>, String> {
    let mut out = Vec::with_capacity(tokens.len() + 16);
    let mut stack: Vec<usize> = vec![0];
    let mut spans = lexer::Token::lexer(source).spanned();
    for token in tokens {
        let span = match spans.next() {
            Some((Ok(_), span)) => span,
            _ => break,
        };
        if !matches!(token, lexer::Token::Newline) {
            let line_start = source[..span.start].rfind('\n').map(|i| i + 1).unwrap_or(0);
            let leading = &source[line_start..span.start];
            // Пустой leading = токен в колонке 0: это тоже отступ (Dedent).
            // Токен в середине строки окружён не-пробелами — его пропускаем.
            if leading.bytes().all(|b| b == b' ' || b == b'\t') {
                let col: usize = leading
                    .bytes()
                    .map(|b| if b == b'\t' { 4 } else { 1 })
                    .sum();
                let top = *stack.last().unwrap_or(&0);
                if col > top {
                    stack.push(col);
                    out.push(lexer::Token::Indent);
                } else if col < top {
                    while *stack.last().unwrap_or(&0) > col {
                        stack.pop();
                        out.push(lexer::Token::Dedent);
                    }
                    if *stack.last().unwrap_or(&0) != col {
                        return Err(format!(
                            "строка {}: отступ не сходится с предыдущими блоками",
                            source[..span.start].bytes().filter(|b| *b == b'\n').count() + 1
                        ));
                    }
                }
            }
        }
        out.push(token);
    }
    while stack.len() > 1 {
        stack.pop();
        out.push(lexer::Token::Dedent);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_are_skipped() {
        let ast = parse_source("# привет комментарий\na = 5\nprint(a)\n");
        assert!(ast.is_ok(), "{ast:?}");
    }

    #[test]
    fn builtins_generate() {
        let rust = codegen::compile_to_string(
            parse_source("a = arg(1)\nsay(a)\n").unwrap(),
        );
        assert!(rust.contains("spark_arg"), "{rust}");
        assert!(rust.contains("spark_say"), "{rust}");
    }

    #[test]
    fn for_over_array_and_vec_var() {
        let rust = codegen::compile_to_string(
            parse_source("items = [\"a\", \"b\"]\nfor x in items:\n    print(x)\nfor y in [1, 2]:\n    print(y)\n").unwrap(),
        );
        assert!(rust.contains("for x in items {"), "{rust}");
        assert!(rust.contains("for y in vec![1, 2] {"), "{rust}");
    }

    #[test]
    fn return_literal_becomes_string() {
        let rust = codegen::compile_to_string(
            parse_source("def f():\n    return 42\nprint(f())\n").unwrap(),
        );
        assert!(rust.contains("return 42.to_string()"), "{rust}");
    }

    #[test]
    fn os_automation_builtins_map_to_prelude() {
        let rust = codegen::compile_to_string(parse_source(
            "send_hotkey(\"CTRL+SHIFT+T\")\ntype_text(\"привет\")\nwindow_focus(\"chromium\")\ndesktop_switch(2)\nvolume_set(50)\nprint(battery())\n",
        ).unwrap());
        for fn_name in [
            "spark_send_hotkey", "spark_type_text", "spark_window_focus",
            "spark_desktop_switch", "spark_volume_set", "spark_battery",
        ] {
            assert!(rust.contains(fn_name), "{fn_name} отсутствует:\n{rust}");
        }
        // Русские синонимы ведут к тем же функциям.
        let rust = codegen::compile_to_string(
            parse_source("сочетание(\"CTRL+T\")\nстол(3)\nскриншот(\"/tmp/a.png\")\n").unwrap(),
        );
        assert!(rust.contains("spark_send_hotkey"), "{rust}");
        assert!(rust.contains("spark_desktop_switch"), "{rust}");
        assert!(rust.contains("spark_screenshot"), "{rust}");
    }

    #[test]
    fn python_string_methods_compile_to_rust() {
        let rust = codegen::compile_to_string(parse_source(
            "s = \"Привет Мир\"\nprint(s.upper())\nprint(s.lower())\nprint(s.replace(\"Мир\", \"Свет\"))\nprint(s.contains(\"Мир\"))\nprint(s.len())\nprint(s.startswith(\"При\"))\nparts = s.split(\" \")\nprint(parts[0])\n",
        ).unwrap());
        assert!(rust.contains(".to_uppercase()"), "{rust}");
        assert!(rust.contains(".to_lowercase()"), "{rust}");
        assert!(rust.contains(".replace(\"Мир\", \"Свет\")"), "{rust}");
        assert!(rust.contains(".contains(\"Мир\")"), "{rust}");
        assert!(rust.contains(".chars().count()"), "{rust}");
        assert!(rust.contains(".starts_with(\"При\")"), "{rust}");
        // split собирает Vec<String> — индексация и for по нему работают.
        assert!(
            rust.contains(".split(\" \").map(|s| s.to_string()).collect::<Vec<String>>()"),
            "{rust}"
        );
    }

    #[test]
    fn prelude_embeds_os_automation_functions() {
        // Прелюдия обязана попадать в сгенерированный код — иначе
        // плагин с window_focus не соберётся.
        let rust = codegen::compile_to_string(parse_source("print(1)\n").unwrap());
        for fn_name in [
            "fn spark_send_hotkey", "fn spark_type_text", "fn spark_mouse_click",
            "fn spark_window_list", "fn spark_window_focus", "fn spark_desktop_switch",
            "fn spark_volume_set", "fn spark_brightness_set", "fn spark_screenshot",
            "fn spark_clip_get", "fn spark_http_get", "fn spark_app_open",
            "fn spark_battery", "fn spark_sleep_ms", "fn spark_out",
        ] {
            assert!(rust.contains(fn_name), "{fn_name} нет в прелюдии");
        }
    }

    /// Прогнать скрипт интерпретатором: рантайм без rustc.
    fn run(script: &str, args: &[&str]) -> Result<(Vec<String>, Vec<String>), String> {
        let ast = parse_source(script)?;
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        interp::interpret(&ast, &args)
            .map(|out| (out.say, out.prints))
    }

    #[test]
    fn interpreter_runs_plugin_subset() {
        // Ветки, аргументы, методы, конкатенация.
        let (say, _) = run(
            "т = arg(1)\nif т == \"\":\n    say(\"пусто\")\nelif т == \"мир\":\n    say(\"мир-мир\")\nelse:\n    say(\"привет \" + т.upper())\n",
            &["Павел"],
        )
        .unwrap();
        assert_eq!(say, vec!["привет ПАВЕЛ".to_string()]);
        let (say, _) = run(
            "т = arg(1)\nif т == \"\":\n    say(\"пусто\")\nelif т == \"мир\":\n    say(\"мир-мир\")\nelse:\n    say(\"другое\")\n",
            &["мир"],
        )
        .unwrap();
        assert_eq!(say, vec!["мир-мир".to_string()]);
    }

    #[test]
    fn interpreter_loops_and_functions() {
        // for с break/else, while, рекурсия, значения по умолчанию.
        let (say, prints) = run(
            "for i in [1, 2, 3]:\n    if i == 3:\n        break\n    print(i)\n\ndef сумма(a, b=0):\n    if a <= 0:\n        return b\n    return сумма(a - 1, b + a)\n\nprint(сумма(4))\nsay(\"готово\")\n",
            &[],
        )
        .unwrap();
        assert_eq!(prints, vec!["1".to_string(), "2".to_string(), "10".to_string()]);
        assert_eq!(say, vec!["готово".to_string()]);
        // Блок for не глотает строки после себя: dedent-границы.
        let (_, prints) = run("for i in [1, 2]:\n    print(i)\nprint(\"хвост\")\n", &[]).unwrap();
        assert_eq!(prints, vec!["1", "2", "хвост"]);
    }

    #[test]
    fn interpreter_builtins_and_os_automation() {
        let (say, _) = run("say(\"заряд \" + battery() + \"%\")", &[]).unwrap();
        assert!(say[0].starts_with("заряд "), "{say:?}");
        // Буфер обмена зависит от окружения (wl-paste без Wayland пуст):
        // для теста обход через файлы — те же built-ins.
        let tmp = std::env::temp_dir().join(format!("spark-interp-{}", std::process::id()));
        let tmps = tmp.to_string_lossy().to_string();
        let (say, _) = run(
            &format!("write_file(\"{tmps}\", \"тест\")\nsay(\"файл: \" + read_file(\"{tmps}\"))"),
            &[],
        )
        .unwrap();
        assert_eq!(say[0], "файл: тест");
        let _ = std::fs::remove_file(&tmp);
        // range и len и in.
        let (_, prints) = run("for i in range(3):\n    print(i)\nprint(\"ab\".len())\nprint(\"оп\" in \"снегопад\")", &[]).unwrap();
        assert_eq!(prints, vec!["0", "1", "2", "2", "True"]);
    }

    #[test]
    fn interpreter_try_except_and_honest_gaps() {
        let (say, _) = run(
            "try:\n    x = 1 / 0\nexcept:\n    say(\"поймала\")\n",
            &[],
        )
        .unwrap();
        assert_eq!(say, vec!["поймала".to_string()]);
        // Словари и классы честно требуют rustc, а не молча ломаются.
        assert!(run("d = {\"a\": 1}\n", &[]).is_err());
        assert!(run("class Точка:\n    x = 1\n", &[]).is_err());
    }

    #[test]
    fn generated_os_plugin_code_is_valid_rust() {
        // Полный мини-плагин: сгенерированный код должен компилироваться
        // rustc'ом — это и есть проверка кроссплатформенной прелюдии.
        let source = "name = arg(1)\nif name == \"\":\n    say(\"укажи имя окна\")\nelse:\n    r = window_focus(name)\n    say(r)\n";
        let ast = parse_source(source).unwrap();
        let rust = codegen::compile_to_string(ast);
        let dir = std::env::temp_dir().join(format!("spark-codegen-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let rs = dir.join("plugin.rs");
        std::fs::write(&rs, &rust).unwrap();
        let out = std::process::Command::new("rustc")
            .arg("--edition").arg("2024")
            .arg("--crate-name").arg("plugin")
            .arg("--emit").arg("metadata")
            .arg(rs.to_str().unwrap())
            .arg("--out-dir").arg(dir.to_str().unwrap())
            .output()
            .expect("rustc доступен в тестовом окружении");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            out.status.success(),
            "сгенерированный плагин не компилируется:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

