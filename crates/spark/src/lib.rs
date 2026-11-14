//! Spark Lang — скриптовый язык с питон-like синтаксисом, вендор из
//! github.com/pavel807/Spark. Компилирует .sk в Rust и собирает rustc'ом.
//!
//! Джарвис использует библиотеку для плагинов-автоматизаций: исходник
//! плагина → AST → Rust-код → бинарник (кэш по хэшу исходника). Бинарник
//! `spark` CLI сохранён как есть.

pub mod ast;
pub mod codegen;
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
    let mut parser = parser::Parser::new(tokens);
    Ok(parser.parse())
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
            .arg("--edition").arg("2021")
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
