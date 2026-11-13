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
}
