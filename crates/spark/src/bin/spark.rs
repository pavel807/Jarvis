use spark::{ast, codegen, lexer, parser};

use logos::Logos;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

struct ImportResolver {
    loaded_modules: HashMap<String, Vec<ast::Stmt>>,
}

impl ImportResolver {
    fn new() -> Self {
        Self {
            loaded_modules: HashMap::new(),
        }
    }

    fn resolve_import(&mut self, module: &str, items: Option<Vec<String>>, base_path: &Path) -> Result<Vec<ast::Stmt>, String> {
        let key = if let Some(ref i) = items {
            format!("{}:{}", module, i.join(","))
        } else {
            module.to_string()
        };
        
        if let Some(cached) = self.loaded_modules.get(&key) {
            return Ok(cached.clone());
        }
        
        let module_path = base_path.parent()
            .unwrap_or(base_path)
            .join(format!("{}.sk", module));
        
        if !module_path.exists() {
            return Err(format!("Модуль '{}' не найден", module));
        }
        
        let source = fs::read_to_string(&module_path)
            .map_err(|e| format!("Не удалось прочитать {}: {}", module_path.display(), e))?;
        
        let tokens: Vec<lexer::Token> = lexer::Token::lexer(&source)
            .map(|res| res.map_err(|e| format!("Ошибка лексера: {:?}", e)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e)?;
        
        let mut module_parser = parser::Parser::new(tokens);
        let module_ast = module_parser.parse();
        
        let result = if let Some(needed) = items {
            module_ast.into_iter()
                .filter(|stmt| {
                    if let ast::Stmt::Def { name, .. } = stmt {
                        needed.contains(name)
                    } else {
                        false
                    }
                })
                .collect()
        } else {
            module_ast
        };
        
        self.loaded_modules.insert(key, result.clone());
        Ok(result)
    }
}

fn resolve_imports(ast: Vec<ast::Stmt>, main_file: &str) -> Result<Vec<ast::Stmt>, String> {
    let base_path = Path::new(main_file);
    let mut import_resolver = ImportResolver::new();
    let mut result = Vec::new();
    let mut imports = Vec::new();
    
    for stmt in ast {
        match stmt {
            ast::Stmt::Import { module, items, alias } => {
                let imported = import_resolver.resolve_import(&module, items, base_path)?;
                
                if let Some(aliased) = alias {
                    for imp in imported {
                        if let ast::Stmt::Def { name: _, args, body } = imp {
                            let new_stmt = ast::Stmt::Def {
                                name: aliased.clone(),
                                args,
                                body,
                            };
                            imports.push(new_stmt);
                        }
                    }
                } else {
                    imports.extend(imported);
                }
            }
            _ => result.push(stmt),
        }
    }
    
    result.extend(imports);
    Ok(result)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Использование: spark <файл.sk> [-o выходной_файл] [--run] [--release[=1|2|3] | --dev[=1|2|3]]");
        return;
    }

    let input_file = &args[1];

    let path = Path::new(input_file);
    if path.extension().map(|e| e != "sk").unwrap_or(true) {
        eprintln!("❌ Ошибка: ожидается файл с расширением .sk (Spark Lang)");
        std::process::exit(1);
    }

    let mut output_name = "app";
    let mut should_run = false;
    let mut opt_level = 1;

    for i in 1..args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            output_name = &args[i + 1];
        }
        if args[i] == "--run" {
            should_run = true;
        }
        if args[i].starts_with("--release") || args[i] == "-r" {
            opt_level = 3;
            if let Some(level) = args[i].split('=').nth(1) {
                if let Ok(l) = level.parse::<i32>() {
                    opt_level = l.clamp(1, 3);
                }
            }
        }
    }

    let source = fs::read_to_string(input_file).expect("❌ Файл не найден");

    let tokens: Vec<lexer::Token> = lexer::Token::lexer(&source)
        .map(|res| res.expect("Ошибка лексера"))
        .collect();

    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse();

    let resolved = match resolve_imports(ast, input_file) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("❌ Ошибка импорта: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = codegen::compile_to_binary(resolved, output_name, opt_level) {
        eprintln!("❌ Ошибка компиляции:\n{e}");
        std::process::exit(1);
    }

    if should_run {
        println!("🏃 Запуск {}...", output_name);
        let mut child = Command::new(format!("./{}", output_name))
            .spawn()
            .expect("❌ Не удалось запустить программу");
        child.wait().expect("❌ Программа завершилась с ошибкой");
    }
}