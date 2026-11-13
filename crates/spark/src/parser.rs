use crate::ast::{Expr, Stmt};
use crate::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        while !self.is_at_end() {
            if let Some(stmt) = self.parse_statement() {
                stmts.push(stmt);
            }
        }
        stmts
    }

    fn parse_statement(&mut self) -> Option<Stmt> {
        match self.peek() {
            Some(Token::Newline) => {
                self.advance();
                None
            }
            Some(Token::If) => Some(self.parse_if()),
            Some(Token::For) => Some(self.parse_for()),
            Some(Token::While) => Some(self.parse_while()),
            Some(Token::Def) => Some(self.parse_def()),
            Some(Token::Class) => Some(self.parse_class()),
            Some(Token::Try) => Some(self.parse_try()),
            Some(Token::With) => Some(self.parse_with()),
            Some(Token::Raise) => Some(self.parse_raise()),
            Some(Token::Assert) => Some(self.parse_assert()),
            Some(Token::Break) => {
                self.advance();
                self.consume_newline();
                Some(Stmt::Break)
            }
            Some(Token::Continue) => {
                self.advance();
                self.consume_newline();
                Some(Stmt::Continue)
            }
            Some(Token::Pass) => {
                self.advance();
                self.consume_newline();
                Some(Stmt::Pass)
            }
            Some(Token::Return) => Some(self.parse_return()),
            Some(Token::Import) => Some(self.parse_import()),
            Some(Token::From) => Some(self.parse_from_import()),
            Some(Token::Global) => Some(self.parse_global()),
            Some(Token::Nonlocal) => Some(self.parse_nonlocal()),
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.advance();
                match self.peek() {
                    Some(Token::Assign) => {
                        self.advance();
                        let value = self.parse_expr();
                        self.consume_newline();
                        Some(Stmt::Assign { target: name, value })
                    }
                    Some(Token::PlusAssign) => {
                        self.advance();
                        let value = self.parse_expr();
                        self.consume_newline();
                        Some(Stmt::AugAssign { target: name, op: "+".to_string(), value })
                    }
                    Some(Token::MinusAssign) => {
                        self.advance();
                        let value = self.parse_expr();
                        self.consume_newline();
                        Some(Stmt::AugAssign { target: name, op: "-".to_string(), value })
                    }
                    Some(Token::MulAssign) => {
                        self.advance();
                        let value = self.parse_expr();
                        self.consume_newline();
                        Some(Stmt::AugAssign { target: name, op: "*".to_string(), value })
                    }
                    Some(Token::DivAssign) => {
                        self.advance();
                        let value = self.parse_expr();
                        self.consume_newline();
                        Some(Stmt::AugAssign { target: name, op: "/".to_string(), value })
                    }
                    Some(Token::LParen) => {
                        let args = self.parse_call_args();
                        self.consume_newline();
                        Some(Stmt::Expr(Expr::Call {
                            func: Box::new(Expr::Ident(name)),
                            args,
                        }))
                    }
                    _ => {
                        self.consume_newline();
                        Some(Stmt::Expr(Expr::Ident(name)))
                    }
                }
            }
            _ => {
                self.advance();
                None
            }
        }
    }

    fn parse_if(&mut self) -> Stmt {
        self.advance();
        let condition = self.parse_expr();
        self.consume(Token::Colon);
        self.consume_newline();
        let then_branch = self.parse_block();

        let mut elif_branches = Vec::new();
        let mut else_branch = None;

        loop {
            match self.peek() {
                Some(Token::Elif) => {
                    self.advance();
                    let cond = self.parse_expr();
                    self.consume(Token::Colon);
                    self.consume_newline();
                    let body = self.parse_block();
                    elif_branches.push((cond, body));
                }
                Some(Token::Else) => {
                    self.advance();
                    self.consume(Token::Colon);
                    self.consume_newline();
                    else_branch = Some(self.parse_block());
                    break;
                }
                Some(Token::Newline) => {
                    self.advance();
                }
                _ => break,
            }
        }

        Stmt::If {
            condition,
            then_branch,
            elif_branches,
            else_branch,
        }
    }

    fn parse_for(&mut self) -> Stmt {
        self.advance();
        let var = match self.advance() {
            Some(Token::Ident(n)) => n,
            _ => panic!("Ожидалось имя переменной"),
        };
        self.consume(Token::In);
        let iter = self.parse_expr();
        self.consume(Token::Colon);
        self.consume_newline();
        let body = self.parse_block();

        let mut else_body = None;
        if self.peek() == Some(Token::Else) {
            self.advance();
            self.consume(Token::Colon);
            self.consume_newline();
            else_body = Some(self.parse_block());
        }

        Stmt::For { var, iter, body, else_body }
    }

    fn parse_while(&mut self) -> Stmt {
        self.advance();
        let condition = self.parse_expr();
        self.consume(Token::Colon);
        self.consume_newline();
        let body = self.parse_block();

        Stmt::While { condition, body }
    }

    fn parse_def(&mut self) -> Stmt {
        self.advance();
        let name = match self.advance() {
            Some(Token::Ident(n)) => n,
            _ => panic!("Ожидалось имя функции"),
        };
        self.consume(Token::LParen);
        let mut args = Vec::new();
        
        loop {
            match self.peek() {
                Some(Token::Ident(n)) => {
                    let arg_name = n.clone();
                    self.advance();
                    let default = if self.peek() == Some(Token::Assign) {
                        self.advance();
                        Some(self.parse_expr())
                    } else {
                        None
                    };
                    args.push((arg_name, default));
                    if self.peek() == Some(Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                Some(Token::RParen) => {
                    break;
                }
                _ => break,
            }
        }
        self.consume(Token::RParen);
        self.consume(Token::Colon);
        self.consume_newline();
        let body = self.parse_block();

        Stmt::Def { name, args, body }
    }

    fn parse_class(&mut self) -> Stmt {
        self.advance();
        let name = match self.advance() {
            Some(Token::Ident(n)) => n,
            _ => panic!("Ожидалось имя класса"),
        };
        
        let mut bases = Vec::new();
        if self.peek() == Some(Token::LParen) {
            self.advance();
            loop {
                match self.peek() {
                    Some(Token::Ident(n)) => {
                        bases.push(n);
                        self.advance();
                        if self.peek() == Some(Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    _ => break,
                }
            }
            self.consume(Token::RParen);
        }
        
        self.consume(Token::Colon);
        self.consume_newline();
        let body = self.parse_block();

        Stmt::Class { name, bases, body }
    }

    fn parse_try(&mut self) -> Stmt {
        self.advance();
        self.consume(Token::Colon);
        self.consume_newline();
        let body = self.parse_block();

        let mut except_branches = Vec::new();
        let mut else_branch = None;
        let mut finally_body = None;

        loop {
            match self.peek() {
                Some(Token::Except) => {
                    self.advance();
                    let mut exc_type: Option<String> = None;
                    let mut alias: Option<String> = None;
                    
                    if let Some(Token::Ident(id)) = self.peek() {
                        exc_type = Some(id.clone());
                        self.advance();
                        if self.peek() == Some(Token::As) {
                            self.advance();
                            if let Some(Token::Ident(alias_name)) = self.advance() {
                                alias = Some(alias_name);
                            }
                        }
                    } else if self.peek() == Some(Token::As) {
                        self.advance();
                        if let Some(Token::Ident(alias_name)) = self.advance() {
                            alias = Some(alias_name);
                        }
                    }
                    self.consume(Token::Colon);
                    self.consume_newline();
                    let except_body = self.parse_block();
                    except_branches.push((exc_type, alias, except_body));
                }
                Some(Token::Else) => {
                    self.advance();
                    self.consume(Token::Colon);
                    self.consume_newline();
                    else_branch = Some(self.parse_block());
                }
                Some(Token::Finally) => {
                    self.advance();
                    self.consume(Token::Colon);
                    self.consume_newline();
                    finally_body = Some(self.parse_block());
                    break;
                }
                _ => break,
            }
        }

        Stmt::Try {
            body,
            except_branches,
            else_branch,
            finally_body,
        }
    }

    fn parse_with(&mut self) -> Stmt {
        self.advance();
        let mut items = Vec::new();
        
        loop {
            let expr = self.parse_expr();
            let alias = if self.peek() == Some(Token::As) {
                self.advance();
                match self.advance() {
                    Some(Token::Ident(n)) => Some(n),
                    _ => None,
                }
            } else {
                None
            };
            items.push((expr, alias));
            
            if self.peek() == Some(Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(Token::Colon);
        self.consume_newline();
        let body = self.parse_block();

        Stmt::With { items, body }
    }

    fn parse_raise(&mut self) -> Stmt {
        self.advance();
        let expr = if self.peek() == Some(Token::Newline) || self.peek().is_none() {
            None
        } else {
            Some(self.parse_expr())
        };
        self.consume_newline();
        Stmt::Raise(expr)
    }

    fn parse_assert(&mut self) -> Stmt {
        self.advance();
        let cond = self.parse_expr();
        let msg = if self.peek() == Some(Token::Comma) {
            self.advance();
            Some(self.parse_expr())
        } else {
            None
        };
        self.consume_newline();
        Stmt::Assert(cond, msg)
    }

    fn parse_return(&mut self) -> Stmt {
        self.advance();
        let expr = if self.peek() == Some(Token::Newline) || self.peek().is_none() {
            None
        } else {
            Some(self.parse_expr())
        };
        self.consume_newline();
        Stmt::Return(expr)
    }

    fn parse_import(&mut self) -> Stmt {
        self.advance();
        let module = match self.advance() {
            Some(Token::Ident(n)) => n,
            _ => panic!("Ожидалось имя модуля"),
        };
        
        let items = None;
        let alias = if self.peek() == Some(Token::As) {
            self.advance();
            match self.advance() {
                Some(Token::Ident(n)) => Some(n),
                _ => None,
            }
        } else {
            None
        };
        
        self.consume_newline();
        Stmt::Import { module, items, alias }
    }

    fn parse_from_import(&mut self) -> Stmt {
        self.advance();
        let module = match self.advance() {
            Some(Token::Ident(n)) => n,
            _ => panic!("Ожидалось имя м��дуля"),
        };
        
        self.consume(Token::Import);
        
        let mut items = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Ident(n)) => {
                    items.push(n);
                    self.advance();
                    if self.peek() == Some(Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        
        let alias = if self.peek() == Some(Token::As) {
            self.advance();
            match self.advance() {
                Some(Token::Ident(n)) => Some(n),
                _ => None,
            }
        } else {
            None
        };
        
        self.consume_newline();
        Stmt::Import { module, items: Some(items), alias }
    }

    fn parse_global(&mut self) -> Stmt {
        self.advance();
        let mut vars = Vec::new();
        loop {
            match self.advance() {
                Some(Token::Ident(n)) => {
                    vars.push(n);
                    if self.peek() == Some(Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        self.consume_newline();
        Stmt::Global(vars)
    }

    fn parse_nonlocal(&mut self) -> Stmt {
        self.advance();
        let mut vars = Vec::new();
        loop {
            match self.advance() {
                Some(Token::Ident(n)) => {
                    vars.push(n);
                    if self.peek() == Some(Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        self.consume_newline();
        Stmt::Nonlocal(vars)
    }

    fn parse_block(&mut self) -> Vec<Stmt> {
        let mut statements = Vec::new();
        while !self.is_at_end() {
            match self.peek() {
                Some(Token::Return) => {
                    statements.push(self.parse_return());
                    break;
                }
                Some(Token::Else) | Some(Token::Elif) => {
                    break;
                }
                Some(Token::Newline) => {
                    self.advance();
                    if statements.len() == 0 {
                        continue;
                    }
                    if let Some(Token::Else) | Some(Token::Elif) = self.peek() {
                        break;
                    }
                }
                Some(Token::Def) | Some(Token::For) | Some(Token::While) | Some(Token::If) => {
                    break;
                }
                _ => {
                    if let Some(stmt) = self.parse_statement() {
                        statements.push(stmt);
                    } else if self.peek() == Some(Token::Newline) {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }
        statements
    }

    fn parse_call_args(&mut self) -> Vec<Expr> {
        self.consume(Token::LParen);
        let mut args = Vec::new();
        loop {
            match self.peek() {
                Some(Token::RParen) => {
                    break;
                }
                _ => {
                    let arg = self.parse_expr();
                    args.push(arg);
                    if self.peek() == Some(Token::Comma) {
                        self.advance();
                    }
                }
            }
        }
        self.consume(Token::RParen);
        args
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_lambda()
    }

    fn parse_lambda(&mut self) -> Expr {
        if self.peek() == Some(Token::Lambda) {
            self.advance();
            let mut args = Vec::new();
            loop {
                match self.peek() {
                    Some(Token::Ident(n)) => {
                        args.push(n);
                        self.advance();
                        if self.peek() == Some(Token::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    _ => break,
                }
            }
            self.consume(Token::Colon);
            let body = Box::new(self.parse_expr());
            return Expr::Lambda { args, body };
        }
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Expr {
        let mut left = self.parse_or();
        if self.peek() == Some(Token::If) {
            self.advance();
            let cond = left;
            left = self.parse_expr();
            self.consume(Token::Else);
            let else_ = self.parse_expr();
            return Expr::Ternary {
                cond: Box::new(cond),
                then: Box::new(left),
                else_: Box::new(else_),
            };
        }
        left
    }

    fn parse_or(&mut self) -> Expr {
        let mut left = self.parse_and();
        while let Some(op) = self.get_binary_op() {
            if op == "or" {
                self.advance();
                let right = self.parse_and();
                left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
            } else {
                break;
            }
        }
        left
    }

    fn parse_and(&mut self) -> Expr {
        let mut left = self.parse_not();
        while let Some(op) = self.get_binary_op() {
            if op == "and" {
                self.advance();
                let right = self.parse_not();
                left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
            } else {
                break;
            }
        }
        left
    }

    fn parse_not(&mut self) -> Expr {
        if self.peek() == Some(Token::Not) {
            self.advance();
            let expr = self.parse_not();
            return Expr::UnaryOp("not".to_string(), Box::new(expr));
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Expr {
        let mut left = self.parse_addition();
        while let Some(op) = self.get_binary_op() {
            if op == "==" || op == "!=" || op == ">" || op == "<" || op == ">=" || op == "<=" || op == "in" || op == "is" {
                self.advance();
                let right = self.parse_addition();
                left = Expr::Compare(Box::new(left), op, Box::new(right));
            } else {
                break;
            }
        }
        left
    }

    fn parse_addition(&mut self) -> Expr {
        let mut left = self.parse_multiplication();
        while let Some(op) = self.get_binary_op() {
            if op == "+" || op == "-" {
                self.advance();
                let right = self.parse_multiplication();
                left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
            } else {
                break;
            }
        }
        left
    }

    fn parse_multiplication(&mut self) -> Expr {
        let mut left = self.parse_unary();
        while let Some(op) = self.get_binary_op() {
            if op == "*" || op == "/" || op == "%" || op == "//" {
                self.advance();
                let right = self.parse_unary();
                left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
            } else {
                break;
            }
        }
        left
    }

    fn parse_unary(&mut self) -> Expr {
        if let Some(op) = self.get_unary_op() {
            self.advance();
            let expr = self.parse_unary();
            return Expr::UnaryOp(op, Box::new(expr));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Expr {
        let mut expr = self.parse_primary();

        loop {
            match self.peek() {
                Some(Token::LBrack) => {
                    self.advance();
                    let index = self.parse_expr();
                    self.consume(Token::RBrack);
                    expr = Expr::Index {
                        array: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                Some(Token::Dot) => {
                    self.advance();
                    if let Some(Token::Ident(method)) = self.advance() {
                        if self.peek() == Some(Token::LParen) {
                            let args = self.parse_call_args();
                            expr = Expr::MethodCall {
                                receiver: Box::new(expr),
                                method,
                                args,
                            };
                        } else {
                            expr = Expr::MethodCall {
                                receiver: Box::new(expr),
                                method,
                                args: vec![],
                            };
                        }
                    }
                }
                Some(Token::LParen) => {
                    let args = self.parse_call_args();
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                    };
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_primary(&mut self) -> Expr {
        let token = self.advance().expect("Неожиданный конец файла");
        match token {
            Token::Number(n) => Expr::Number(n),
            Token::Float(s) => Expr::Float(s.parse().unwrap_or(0.0)),
            Token::HexNumber(s) => Expr::Number(i64::from_str_radix(&s[2..], 16).unwrap_or(0)),
            Token::OctNumber(s) => Expr::Number(i64::from_str_radix(&s[2..], 8).unwrap_or(0)),
            Token::BinNumber(s) => Expr::Number(i64::from_str_radix(&s[2..], 2).unwrap_or(0)),
            Token::Str(s) => Expr::Str(s),
            Token::StrSingle(s) => Expr::Str(s),
            Token::Ident(i) => Expr::Ident(i),
            Token::True => Expr::Bool(true),
            Token::False => Expr::Bool(false),
            Token::None => Expr::None,
            Token::LParen => {
                if self.peek() == Some(Token::RParen) {
                    self.advance();
                    return Expr::Tuple(vec![]);
                }
                let first = self.parse_expr();
                if self.peek() == Some(Token::Comma) {
                    self.advance();
                    let mut elements = vec![first];
                    loop {
                        if self.peek() == Some(Token::RParen) {
                            self.advance();
                            break;
                        }
                        elements.push(self.parse_expr());
                        if self.peek() == Some(Token::Comma) {
                            self.advance();
                        } else if self.peek() == Some(Token::RParen) {
                            self.advance();
                            break;
                        } else {
                            break;
                        }
                    }
                    Expr::Tuple(elements)
                } else {
                    self.consume(Token::RParen);
                    first
                }
            }
            Token::LBrack => {
                if self.peek() == Some(Token::RBrack) {
                    self.advance();
                    return Expr::Array(vec![]);
                }
                let elements = self.parse_list_elements();
                self.consume(Token::RBrack);
                Expr::Array(elements)
            }
            Token::LBrace => {
                if self.peek() == Some(Token::Colon) {
                    return self.parse_dict();
                }
                self.parse_set_or_dict()
            }
            _ => panic!("Неподдерживаемое выражение: {:?}", token),
        }
    }

    fn parse_list_elements(&mut self) -> Vec<Expr> {
        let mut elements = vec![];
        loop {
            if self.peek() == Some(Token::RBrack) || self.peek().is_none() {
                break;
            }
            elements.push(self.parse_expr());
            if self.peek() == Some(Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        elements
    }

    fn parse_dict(&mut self) -> Expr {
        self.advance();
        let mut pairs = vec![];
        loop {
            if self.peek() == Some(Token::RBrace) {
                self.advance();
                break;
            }
            let key = self.parse_expr();
            self.consume(Token::Colon);
            let value = self.parse_expr();
            pairs.push((key, value));
            if self.peek() == Some(Token::Comma) {
                self.advance();
            } else if self.peek() == Some(Token::RBrace) {
                self.advance();
                break;
            } else {
                break;
            }
        }
        Expr::Dict(pairs)
    }

    fn parse_set_or_dict(&mut self) -> Expr {
        let mut elements = vec![];
        loop {
            if self.peek() == Some(Token::RBrace) {
                self.advance();
                break;
            }
            elements.push(self.parse_expr());
            if self.peek() == Some(Token::Comma) {
                self.advance();
            } else if self.peek() == Some(Token::RBrace) {
                self.advance();
                break;
            } else {
                break;
            }
        }
        Expr::Set(elements)
    }

    fn get_binary_op(&self) -> Option<String> {
        match self.peek() {
            Some(Token::Plus) => Some("+".to_string()),
            Some(Token::Minus) => Some("-".to_string()),
            Some(Token::Mul) => Some("*".to_string()),
            Some(Token::Div) => Some("/".to_string()),
            Some(Token::Mod) => Some("%".to_string()),
            Some(Token::FloorDiv) => Some("//".to_string()),
            Some(Token::Pow) => Some("**".to_string()),
            Some(Token::Eq) => Some("==".to_string()),
            Some(Token::Neq) => Some("!=".to_string()),
            Some(Token::Greater) => Some(">".to_string()),
            Some(Token::Less) => Some("<".to_string()),
            Some(Token::Geq) => Some(">=".to_string()),
            Some(Token::Leq) => Some("<=".to_string()),
            Some(Token::And) => Some("and".to_string()),
            Some(Token::Or) => Some("or".to_string()),
            Some(Token::In) => Some("in".to_string()),
            Some(Token::Is) => Some("is".to_string()),
            _ => None,
        }
    }

    fn get_unary_op(&self) -> Option<String> {
        match self.peek() {
            Some(Token::Not) => Some("not".to_string()),
            Some(Token::Minus) => Some("-".to_string()),
            Some(Token::Plus) => Some("+".to_string()),
            _ => None,
        }
    }

    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.pos).cloned()
    }

    fn advance(&mut self) -> Option<Token> {
        let t = self.peek();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn consume(&mut self, expected: Token) {
        if self.peek() == Some(expected.clone()) {
            self.advance();
        } else {
            panic!("Ожидалось {:?}, но найдено {:?}", expected, self.peek());
        }
    }

    fn consume_newline(&mut self) {
        while self.peek() == Some(Token::Newline) {
            self.advance();
        }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }
}