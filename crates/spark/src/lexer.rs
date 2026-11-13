use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\f]+")]
// Комментарий — # до конца строки. Без этого лексер падал на первом #:
// а шапка плагина (имя/описание/аргументы) живёт именно в комментариях.
// (skip живёт в атрибуте #[logos]: #[regex] на enum logos молча игнорирует.)
#[logos(skip(r"#[^\n]*", allow_greedy = true))]
pub enum Token {
    #[token("def")]
    Def,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("elif")]
    Elif,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("while")]
    While,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,
    #[token("True")]
    True,
    #[token("False")]
    False,
    #[token("None")]
    None,
    #[token("import")]
    Import,
    #[token("from")]
    From,
    #[token("class")]
    Class,
    #[token("try")]
    Try,
    #[token("except")]
    Except,
    #[token("finally")]
    Finally,
    #[token("raise")]
    Raise,
    #[token("with")]
    With,
    #[token("as")]
    As,
    #[token("pass")]
    Pass,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("lambda")]
    Lambda,
    #[token("yield")]
    Yield,
    #[token("global")]
    Global,
    #[token("nonlocal")]
    Nonlocal,
    #[token("del")]
    Del,
    #[token("assert")]
    Assert,
    #[token("is")]
    Is,

    // Идентификаторы в юникоде: русские имена функций и переменных —
    // основа автоматизаций на Spark («def состояние():» компилируется
    // в легальный rust-идентификатор).
    #[regex(r"[\p{XID_Start}_][\p{XID_Continue}_]*", |lex| lex.slice().to_string())]
    Ident(String),

    #[regex(r"0[xX][0-9a-fA-F]+", |lex| lex.slice().to_string())]
    HexNumber(String),

    #[regex(r"0[oO][0-7]+", |lex| lex.slice().to_string())]
    OctNumber(String),

    #[regex(r"0[bB][01]+", |lex| lex.slice().to_string())]
    BinNumber(String),

    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().to_string())]
    Float(String),

    #[regex(r"[0-9]+[eE][+-]?[0-9]+", |lex| lex.slice().to_string())]
    ExpNumber(String),

    #[regex("[0-9]+", |lex| lex.slice().parse::<i64>().unwrap())]
    Number(i64),

    #[regex(r#""[^"\\]*(?:\\.[^"\\]*)*""#, |lex| {
        let s = lex.slice();
        let inner = &s[1..s.len()-1];
        let unescaped = inner.replace("\\\"", "\"").replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\");
        unescaped
    })]
    Str(String),

    #[regex(r#"'[^'\\]*(?:\\.[^'\\]*)*'"#, |lex| {
        let s = lex.slice();
        let inner = &s[1..s.len()-1];
        let unescaped = inner.replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\");
        unescaped
    })]
    StrSingle(String),

    #[token("==")]
    Eq,
    #[token("!=")]
    Neq,
    #[token("<=")]
    Leq,
    #[token(">=")]
    Geq,
    #[token(">")]
    Greater,
    #[token("<")]
    Less,
    #[token("=")]
    Assign,
    #[token("+=")]
    PlusAssign,
    #[token("-=")]
    MinusAssign,
    #[token("*=")]
    MulAssign,
    #[token("/=")]
    DivAssign,
    #[token("//=")]
    FloorDivAssign,
    #[token("%=")]
    ModAssign,
    #[token("**=")]
    PowAssign,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Mul,
    #[token("/")]
    Div,
    #[token("%")]
    Mod,
    #[token("//")]
    FloorDiv,
    #[token("**")]
    Pow,

    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("[")]
    LBrack,
    #[token("]")]
    RBrack,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token("->")]
    Arrow,

    #[regex(r"\n|(\r\n)")]
    Newline,
}