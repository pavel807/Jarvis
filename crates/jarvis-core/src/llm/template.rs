//! Сборка промпта по ChatML-шаблону.
//!
//! Qwen3 использует ChatML (`<|im_start|>role\n...<|im_end|>`), bos-токена нет.
//! Для моделей семейства llama (например MiniCPM5) шаблон начинается с `<s>`.
//! Размышления выключены (enable_thinking=false в терминах шаблона):
//! после промпта ассистента подставляется пустой блок
//! `<think>\n\n</think>\n\n`, чтобы модель сразу отвечала по делу.
//! Остаточные блоки, если модель всё же начнёт «думать», фильтрует
//! `llm::strip_think` на лету.

/// Пара «вопрос → ответ» из предыдущих обменов для контекста диалога.
pub type ChatTurn = (String, String);

/// Литерал bos-токена моделей llama-семейства (tokenizer_config.json); у Qwen3 bos нет.
pub const MINICPM_BOS: &str = "<s>";

/// Разворачивает диалог system + история + новый вопрос в готовую строку
/// для токенизатора.
///
/// Специальные токены добавляет сам шаблон, поэтому кодирование выполняется
/// с `add_special_tokens=false`. `bos` — литерал bos-токена модели
/// (`<s>` у llama-семейства, None у Qwen3).
pub fn build_chat_prompt(system: &str, history: &[ChatTurn], user: &str, bos: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(bos) = bos {
        out.push_str(bos);
    }
    out.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    for (question, answer) in history {
        out.push_str(&format!(
            "<|im_start|>user\n{question}<|im_end|>\n\
             <|im_start|>assistant\n{answer}<|im_end|>\n"
        ));
    }
    out.push_str(&format!(
        "<|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_prompt_without_history() {
        let prompt = build_chat_prompt("Ты ассистент.", &[], "привет", None);
        assert_eq!(
            prompt,
            "<|im_start|>system\nТы ассистент.<|im_end|>\n<|im_start|>user\nпривет<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn chatml_prompt_with_bos() {
        let prompt = build_chat_prompt("Ты ассистент.", &[], "привет", Some("<s>"));
        assert!(prompt.starts_with("<s><|im_start|>system"));
    }

    #[test]
    fn chatml_prompt_with_history() {
        let history = vec![
            ("который час".to_string(), "Полдень.".to_string()),
        ];
        let prompt = build_chat_prompt("Ты ассистент.", &history, "а в Лондоне?", None);
        assert!(prompt.contains("<|im_start|>user\nкоторый час<|im_end|>\n<|im_start|>assistant\nПолдень.<|im_end|>\n"));
        // Новый вопрос идёт после истории.
        let user_pos = prompt.find("а в Лондоне?").unwrap();
        let hist_pos = prompt.find("который час").unwrap();
        assert!(hist_pos < user_pos);
    }
}
