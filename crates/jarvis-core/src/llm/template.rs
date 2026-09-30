//! Сборка промпта по ChatML-шаблону.
//!
//! Qwen3 и MiniCPM5 используют ChatML (`<|im_start|>role\n...<|im_end|>`).
//! Отличия: Qwen3 не добавляет bos-токен, MiniCPM5 начинает с `<s>`.
//! Размышления выключены (enable_thinking=false в терминах шаблона):
//! после промпта ассистента подставляется пустой блок
//! `<think>\n\n</think>\n\n`, чтобы модель сразу отвечала по делу.
//! Остаточные блоки, если модель всё же начнёт «думать», фильтрует
//! `llm::strip_think` на лету.

/// Литерал bos-токена MiniCPM5 (tokenizer_config.json); у Qwen3 bos нет.
pub const MINICPM_BOS: &str = "<s>";

/// Разворачивает диалог system+user в готовую строку для токенизатора.
///
/// Специальные токены добавляет сам шаблон, поэтому кодирование выполняется
/// с `add_special_tokens=false`. `bos` — литерал bos-токена модели
/// (`<s>` у MiniCPM5, None у Qwen3).
pub fn build_chat_prompt(system: &str, user: &str, bos: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(bos) = bos {
        out.push_str(bos);
    }
    out.push_str(&format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_prompt_without_bos() {
        let prompt = build_chat_prompt("Ты ассистент.", "привет", None);
        assert_eq!(
            prompt,
            "<|im_start|>system\nТы ассистент.<|im_end|>\n<|im_start|>user\nпривет<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn chatml_prompt_with_bos() {
        let prompt = build_chat_prompt("Ты ассистент.", "привет", Some("<s>"));
        assert!(prompt.starts_with("<s><|im_start|>system"));
    }
}
