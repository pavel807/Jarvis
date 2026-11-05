//! Сборка промпта по MiniCPM-шаблону (MiniCPM5-1B-ru-v3).
//!
//! Модель использует ChatML-подобный формат с маркерами
//! `<|im_start|>role\n...<|im_end|>`. BOS — классический `<s>` (id 0),
//! EOS — `</s>` (id 1) и `<|im_end|>` (id 130073).
//!
//! Токены `<|start_header_id|>` / `<|eot_id|>` из Llama-3 здесь
//! **отсутствуют** в словаре — их использование давало мусор из
//! галлюцинированных `<|content_*|>`.

/// Пара «вопрос → ответ» из предыдущих обменов.
pub type ChatTurn = (String, String);

/// BOS-токен MiniCPM5-1B (классический `<s>`, id 0).
pub const MINICPM_BOS: &str = "<s>";

/// Префикс ответа ассистента Qwen3.5: шаблон не-thinking режима принудительно
/// закрывает блок рассуждений, чтобы модель не тратила токены на `<think>`.
pub const QWEN_ASSISTANT_PREFIX: &str = "<think>\n\n</think>\n\n";

/// Разворачивает диалог system + история + новый вопрос в готовую строку
/// для токенизатора. Специальные токены добавляет сам шаблон, поэтому
/// кодирование выполняется с `add_special_tokens=false`.
///
/// `assistant_prefix` дописывается сразу после заголовка
/// `<|im_start|>assistant\n`: пустая строка у MiniCPM5,
/// [`QWEN_ASSISTANT_PREFIX`] у Qwen3.5.
pub fn build_chat_prompt(
    system: &str,
    history: &[ChatTurn],
    user: &str,
    bos: Option<&str>,
    assistant_prefix: &str,
) -> String {
    let mut out = String::new();
    if let Some(bos) = bos {
        out.push_str(bos);
    }
    // System
    out.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    // История: в ответах ассистента повторяем assistant_prefix — ровно так
    // выглядел реальный контекст при генерации. Без этого токены прошлых
    // ходов не совпадают с уже посчитанным префиксным KV-кэшем, и каждый
    // ход/шаг агента префиллится с нуля (секунды на ровном месте).
    for (q, a) in history {
        out.push_str(&format!(
            "<|im_start|>user\n{q}<|im_end|>\n\
             <|im_start|>assistant\n{assistant_prefix}{a}<|im_end|>\n"
        ));
    }
    // Новый вопрос и открытый заголовок ассистента
    out.push_str(&format!(
        "<|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n{assistant_prefix}"
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minicpm_prompt_without_history() {
        let prompt = build_chat_prompt("Ты ассистент.", &[], "привет", None, "");
        assert_eq!(
            prompt,
            "<|im_start|>system\nТы ассистент.<|im_end|>\n\
             <|im_start|>user\nпривет<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn minicpm_prompt_with_bos() {
        let prompt = build_chat_prompt("Ты ассистент.", &[], "привет", Some("<s>"), "");
        assert!(prompt.starts_with("<s><|im_start|>system"));
    }

    #[test]
    fn minicpm_prompt_with_history() {
        let history = vec![("который час".to_string(), "Полдень.".to_string())];
        let prompt = build_chat_prompt("Ты ассистент.", &history, "а в Лондоне?", None, "");
        assert!(prompt.contains(
            "<|im_start|>user\nкоторый час<|im_end|>\n\
             <|im_start|>assistant\nПолдень.<|im_end|>\n"
        ));
        let user_pos = prompt.find("а в Лондоне?").unwrap();
        let hist_pos = prompt.find("который час").unwrap();
        assert!(hist_pos < user_pos);
    }

    #[test]
    fn qwen_prompt_closes_think() {
        let prompt = build_chat_prompt("Ты ассистент.", &[], "привет", None, QWEN_ASSISTANT_PREFIX);
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!prompt.starts_with("<s>"));
    }

    #[test]
    fn qwen_history_turns_keep_assistant_prefix() {
        // Префикс должен стоять и в исторических ответах: иначе токены
        // прошлого хода не совпадают с префиксным KV-кэшем, и кэш теряется
        // на каждом шаге (полный префилл заново).
        let history = vec![(
            "открой сайт".to_string(),
            "<tool>{\"name\":\"open_url\"}</tool>".to_string(),
        )];
        let prompt = build_chat_prompt("sys", &history, "дальше", None, QWEN_ASSISTANT_PREFIX);
        assert!(
            prompt.contains(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool>{\"name\":\"open_url\"}</tool><|im_end|>"
            ),
            "{prompt}"
        );
        // Второй ход начинается ровно с первого — префикс для кэша совпадает.
        let first = build_chat_prompt("sys", &[], "открой сайт", None, QWEN_ASSISTANT_PREFIX);
        assert!(prompt.starts_with(&first));
    }
}
