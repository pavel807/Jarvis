//! Чат-путь Holo-3.1-4B-Thinking: сборка промпта по официальному шаблону
//! (text-only), генерация — обычная и спекулятивная.
//!
//! Шаблон воспроизводит `chat_template.json` чекпойнта: ChatML
//! (`<|im_start|>/<|im_end|>`), блок `# Tools` с JSON-схемами в `<tools>`,
//! tool-ответы как `<tool_response>` внутри user-хода. Generation prompt
//! открывает `<think>` — Thinking-модель рассуждает перед ответом; закрытый
//! блок `<think>\n\n</think>` (стандартный механизм Qwen3) отключает
//! рассуждения для быстрых голосовых ответов.

use super::hybrid::HybridModel;
use super::runtime::{SamplingParams, StreamEvent};
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use tokenizers::Tokenizer;

/// Одно сообщение диалога в терминах шаблона Holo-3.1 (text-only).
#[derive(Debug, Clone)]
pub enum ChatMessage {
    System(String),
    User(String),
    /// Ответ ассистента в истории (без think-блока: он в историю не пишется).
    Assistant(String),
    /// Результат инструмента: рендерится как `<tool_response>` внутри
    /// user-хода, ровно как в официальном шаблоне.
    ToolResult(String),
}

// =========================================================================
// Holo-3.1-4B (Qwen3.5-гибрид): чат-путь с heredoc-инструментами
// =========================================================================

/// Блок инструкций инструментов из официального шаблона Holo/Qwen3.5:
/// вызовы — heredoc-форматом `<function=имя><parameter=ключ>…</parameter>`.
pub fn holo_tools_block(tools: &[String]) -> String {
    let mut out = String::from(
        "# Tools\n\nYou have access to the following functions:\n\n<tools>",
    );
    for schema in tools {
        out.push('\n');
        out.push_str(schema);
    }
    out.push_str(
        "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format \
         with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
         <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
         <parameter=example_parameter_2>\nThis is the value for the second parameter\n\
         that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n\
         Reminder:\n\
         - Function calls MUST follow the specified format: an inner <function=...></function> \
         block must be nested within <tool_call></tool_call> XML tags\n\
         - Required parameters MUST be specified\n\
         - You may provide optional reasoning for your function call in natural language \
         BEFORE the function call, but NOT after\n\
         - If there is no function call available, answer the question like normal with your \
         current knowledge and do not tell the user about function calls\n\
         </IMPORTANT>",
    );
    out
}

/// Чат-движок Holo-3.1-4B: гибридный декодер + сборка промпта по официальному
/// шаблону Qwen3.5.
pub struct TextEngine35 {
    model: HybridModel,
}

impl TextEngine35 {
    pub fn load(dir: &Path, device: candle_core::Device) -> Result<Self> {
        Ok(Self { model: HybridModel::load(dir, device)? })
    }

    pub fn model(&self) -> &HybridModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut HybridModel {
        &mut self.model
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.model.tokenizer()
    }

    /// Сборка промпта по официальному шаблону Qwen3.5 (heredoc-инструменты).
    pub fn build_prompt(
        messages: &[ChatMessage],
        tools: Option<&[String]>,
        thinking: bool,
    ) -> String {
        let mut out = String::new();
        if let Some(tool_schemas) = tools
            && !tool_schemas.is_empty()
        {
            out.push_str("<|im_start|>system\n");
            if let Some(ChatMessage::System(sys)) = messages.first() {
                out.push_str(sys);
                out.push_str("\n\n");
            }
            out.push_str(&holo_tools_block(tool_schemas));
            out.push_str("<|im_end|>\n");
        } else if let Some(ChatMessage::System(sys)) = messages.first() {
            out.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
        }
        for message in messages {
            match message {
                ChatMessage::System(_) => {}
                ChatMessage::User(text) => {
                    out.push_str(&format!("<|im_start|>user\n{text}<|im_end|>\n"));
                }
                ChatMessage::Assistant(text) => {
                    out.push_str(&format!("<|im_start|>assistant\n{text}<|im_end|>\n"));
                }
                ChatMessage::ToolResult(text) => {
                    out.push_str(&format!(
                        "<|im_start|>user\n<tool_response>\n{text}\n</tool_response><|im_end|>\n"
                    ));
                }
            }
        }
        if thinking {
            out.push_str("<|im_start|>assistant\n<think>\n");
        } else {
            out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
        out
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer()
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Токенизация промпта Holo: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Обычная генерация ответа (без draft).
    #[allow(clippy::too_many_arguments)]
    pub fn reply(
        &mut self,
        messages: &[ChatMessage],
        tools: Option<&[String]>,
        thinking: bool,
        params: &SamplingParams,
        cancel: Option<&AtomicBool>,
        sink: impl FnMut(&StreamEvent) -> bool,
    ) -> Result<String> {
        Ok(self.reply_report(messages, tools, thinking, params, cancel, sink)?.text)
    }

    /// Как [`reply`](Self::reply), но с полным отчётом генерации (токены,
    /// скорости) — для бенчмарков и диагностики.
    #[allow(clippy::too_many_arguments)]
    pub fn reply_report(
        &mut self,
        messages: &[ChatMessage],
        tools: Option<&[String]>,
        thinking: bool,
        params: &SamplingParams,
        cancel: Option<&AtomicBool>,
        sink: impl FnMut(&StreamEvent) -> bool,
    ) -> Result<super::runtime::GenerationOutput> {
        let prompt = Self::build_prompt(messages, tools, thinking);
        let ids = self.tokenize(&prompt)?;
        super::runtime::generate(&mut self.model, &ids, params, cancel, sink)
    }
}

/// Отделяет think-блок от ответа. Генерация в thinking-режиме начинается
/// внутри `<think>`, поэтому «рассуждение» — это всё до `</think>`,
/// а ответ — после. В non-thinking режиме рассуждения нет.
pub fn split_think(text: &str) -> (Option<String>, String) {
    match text.find("</think>") {
        Some(pos) => {
            let reasoning = text[..pos].trim().trim_start_matches("<think>").trim().to_string();
            let answer = text[pos + "</think>".len()..].trim().to_string();
            (Some(reasoning), answer)
        }
        None => {
            let trimmed = text.trim();
            if trimmed.starts_with("<think>") || trimmed.contains("<think>") {
                (Some(trimmed.trim_start_matches("<think>").trim().to_string()), String::new())
            } else {
                (None, trimmed.to_string())
            }
        }
    }
}

#[cfg(test)]
mod holo_tests {
    use super::*;

    #[test]
    fn holo_prompt_embeds_tools_and_think() {
        let tools = vec![r#"{"type":"function","function":{"name":"click"}}"#.to_string()];
        let messages = vec![
            ChatMessage::System("Ты Джарвис.".into()),
            ChatMessage::User("привет".into()),
        ];
        let p = TextEngine35::build_prompt(&messages, Some(&tools), true);
        assert!(p.starts_with("<|im_start|>system\nТы Джарвис.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>"));
        assert!(p.contains("<function=example_function_name>"));
        assert!(p.contains("</IMPORTANT><|im_end|>\n"));
        assert!(p.contains("<|im_start|>user\nпривет<|im_end|>\n"));
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n"));
        let p2 = TextEngine35::build_prompt(&messages, None, false);
        assert!(p2.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert!(!p2.contains("# Tools"));
    }
}
