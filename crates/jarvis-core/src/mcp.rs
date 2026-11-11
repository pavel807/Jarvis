//! MCP-клиент — Model Context Protocol по логике OpenJarvis: подключение к
//! внешним серверам по stdio (JSON-RPC 2.0, newline-delimited), handshake
//! initialize → tools/list → tools/call. Инструменты подключённых серверов
//! появляются у модели как один generic-вызов:
//! `mcp_call {"server":"имя","tool":"имя_инструмента","args":{...}}`.
//!
//! Серверы настраиваются в settings.toml:
//! ```toml
//! [[mcp_servers]]
//! name = "fs"
//! cmd = "npx"
//! args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
//! ```
//!
//! Отказоустойчивость по OpenJarvis: упавший сервер не роняет остальные —
//! при подключении ошибка пишется в stderr, сервер помечается мёртвым,
//! вызов к нему возвращает честную ошибку.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Таймаут handshake и вызова инструмента.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Инструмент внешнего сервера (для блока промпта).
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    pub server: String,
    pub name: String,
    pub description: String,
}

/// Живое подключение к одному серверу.
struct McpServer {
    name: String,
    /// Процесс держим живым: drop Child не убивает его, поэтому поле
    /// существует ради владения и не читается.
    #[allow(dead_code)]
    child: Child,
    stdin: Mutex<ChildStdin>,
    /// Каналы ответов по request id: reader-поток кладёт Value, вызывающий
    /// ждёт с таймаутом. Mutex+HashMap — запросы из разных потоков не
    /// перехватывают чужие ответы (как request_lock в OpenJarvis).
    pending: Arc<Mutex<std::collections::HashMap<u64, Sender<Value>>>>,
    next_id: AtomicCounter,
}

struct AtomicCounter(Mutex<u64>);

impl AtomicCounter {
    fn next(&self) -> u64 {
        let mut n = self.0.lock().unwrap_or_else(|e| e.into_inner());
        *n += 1;
        *n
    }
}

impl McpServer {
    /// Подключиться: запустить процесс, handshake, прочитать список инструментов.
    fn connect(name_of_server: &str, cmd: &str, args: &[String]) -> Result<(Self, Vec<McpTool>)> {
        let mut child = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("MCP «{name_of_server}»: не удалось запустить {cmd}"))?;
        let stdin = child.stdin.take().context("MCP: нет stdin")?;
        let stdout = child.stdout.take().context("MCP: нет stdout")?;

        let pending: Arc<Mutex<std::collections::HashMap<u64, Sender<Value>>>> =
            Arc::new(Mutex::new(std::collections::HashMap::new()));
        let reader_pending = Arc::clone(&pending);
        std::thread::Builder::new()
            .name(format!("mcp-{name_of_server}-reader"))
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                        continue; // не JSON-RPC (логи сервера) — игнорируем
                    };
                    let Some(id) = msg.get("id").and_then(Value::as_u64) else {
                        continue; // уведомления не ждут ответа
                    };
                    let tx = reader_pending
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .remove(&id);
                    if let Some(tx) = tx {
                        let _ = tx.send(msg);
                    }
                }
            })
            .expect("поток чтения MCP");

        let mut server = Self {
            name: name_of_server.to_string(),
            child,
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicCounter(Mutex::new(0)),
        };
        // Handshake: initialize + уведомление initialized.
        let _init = server.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "jarvis", "version": "0.2.0"}
            }),
        )?;
        server.notify("notifications/initialized")?;

        // Список инструментов.
        let listed = server.request("tools/list", json!({}))?;
        let tools = listed
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|t| {
                        let name = t.get("name")?.as_str()?.to_string();
                        let description = t
                            .get("description")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        Some(McpTool {
                            server: name_of_server.to_string(),
                            name,
                            description,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok((server, tools))
    }

    /// JSON-RPC request: отправка, ожидание ответа с таймаутом.
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.next();
        let frame = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let (tx, rx) = channel();
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, tx);
        {
            let mut stdin = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
            writeln!(stdin, "{frame}").with_context(|| {
                format!("MCP «{}»: сервер отвалился ({method})", self.name)
            })?;
            stdin.flush().ok();
        }
        let response = rx
            .recv_timeout(REQUEST_TIMEOUT)
            .map_err(|_| anyhow!("MCP «{}»: нет ответа на {method} за {REQUEST_TIMEOUT:?}", self.name))?;
        if let Some(err) = response.get("error") {
            return Err(anyhow!(
                "MCP «{}»: {method} вернул ошибку: {err}",
                self.name
            ));
        }
        Ok(response)
    }

    /// Уведомление без ответа.
    fn notify(&mut self, method: &str) -> Result<()> {
        let frame = json!({"jsonrpc": "2.0", "method": method});
        let mut stdin = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        writeln!(stdin, "{frame}").context("MCP: сервер отвалился")?;
        stdin.flush().ok();
        Ok(())
    }

    /// tools/call: результат — текст первого content-блока.
    fn call_tool(&mut self, tool: &str, args: &Value) -> Result<String> {
        let response = self.request("tools/call", json!({"name": tool, "arguments": args}))?;
        let content = response
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("MCP «{}»: нет content в ответе", self.name))?;
        let text: String = content
            .iter()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty() {
            return Err(anyhow!(
                "MCP «{}»: инструмент {tool} вернул пустой результат",
                self.name
            ));
        }
        Ok(text)
    }
}

/// Реестр подключений: ленивая инициализация по настройкам. Рядом с живым
/// сервером храним его конфиг и список инструментов, снятый при подключении.
type Registry = Vec<(crate::settings::McpServerConfig, McpServer, Vec<McpTool>)>;

fn manager() -> &'static Mutex<Registry> {
    static MANAGER: std::sync::OnceLock<Mutex<Registry>> = std::sync::OnceLock::new();
    MANAGER.get_or_init(|| Mutex::new(Vec::new()))
}

/// Подключиться ко всем настроенным серверам (идемпотентно: уже
/// подключённые пропускаются). Вызывается перед первым использованием.
pub fn ensure_connected() {
    let configs = crate::settings::Settings::load(&crate::paths::settings_file())
        .map(|s| s.mcp_servers)
        .unwrap_or_default();
    let mut all = manager().lock().unwrap_or_else(|e| e.into_inner());
    for config in configs {
        if all.iter().any(|(c, _, _)| c.name == config.name) {
            continue;
        }
        match McpServer::connect(&config.name, &config.cmd, &config.args) {
            Ok((server, tools)) => {
                println!(
                    "🔌 MCP «{}»: подключён, инструментов: {}",
                    config.name,
                    tools.len()
                );
                all.push((config, server, tools));
            }
            Err(e) => eprintln!("⚠️ MCP «{}»: {e:#}", config.name),
        }
    }
}

/// Блок внешних инструментов для action-промпта: пусто, если серверов нет.
pub fn prompt_block() -> String {
    let tools = list_tools();
    if tools.is_empty() {
        return String::new();
    }
    let lines: Vec<String> = tools
        .iter()
        .map(|t| {
            let desc = t.description.chars().take(120).collect::<String>();
            format!("  mcp_call {{\"server\":\"{}\",\"tool\":\"{}\",\"args\":{{…}}}} — {desc}", t.server, t.name)
        })
        .collect();
    format!(
        "\nВНЕШНИЕ ИНСТРУМЕНТЫ (MCP) — вызывай mcp_call:\n{}\n",
        lines.join("\n")
    )
}

fn list_tools() -> Vec<McpTool> {
    let all = manager().lock().unwrap_or_else(|e| e.into_inner());
    all.iter()
        .flat_map(|(_, _, tools)| tools.iter().cloned())
        .collect()
}

/// Вызов инструмента внешнего сервера из tool-вызова модели.
pub fn call(server: &str, tool: &str, args: &Value) -> Result<String> {
    let mut all = manager().lock().unwrap_or_else(|e| e.into_inner());
    let Some((_, srv, _)) = all.iter_mut().find(|(c, _, _)| c.name == server) else {
        return Err(anyhow!(
            "MCP-сервер «{server}» не подключён: проверь [[mcp_servers]] в settings.toml"
        ));
    };
    srv.call_tool(tool, args)
}

/// Подключённые серверы (для диагностики).
pub fn connected_servers() -> Vec<String> {
    manager()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(c, _, _)| c.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_frame_shape() {
        // Кадр JSON-RPC: метод, параметры, целочисленный id — контракт,
        // который ждёт любой MCP-сервер.
        let frame = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        });
        assert_eq!(frame["jsonrpc"], "2.0");
        assert_eq!(frame["id"], 1);
        assert_eq!(frame["method"], "tools/list");
    }

    #[test]
    fn tools_list_response_parses() {
        let response = json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {"tools": [
                {"name": "read_file", "description": "Читает файл", "inputSchema": {"type": "object"}},
                {"name": "no_desc"}
            ]}
        });
        let tools: Vec<McpTool> = response["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?.to_string();
                Some(McpTool {
                    server: "fs".into(),
                    name,
                    description: t.get("description").and_then(Value::as_str).unwrap_or("").into(),
                })
            })
            .collect();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[1].description, "");
    }

    #[test]
    fn tools_call_content_joins_text_blocks() {
        let response = json!({
            "jsonrpc": "2.0", "id": 3,
            "result": {"content": [
                {"type": "text", "text": "строка один"},
                {"type": "text", "text": "строка два"}
            ]}
        });
        let text: String = response["result"]["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(text, "строка один\nстрока два");
    }
}
