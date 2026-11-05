//! Запуск команд оболочки: одна логика поверх `sh -c` / `cmd /C`.
//!
//! Модель пишет команды в терминологии POSIX (`ls`, `cd`, `grep`), потому что
//! промпт ей это подсказывает. На Windows честный исполнимый вариант — `cmd`,
//! а разницу в синтаксисе модель видит из строки «(windows)» в промпте
//! действий: притворяться, что `ls` работает везде, — хуже, чем сказать как есть.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Оболочка системы: (программа, флаг команды).
pub fn command_line() -> (&'static str, &'static str) {
    #[cfg(unix)]
    {
        ("sh", "-c")
    }
    #[cfg(windows)]
    {
        ("cmd", "/C")
    }
}

/// Выполняет команду и ждёт до `timeout`. Успех — только код 0;
/// иначе возвращается причина с выводом, а не молчаливый «готово».
pub fn run(cmd: &str, timeout: Duration) -> Result<String, String> {
    let (exe, flag) = command_line();
    let mut child = Command::new(exe)
        .arg(flag)
        .arg(cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child.wait_with_output().map_err(|e| e.to_string())?;
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();

                if status.success() {
                    return Ok(stdout);
                }
                // Не-нулевой код выхода — это неудача, даже если stderr пуст:
                // раньше такой случай молча выдавали за успех («Готово»).
                let code = status.code().unwrap_or(-1);
                if stderr.is_empty() {
                    if stdout.trim().is_empty() {
                        return Err(format!("команда завершилась с кодом {code}, вывода нет"));
                    }
                    return Err(format!(
                        "команда завершилась с кодом {code}, вывод: {}",
                        stdout.trim().chars().take(200).collect::<String>()
                    ));
                }
                return Err(format!("код {code}: {}", stderr.trim()));
            }
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    // Дожидаемся завершения убитого процесса: без wait он
                    // остаётся зомби, а пайпы — незакрытыми.
                    let _ = child.wait();
                    return Err(format!(
                        "команда выполнялась дольше {} сек, прервано",
                        timeout.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// Запускает команду в фоне, не дожидаясь результата (для команд из
/// commands.toml, где ожидание бессмысленно).
pub fn spawn(cmd: &str) -> bool {
    let (exe, flag) = command_line();
    Command::new(exe)
        .arg(flag)
        .arg(cmd)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}

/// Запускает «скрипт для текущей ОС» в фоне: на macOS — AppleScript через
/// osascript (тип действия `stop` в commands.toml исторически хранит именно
/// скрипты), на Windows — PowerShell, на Linux — sh. Возвращает, удалось ли
/// отправить на исполнение.
pub fn spawn_script(script: &str) -> bool {
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("osascript");
        c.args(["-e", script]);
        c
    };
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("powershell");
        c.args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ]);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.args(["-c", script]);
        c
    };

    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}