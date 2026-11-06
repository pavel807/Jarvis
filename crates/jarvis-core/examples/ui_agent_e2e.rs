//! E2E GUI-агента через весь стек: сервис → реальный скриншот → процессор →
//! vision-башня → фьюжн → гибридный декодер → описание экрана.
//! Run: JARVIS_HOME=<repo> cargo run --release -p jarvis-core --example ui_agent_e2e
fn main() -> anyhow::Result<()> {
    let (events_tx, events_rx) = std::sync::mpsc::channel::<jarvis_core::events::JarvisEvent>();
    std::thread::spawn(move || {
        while let Ok(ev) = events_rx.recv() {
            if let jarvis_core::events::JarvisEvent::Log { message } = ev {
                println!("  [log] {message}");
            }
        }
    });

    let handle = jarvis_core::holo::spawn_service(events_tx);
    jarvis_core::holo::set_service(handle);

    let st = jarvis_core::holo::with_service(|h| Ok(h.status()))?;
    println!("Статус: {:?} — {}", st.stage, st.detail);
    if !st.downloaded {
        anyhow::bail!("Holo-3.1-4B не скачана: {}", st.detail);
    }

    println!("\n=== screen_look: что на экране ===");
    let started = std::time::Instant::now();
    let description: String =
        jarvis_core::holo::with_service(|h| {
            Ok(h.look("Опиши кратко, что открыто на экране.")?)
        })?;
    println!("Описание ({:.1} с): {description}", started.elapsed().as_secs_f64());

    println!("\n=== ui_task: агентная задача целиком (таймаут 240 с) ===");
    let started = std::time::Instant::now();
    let result = jarvis_core::holo::with_service(|h| {
        Ok(h.run_task_full("Open Safari and go to aviasales.ru", None)?)
    })?;
    println!(
        "Итог: {} (успех: {}) за {:.1} с, шагов {}, действий {}",
        result.summary,
        result.succeeded(),
        started.elapsed().as_secs_f64(),
        result.steps,
        result.total_actions,
    );
    for line in &result.evidence {
        println!("  {line}");
    }

    println!("\n✅ E2E GUI-агента пройден");
    Ok(())
}
