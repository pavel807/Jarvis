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
    let report =
        jarvis_core::holo::with_service(|h| Ok(h.run_task("Open Safari and go to aviasales.ru")?))?;
    println!(
        "Итог: {:?} за {:.1} с, шагов {}, действий {}",
        report.termination,
        started.elapsed().as_secs_f64(),
        report.steps.len(),
        report.total_actions,
    );
    for step in &report.steps {
        println!("  шаг {}: {}", step.step + 1, step.log.join(" | "));
    }

    println!("\n✅ E2E GUI-агента пройден");
    Ok(())
}
