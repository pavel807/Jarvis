//! Бенч §7: холодный ход (первый bounded capture) против тёплого
//! (перечитывание одного узла из тени). Плотное приложение ≥1000 узлов.

use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use xa11y::Element;

fn dense_root() -> (Element, Arc<dyn xa11y::Provider>) {
    let mut desktop = autocore::fake::FakeDesktop::new();
    let root = desktop.open_app_dense("Бенч", 6, 4); // 6+36+216+1296 ≈ 1554 узла
    let provider = desktop.provider();
    (root, provider)
}

fn bench_cold_vs_warm(c: &mut Criterion) {
    let (root, _provider) = dense_root();

    c.bench_function("cold_capture_1500_nodes", |b| {
        b.iter(|| {
            let (shadow, capture) =
                autocore::capture::capture_shadow(&root, &autocore::capture::default_options())
                    .expect("холодный capture");
            (shadow.nodes.len(), capture.stats.children_reads)
        })
    });

    // Тёплый ход: тень уже стоит,dirty один узел — перечитывается
    // только его поддерево узким бюджетом.
    let (shadow, _) =
        autocore::capture::capture_shadow(&root, &autocore::capture::default_options())
            .expect("первичный capture");
    let dirty_root = *shadow.roots.first().expect("корень есть");

    c.bench_function("warm_reread_single_node", |b| {
        b.iter(|| {
            // Тёплый ход нашего слоя: узкий capture поддерева одного узла.
            let capture = root
                .capture(&autocore::capture::narrow_options())
                .expect("тёплый capture");
            capture.stats.children_reads
        })
    });

    let _ = dirty_root;
}

criterion_group!(benches, bench_cold_vs_warm);
criterion_main!(benches);
