// Системные пути для espeak-rs-sys (см. Kokoros/kokoros/build.rs):
// на Linux его build-скрипт не добавляет стандартные пути к libsonic/libpcaudio,
// на macOS нужен явный линк на libc++.

fn main() {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-search=/usr/lib");
        println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=/usr/lib64");
        println!("cargo:rustc-link-lib=dylib=sonic");
        println!("cargo:rustc-link-lib=dylib=pcaudio");
    }

    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    }
}
