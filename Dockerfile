# Сборка Jarvis в контейнере: воспроизводимый путь, независимый от
# дистрибутива хоста. Артефакты падают в ./dist-out (том compose).
#
#   docker compose run --rm build
#
FROM archlinux:base-devel

# Зависимости Tauri на Linux (webkit2gtk-4.1), Rust, bun.
RUN pacman -Syu --noconfirm \
        webkit2gtk-4.1 \
        gtk3 \
        alsa-lib \
        libpulse \
        clang \
        pkgconf \
        curl \
        wget \
        file \
        git \
        git-lfs \
        unzip \
    && pacman -Scc --noconfirm || true

# Rust через rustup (пакет rust в Arch тянет меньше, но rustup даёт
# фиксированную тулчейн-версию для воспроизводимости).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal

# bun: единственный поддержанный пакетный менеджер фронтенда.
RUN curl -fsSL https://bun.sh/install | bash \
    && mv /root/.bun/bin/bun /usr/local/bin/bun

WORKDIR /src

# Скрипт сборки вызывает compose (см. compose.yml): он монтирует исходники
# и забирает артефакты в dist-out/.
CMD ["bash", "-lc", "bun install --frozen-lockfile && bun tauri build && mkdir -p /out && cp -v src-tauri/target/release/jarvis-app /out/ && cp -rv src-tauri/target/release/bundle/* /out/ 2>/dev/null || true"]
