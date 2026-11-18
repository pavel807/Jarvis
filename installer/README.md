# Jarvis Installer

Графический установщик Jarvis вместо `install.sh` / `uninstall.sh`: приложение [nsh](https://nsh.dev) в режиме
`mixed` — интерфейс на TSX (`App.tsx`), вся работа с процессами и файлами в Rust-крейте рядом
(`src/lib.rs`, экспорт `Installer` в `jarvis-installer/native`). Дизайн — тёмный Apple Pro: глубокий чёрный, стеклянный орб-монограмма, белая кнопка как на apple.com.

## Запуск

```sh
nsh run          # из папки installer/; окно десктопа
nsh check        # cargo check + типы
nsh test         # headless-тесты интерфейса
```

## Что делает

- **Установка** (по умолчанию): проверка системы, на Linux — рантайм-библиотеки Tauri
  (webkit2gtk-4.1, libpulse; недостающие — через `pkexec pacman`), загрузка готового бинарника
  из GitHub Releases (`releases/latest/download/jarvis-<платформа>.tar.gz`, схема — в
  `.github/workflows/release.yml`), установка `jarvis` / `jarvis-cli` в `~/.local/bin`,
  каталоги данных в `~/.local/share/Jarvis`, плагины `.sk` без перезаписи, ярлык в меню (Linux).
  Bun и cargo не нужны.
- **Собрать из исходников** — тумблер-фолбэк, если релиз недоступен: `bun install` +
  `bun tauri build` в `app/`, нужны bun и cargo.
- **Удаление**: бинарь и ярлык; галочка «Стереть данные, модели и плагины» сносит
  `~/.local/share/Jarvis` целиком (там голосовые сэмплы — без галочки остаются).

Пути можно переопределить: `JARVIS_REPO` (корень репозитория), `JARVIS_PREFIX`, `XDG_DATA_HOME`.

Журнал каждого шага виден в окне; ответы Rust-методов — JSON-строки `{"ok":bool,"log":string}`.
