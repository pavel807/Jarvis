//! Генератор базового датасета для обучения voice conversion.
//!
//! Синтезирует речь Silero TTS v5 всеми доступными голосами и раскладывает
//! результат по папкам `<out>/<voice>/*.wav`. Каждая папка = один диктор
//! для speaker encoder.
//!
//! Запуск:
//! ```text
//! cargo run --release --example silero_dataset -- \
//!     --out JARVIS-Voice-Engine/dataset/silero --per-voice 40
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

use jarvis_core::tts_silero::{SileroEngine, SILERO_VOICES};

/// Фразы для обучения: разные типы слов — числа, даты, термины, вопросы.
const PHRASES: &[&str] = &[
    "Привет, это проверка голоса для системы.",
    "Сегодня вторник, девятнадцатое ноября.",
    "Температура за окном плюс три градуса.",
    "Запускаю сборку проекта, это займёт несколько минут.",
    "Готов принять вашу команду.",
    "Обрабатываю запрос к базе данных.",
    "Найдено три файла в директории.",
    "Проверка связи с сервером завершена успешно.",
    "Сто двадцать три километра, двенадцать минут.",
    "Открываю панель управления.",
    "Восемьсот десять мегабайт свободного места.",
    "Это тестовое сообщение для оценки качества.",
    "Пожалуйста, повторите запрос.",
    "Алгоритм сортировки работает корректно.",
    "Установите новую версию программы.",
    "Принято, перехожу к выполнению.",
    "Интерфейс полностью готов к работе.",
    "Выполнение завершено без ошибок.",
    "Голосовой помощникJarvis слушает вас.",
    "Настройка параметров синтеза речи.",
    "Система загрузила четыреста мегабайт данных.",
    "Проверьте, пожалуйста, соединение.",
    "Закрываю задачу, спасибо за внимание.",
    "Мне нужно немного времени на обработку.",
    "Девяносто пятый процент операций завершён.",
    "Перехожу на резервный канал связи.",
    "Речь распознана корректно.",
    "Начинаю воспроизведение аудиозаписи.",
    "Стоимость составляет две тысячи рублей.",
    "Пожалуйста, говорите медленнее.",
];

struct Args {
    out: PathBuf,
    per_voice: usize,
}

fn parse_args() -> Result<Args> {
    let mut out = PathBuf::from("JARVIS-Voice-Engine/dataset/silero");
    let mut per_voice = 40usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => {
                out = PathBuf::from(
                    it.next().context("после --out ожидался путь")?,
                );
            }
            "--per-voice" => {
                per_voice = it
                    .next()
                    .context("после --per-voice ожидалось число")?
                    .parse()
                    .context("не удалось разобрать число")?;
            }
            other => anyhow::bail!("неизвестный аргумент: {other}"),
        }
    }
    Ok(Args { out, per_voice })
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for s in samples {
        let v = s.clamp(-1.0, 1.0);
        writer.write_sample((v * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;

    let mut total = 0usize;
    for (voice, _desc) in SILERO_VOICES {
        let dir = args.out.join(voice);
        fs::create_dir_all(&dir)?;

        let engine = match SileroEngine::new(voice) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("пропускаю голос {voice}: {err:#}");
                continue;
            }
        };

        let mut written = 0usize;
        for i in 0..args.per_voice {
            let phrase = PHRASES[i % PHRASES.len()];
            // Небольшой разброс скорости — речь не должна быть монотонной.
            let speed = 0.95 + (i % 5) as f32 * 0.025;
            let samples = match engine.synthesize(phrase, speed) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("  ! {voice}: {err:#}");
                    continue;
                }
            };
            let path = dir.join(format!("{:03}.wav", i));
            write_wav(&path, &samples, engine.sample_rate())?;
            written += 1;
            total += 1;
        }
        println!("{voice}: {written} файлов");
    }
    println!("всего {total} файлов в {}", args.out.display());
    Ok(())
}
