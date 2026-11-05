import { store } from "./store.svelte";

/**
 * Единое описание состояния ассистента.
 *
 * Одна и та же фраза идёт в боковую панель и в шапку главной —
 * раньше эти надписи жили в разных файлах и расходились.
 *
 * tone: off — не слушает, idle — ждёт команду, live — слушает,
 * busy — идёт работа без звука (загрузка моделей, распознавание, генерация),
 * speaking — Джарвис сейчас говорит, error — подсистема не поднялась.
 */
export type Tone = "off" | "idle" | "live" | "busy" | "speaking" | "error";

/** Состояние компактного индикатора (StatusOrb). */
export type OrbState = "off" | "idle" | "listening" | "processing" | "speaking" | "error";

/**
 * Ассистент работает с фразой: распознаёт сказанное или готовит ответ.
 *
 * Важно для визуализации: в этой фазе звука уже нет, но индикатор должен
 * продолжать двигаться, иначе пауза выглядит как зависший интерфейс.
 */
export function assistantBusy(): boolean {
  return (
    (!store.speaking && store.engineState === "processing") ||
    (!store.speaking && store.processing) ||
    (!store.speaking && store.chat.busy)
  );
}

export function getAssistant(): { tone: Tone; short: string; full: string } {
  if (!store.status.engineRunning) {
    return { tone: "busy", short: "Запускаюсь…", full: "Загружаю модели — это займёт минуту" };
  }
  if (store.status.paused) {
    return { tone: "off", short: "На паузе", full: "Прослушивание на паузе" };
  }
  if (store.status.llmStatus === "error") {
    return { tone: "error", short: "Ошибка", full: "Модель ответов не загрузилась — журнал в «Помощи»" };
  }
  // Речь важнее остальных фаз: пока Джарвис говорит, он не «готов».
  if (store.speaking) {
    return { tone: "speaking", short: "Говорю", full: "Отвечаю…" };
  }
  if (store.status.llmStatus === "loading") {
    return { tone: "busy", short: "Запускаюсь…", full: "Загружаю модель ответов" };
  }
  if (store.chat.busy) {
    return { tone: "busy", short: "Думаю…", full: "Готовлю ответ…" };
  }
  if (assistantBusy()) {
    return { tone: "busy", short: "Разбираю…", full: "Разбираю сказанное…" };
  }
  if (store.engineState === "listening") {
    return { tone: "live", short: "Слушаю", full: "Слушаю вас" };
  }
  return { tone: "idle", short: "Готов", full: "Жду вашего вызова" };
}

/** Состояние для реактора: тон плюс нюансы, которые видит только сцена. */
export function getOrbState(): OrbState {
  const { tone } = getAssistant();
  if (tone === "live") return "listening";
  if (tone === "busy") return "processing";
  if (tone === "speaking") return "speaking";
  if (tone === "error") return "error";
  return tone;
}

/** Короткий технический статус голосовой зоны (моноширинный капс). */
export function voiceStatusLabel(): string {
  if (!store.status.engineRunning) return "ЗАГРУЗКА СИСТЕМ";
  if (store.status.paused) return "МИКРОФОН ВЫКЛ";
  if (store.status.llmStatus === "error") return "ОШИБКА ЯДРА";
  if (store.speaking) return "ПЕРЕДАЧА";
  if (store.chat.busy) return "ОБРАБОТКА ЗАПРОСА";
  if (assistantBusy()) return "АНАЛИЗ ФРАЗЫ";
  if (store.engineState === "listening") return "СЛУШАЮ";
  return "ГОТОВ";
}
