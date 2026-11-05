import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

/**
 * Состояние приложения.
 *
 * Три слоя:
 *  - `store` — то, что приходит от движка: статус, события, история диалога;
 *  - `settings` — то, что пользователь меняет сам;
 *  - `ui` — навигация и прочее локальное состояние интерфейса.
 *
 * События движка переводятся в человеческие фразы сразу при получении:
 * страницам не нужно знать про `llm_status` и `audio_level`.
 *
 * Файл называется `.svelte.ts`, чтобы Svelte 5 скомпилировал руны.
 */

export interface EngineEvent {
  type: string;
  [key: string]: unknown;
}

export interface EngineStatus {
  engineRunning: boolean;
  paused: boolean;
  llmStatus: string;
  ttsReady: boolean;
  /** Рабочая папка агента: от неё считаются пути файловых инструментов. */
  cwd: string;
}

export interface Settings {
  // Детекция фразы (в интерфейсе не показывается — правится в settings.toml)
  silence_rms_threshold: number;
  silence_timeout_secs: number;
  min_command_duration_secs: number;
  max_command_duration_secs: number;
  followup_window_secs: number;
  // Wake word
  wake_threshold: number;
  wakeword_model: string;
  // Ответы
  llm_enabled: boolean;
  /** Бэкенд LLM: "uzu" (Metal, модели каталога Mirai) или "candle". */
  llm_backend: string;
  /** Модель LLM (имя меняет смысл в зависимости от бэкенда). */
  llm_model: string;
  web_search_auto: boolean;
  llm_normalize: boolean;
  llm_temperature: number;
  /** Страховочный лимит длины ответа в токенах; 0 — без лимита. */
  llm_max_tokens: number;
  // Голос
  tts_voice: string;
  tts_speed: number;
  // Отклик
  ack_before_llm: boolean;
  /** Помнить темы, от которых разговор ушёл в сторону. */
  topic_memory: boolean;
  /** Фильтр говорящего: в ответы идёт только голос владельца. */
  speaker_filter: boolean;
  /** Порог косинусного сходства с голосовым профилем владельца. */
  speaker_threshold: number;
}

export interface Toast {
  id: number;
  text: string;
  kind: "info" | "ok" | "err";
}

/** Реплика в общей истории диалога на главной. */
export interface Turn {
  id: number;
  /** user — сказал человек; assistant — ответил Джарвис; note — служебная пометка. */
  role: "user" | "assistant" | "note";
  text: string;
  /** Откуда реплика: голос, клавиатура или итог выполненного действия. */
  source: "voice" | "text" | "action";
  /** Ответ ещё печатается: показываем каретку, реплику можно перебить. */
  streaming?: boolean;
  /** Неудача: ошибка движка вместо ответа. */
  error?: boolean;
}

function now(): string {
  return new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

let ids = 0;

export const store = $state({
  page: "home" as string,
  /** Страница, с которой пришли в подраздел — для кнопки «назад». */
  pageBack: "more" as string,
  /** Боковая панель свернута пользователем (иконки без подписей). */
  sidebarCompact: false as boolean,
  /**
   * Идёт сборка реактора на запуске: интерфейс скрыт до розжига.
   * Снимает флаг сам реактор (ArcReactor), страховка — таймер в App.
   */
  booting: true as boolean,
  /**
   * Просьба «проиграй анимацию реактора» в уже запущенном интерфейсе.
   * Флаг поднимает событие движка, снимает его реактор на розжиге.
   */
  introRequest: false as boolean,
  status: {
    engineRunning: false,
    paused: false,
    llmStatus: "loading",
    ttsReady: false,
    cwd: "",
  } as EngineStatus,
  /** Что делает движок: wake_word | listening | processing. */
  engineState: "wake_word" as string,
  /**
   * Движок распознаёт сказанное. Отдельный флаг, потому что состояние
   * «processing» появилось в движке позже: если его нет, флаг поднимает
   * приход расшифровки — так интерфейс не остаётся без сигнала.
   */
  processing: false as boolean,
  transcript: "" as string,
  wakeScore: 0 as number,
  audioLevel: 0 as number,
  /**
   * Откуда пришёл уровень звука: «mic» — говорит пользователь, «tts» — сам
   * Джарвис. Без этого индикатор одинаково реагировал бы на свою же речь.
   */
  audioSource: "mic" as string,
  /** Джарвис сейчас озвучивает ответ. В этой фазе его можно перебить. */
  speaking: false as boolean,
  /** Команда, которую движок распознал дословно. */
  matchedCommand: "" as string,
  chat: {
    question: "" as string,
    streaming: "" as string,
    done: "" as string,
    busy: false,
    /** Последняя ошибка движка во время ожидания ответа. */
    error: "" as string,
  },
  /**
   * Единая история диалога: голос, клавиатура, итоги действий.
   * Живёт до закрытия окна, очищается кнопкой на главной.
   */
  history: [] as Turn[],
  /** Момент последней реплики в диалоге — для телеметрии на главной. */
  lastActivity: 0 as number,
  trainProgress: { epoch: 0, total: 0 } as { epoch: number; total: number },
  recording: false,
  /**
   * Доступ к микрофону: false — движок ждёт разрешения (показываем меню),
   * true — получено (меню закрывается), null — вопросов нет.
   */
  micPermission: null as boolean | null,
  /** Технический журнал — целиком показывается только в «Помощи». */
  logs: [] as string[],
  toasts: [] as Toast[],
});

export function toast(text: string, kind: Toast["kind"] = "info", ms = 2800) {
  const item: Toast = { id: ++ids, text, kind };
  store.toasts.push(item);
  setTimeout(() => {
    const i = store.toasts.findIndex((t) => t.id === item.id);
    if (i !== -1) store.toasts.splice(i, 1);
  }, ms);
}

/** Вторичные экраны: открываются из разделов и требуют кнопку «назад». */
export const SUB_PAGES = ["commands", "voice", "wakeword", "help"];

/** Четыре раздела нижней навигации. */
export const NAV_PAGES = ["home", "memory", "system", "settings"];

export function go(page: string) {
  // Запоминаем раздел, откуда ушли, чтобы «назад» вела именно туда.
  if (!SUB_PAGES.includes(page)) store.pageBack = page;
  store.page = page;
}

// ---------- История диалога ----------

/** Добавляет реплику в историю (используют и события движка, и чат UI). */
export function pushTurn(turn: Omit<Turn, "id">): Turn {
  const item: Turn = { id: ++ids, ...turn };
  store.history.push(item);
  store.lastActivity = Date.now();
  if (store.history.length > 200) store.history.splice(0, store.history.length - 200);
  return item;
}

function noteTurn(text: string) {
  pushTurn({ role: "note", text, source: "action" });
}

/**
 * Финальный ответ ассистента: если ответ уже печатался (стриминг),
 * обновляем его, иначе добавляем реплику.
 */
function completeReply(text: string, source: Turn["source"] = "voice") {
  if (!text) return;
  store.lastActivity = Date.now();
  const last = store.history[store.history.length - 1];
  if (last && last.role === "assistant" && last.streaming) {
    last.text = text;
    last.streaming = false;
    return;
  }
  pushTurn({ role: "assistant", text, source, streaming: false });
}

function clearHistory() {
  store.history.length = 0;
  store.matchedCommand = "";
  store.transcript = "";
  store.chat.question = "";
  store.chat.streaming = "";
  store.chat.done = "";
  store.chat.error = "";
}

// ---------- Настройки ----------

export const settings = $state<Settings>({
  silence_rms_threshold: 0.018,
  silence_timeout_secs: 0.7,
  min_command_duration_secs: 0.5,
  max_command_duration_secs: 8,
  followup_window_secs: 8,
  wake_threshold: 0.6,
  wakeword_model: "jarvis_model_2.rpw",
  llm_enabled: true,
  llm_backend: "uzu",
  llm_model: "qwen3.5-2b-mirai-m",
  web_search_auto: true,
  llm_normalize: true,
  llm_temperature: 0.6,
  llm_max_tokens: 0,
  tts_voice: "ru_dmitriy",
  tts_speed: 1.0,
  ack_before_llm: false,
  topic_memory: true,
  speaker_filter: true,
  speaker_threshold: 0.6,
});

export async function loadSettings() {
  try {
    Object.assign(settings, await invoke<Settings>("get_settings"));
  } catch {
    // Движок ещё не поднят — остаются значения по умолчанию.
  }
}

/**
 * Меняет настройки и сразу сохраняет. Бэкенд ждёт структуру целиком, поэтому
 * отправляем весь объект: поля, которых нет в интерфейсе, сохраняются как есть.
 */
export async function patchSettings(patch: Partial<Settings>) {
  const previous = { ...settings };
  Object.assign(settings, patch);
  try {
    await invoke("save_settings", { settings: { ...settings } });
  } catch (e) {
    Object.assign(settings, previous);
    toast(`Не удалось сохранить: ${e}`, "err");
  }
}

// ---------- Инициализация ----------

export async function initEvents() {
  await listen<EngineEvent>("engine-event", (event) => {
    const ev = event.payload;
    switch (ev.type) {
      case "state_changed": {
        const state = (ev.state as string) ?? "wake_word";
        store.engineState = state;
        if (state === "processing") store.processing = true;
        else if (state === "wake_word" || state === "listening") store.processing = false;
        break;
      }
      case "wake_word_detected":
        store.wakeScore = (ev.score as number) ?? 0;
        store.transcript = "";
        store.processing = false;
        break;
      case "transcript":
        // Расшифровка приходит один раз на фразу — это финальный текст.
        store.transcript = (ev.text as string) ?? "";
        if (store.transcript.trim()) {
          pushTurn({ role: "user", text: store.transcript, source: "voice" });
        }
        store.processing = true;
        break;
      case "topic_resumed":
        noteTurn(`Вернулись к теме: ${((ev.topic as string) ?? "").trim()}`);
        store.logs.unshift(`${now()}  🧠 Возврат к теме: ${ev.topic}`);
        break;
      case "action_started": {
        const summary = ((ev.summary as string) ?? "").slice(0, 120);
        store.logs.unshift(`${now()}  🔧 ${(ev.tool as string) ?? ""} ${summary}`);
        if (store.logs.length > 300) store.logs.length = 300;
        break;
      }
      case "action_result": {
        const summary = ((ev.summary as string) ?? "").slice(0, 300);
        store.matchedCommand = "";
        // Итог действия Джарвис проговаривает — показываем ту же фразу.
        if (summary) {
          pushTurn({ role: "assistant", text: summary, source: "action" });
        }
        store.logs.unshift(`${now()}  ${(ev.ok as boolean) ? "✅" : "❗"} ${summary}`);
        if (store.logs.length > 300) store.logs.length = 300;
        break;
      }
      case "command_matched":
        // Дословное совпадение: без этой пометки казалось, что Джарвис
        // молча проигнорировал просьбу.
        store.matchedCommand = (ev.name as string) ?? "";
        noteTurn(`Команда «${store.matchedCommand}»`);
        store.logs.unshift(`${now()}  ⚡ Команда: ${store.matchedCommand}`);
        break;
      case "command_not_found":
        store.logs.unshift(`${now()}  ❓ Команда не найдена: ${(ev.text as string) ?? ""}`);
        break;
      case "reactor_intro":
        // Реактор живёт только на главной: без возврата просьба осталась бы
        // невыполненной, если пользователь был в другом разделе.
        store.introRequest = true;
        store.booting = true;
        go("home");
        break;
      case "llm_status":
        if ((ev.status as string) === "thinking") store.chat.busy = true;
        void refreshStatus();
        break;
      case "llm_chunk": {
        // Приходят нарастающие фрагменты: печатаем каретку на конце.
        const text = (ev.text as string) ?? "";
        store.chat.streaming = text;
        store.chat.busy = true;
        store.chat.error = "";
        store.lastActivity = Date.now();
        const last = store.history[store.history.length - 1];
        if (last && last.role === "assistant" && last.streaming) {
          last.text = text;
        } else {
          pushTurn({ role: "assistant", text, source: "voice", streaming: true });
        }
        break;
      }
      case "llm_done":
        store.chat.done = (ev.text as string) ?? "";
        store.chat.streaming = "";
        store.chat.busy = false;
        store.chat.error = "";
        store.processing = false;
        completeReply(store.chat.done);
        break;
      case "audio_level":
        store.audioLevel = (ev.level as number) ?? 0;
        store.audioSource = (ev.source as string) ?? "mic";
        break;
      case "train_progress":
        store.trainProgress = {
          epoch: (ev.epoch as number) ?? 0,
          total: (ev.total as number) ?? 0,
        };
        break;
      case "train_done":
        store.trainProgress = { epoch: 0, total: 0 };
        break;
      case "record_saved":
        store.recording = false;
        break;
      case "mic_permission":
        store.micPermission = (ev.granted as boolean) ?? false;
        if (store.micPermission) toast("Микрофон подключён", "ok");
        break;
      case "log":
        store.logs.unshift(`${now()}  ${(ev.message as string) ?? ""}`);
        if (store.logs.length > 300) store.logs.length = 300;
        break;
      case "error": {
        const message = (ev.message as string) ?? "Неизвестная ошибка";
        const wasChatPending = store.chat.busy || Boolean(store.chat.question);
        store.recording = false;
        store.chat.busy = false;
        store.processing = false;
        // Чат ждал ответ — вместо вечно крутящегося спиннера покажем ошибку.
        if (wasChatPending) store.chat.error = message;
        // Ошибка прерывает и печатающийся ответ: реплика не должна висеть
        // с кареткой навсегда.
        const last = store.history[store.history.length - 1];
        if (last && last.role === "assistant" && last.streaming && !last.text) {
          store.history.pop();
        } else if (last && last.role === "assistant" && last.streaming) {
          last.streaming = false;
        }
        if (wasChatPending) {
          pushTurn({ role: "assistant", text: `Не получилось: ${message}`, source: "text", error: true });
        }
        store.logs.unshift(`${now()}  ❗ ${message}`);
        toast(message, "err", 5000);
        break;
      }
      default:
        break;
    }
  });

  // Фаза «говорит» отдельным событием: у движка нет для неё состояния, а без
  // неё в статусе было «Готов», пока Джарвис ещё договаривал фразу.
  await listen<{ speaking: boolean }>("speech", (event) => {
    store.speaking = Boolean(event.payload?.speaking);
    // Озвучка закончилась — печатающийся ответ, если он ещё висит, закрыт.
    if (!store.speaking) {
      const last = store.history[store.history.length - 1];
      if (last && last.role === "assistant" && last.streaming && !store.chat.busy) {
        last.streaming = false;
      }
    }
  });

  setInterval(refreshStatus, 2000);
  await Promise.all([refreshStatus(), loadSettings()]);
}

export async function refreshStatus() {
  try {
    store.status = await invoke<EngineStatus>("status");
  } catch {
    // движок ещё не поднят
  }
}

export { clearHistory, noteTurn };
