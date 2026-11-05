/**
 * Мок Tauri IPC для разработки интерфейса в обычном браузере.
 *
 * Приложение — desktop-окно (1240×820), поэтому верстать и проверять дизайн
 * удобнее через `bun run dev` и http://localhost:1420/dev/preview.html:
 * настоящий движок (модели на 4 ГБ) для этого не нужен.
 *
 * Мок подменяет ровно то, что использует фронтенд: `invoke` и шину событий.
 * Данные подобраны так, чтобы страницы выглядели как в жизни: список команд
 * из commands.toml, голоса Silero, сэмплы wake word.
 *
 * Состояние можно переключать из консоли браузера: `window.jarvisMock.scenario(...)`.
 */

type Handler = (payload: unknown) => void;

const handlers = new Map<number, Handler>();
let nextCallbackId = 1;

const VOICES: [string, string][] = [
  ["ru_eduard", "мужской, основной"],
  ["ru_dmitriy", "мужской"],
  ["ru_alexandr", "мужской"],
  ["ru_bogdan", "мужской"],
  ["ru_igor", "мужской"],
  ["ru_marat", "мужской"],
  ["ru_roman", "мужской"],
  ["ru_gamat", "мужской"],
  ["ru_aigul", "женский"],
  ["ru_albina", "женский"],
  ["ru_alfia", "женский"],
  ["ru_alfia2", "женский"],
  ["ru_ekaterina", "женский"],
  ["ru_vika", "женский"],
  ["ru_karina", "женский"],
  ["ru_kermen", "женский"],
  ["ru_miyau", "женский"],
  ["ru_nurgul", "женский"],
  ["ru_oksana", "женский"],
  ["ru_onaoy", "женский"],
  ["ru_ramilia", "женский"],
  ["ru_safarhuja", "женский"],
  ["ru_saida", "женский"],
  ["ru_sibday", "женский"],
  ["ru_zara", "женский"],
  ["ru_zhadyra", "женский"],
  ["ru_zhazira", "женский"],
  ["ru_zinaida", "женский"],
  ["ru_kejilgan", "женский"],
];

const settings = {
  silence_rms_threshold: 0.018,
  silence_timeout_secs: 0.7,
  min_command_duration_secs: 0.5,
  max_command_duration_secs: 3,
  followup_window_secs: 8,
  wake_threshold: 0.6,
  wakeword_model: "jarvis_model_2.rpw",
  llm_enabled: true,
  llm_backend: "uzu",
  llm_model: "qwen3.5-2b-mirai-l",
  web_search_auto: true,
  llm_normalize: true,
  llm_temperature: 0.6,
  llm_max_tokens: 256,
  tts_voice: "ru_dmitriy",
  tts_speed: 0.95,
  ack_before_llm: false,
  topic_memory: true,
  speaker_filter: true,
  speaker_threshold: 0.6,
};

const commands = {
  global: {
    reply_success: "resources/audio/ok{}.wav",
    reply_error: "resources/audio/reply1.wav",
    reply_unknown: "resources/audio/reply1.wav",
    reply_ack: "resources/audio/reply1.wav",
    fuzzy_min_score: 0.72,
  },
  commands: [
    {
      name: "YouTube",
      trigger_words: ["youtube", "ютуб", "видео"],
      action_type: "url",
      action_value: "https://www.youtube.com",
      enabled: true,
    },
    {
      name: "Найти в интернете",
      trigger_words: ["найди", "поищи", "загугли"],
      action_type: "web",
      action_value: "",
      enabled: true,
    },
    {
      name: "Adobe Premiere Pro",
      trigger_words: ["premiere pro", "монтаж", "видеомонтаж"],
      action_type: "app",
      action_value: "Adobe Premiere Pro 2026",
      enabled: true,
    },
    {
      name: "Telegram",
      trigger_words: ["telegram", "телеграм", "телега"],
      action_type: "url",
      action_value: "https://web.telegram.org",
      enabled: false,
    },
    {
      name: "Замолчать",
      trigger_words: ["замолчи", "стоп", "хватит"],
      action_type: "stop",
      action_value: "",
      enabled: true,
    },
    {
      name: "Погода в браузере",
      trigger_words: ["погода"],
      action_type: "search",
      action_value: "https://www.google.com/search?q={}",
      enabled: true,
    },
  ],
};

const samples = {
  train: [
    ["train/jarvis1/wake_1.wav", 1.4],
    ["train/jarvis1/wake_2.wav", 1.2],
    ["train/jarvis1/wake_3.wav", 1.6],
  ] as [string, number][],
  test: [["test/jarvis1/wake_1.wav", 1.3]] as [string, number][],
};

const tasks = [
  { id: 1, text: "Собрать обратную связь по интерфейсу Mark V", done: false },
  { id: 2, text: "Проверить приоритеты команд после чистки списка", done: false },
  { id: 3, text: "Обновить образцы wake word на новый микрофон", done: true },
];

let taskSeq = tasks.length;

let engineState = {
  engineRunning: true,
  paused: false,
  llmStatus: "ready",
  ttsReady: true,
  // Рабочая папка агента: у живого приложения это домашний каталог.
  cwd: "/Users/user/Projects",
};

/**
 * Слушатели событий Tauri. `invoke('plugin:event|listen')` получает не сам
 * обработчик, а его id из `transformCallback` — поэтому запоминаем и функцию,
 * иначе эмуляция события вызвала бы «listener.handler is not a function».
 */
const listeners = new Map<number, { event: string; handler: Handler }>();
let nextListenerId = 1;

function fire(event: string, payload: unknown) {
  for (const listener of listeners.values()) {
    if (listener.event === event) listener.handler({ event, id: 0, payload });
  }
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function invoke(cmd: string, args?: Record<string, unknown>): Promise<unknown> {
  switch (cmd) {
    case "status":
      return { ...engineState };
    case "set_paused":
      engineState.paused = Boolean(args?.paused);
      return null;
    case "get_cwd":
      return engineState.cwd;
    case "set_cwd":
      engineState.cwd = String(args?.path ?? engineState.cwd);
      return engineState.cwd;
    case "get_commands":
      return structuredClone(commands);
    case "save_commands":
      return null;
    case "get_settings":
      return { ...settings };
    case "save_settings":
      return null;
    case "get_logs":
      return [];
    case "tasks_list":
      return structuredClone(tasks);
    case "tasks_add":
      taskSeq += 1;
      tasks.push({ id: taskSeq, text: String(args?.text ?? "Без названия"), done: false });
      return structuredClone(tasks[tasks.length - 1]);
    case "tasks_set_done": {
      const id = Number(args?.id ?? 0);
      const task = tasks.find((t) => t.id === id);
      if (task) task.done = Boolean(args?.done);
      return task ? structuredClone(task) : null;
    }
    case "tasks_remove": {
      const id = Number(args?.id ?? 0);
      const index = tasks.findIndex((t) => t.id === id);
      if (index === -1) return null;
      const [removed] = tasks.splice(index, 1);
      return structuredClone(removed);
    }
    case "tasks_open":
      return null;
    case "tts_voices":
      return VOICES;
    case "tts_test":
      return null;
    case "open_external":
      return null;
    case "ww_samples":
      return structuredClone(samples);
    case "ww_delete":
      samples.train = samples.train.filter(([p]) => p !== args?.path);
      return null;
    case "ww_test":
      return 0.78;
    case "ww_record":
    case "ww_build":
    case "ww_apply":
    case "llm_ask":
    case "llm_cancel":
      return null;
    case "plugin:event|listen": {
      // handler приходит как id, зарегистрированный через transformCallback:
      // достаём саму функцию по этому id.
      const id = Number(args?.handler ?? 0);
      const handler = handlers.get(id);
      if (handler) {
        const listenerId = nextListenerId++;
        listeners.set(listenerId, { event: String(args?.event ?? ""), handler });
        return listenerId;
      }
      return id;
    }
    case "plugin:event|unlisten":
      listeners.delete(Number(args?.eventId));
      return null;
    default:
      return null;
  }
}

/** Сценарии для проверки состояний интерфейса без живого движка. */
const scenarios = {
  /** Готов, ждёт «Джарвис» — стартовый экран. */
  idle() {
    engineState = {
      engineRunning: true,
      paused: false,
      llmStatus: "ready",
      ttsReady: true,
      cwd: engineState.cwd,
    };
    // Фазу «говорит» движок сообщает отдельным событием, а не состоянием.
    fire("speech", { speaking: false });
  },
  /** Модель ещё грузится. */
  loading() {
    engineState = {
      engineRunning: true,
      paused: false,
      llmStatus: "loading",
      ttsReady: false,
      cwd: engineState.cwd,
    };
  },
  /** Слушает команду после wake word. */
  listening() {
    scenarios.idle();
    fire("engine-event", { type: "state_changed", state: "listening" });
    fire("engine-event", { type: "audio_level", level: 0.42, source: "mic" });
  },
  /** Распознал фразу, готовит ответ. */
  thinking() {
    scenarios.idle();
    fire("engine-event", { type: "state_changed", state: "listening" });
    fire("engine-event", { type: "transcript", text: "какая сегодня погода в москве" });
    fire("engine-event", { type: "llm_status", status: "thinking" });
  },
  /** Отвечает и озвучивает: в статусе «Говорю», есть кнопка «Остановить». */
  answering() {
    scenarios.thinking();
    fire("engine-event", { type: "llm_chunk", text: "Сегодня в Москве" });
    fire("speech", { speaking: true });
    fire("engine-event", { type: "audio_level", level: 0.55, source: "tts" });
  },
  /** Речь закончилась, движок распознаёт фразу. */
  processing() {
    scenarios.idle();
    fire("engine-event", { type: "wake_word_detected", name: "jarvis", score: 0.84 });
    fire("engine-event", { type: "state_changed", state: "listening" });
    fire("engine-event", { type: "transcript", text: "какая сегодня погода в москве" });
    fire("engine-event", { type: "state_changed", state: "processing" });
  },
  /** Полный цикл: услышал, распознал, ответил. */
  done() {
    scenarios.processing();
    fire("engine-event", { type: "command_matched", name: "Найти в интернете" });
    fire("engine-event", { type: "llm_status", status: "thinking" });
    fire("engine-event", { type: "llm_chunk", text: "Сегодня в Москве" });
    fire("engine-event", {
      type: "llm_done",
      text:
        "Сегодня в Москве около **+7 градусов**, облачно с прояснениями.\n\nВетер небольшой — до `3 м/с`. Теплее одеться не помешает.",
    });
    fire("engine-event", { type: "state_changed", state: "listening" });
    fire("engine-event", { type: "log", message: "TTS-сборка: silero v5, голос «ru_dmitriy»" });
  },
  /** Действие инструментом: видно и услышанное, и итог действия. */
  action() {
    scenarios.idle();
    fire("engine-event", { type: "wake_word_detected", name: "jarvis", score: 0.91 });
    fire("engine-event", { type: "state_changed", state: "listening" });
    fire("engine-event", { type: "transcript", text: "сверни зум" });
    fire("engine-event", { type: "action_started", tool: "window", summary: "minimize зум" });
    fire("engine-event", { type: "action_result", ok: true, summary: "Свернул окно Zoom" });
    fire("speech", { speaking: true });
  },
  /** Прослушивание на паузе: статус «На паузе» и кнопка возврата. */
  paused() {
    engineState = { ...engineState, paused: true };
    fire("engine-event", { type: "llm_status", status: "ready" });
  },
  /** Ошибка модели — статус «Ошибка» и индикатор. */
  failed() {
    engineState = {
      engineRunning: true,
      paused: false,
      llmStatus: "error",
      ttsReady: false,
      cwd: engineState.cwd,
    };
    fire("engine-event", { type: "error", message: "Не удалось загрузить модель распознавания речи" });
  },
  /** Чат: вопрос из поля ввода и печатающийся ответ с Markdown. */
  chat() {
    scenarios.idle();
    fire("engine-event", { type: "transcript", text: "открой ютуб" });
    fire("engine-event", { type: "command_matched", name: "YouTube" });
    scenarios.idle();
    store.chat.busy = true;
    store.history.push({ id: 9001, role: "user", text: "Сравни candle и llama.cpp в двух словах", source: "text" });
    fire("engine-event", { type: "llm_chunk", text: "Коротко:" });
  },
};

import { store } from "../src/store.svelte";

(globalThis as Record<string, unknown>).jarvisMock = { scenarios, fire, invoke, commands, settings };

// Точка входа @tauri-apps/api — единственное, что нужно подменить.
(globalThis as Record<string, unknown>).__TAURI_INTERNALS__ = {
  invoke,
  transformCallback(callback: Handler, once = false) {
    const id = nextCallbackId++;
    handlers.set(id, once ? (payload) => { handlers.delete(id); callback(payload); } : callback);
    return id;
  },
  unregisterCallback(id: number) {
    handlers.delete(id);
  },
  convertFileSrc: (path: string) => path,
  metadata: { currentWindow: { label: "main" }, currentWebview: { label: "main" } },
};

// Страница и сценарий задаются в адресе: удобно снимать скриншоты всех экранов.
const params = new URLSearchParams(location.search);
if (params.get("page")) store.page = params.get("page") as string;
if (params.get("log")) store.logs.push(...params.get("log")!.split("|"));

// Сценарий проигрываем с задержкой: приложение должно успеть подписаться
// на события, иначе они уйдут в пустоту.
const scenario = params.get("scenario") as keyof typeof scenarios | null;
if (scenario && scenarios[scenario]) {
  await delay(400);
  scenarios[scenario]();
}
