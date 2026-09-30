// Общее состояние UI: события движка + статус.
import { reactive } from "vue";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

export interface EngineEvent {
  type: string;
  [key: string]: unknown;
}

export const store = reactive({
  page: "dashboard" as string,
  status: {
    engineRunning: false,
    paused: false,
    llmStatus: "loading",
    ttsReady: false,
  },
  transcript: "" as string,
  wakeScore: 0 as number,
  chat: {
    question: "" as string,
    streaming: "" as string,
    done: "" as string,
    busy: false,
  },
  trainProgress: { epoch: 0, total: 0 } as { epoch: number; total: number },
  recording: false,
  logs: [] as string[],
});

export async function initEvents() {
  await listen<EngineEvent>("engine-event", (event) => {
    const ev = event.payload;
    switch (ev.type) {
      case "state_changed":
        break;
      case "wake_word_detected":
        store.wakeScore = (ev.score as number) ?? 0;
        break;
      case "transcript":
        store.transcript = (ev.text as string) ?? "";
        break;
      case "llm_chunk":
        store.chat.streaming = (ev.text as string) ?? "";
        store.chat.busy = true;
        break;
      case "llm_done":
        store.chat.done = (ev.text as string) ?? "";
        store.chat.streaming = "";
        store.chat.busy = false;
        break;
      case "llm_status":
        refreshStatus();
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
      case "error":
        store.logs.unshift(`❗ ${(ev.message as string) ?? ""}`);
        store.recording = false;
        break;
      default:
        store.logs.unshift(JSON.stringify(ev));
    }
  });

  // Периодический опрос статуса.
  setInterval(refreshStatus, 2000);
  await refreshStatus();
}

export async function refreshStatus() {
  try {
    store.status = await invoke("status");
  } catch {
    // движок ещё не поднят
  }
}
