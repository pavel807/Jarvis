<script setup lang="ts">
import { computed } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { store } from "../store";
import ArcReactor from "../components/ArcReactor.vue";
import PageIcon from "../components/PageIcon.vue";

// Состояние реактора: disconnected → idle → active (слушаю/отвечаю)
const reactorState = computed<"disconnected" | "idle" | "active">(() => {
  if (!store.status.engineRunning) return "disconnected";
  if (store.status.paused) return "idle";
  if (store.chat.busy || store.transcript) return "active";
  return "idle";
});

const stateLabel = computed(() => {
  if (!store.status.engineRunning) return "Движок остановлен";
  if (store.status.paused) return "Пауза";
  if (store.chat.busy) return "Отвечаю…";
  if (reactorState.value === "active") return "Слушаю команду";
  return "Скажите «Джарвис»";
});

async function togglePause() {
  const next = !store.status.paused;
  await invoke("set_paused", { paused: next });
  store.status.paused = next;
}
</script>

<template>
  <h1><PageIcon name="wave" /> Дашборд</h1>
  <p class="page-hint">Скажите «Джарвис» и дайте команду или задайте вопрос.</p>

  <div class="card" style="text-align: center; overflow: hidden">
    <div class="reactor-box">
      <ArcReactor :state="reactorState" theme="cyan" />
    </div>
    <div class="state-text" style="margin-top: 4px">{{ stateLabel }}</div>
  </div>

  <div class="card">
    <div class="row" style="gap: 12px">
      <div>
        <div class="dim" style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px">LLM</div>
        <span class="badge" :class="store.status.llmStatus === 'ready' ? 'ok' : 'warn'">
          {{ store.status.llmStatus }}
        </span>
      </div>
      <div>
        <div class="dim" style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px">TTS</div>
        <span class="badge" :class="store.status.ttsReady ? 'ok' : 'warn'">
          {{ store.status.ttsReady ? "готов" : "загрузка…" }}
        </span>
      </div>
      <div>
        <div class="dim" style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px">Wake</div>
        <span v-if="store.wakeScore > 0" class="badge ok">
          {{ (store.wakeScore * 100).toFixed(0) }}%
        </span>
        <span v-else class="badge">—</span>
      </div>
    </div>

    <button class="primary" style="width: 100%; margin-top: 12px" @click="togglePause">
      {{ store.status.paused ? "▶ Возобновить слушание" : "⏸ Пауза слушания" }}
    </button>

    <h2>Последняя фраза</h2>
    <div class="mono" style="min-height: 20px; user-select: text">
      {{ store.transcript || "— тишина —" }}
    </div>
  </div>

  <div class="card">
    <h2>Журнал</h2>
    <div class="log">{{ store.logs.slice(0, 60).join("\n") || "— пусто —" }}</div>
  </div>
</template>

<style scoped>
.main-content {
  display: flex;
  flex-direction: column;
}

.reactor-box {
  display: flex;
  justify-content: center;
  height: calc(var(--reactor-scale, 1) * 305px);
  overflow: visible;
}
</style>
