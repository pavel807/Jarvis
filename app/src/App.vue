<script setup lang="ts">
import { computed, onMounted } from "vue";
import { store, initEvents } from "./store";
import Dashboard from "./pages/Dashboard.vue";
import WakeWord from "./pages/WakeWord.vue";
import Commands from "./pages/Commands.vue";
import Voice from "./pages/Voice.vue";
import Chat from "./pages/Chat.vue";
import SettingsPage from "./pages/SettingsPage.vue";
import Help from "./pages/Help.vue";

const pages: Record<string, string> = {
  dashboard: "Дашборд",
  wakeword: "Wake Word",
  commands: "Команды",
  voice: "Голос",
  chat: "Чат",
  settings: "Настройки",
  help: "Помощь",
};

// Синее свечение фона, когда ассистент слушает или отвечает.
const assistActive = computed(
  () =>
    store.status.engineRunning &&
    !store.status.paused &&
    (store.chat.busy || !!store.transcript),
);

onMounted(initEvents);
</script>

<template>
  <div class="app-root" :class="{ 'assist-active': assistActive }">
    <header class="topbar">
      <div class="logo">
        J.A.R.V.I.S
        <small>voice assistant <span class="v-badge">v0.2</span></small>
      </div>
      <nav class="topnav">
        <button
          v-for="(title, key) in pages"
          :key="key"
          :class="{ active: store.page === key }"
          @click="store.page = key"
        >
          {{ title }}
        </button>
      </nav>
    </header>

    <main class="content">
      <Dashboard v-if="store.page === 'dashboard'" />
      <WakeWord v-else-if="store.page === 'wakeword'" />
      <Commands v-else-if="store.page === 'commands'" />
      <Voice v-else-if="store.page === 'voice'" />
      <Chat v-else-if="store.page === 'chat'" />
      <SettingsPage v-else-if="store.page === 'settings'" />
      <Help v-else-if="store.page === 'help'" />
    </main>
  </div>
</template>
