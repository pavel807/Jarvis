<script setup lang="ts">
import { ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { store } from "../store";
import PageIcon from "../components/PageIcon.vue";

const question = ref("");

async function ask() {
  if (!question.value.trim() || store.chat.busy) return;
  store.chat.question = question.value;
  store.chat.streaming = "";
  store.chat.done = "";
  store.chat.busy = true;
  await invoke("llm_ask", { prompt: question.value });
  question.value = "";
}

function stop() {
  invoke("llm_cancel");
}
</script>

<template>
  <h1><PageIcon name="chat" /> Чат с ИИ</h1>
  <p class="page-hint">Локальная модель отвечает без интернета — ровно так же, как на голосовые вопросы.</p>
  <p class="dim" style="margin-bottom: 14px">
    Локальная модель отвечает так же, как на голосовые вопросы — без интернета.
  </p>

  <div class="card">
    <div class="row">
      <input
        type="text"
        v-model="question"
        placeholder="Спросите что-нибудь…"
        @keyup.enter="ask"
        :disabled="store.chat.busy"
      />
      <button class="primary" :disabled="store.chat.busy" @click="ask">Спросить</button>
      <button v-if="store.chat.busy" class="danger" @click="stop">Стоп</button>
    </div>
  </div>

  <div v-if="store.chat.question" class="card">
    <div class="dim">Вопрос:</div>
    <div>{{ store.chat.question }}</div>
  </div>

  <div v-if="store.chat.streaming" class="card">
    <div class="dim">Ответ (генерация):</div>
    <div class="mono">{{ store.chat.streaming }}</div>
  </div>

  <div v-if="store.chat.done" class="card">
    <div class="dim">Ответ:</div>
    <div>{{ store.chat.done }}</div>
  </div>
</template>
