<script setup lang="ts">
import { onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import PageIcon from "../components/PageIcon.vue";

const voices = ref<[string, string][]>([]);
const testText = ref("Привет! Я Джарвис, твой голосовой ассистент. Говорю спокойно и понятно.");

onMounted(async () => {
  voices.value = await invoke<[string, string][]>("tts_voices");
});
</script>

<template>
  <h1><PageIcon name="speaker" /> Голос</h1>
  <p class="page-hint">Проверьте, как звучит синтез речи, и посмотрите доступные голоса.</p>

  <div class="card">
    <p class="dim">
      Синтез речи — Qwen3-TTS на чистом Rust (candle + Metal), естественная
      русская интонация. Голос выбирается на странице «Настройки» и применяется
      сразу.
    </p>
    <h2>Доступные голоса</h2>
    <div class="row">
      <span v-for="v in voices" :key="v[0]" class="badge">{{ v[0] }} — {{ v[1] }}</span>
      <span v-if="!voices.length" class="dim">Список недоступен</span>
    </div>
  </div>

  <div class="card">
    <h2>Проверка озвучки</h2>
    <textarea v-model="testText" />
    <div class="row" style="margin-top: 10px">
      <button class="primary" @click="invoke('tts_test', { text: testText })">
        🔊 Озвучить
      </button>
      <span class="dim">звук пойдёт в очередь плеера (до 120 символов кэшируется)</span>
    </div>
  </div>
</template>
