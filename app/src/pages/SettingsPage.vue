<script setup lang="ts">
import { onMounted, reactive, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import PageIcon from "../components/PageIcon.vue";

interface Settings {
  silence_rms_threshold: number;
  silence_timeout_secs: number;
  min_command_duration_secs: number;
  max_command_duration_secs: number;
  followup_window_secs: number;
  wake_threshold: number;
  wakeword_model: string;
  llm_enabled: boolean;
  web_search_auto: boolean;
  llm_normalize: boolean;
  llm_temperature: number;
  llm_max_tokens: number;
  tts_engine: string;
  tts_voice: string;
  tts_speed: number;
  ack_before_llm: boolean;
}

const settings = reactive<Settings>({
  silence_rms_threshold: 0.018,
  silence_timeout_secs: 0.7,
  min_command_duration_secs: 0.5,
  max_command_duration_secs: 8,
  followup_window_secs: 8,
  wake_threshold: 0.6,
  wakeword_model: "jarvis_model_2.rpw",
  llm_enabled: true,
  web_search_auto: true,
  llm_normalize: true,
  llm_temperature: 0.6,
  llm_max_tokens: 256,
  tts_engine: "piper",
  tts_voice: "ruslan",
  tts_speed: 1.0,
  ack_before_llm: false,
});

// Голоса приходят из бэкенда для активного движка, чтобы списки не расходились.
const voices = ref<[string, string][]>([]);
const saved = ref(false);

function voiceLabel(id: string, desc: string): string {
  const title = id.charAt(0).toUpperCase() + id.slice(1).replace(/_/g, " ");
  return `${title} — ${desc}`;
}

async function loadVoices() {
  voices.value = await invoke<[string, string][]>("tts_voices", { engine: settings.tts_engine });
}

onMounted(async () => {
  Object.assign(settings, await invoke<Settings>("get_settings"));
  await loadVoices();
});

async function save() {
  await invoke("save_settings", { settings });
  saved.value = true;
  setTimeout(() => (saved.value = false), 1500);
}
</script>

<template>
  <h1><PageIcon name="gear" /> Настройки</h1>
  <p class="page-hint">Пороги распознавания и параметры голоса. Применяются сразу после сохранения.</p>

  <template v-if="settings">
    <div class="card">
      <h2>Детекция фразы</h2>
      <div class="row">
        <div style="flex: 1; min-width: 180px">
          <label>Порог тишины (RMS)</label>
          <input type="number" step="0.001" v-model.number="settings.silence_rms_threshold" />
        </div>
        <div style="flex: 1; min-width: 180px">
          <label>Тишина завершает фразу, сек</label>
          <input type="number" step="0.1" v-model.number="settings.silence_timeout_secs" />
        </div>
      </div>
      <div class="row">
        <div style="flex: 1; min-width: 180px">
          <label>Мин. длительность фразы, сек</label>
          <input type="number" step="0.1" v-model.number="settings.min_command_duration_secs" />
        </div>
        <div style="flex: 1; min-width: 180px">
          <label>Макс. длительность фразы, сек</label>
          <input type="number" step="0.5" v-model.number="settings.max_command_duration_secs" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Wake word и диалог</h2>
      <div class="row">
        <div style="flex: 1; min-width: 180px">
          <label>Порог wake word</label>
          <input type="number" step="0.05" min="0.1" max="1" v-model.number="settings.wake_threshold" />
        </div>
        <div style="flex: 1; min-width: 180px">
          <label>Окно прямого слушания, сек</label>
          <input type="number" step="1" v-model.number="settings.followup_window_secs" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Локальный LLM (Qwen3-1.7B)</h2>
      <div class="row" style="flex-direction: column; align-items: flex-start; gap: 6px">
        <label style="display: flex; align-items: center; gap: 8px; margin: 0">
          <input type="checkbox" v-model="settings.llm_enabled" style="width: auto" />
          Отвечать на любые вопросы локальной моделью
        </label>
        <label style="display: flex; align-items: center; gap: 8px; margin: 0">
          <input type="checkbox" v-model="settings.web_search_auto" style="width: auto" />
          Искать в интернете свежую информацию (новости, погода, курсы)
        </label>
        <label style="display: flex; align-items: center; gap: 8px; margin: 0">
          <input type="checkbox" v-model="settings.llm_normalize" style="width: auto" />
          Нейро-нормализация озвучки: числа, единицы и сокращения через ИИ
        </label>
        <label style="display: flex; align-items: center; gap: 8px; margin: 0">
          <input type="checkbox" v-model="settings.ack_before_llm" style="width: auto" />
          Звук-подтверждение перед ответом ИИ (выкл. = быстрее первый звук)
        </label>
      </div>
      <div class="row">
        <div style="flex: 1; min-width: 180px">
          <label>Температура</label>
          <input type="number" step="0.1" min="0" max="2" v-model.number="settings.llm_temperature" />
        </div>
        <div style="flex: 1; min-width: 180px">
          <label>Макс. токенов</label>
          <input type="number" step="32" min="64" v-model.number="settings.llm_max_tokens" />
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Голос TTS</h2>
      <label>Движок</label>
      <select v-model="settings.tts_engine" @change="loadVoices">
        <option value="kokoro">Kokoro — быстрый, качественный, EN+RU (CPU)</option>
        <option value="piper">Piper — очень быстрый (CPU)</option>
        <option value="qwen3">Qwen3-TTS — естественная интонация (Metal)</option>
      </select>
      <label>Голос</label>
      <select v-model="settings.tts_voice">
        <option v-for="[id, desc] in voices" :key="id" :value="id">{{ voiceLabel(id, desc) }}</option>
      </select>
      <label>Скорость речи ({{ settings.tts_speed.toFixed(2) }}× — меньше = спокойнее)</label>
      <input type="range" min="0.7" max="1.2" step="0.05" v-model.number="settings.tts_speed" style="width: 100%" />
      <p class="dim" style="margin-top: 8px">
        Движок и голос применяются при сохранении (модель скачивается при первом запуске).
      </p>
    </div>

    <div class="row">
      <button class="primary" @click="save">💾 Сохранить</button>
      <span v-if="saved" class="badge ok">сохранено и применено</span>
    </div>
  </template>
</template>
