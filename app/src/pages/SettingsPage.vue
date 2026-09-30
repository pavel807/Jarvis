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
  llm_enabled: boolean;
  web_search_auto: boolean;
  llm_normalize: boolean;
  llm_temperature: number;
  llm_max_tokens: number;
  tts_voice: string;
  tts_speed: number;
}

const settings = reactive<Settings>({
  silence_rms_threshold: 0.018,
  silence_timeout_secs: 1.3,
  min_command_duration_secs: 0.5,
  max_command_duration_secs: 8,
  followup_window_secs: 8,
  wake_threshold: 0.6,
  llm_enabled: true,
  web_search_auto: true,
  llm_normalize: true,
  llm_temperature: 0.6,
  llm_max_tokens: 256,
  tts_voice: "ryan",
  tts_speed: 1.0,
});

const saved = ref(false);

onMounted(async () => {
  Object.assign(settings, await invoke<Settings>("get_settings"));
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
      <h2>Голос TTS (Qwen3-TTS)</h2>
      <label>Голос</label>
      <select v-model="settings.tts_voice">
        <option value="ryan">Ryan — мужской, тёплый</option>
        <option value="aiden">Aiden — мужской, средний</option>
        <option value="eric">Eric — мужской, глубокий</option>
        <option value="dylan">Dylan — мужской, молодой</option>
        <option value="uncle_fu">Uncle Fu — мужской, низкий</option>
        <option value="serena">Serena — женский, мягкий</option>
        <option value="vivian">Vivian — женский, тёплый</option>
        <option value="ono_anna">Ono Anna — женский, высокий</option>
        <option value="sohee">Sohee — женский, средний</option>
      </select>
      <label>Скорость речи ({{ settings.tts_speed.toFixed(2) }}× — меньше = спокойнее)</label>
      <input type="range" min="0.7" max="1.2" step="0.05" v-model.number="settings.tts_speed" style="width: 100%" />
      <p class="dim" style="margin-top: 8px">
        Голос и скорость применяются сразу при сохранении.
      </p>
    </div>

    <div class="row">
      <button class="primary" @click="save">💾 Сохранить</button>
      <span v-if="saved" class="badge ok">сохранено и применено</span>
    </div>
  </template>
</template>
