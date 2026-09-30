<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { store } from "../store";
import PageIcon from "../components/PageIcon.vue";

interface Sample {
  path: string;
  duration: number;
}

const train = ref<Sample[]>([]);
const test = ref<Sample[]>([]);
const building = ref(false);
const testFile = ref("");
const testScore = ref<number | null>(null);
const notice = ref("");

const phraseSamples = () => train.value.filter((s) => !s.path.includes("[none]"));

async function refresh() {
  const s = await invoke<{ train: [string, number][]; test: [string, number][] }>("ww_samples");
  train.value = s.train.map(([path, duration]) => ({ path, duration }));
  test.value = s.test.map(([path, duration]) => ({ path, duration }));
}

async function record(target: string, kind: string) {
  store.recording = true;
  notice.value = "";
  try {
    await invoke("ww_record", { target, kind });
  } catch (e) {
    alert(String(e));
    store.recording = false;
  }
}

async function remove(path: string) {
  await invoke("ww_delete", { path });
  await refresh();
}

async function buildModel() {
  building.value = true;
  notice.value = "";
  try {
    await invoke("ww_build");
  } catch (e) {
    alert(String(e));
    building.value = false;
  }
}

async function applyModel() {
  try {
    await invoke("ww_apply", { path: null });
    notice.value = "Готово — модель применяется при следующем переходе в режим wake word.";
  } catch (e) {
    alert(String(e));
  }
}

async function runTest() {
  if (!testFile.value) return;
  try {
    testScore.value = await invoke<number>("ww_test", { path: testFile.value });
  } catch (e) {
    alert(String(e));
  }
}

let unlisten: UnlistenFn | null = null;

onMounted(async () => {
  await refresh();
  unlisten = await listen("engine-event", (e: { payload: { type: string } }) => {
    if (e.payload.type === "record_saved") {
      store.recording = false;
      refresh();
    } else if (e.payload.type === "train_done") {
      building.value = false;
      notice.value = "Модель построена. Нажмите «Применить модель».";
    } else if (e.payload.type === "error") {
      building.value = false;
    }
  });
});

onUnmounted(() => {
  // Без отписки каждый вход на страницу вешал бы ещё один обработчик.
  unlisten?.();
  unlisten = null;
});
</script>

<template>
  <h1><PageIcon name="mic" /> Wake Word</h1>
  <p class="page-hint">
    Шаг 1: запишите 3–8 сэмплов фразы «Джарвис». Шаг 2: постройте модель (секунды).
    Шаг 3: проверьте и примените.
  </p>

  <div class="card">
    <h2>Шаг 1 — сэмплы фразы</h2>
    <p class="dim" style="margin-bottom: 8px">
      Говорите «Джарвис» естественно, с разной интонацией. Запись остановится сама после паузы.
    </p>
    <div class="row">
      <button class="primary" :disabled="store.recording" @click="record('train', 'wake')">
        🎙 Записать «Джарвис»
      </button>
      <span v-if="store.recording" class="badge warn">идёт запись…</span>
      <span class="badge">записано: {{ phraseSamples().length }}/8</span>
    </div>
    <table style="margin-top: 10px" v-if="phraseSamples().length">
      <tr v-for="s in phraseSamples()" :key="s.path">
        <td class="mono">{{ s.path.split("/").pop() }}</td>
        <td class="dim">{{ s.duration.toFixed(1) }} с</td>
        <td><button class="danger" @click="remove(s.path)">удалить</button></td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h2>Шаг 2 — построить модель (DTW)</h2>
    <p class="dim" style="margin-bottom: 8px">
      Без нейросети: сравнение голосовых отпечатков записей. Строится мгновенно.
    </p>
    <button
      class="primary"
      :disabled="building || store.recording || phraseSamples().length < 3"
      @click="buildModel"
    >
      {{ building ? "Строю…" : "🧩 Построить модель" }}
    </button>
  </div>

  <div class="card">
    <h2>Шаг 3 — проверить и применить</h2>
    <p class="dim" style="margin-bottom: 8px">
      Запишите проверочный сэмпл и убедитесь, что модель узнаёт вас (score выше ~0.5).
    </p>
    <div class="row">
      <button class="ghost" :disabled="store.recording" @click="record('test', 'wake')">
        🎙 Проверочная запись
      </button>
      <button class="ghost" :disabled="!testFile" @click="runTest">Проверить</button>
      <span v-if="testScore !== null" class="badge" :class="testScore > 0.5 ? 'ok' : 'warn'">
        score: {{ testScore.toFixed(3) }}
      </span>
    </div>

    <table style="margin-top: 10px" v-if="test.length">
      <tr v-for="s in test" :key="s.path">
        <td class="mono">{{ s.path.split("/").pop() }}</td>
        <td class="dim">{{ s.duration.toFixed(1) }} с</td>
        <td><button class="ghost" @click="testFile = s.path">выбрать</button></td>
        <td><button class="danger" @click="remove(s.path)">удалить</button></td>
      </tr>
    </table>

    <div class="row" style="margin-top: 14px">
      <button class="primary" @click="applyModel">✓ Применить модель</button>
    </div>
    <p v-if="notice" class="dim" style="margin-top: 8px">{{ notice }}</p>
  </div>
</template>
