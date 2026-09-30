<script setup lang="ts">
import { onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import PageIcon from "../components/PageIcon.vue";

interface Command {
  name: string;
  trigger_words: string[];
  action_type: string;
  action_value: string;
  action_fallback?: string | null;
}

interface Config {
  global: {
    reply_success: string;
    reply_error: string;
    reply_unknown: string;
    reply_ack?: string;
  };
  commands: Command[];
}

const config = ref<Config | null>(null);
const selected = ref<number | null>(null);

const actionTypes = ["url", "search", "app", "stop", "exit", "command"];

async function refresh() {
  config.value = await invoke<Config>("get_commands");
}

function add() {
  if (!config.value) return;
  config.value.commands.push({
    name: "Новая команда",
    trigger_words: ["триггер"],
    action_type: "url",
    action_value: "https://",
  });
  selected.value = config.value.commands.length - 1;
}

function remove(idx: number) {
  if (!config.value) return;
  config.value.commands.splice(idx, 1);
  selected.value = null;
}

function move(idx: number, delta: number) {
  if (!config.value) return;
  const cmds = config.value.commands;
  const to = idx + delta;
  if (to < 0 || to >= cmds.length) return;
  [cmds[idx], cmds[to]] = [cmds[to], cmds[idx]];
  selected.value = to;
}

async function save() {
  if (!config.value) return;
  try {
    await invoke("save_commands", { config: config.value });
    alert("Сохранено — команды перезагружены на лету");
  } catch (e) {
    alert(String(e));
  }
}

onMounted(refresh);
</script>

<template>
  <h1><PageIcon name="commands" /> Команды</h1>
  <p class="page-hint">
    Что делает ассистент по вашим словам. Порядок в списке = приоритет:
    первые команды выигрывают при совпадении триггеров.
  </p>

  <p v-if="!config" class="dim">Загрузка команд…</p>
  <template v-else>
    <div class="row" style="margin-bottom: 12px">
      <button class="primary" @click="save">💾 Сохранить</button>
      <button class="ghost" @click="add">+ Добавить</button>
    </div>

    <table>
      <thead>
        <tr>
          <th style="width: 30px"></th>
          <th>Имя</th>
          <th>Триггеры</th>
          <th>Тип</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(cmd, idx) in config.commands"
          :key="idx"
          :style="selected === idx ? 'background: var(--bg-hover)' : ''"
          @click="selected = idx"
        >
          <td class="dim">{{ idx + 1 }}</td>
          <td>{{ cmd.name }}</td>
          <td class="dim mono">{{ cmd.trigger_words.join(", ") }}</td>
          <td><span class="badge">{{ cmd.action_type }}</span></td>
        </tr>
      </tbody>
    </table>

    <div v-if="selected !== null && config.commands[selected]" class="card" style="margin-top: 16px">
      <h2>Редактирование</h2>
      <label>Имя</label>
      <input type="text" v-model="config.commands[selected].name" />
      <label>Триггеры (через запятую)</label>
      <textarea
        :value="config.commands[selected].trigger_words.join(', ')"
        @input="
          config!.commands[selected!].trigger_words = ($event.target as HTMLTextAreaElement).value
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        "
      />
      <div class="row">
        <div style="flex: 1; min-width: 140px">
          <label>Тип действия</label>
          <select v-model="config.commands[selected].action_type">
            <option v-for="t in actionTypes" :key="t" :value="t">{{ t }}</option>
          </select>
        </div>
        <div style="flex: 2; min-width: 240px">
          <label>Значение (URL / имя приложения / команда)</label>
          <input type="text" v-model="config.commands[selected].action_value" />
        </div>
      </div>
      <div class="row" style="margin-top: 12px">
        <button class="danger" @click="remove(selected)">Удалить команду</button>
        <button class="ghost" @click="move(selected, -1)">↑ выше</button>
        <button class="ghost" @click="move(selected, 1)">↓ ниже</button>
      </div>
    </div>
  </template>
</template>
