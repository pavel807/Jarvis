<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { store, toast } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";
  import Markdown from "../components/Markdown.svelte";

  /**
   * Память: история диалога этой сессии и задачник (tasks.md), который
   * Джарвис и пользователь ведут вместе. Всё содержимое — реальное.
   */

  interface TaskDto {
    id: number;
    text: string;
    done: boolean;
  }

  let tasks = $state<TaskDto[] | null>(null);
  let newTask = $state("");
  let adding = $state(false);

  const openTasks = $derived((tasks ?? []).filter((t) => !t.done));
  const doneTasks = $derived((tasks ?? []).filter((t) => t.done));

  async function refresh() {
    try {
      tasks = await invoke<TaskDto[]>("tasks_list");
    } catch {
      tasks = [];
    }
  }

  onMount(() => {
    void refresh();
  });

  async function addTask() {
    const text = newTask.trim();
    if (!text || adding) return;
    adding = true;
    try {
      const created = await invoke<TaskDto>("tasks_add", { text });
      tasks = [...(tasks ?? []), created];
      newTask = "";
      toast("Задача добавлена в задачник", "ok", 1600);
    } catch (e) {
      toast(`Не получилось: ${e}`, "err");
    } finally {
      adding = false;
    }
  }

  async function toggleTask(task: TaskDto) {
    if (!tasks) return;
    const done = !task.done;
    try {
      const updated = await invoke<TaskDto | null>("tasks_set_done", { id: task.id, done });
      if (updated) {
        tasks = tasks.map((t) => (t.id === task.id ? updated : t));
      }
    } catch (e) {
      toast(`Не получилось: ${e}`, "err");
    }
  }

  async function removeTask(task: TaskDto) {
    if (!tasks) return;
    try {
      await invoke("tasks_remove", { id: task.id });
      tasks = tasks.filter((t) => t.id !== task.id);
      // Номера сквозные — после удаления перечитываем, чтобы не врать.
      await refresh();
    } catch (e) {
      toast(`Не получилось: ${e}`, "err");
    }
  }

  async function openTasksFile() {
    try {
      await invoke("tasks_open");
    } catch (e) {
      toast(`Не получилось: ${e}`, "err");
    }
  }

  const metaFor = (turn: (typeof store.history)[number]) => {
    if (turn.role === "user") return { cls: "is-user", label: turn.source === "text" ? "ввод" : "голос" };
    if (turn.role === "note") return { cls: "is-action", label: "система" };
    if (turn.source === "action") return { cls: "is-action", label: "действие" };
    return { cls: "is-ai", label: "J.A.R.V.I.S" };
  };
</script>

<div class="page">
  <!-- История диалога -->
  <div class="section-title">История диалога сессии</div>
  {#if !store.history.length}
    <div class="card">
      <div class="empty">
        <AppIcon name="memory" size={22} />
        Диалога ещё не было.<br />
        Скажите «Джарвис» или напишите на главной — история появится здесь.
      </div>
    </div>
  {:else}
    <div class="card card--flush history-card">
      {#each store.history as turn (turn.id)}
        <div
          class="log-line log-line--{turn.role}"
          class:log-line--error={turn.error}
        >
          <span class="log-line__meta {metaFor(turn).cls}">
            {metaFor(turn).label}
          </span>
          {#if turn.role === "assistant" && !turn.error}
            <div class="log-line__text">
              <Markdown text={turn.text} />
              {#if turn.streaming}<span class="caret"></span>{/if}
            </div>
          {:else}
            <div class="log-line__text">{turn.text}</div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}

  <!-- Задачник -->
  <div class="section-title">Задачник</div>
  <div class="card">
    <div class="task-add">
      <input
        bind:value={newTask}
        class="input"
        type="text"
        placeholder="Новая задача…"
        aria-label="Текст новой задачи"
        onkeydown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            void addTask();
          }
        }}
      />
      <button
        class="btn btn--primary btn--sm"
        disabled={!newTask.trim() || adding}
        onclick={addTask}
      >
        <AppIcon name="plus" size={14} />
        Добавить
      </button>
    </div>

    {#if tasks === null}
      <div class="empty"><span class="spinner"></span>Читаю задачник…</div>
    {:else}
      {#if openTasks.length}
        <div class="task-list">
          {#each openTasks as task (task.id)}
            <div class="task-row">
              <span class="task-row__num">{task.id}</span>
              <button
                class="task-row__check"
                title="Отметить выполненной"
                aria-label={`Выполнить: ${task.text}`}
                onclick={() => toggleTask(task)}
              ></button>
              <span class="task-row__text">{task.text}</span>
              <button
                class="icon-btn icon-btn--danger"
                title="Удалить"
                aria-label={`Удалить: ${task.text}`}
                onclick={() => removeTask(task)}
              >
                <AppIcon name="trash" size={14} />
              </button>
            </div>
          {/each}
        </div>
      {:else}
        <p class="hint" style="margin: 10px 0 2px">
          Открытых задач нет. Джарвис тоже может добавлять — скажите «добавь задачу …».
        </p>
      {/if}

      {#if doneTasks.length}
        <div class="section-title" style="margin-top: 14px">Выполнено · {doneTasks.length}</div>
        <div class="task-list">
          {#each doneTasks as task (task.id)}
            <div class="task-row is-done">
              <span class="task-row__num">{task.id}</span>
              <button
                class="task-row__check is-done"
                title="Вернуть в работу"
                aria-label={`Вернуть в работу: ${task.text}`}
                onclick={() => toggleTask(task)}
              >
                <AppIcon name="check" size={11} />
              </button>
              <span class="task-row__text">{task.text}</span>
              <button
                class="icon-btn icon-btn--danger"
                title="Удалить"
                aria-label={`Удалить: ${task.text}`}
                onclick={() => removeTask(task)}
              >
                <AppIcon name="trash" size={14} />
              </button>
            </div>
          {/each}
        </div>
      {/if}
    {/if}

    <button class="btn btn--ghost btn--sm" style="margin-top: 12px" onclick={openTasksFile}>
      <AppIcon name="book" size={14} />
      Открыть tasks.md в редакторе
    </button>
  </div>
</div>

<style>
  .history-card {
    max-height: 380px;
    overflow-y: auto;
  }

  .task-add {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
  }

  .task-add .input {
    flex: 1;
    min-width: 0;
  }
</style>
