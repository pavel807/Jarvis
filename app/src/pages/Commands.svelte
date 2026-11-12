<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { toast, go, store } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";
  import Modal from "../components/Modal.svelte";

  interface Command {
    name: string;
    trigger_words: string[];
    action_type: string;
    action_value: string;
    action_fallback?: string | null;
    enabled: boolean;
  }

  interface DraftCommand extends Omit<Command, "action_fallback"> {
    action_fallback: string;
  }

  interface Config {
    /** Глобальные ответы и служебные поля — не показываем, но сохраняем как есть. */
    global: Record<string, unknown>;
    commands: Command[];
  }

  /**
   * Типы действий в человеческих словах. Порядок в select — от частого к редкому.
   */
  const ACTION_META: Record<
    string,
    { label: string; icon: string; hint: string; valueLabel?: string; placeholder?: string }
  > = {
    app: {
      label: "Запустить приложение",
      icon: "grid",
      valueLabel: "Название программы",
      placeholder: "Safari",
      hint: "Название как в папке «Программы» — например Safari или Microsoft Excel.",
    },
    url: {
      label: "Открыть сайт",
      icon: "link",
      valueLabel: "Адрес сайта",
      placeholder: "https://example.com",
      hint: "Страница откроется в браузере по умолчанию.",
    },
    search: {
      label: "Найти в браузере",
      icon: "search",
      valueLabel: "Адрес поиска",
      placeholder: "https://www.google.com/search?q={}",
      hint: "Вместо {} подставится то, что вы сказали после команды.",
    },
    web: {
      label: "Найти в интернете и рассказать",
      icon: "globe",
      hint: "Джарвис поищет в интернете и коротко ответит голосом. Ничего указывать не нужно.",
    },
    path: {
      label: "Открыть файл или папку",
      icon: "folder",
      valueLabel: "Путь",
      placeholder: "tasks.md",
      hint: "Относительный путь считается от папки с данными Джарвиса.",
    },
    stop: {
      label: "Замолчать",
      icon: "stop",
      hint: "Прерывает озвучку и подготовку ответа.",
    },
    exit: {
      label: "Закрыть Джарвис",
      icon: "power",
      hint: "Полностью завершает приложение.",
    },
    command: {
      label: "Выполнить в терминале",
      icon: "terminal",
      valueLabel: "Команда",
      placeholder: "say привет",
      hint: "Для продвинутых: строка выполнится в системной оболочке.",
    },
  };

  const ACTION_ORDER = ["app", "url", "search", "web", "path", "stop", "exit", "command"];

  type SortMode = "priority" | "name";

  function blankDraft(): DraftCommand {
    return {
      name: "",
      trigger_words: [],
      action_type: "app",
      action_value: "",
      action_fallback: "",
      enabled: true,
    };
  }

  let config = $state<Config | null>(null);
  let query = $state("");
  let filterType = $state("all");
  let sortMode = $state<SortMode>("priority");
  let draft = $state<DraftCommand>(blankDraft());
  let draftOpen = $state(false);
  let draftIndex = $state(-1);
  let triggerInput = $state("");
  let saving = $state(false);
  let confirmRemove = $state(false);

  // ---------- Drag-сортировка: порядок в списке = приоритет распознавания ----------
  let dragIndex = $state(-1);
  let overIndex = $state(-1);

  const meta = (type: string) => ACTION_META[type] ?? ACTION_META.app;

  /** Короткое описание действия для строки списка. */
  function actionSummary(cmd: Command): string {
    const value = cmd.action_value?.trim();
    switch (cmd.action_type) {
      case "url": {
        try {
          return `Откроет ${new URL(value).host.replace(/^www\./, "")}`;
        } catch {
          return value ? `Откроет ${value}` : "Откроет сайт";
        }
      }
      case "search":
        return "Найдёт в браузере то, что вы скажете";
      case "web":
        return "Найдёт в интернете и расскажет";
      case "app":
        return value ? `Запустит «${value}»` : "Запустит приложение";
      case "path":
        return value ? `Откроет ${value}` : "Откроет файл";
      case "command":
        return value ? `Выполнит: ${value}` : "Выполнит системную команду";
      case "stop":
        return "Прервёт ответ";
      case "exit":
        return "Закроет Джарвис";
      default:
        return `Неизвестное действие: ${cmd.action_type}`;
    }
  }

  /** Отфильтрованный список с исходными индексами — по ним правим и сортируем. */
  const filtered = $derived.by(() => {
    if (!config) return [];
    const q = query.trim().toLowerCase();
    const indexed = config.commands
      .map((cmd, index) => ({ cmd, index }))
      .filter(({ cmd }) => {
        if (filterType !== "all" && cmd.action_type !== filterType) return false;
        if (!q) return true;
        return (
          cmd.name.toLowerCase().includes(q) ||
          cmd.trigger_words.some((t) => t.toLowerCase().includes(q)) ||
          (cmd.action_value ?? "").toLowerCase().includes(q)
        );
      });
    if (sortMode === "name") {
      indexed.sort((a, b) => a.cmd.name.localeCompare(b.cmd.name, "ru"));
    }
    return indexed;
  });

  const triggerSummary = (cmd: Command) =>
    cmd.trigger_words.slice(0, 3).join(" · ") + (cmd.trigger_words.length > 3 ? " · …" : "");

  /**
   * Команды по разделам — по типу действия. Порядок внутри раздела — общий
   * приоритет (или алфавит, если выбрано), разделы — в порядке ACTION_ORDER.
   */
  const grouped = $derived.by(() => {
    const groups: { type: string; rows: { cmd: Command; index: number }[] }[] = [];
    for (const row of filtered) {
      const g = groups.find((x) => x.type === row.cmd.action_type);
      if (g) g.rows.push(row);
      else groups.push({ type: row.cmd.action_type, rows: [row] });
    }
    return groups.sort(
      (a, b) =>
        ACTION_ORDER.indexOf(a.type) - ACTION_ORDER.indexOf(b.type) ||
        a.type.localeCompare(b.type),
    );
  });

  async function refresh() {
    config = await invoke<Config>("get_commands");
  }

  onMount(() => {
    void refresh();
  });

  function openNew() {
    draft = blankDraft();
    draftIndex = -1;
    triggerInput = "";
    confirmRemove = false;
    draftOpen = true;
  }

  function openEdit(cmd: Command) {
    if (!config) return;
    const index = config.commands.indexOf(cmd);
    if (index === -1) return;
    draft = { ...$state.snapshot(cmd), action_fallback: cmd.action_fallback ?? "" };
    draftIndex = index;
    triggerInput = "";
    confirmRemove = false;
    draftOpen = true;
  }

  function closeDraft() {
    draftOpen = false;
    draftIndex = -1;
    confirmRemove = false;
  }

  function addTrigger() {
    const word = triggerInput.trim().toLowerCase();
    if (!word) return;
    if (!draft.trigger_words.includes(word)) draft.trigger_words.push(word);
    triggerInput = "";
  }

  function removeTrigger(index: number) {
    draft.trigger_words.splice(index, 1);
  }

  async function persist(silent = false) {
    if (!config) return;
    saving = true;
    try {
      await invoke("save_commands", { config: $state.snapshot(config) });
      if (!silent) toast("Команды сохранены и уже действуют", "ok");
    } catch (e) {
      toast(`Не удалось сохранить: ${e}`, "err", 5000);
      await refresh();
    } finally {
      saving = false;
    }
  }

  async function commitDraft() {
    if (!draftOpen || !config) return;
    const cmd = draft;

    if (!cmd.name.trim()) return toast("Дайте команде название", "err");
    if (!cmd.trigger_words.length) return toast("Добавьте хотя бы одно слово-триггер", "err");
    if (meta(cmd.action_type).placeholder !== undefined && !cmd.action_value.trim()) {
      return toast("Заполните поле действия", "err");
    }

    const fallback = cmd.action_fallback.trim();
    const payload: Command = {
      ...cmd,
      name: cmd.name.trim(),
      action_fallback: fallback ? fallback : null,
    };
    const created = draftIndex === -1;
    if (created) config.commands.push(payload);
    else config.commands[draftIndex] = payload;

    closeDraft();
    await persist();
    if (created) toast(`Команда «${cmd.name.trim()}» добавлена`, "ok");
  }

  async function toggleEnabled(cmd: Command, value: boolean) {
    cmd.enabled = value;
    await persist(true);
    toast(value ? `«${cmd.name}» включена` : `«${cmd.name}» выключена`, "info", 1600);
  }

  async function removeCommand(index: number) {
    if (!config) return;
    const removed = config.commands[index];
    if (!removed) return;
    config.commands.splice(index, 1);
    closeDraft();
    await persist();
    toast(`Команда «${removed.name}» удалена`, "ok");
  }

  /** Перестановка перетаскиванием: drop на другую строку. */
  async function dropOn(target: number) {
    const from = dragIndex;
    dragIndex = -1;
    overIndex = -1;
    if (!config || from < 0 || from === target || sortMode !== "priority") return;
    const [moved] = config.commands.splice(from, 1);
    config.commands.splice(target, 0, moved);
    await persist(true);
  }
</script>

<div class="page">
  <div class="page-head">
    <button class="icon-btn back-btn" title="Назад (Esc)" aria-label="Назад" onclick={() => go(store.pageBack)}>
      <AppIcon name="chevronLeft" size={17} />
    </button>
    <div class="page-head__body">
      <h1 class="page-title">Команды</h1>
    </div>
    <button class="btn btn--primary" onclick={openNew}>
      <AppIcon name="plus" size={16} />
      Новая команда
    </button>
  </div>
  <p class="page-sub commands-sub">
    Джарвис выполняет команду, когда слышит одно из её слов. Чем выше строка,
    тем раньше команда срабатывает — порядок можно менять перетаскиванием.
  </p>

  <div class="toolbar">
    <div class="search-box">
      <span class="search-box__icon"><AppIcon name="search" size={15} /></span>
      <input
        bind:value={query}
        class="input"
        type="search"
        placeholder="Поиск по названию, словам, действию…"
        aria-label="Поиск по командам"
      />
    </div>
    <select bind:value={filterType} class="select" aria-label="Фильтр по действию">
      <option value="all">Все действия</option>
      {#each ACTION_ORDER as type (type)}
        <option value={type}>{meta(type).label}</option>
      {/each}
    </select>
    <select bind:value={sortMode} class="select" aria-label="Сортировка">
      <option value="priority">По приоритету</option>
      <option value="name">По алфавиту</option>
    </select>
    <div class="toolbar__spacer"></div>
    {#if config}
      <span class="badge" title="Активных команд из всех">
        {config.commands.filter((c) => c.enabled).length} из {config.commands.length}
      </span>
    {/if}
  </div>

  {#if !config}
    <div class="card card--outline"><div class="empty"><span class="spinner"></span>Загружаю команды…</div></div>
  {:else if filtered.length}
    {#each grouped as group (group.type)}
      <div class="section-title cmd-section">{meta(group.type).label}</div>
      <div class="cmd-group">
      {#each group.rows as row (row.cmd.name)}
        {@const cmd = row.cmd}
        {@const index = row.index}
        <div
          class="cmd-row"
          class:is-disabled={!cmd.enabled}
          class:is-dragging={dragIndex === index}
          class:is-over={overIndex === index && dragIndex !== index}
          draggable={sortMode === "priority"}
          role="button"
          tabindex="0"
          aria-label={`Команда ${cmd.name}: ${actionSummary(cmd)}. Открыть настройки`}
          onclick={() => openEdit(cmd)}
          onkeydown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              openEdit(cmd);
            }
          }}
          ondragstart={() => (dragIndex = index)}
          ondragend={() => {
            dragIndex = -1;
            overIndex = -1;
          }}
          ondragover={(event) => {
            event.preventDefault();
            overIndex = index;
          }}
          ondragleave={() => {
            if (overIndex === index) overIndex = -1;
          }}
          ondrop={(event) => {
            event.preventDefault();
            void dropOn(index);
          }}
        >
          {#if sortMode === "priority"}
            <span class="cmd-row__grip" title="Перетащите, чтобы изменить приоритет">
              <AppIcon name="grip" size={16} />
            </span>
          {/if}
          <span class="cmd-row__icon"><AppIcon name={meta(cmd.action_type).icon} size={15} /></span>
          <span class="cmd-row__body">
            <span class="cmd-row__name">{cmd.name}</span>
            <span class="cmd-row__sub">
              <span class="cmd-row__desc">{actionSummary(cmd)}</span>
              <span class="cmd-row__triggers" title={cmd.trigger_words.join(", ")}>
                {triggerSummary(cmd)}
              </span>
            </span>
          </span>
          {#if !cmd.enabled}
            <span class="badge">выключена</span>
          {/if}
          <span class="cmd-row__actions">
            <button
              class="switch"
              class:is-on={cmd.enabled}
              role="switch"
              aria-checked={cmd.enabled}
              aria-label={`Команда ${cmd.name} ${cmd.enabled ? "включена" : "выключена"}`}
              title={cmd.enabled ? "Выключить" : "Включить"}
              onclick={(event) => {
                event.stopPropagation();
                void toggleEnabled(cmd, !cmd.enabled);
              }}
            ></button>
            <button
              class="icon-btn"
              title="Настройки команды"
              aria-label={`Настройки команды ${cmd.name}`}
              onclick={(event) => {
                event.stopPropagation();
                openEdit(cmd);
              }}
            >
              <AppIcon name="pencil" size={15} />
            </button>
          </span>
        </div>
      {/each}
      </div>
    {/each}
  {:else if config.commands.length}
    <div class="card card--outline">
      <div class="empty">
        <AppIcon name="search" size={22} />
        По запросу «{query || meta(filterType).label}» ничего нет.<br />
        Очистите поиск или фильтр.
      </div>
    </div>
  {:else}
    <div class="card card--outline">
      <div class="empty">
        <AppIcon name="commands" size={24} />
        Команд пока нет.<br />
        Добавьте первую — например «открой ютуб».
      </div>
    </div>
  {/if}

  {#if sortMode === "name" && config?.commands.length}
    <p class="hint" style="margin-top: 10px">
      Порядок перетаскиванием работает только в сортировке «По приоритету» — она показывает,
      как команды проверяются голосом.
    </p>
  {/if}
</div>

{#if draftOpen}
  <Modal title={draftIndex === -1 ? "Новая команда" : "Настройка команды"} onclose={closeDraft}>
    {#snippet headActions()}
      {#if draftIndex !== -1}
        <button
          class="icon-btn icon-btn--danger"
          title={confirmRemove ? "Нажмите ещё раз для удаления" : "Удалить команду"}
          aria-label="Удалить команду"
          onclick={() => {
            if (!confirmRemove) {
              confirmRemove = true;
              return;
            }
            void removeCommand(draftIndex);
          }}
        >
          <AppIcon name="trash" size={16} />
        </button>
      {/if}
    {/snippet}

    <div class="field">
      <label class="label" for="cmd-name">Название</label>
      <input id="cmd-name" bind:value={draft.name} class="input" placeholder="Например: YouTube" />
    </div>

    <div class="field">
      <div class="label">Слова, на которые реагировать</div>
      {#if draft.trigger_words.length}
        <div class="chips" style="margin-bottom: 8px">
          {#each draft.trigger_words as word, i (word)}
            <span class="chip chip--accent">
              {word}
              <button class="chip__remove" title="Убрать" onclick={() => removeTrigger(i)}>×</button>
            </span>
          {/each}
        </div>
      {/if}
      <div style="display: flex; gap: 8px">
        <input
          bind:value={triggerInput}
          class="input"
          placeholder="ютуб"
          aria-label="Новое слово-триггер"
          onkeydown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              addTrigger();
            }
          }}
        />
        <button class="btn btn--secondary" disabled={!triggerInput.trim()} onclick={addTrigger}>
          <AppIcon name="plus" size={15} />
          Добавить
        </button>
      </div>
      <p class="hint">Произносите эти слова после «Джарвис». Чем больше вариантов, тем надёжнее.</p>
    </div>

    <div class="field">
      <label class="label" for="cmd-action">Что делать</label>
      <select id="cmd-action" bind:value={draft.action_type} class="select">
        {#each ACTION_ORDER as type (type)}
          <option value={type}>
            {ACTION_META[type].label}
          </option>
        {/each}
      </select>
      <p class="hint">{meta(draft.action_type).hint}</p>
    </div>

    {#if meta(draft.action_type).valueLabel}
      <div class="field">
        <label class="label" for="cmd-value">{meta(draft.action_type).valueLabel}</label>
        <input
          id="cmd-value"
          bind:value={draft.action_value}
          class="input"
          placeholder={meta(draft.action_type).placeholder}
        />
      </div>
    {/if}

    {#if draft.action_type === "app"}
      <div class="field">
        <label class="label" for="cmd-fallback">
          Полный путь, если программа не находится (необязательно)
        </label>
        <input
          id="cmd-fallback"
          bind:value={draft.action_fallback}
          class="input"
          placeholder="/Applications/Zed.app"
        />
      </div>
    {/if}

    {#snippet foot()}
      <button class="btn btn--ghost" onclick={closeDraft}>Отмена</button>
      <div class="spacer"></div>
      <button class="btn btn--primary" disabled={saving} onclick={commitDraft}>
        {draftIndex === -1 ? "Добавить команду" : "Сохранить"}
      </button>
    {/snippet}
  </Modal>
{/if}

<style>
  .cmd-group + .hint {
    margin-top: 10px;
  }

  .back-btn {
    flex-shrink: 0;
    align-self: flex-start;
  }

  /* Подзаголовок под шапкой на всю ширину: с кнопкой в строке заголовка
     он сжимался до узкой колонки и растягивался на восемь строк. */
  .commands-sub {
    margin: -6px 0 14px;
    padding-left: 44px;
  }

  /* Раздел по типу действия: между группами больше воздуха, чем между строками. */
  .cmd-section {
    margin-top: 18px;
  }

  .cmd-section:first-of-type {
    margin-top: 0;
  }
</style>
