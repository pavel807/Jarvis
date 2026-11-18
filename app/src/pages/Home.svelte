<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { store, settings, clearHistory, go, resolveChoice, activeModelLabel, pushTurn, toast } from "../store.svelte";
  import { getAssistant, voiceStatusLabel } from "../assistant";
  import ArcReactor from "../components/ArcReactor.svelte";
  import MicControl from "../components/MicControl.svelte";
  import Waveform from "../components/Waveform.svelte";
  import AppIcon from "../components/AppIcon.svelte";

  /**
   * Главная Mark V: реактор — центр композиции и единственный яркий элемент.
   * Состояние системы читается с реактора и полосы статуса, состояние
   * разговора — у микрофона; дублирующих телеметрических строк нет.
   * Полная история диалога живёт в разделе «Память», здесь — последний обмен.
   */

  const assistant = $derived(getAssistant());
  const statusLabel = $derived(voiceStatusLabel());
  /** Какая модель реально отвечает — по факту загрузки, не по настройкам. */
  const modelLabel = $derived(activeModelLabel(store.status));
  const modelReady = $derived(
    store.status.llmStatus === "ready" && Boolean(store.status.llmBackend),
  );

  /** Последние реплики для «передачи»: вопрос и ответ. */
  const lastUser = $derived.by(() => {
    for (let i = store.history.length - 1; i >= 0; i--) {
      const turn = store.history[i];
      if (turn.role === "user") return turn;
    }
    return null;
  });

  const lastReply = $derived.by(() => {
    for (let i = store.history.length - 1; i >= 0; i--) {
      const turn = store.history[i];
      if (turn.role === "assistant") return turn;
    }
    return null;
  });

  const busy = $derived(store.chat.busy);

  const hasExchange = $derived(Boolean(lastUser || lastReply));
  const wakeWord = $derived(settings.wake_word || "Джарвис");

  // ---------- Текстовый ввод ----------

  /** На тач-устройствах поле открыто сразу: набрать вопрос проще, чем говорить. */
  const isTouch = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  let textOpen = $state(isTouch);
  let draft = $state("");
  let textEl: HTMLTextAreaElement | undefined = $state();

  function toggleText() {
    textOpen = !textOpen;
    if (textOpen) setTimeout(() => textEl?.focus(), 50);
  }

  async function sendText() {
    const text = draft.trim();
    if (!text || busy) return;
    draft = "";
    pushTurn({ role: "user", text, source: "text" });
    try {
      await invoke("send_chat", { text });
    } catch (e) {
      toast(`Не удалось отправить: ${e}`, "err");
    }
  }

  /** Enter — отправить, Shift+Enter — перенос строки. */
  function onTextKey(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendText();
    }
  }
</script>

<div class="home">
  <div class="reactor-stage">
    <ArcReactor />
  </div>

  <!-- Последняя передача: что услышано и что отвечено -->
  <div class="transmission" class:is-empty={!hasExchange}>
    {#if !hasExchange}
      <span class="transmission__text dim">
        Скажите «{wakeWord}» и говорите команду.
      </span>
    {:else}
      {#if lastUser}
        <div class="transmission__row">
          <span class="transmission__text">{lastUser.text}</span>
        </div>
      {/if}
      {#if lastReply}
        <div
          class="transmission__row is-reply"
          class:is-action={lastReply.source === "action"}
          class:is-error={lastReply.error}
        >
          <span class="transmission__text">
            {lastReply.text}
            {#if lastReply.streaming}<span class="caret"></span>{/if}
          </span>
        </div>
      {:else if busy}
        <div class="transmission__row is-reply">
          <span class="transmission__text">
            <span class="thinking-row"><span class="spinner"></span>Обрабатываю запрос…</span>
          </span>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Несколько целей: выбор голосом или кликом по чипу -->
  {#if store.pendingChoice}
    <div class="choice">
      <span class="choice__prompt">{store.pendingChoice.prompt}</span>
      <div class="chips">
        {#each store.pendingChoice.options as option (option.index)}
          <button class="chip choice__chip" onclick={() => resolveChoice(option.index)}>
            {option.label}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Голосовой пульт: статус, кнопка с уровнем звука, спектр -->
  <div class="voice-zone">
    <span class="voice-status" class:is-live={assistant.tone === "live"} class:is-busy={busy} class:is-off={store.status.paused} class:is-err={assistant.tone === "error"}>
      {statusLabel}
    </span>
    <!-- Бейдж активной модели: видно подмену облака локальной моделью. -->
    <span
      class="model-badge"
      class:is-ok={modelReady}
      class:is-err={!modelReady}
      title="Модель, которая сейчас отвечает"
    >
      <span class="model-badge__dot"></span>
      {modelLabel}
    </span>
    <MicControl />
    <!-- Печать вместо голоса: на тач-устройствах открыто сразу. -->
    <button class="icon-btn text-toggle" title={textOpen ? "Скрыть ввод" : "Написать текстом"} aria-label={textOpen ? "Скрыть ввод" : "Написать текстом"} onclick={toggleText}>
      <AppIcon name="pencil" size={15} />
    </button>
    {#if textOpen}
      <div class="text-row">
        <textarea
          class="input text-row__field"
          rows="1"
          bind:this={textEl}
          bind:value={draft}
          onkeydown={onTextKey}
          placeholder="Спросите что угодно — как у Алисы"
          enterkeyhint="send"
        ></textarea>
        <button
          class="btn btn--primary text-row__send"
          aria-label="Отправить"
          disabled={!draft.trim() || busy}
          onclick={() => void sendText()}
        >
          <AppIcon name="send" size={16} />
        </button>
      </div>
    {/if}
    <Waveform />
    {#if store.history.length}
      <div class="voice-links">
        <button class="link-btn" onclick={() => go("memory")}>
          История
          <AppIcon name="chevronRight" size={11} />
        </button>
        <button
          class="icon-btn"
          title="Очистить журнал диалога"
          aria-label="Очистить журнал диалога"
          onclick={clearHistory}
        >
          <AppIcon name="eraser" size={15} />
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  .transmission.is-empty {
    justify-content: center;
  }

  .model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 100%;
    padding: 3px 10px;
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 11px;
    color: var(--text-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .model-badge__dot {
    flex: none;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.7;
  }

  .model-badge.is-ok {
    color: var(--cyan);
    border-color: var(--cyan-ring);
  }

  .model-badge.is-err {
    color: var(--danger);
    border-color: var(--danger-soft);
  }

  /* Текстовый ввод: растягивается по ширине пульта, кнопка — к нему. */
  .text-toggle {
    flex: none;
  }

  .text-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    width: 100%;
    max-width: 560px;
    margin-top: 2px;
  }

  .text-row__field {
    flex: 1;
    min-width: 0;
    resize: none;
    max-height: 120px;
    line-height: 1.4;
  }

  .text-row__send {
    flex: none;
    width: 42px;
    height: 42px;
    padding: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }

  .choice {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    max-width: 640px;
    margin: 0 auto;
    padding: 0 16px;
  }

  .choice__prompt {
    color: var(--text-2);
    font-size: 12.5px;
    line-height: 1.45;
    text-align: center;
  }

  .choice__chip {
    cursor: pointer;
    transition: color 0.14s, border-color 0.14s, background 0.14s;
  }

  .choice__chip:hover {
    background: var(--cyan-soft);
    border-color: var(--cyan-ring);
    color: var(--cyan);
  }

  .voice-links {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }

  .link-btn {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    align-self: flex-start;
    background: none;
    border: none;
    padding: 0;
    color: var(--text-3);
    font-size: 11px;
    cursor: pointer;
    transition: color 0.14s;
  }

  .link-btn:hover {
    color: var(--cyan);
  }

</style>
