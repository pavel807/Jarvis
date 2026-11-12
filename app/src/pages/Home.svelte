<script lang="ts">
  import { store, settings, clearHistory, go } from "../store.svelte";
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
          <span class="transmission__tag">IN</span>
          <span class="transmission__text">{lastUser.text}</span>
        </div>
      {/if}
      {#if lastReply}
        <div
          class="transmission__row is-reply"
          class:is-action={lastReply.source === "action"}
          class:is-error={lastReply.error}
        >
          <span class="transmission__tag">OUT</span>
          <span class="transmission__text">
            {lastReply.text}
            {#if lastReply.streaming}<span class="caret"></span>{/if}
          </span>
        </div>
      {:else if busy}
        <div class="transmission__row is-reply">
          <span class="transmission__tag">OUT</span>
          <span class="transmission__text">
            <span class="thinking-row"><span class="spinner"></span>Обрабатываю запрос…</span>
          </span>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Голосовой пульт: статус, кнопка с уровнем звука, спектр -->
  <div class="voice-zone">
    <span class="voice-status" class:is-live={assistant.tone === "live"} class:is-busy={busy} class:is-off={store.status.paused} class:is-err={assistant.tone === "error"}>
      {statusLabel}
    </span>
    <MicControl />
    <Waveform />
    {#if store.history.length}
      <div class="voice-links">
        <button class="link-btn" onclick={() => go("memory")}>
          Полная история — в разделе «Память»
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
