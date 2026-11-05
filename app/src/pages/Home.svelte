<script lang="ts">
  import { store, clearHistory, go } from "../store.svelte";
  import { getAssistant, voiceStatusLabel } from "../assistant";
  import { settings } from "../store.svelte";
  import ArcReactor from "../components/ArcReactor.svelte";
  import MicControl from "../components/MicControl.svelte";
  import Waveform from "../components/Waveform.svelte";
  import CommandInput from "../components/CommandInput.svelte";
  import AppIcon from "../components/AppIcon.svelte";

  /**
   * Главная Mark V: реактор — центр композиции, вокруг — телеметрия,
   * последняя передача и голосовое управление. Полная история диалога
   * живёт в разделе «Память», здесь — только последний обмен.
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

  /** Телеметрия реактора: только реальные значения. */
  const coreMode = $derived(
    assistant.tone === "error"
      ? "FAULT"
      : store.status.paused
        ? "STANDBY"
        : assistant.tone === "live"
          ? "LISTEN"
          : busy
            ? "COMPUTE"
            : store.speaking
              ? "TRANSMIT"
              : "IDLE",
  );

  const modelLabel = $derived(
    settings.llm_backend === "uzu" ? "UZU · METAL" : "CANDLE",
  );

  const activityLabel = $derived.by(() => {
    if (!store.lastActivity) return "—";
    const minutes = Math.round((Date.now() - store.lastActivity) / 60000);
    if (minutes < 1) return "СЕЙЧАС";
    if (minutes < 60) return `${minutes} МИН`;
    return `${Math.round(minutes / 60)} Ч`;
  });

  const hasExchange = $derived(Boolean(lastUser || lastReply));
</script>

<div class="home">
  <div class="reactor-stage">
    <ArcReactor />
  </div>

  <div class="reactor-info">
    <div class="reactor-info__cell">
      <span class="tech-label">Core</span>
      <span class="reactor-info__value" class:is-energy={!store.status.paused}>{assistant.short}</span>
    </div>
    <div class="reactor-info__cell">
      <span class="tech-label">Mode</span>
      <span class="reactor-info__value">{coreMode}</span>
    </div>
    <div class="reactor-info__cell">
      <span class="tech-label">Activity</span>
      <span class="reactor-info__value">{activityLabel}</span>
    </div>
  </div>

  <!-- Последняя передача: что услышано и что отвечено -->
  <div class="transmission" class:is-empty={!hasExchange}>
    {#if !hasExchange}
      <span class="transmission__text dim">
        Скажите «Джарвис» и говорите команду — или наберите её ниже.
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

  <!-- Голосовое управление -->
  <div class="voice-zone">
    <MicControl />
    <div class="voice-center">
      <span class="voice-status" class:is-live={assistant.tone === "live"} class:is-busy={busy} class:is-off={store.status.paused} class:is-err={assistant.tone === "error"}>
        {statusLabel}
      </span>
      <Waveform />
      {#if store.history.length}
        <button class="link-btn" onclick={() => go("memory")}>
          Полная история — в разделе «Память»
          <AppIcon name="chevronRight" size={11} />
        </button>
      {/if}
    </div>
    {#if store.history.length}
      <button
        class="icon-btn"
        title="Очистить журнал диалога"
        aria-label="Очистить журнал диалога"
        onclick={clearHistory}
      >
        <AppIcon name="eraser" size={15} />
      </button>
    {/if}
  </div>

  <CommandInput />

  <p class="home-foot tech-label">
    {modelLabel} · ЛОКАЛЬНО · {settings.wakeword_model ? "WAKE WORD АКТИВЕН" : "WAKE WORD OFF"}
  </p>
</div>

<style>
  .transmission.is-empty {
    justify-content: center;
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

  .home-foot {
    text-align: center;
    margin: 2px 0 10px;
  }
</style>
