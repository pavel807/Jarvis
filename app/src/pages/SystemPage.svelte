<script lang="ts">
  import { store, go } from "../store.svelte";
  import { getAssistant } from "../assistant";
  import AppIcon from "../components/AppIcon.svelte";

  /**
   * Система: состояния подсистем и диагностика. Технический журнал убран
   * из интерфейса — статус подсистем говорит всё, что нужно пользователю.
   */

  const assistant = $derived(getAssistant());

  const subsystems = $derived.by(() => {
    const running = store.status.engineRunning;
    const llm = store.status.llmStatus;
    return [
      {
        name: "Движок",
        value: running ? "ONLINE" : "OFFLINE",
        tone: running ? "ok" : "err",
      },
      {
        name: "AI-ядро",
        value: llm === "ready" ? "READY" : llm === "error" ? "ERROR" : "LOADING",
        tone: llm === "ready" ? "ok" : llm === "error" ? "err" : "warn",
      },
      {
        name: "Синтез речи",
        value: store.status.ttsReady ? "ONLINE" : "LOADING",
        tone: store.status.ttsReady ? "ok" : "warn",
      },
      {
        name: "Микрофон",
        value: !running
          ? "OFFLINE"
          : store.status.paused
            ? "PAUSED"
            : "LISTENING",
        tone: !running ? "err" : store.status.paused ? "warn" : "ok",
      },
    ] as const;
  });
</script>

<div class="page">
  <div class="section-title">Состояние подсистем</div>
  <div class="card card--flush">
    <div class="status-list">
      {#each subsystems as sub (sub.name)}
        <div class="status-item">
          <span class="status-item__dot" class:is-ok={sub.tone === "ok"} class:is-warn={sub.tone === "warn"} class:is-err={sub.tone === "err"}></span>
          <span class="status-item__name">{sub.name}</span>
          <span class="status-item__value">{sub.value}</span>
        </div>
      {/each}
    </div>
  </div>

  <div class="section-title">Диагностика</div>
  <div class="card card--flush">
    <button class="row-item" onclick={() => go("help")}>
      <span class="row-item__icon"><AppIcon name="help" size={16} /></span>
      <span class="row-item__text">
        <span class="row-item__title">Помощь и типовые проблемы</span>
        <span class="row-item__hint">Как пользоваться и что делать, если не работает</span>
      </span>
      <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
    </button>
    <div class="row-item">
      <span class="row-item__text">
        <span class="row-item__title">Рабочая папка агента</span>
        <span class="row-item__hint mono" style="user-select: text">
          {store.status.cwd || "—"}
        </span>
      </span>
    </div>
  </div>

  <div class="section-title">О программе</div>
  <div class="card">
    <div class="about-row">
      <span class="about-row__mark">J</span>
      <span class="row-item__text">
        <span class="row-item__title">Джарвис 0.2 · Mark V</span>
        <span class="row-item__hint">
          Голосовой ассистент: речь распознаётся и синтезируется локально.
          Ответы готовит модель на выбор — локальная на диске или облачная
          через API. Текущая фаза: {assistant.full.toLowerCase()}.
        </span>
      </span>
    </div>
  </div>
</div>

<style>
  .about-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .about-row__mark {
    flex-shrink: 0;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: var(--cyan-soft);
    border: 1px solid rgba(75, 203, 255, 0.35);
    color: var(--cyan);
    font-family: var(--mono);
    font-weight: 700;
    font-size: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>
