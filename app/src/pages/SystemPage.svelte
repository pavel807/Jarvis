<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { store, go } from "../store.svelte";
  import { getAssistant } from "../assistant";
  import AppIcon from "../components/AppIcon.svelte";

  /**
   * Система: реальные состояния подсистем, технический журнал движка
   * и сведения о данных. Ничего выдуманного: статус — из status-опроса,
   * журнал — из get_logs и живых событий.
   */

  let fileLogs = $state<string[]>([]);
  let showLog = $state(true);

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

  const allLogs = $derived.by(() => {
    // Живые события (новые сверху) + стартовый журнал файла (старые снизу).
    const file = [...fileLogs].reverse();
    return [...store.logs, ...file];
  });

  onMount(async () => {
    try {
      fileLogs = await invoke<string[]>("get_logs");
    } catch {
      // движок ещё не поднят — журнал будет только из живых событий
    }
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
      <div class="status-item">
        <span class="status-item__dot is-ok"></span>
        <span class="status-item__name">Приватность</span>
        <span class="status-item__value">ВСЁ ЛОКАЛЬНО</span>
      </div>
    </div>
  </div>

  <div class="section-title">Технический журнал</div>
  <button class="btn btn--secondary btn--sm" onclick={() => (showLog = !showLog)}>
    <AppIcon name={showLog ? "chevronDown" : "chevronRight"} size={14} />
    {showLog ? "Скрыть журнал" : "Показать журнал"}
  </button>
  {#if showLog}
    <div class="card" style="margin-top: 10px">
      <div class="log">
        {allLogs.length ? allLogs.join("\n") : "Журнал пуст — движок ещё не присылал событий."}
      </div>
    </div>
  {/if}

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
          Голосовой ассистент, который работает целиком на этом компьютере.
          Речь распознаётся и синтезируется локально, ответы готовит модель на диске —
          без отправки данных в облако. Текущая фаза: {assistant.full.toLowerCase()}.
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
