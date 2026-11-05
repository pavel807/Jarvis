<script lang="ts">
  import { store } from "../store.svelte";

  /**
   * Системный статус: четыре реальных индикатора, без выдуманных значений.
   * Данные приходят из status-опроса движка и событий.
   */

  interface Cell {
    name: string;
    value: string;
    tone: "ok" | "energy" | "warn" | "err" | "muted";
  }

  const cells = $derived.by<Cell[]>(() => {
    const running = store.status.engineRunning;
    const paused = store.status.paused;
    const llm = store.status.llmStatus;

    const mic: Cell = !running
      ? { name: "MIC", value: "WAIT", tone: "muted" }
      : paused
        ? { name: "MIC", value: "PAUSED", tone: "warn" }
        : { name: "MIC", value: "LIVE", tone: "energy" };

    const core: Cell = !running
      ? { name: "AI CORE", value: "BOOT", tone: "warn" }
      : llm === "ready"
        ? { name: "AI CORE", value: "READY", tone: "ok" }
        : llm === "error"
          ? { name: "AI CORE", value: "ERROR", tone: "err" }
          : { name: "AI CORE", value: "LOAD", tone: "warn" };

    const voice: Cell = !running
      ? { name: "VOICE", value: "WAIT", tone: "muted" }
      : store.status.ttsReady
        ? { name: "VOICE", value: "ONLINE", tone: "ok" }
        : { name: "VOICE", value: "LOAD", tone: "warn" };

    const link: Cell = running
      ? { name: "ENGINE", value: "ONLINE", tone: "ok" }
      : { name: "ENGINE", value: "OFFLINE", tone: "err" };

    return [core, voice, mic, link];
  });
</script>

<div class="sysstatus" role="status" aria-label="Состояние подсистем">
  {#each cells as cell (cell.name)}
    <div class={`syscell is-${cell.tone}`} title={`${cell.name}: ${cell.value}`}>
      <span class="syscell__top">
        <span class="syscell__led"></span>
        <span class="syscell__name">{cell.name}</span>
      </span>
      <span class="syscell__value">{cell.value}</span>
    </div>
  {/each}
</div>
