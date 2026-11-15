<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { store, listenStart, listenStop } from "../store.svelte";
  import AppIcon from "./AppIcon.svelte";

  /**
   * Кнопка микрофона — главный прибор главной страницы. Кольцо по контуру
   * не украшение: его дуга заполняется реальным уровнем звука от движка,
   * поэтому сразу видно, что микрофон слышит.
   *
   * Два режима:
   * - десктоп: клик включает/выключает постоянное прослушивание
   *   (set_paused), wake-слово активирует Джарвиса сам;
   * - тач (iPhone/iPad): tap-to-talk — первый тап открывает микрофон,
   *   второй останавливает запись и отправляет фразу на распознавание
   *   (FastConformer); wake-слова на телефоне нет.
   */

  let { size = 84 }: { size?: number } = $props();

  let busy = $state(false);

  const touch =
    typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;

  const running = $derived(store.status.engineRunning);
  const paused = $derived(store.status.paused);
  const denied = $derived(store.micPermission === false);
  const listening = $derived(
    touch ? store.listening : !paused && running && !denied,
  );

  /** Контурное кольцо: дуга уровня звука. */
  const R = 46;
  const CIRC = 2 * Math.PI * R;
  const level = $derived(
    listening ? Math.max(0, Math.min(1, store.audioLevel)) : 0,
  );

  const label = $derived(touch
    ? store.listening
      ? "Остановить запись"
      : "Говорить"
    : denied
      ? "Нет доступа к микрофону"
      : !running
        ? "Движок запускается"
        : paused
          ? "Включить прослушивание"
          : "Выключить прослушивание");

  const disabled = $derived(touch ? busy : busy || !running);

  async function toggle() {
    if (busy) return;
    if (touch) {
      // Tap-to-talk: повторный тап останавливает запись и запускает
      // распознавание. Статусы «pending»/«denied» приносит listenStart.
      busy = true;
      try {
        if (store.listening) listenStop();
        else await listenStart();
      } finally {
        busy = false;
      }
      return;
    }
    if (!running) return;
    busy = true;
    const next = !paused;
    try {
      await invoke("set_paused", { paused: next });
      store.status.paused = next;
    } catch {
      // движок ещё не поднят
    } finally {
      busy = false;
    }
  }
</script>

<button
  class="mic"
  class:is-listening={listening}
  class:is-off={paused && running && !denied}
  class:is-error={denied}
  {disabled}
  style:--size="{size}px"
  aria-pressed={listening}
  aria-label={label}
  title={label}
  onclick={toggle}
>
  <svg class="mic__ring" viewBox="0 0 100 100" aria-hidden="true">
    <defs>
      <linearGradient id="mic-ring-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#d9f4ff" />
        <stop offset="1" stop-color="#4bcbff" />
      </linearGradient>
    </defs>
    <circle class="mic__track" cx="50" cy="50" r={R} />
    <circle
      class="mic__level"
      cx="50"
      cy="50"
      r={R}
      style:stroke-dasharray="{CIRC}"
      style:stroke-dashoffset={CIRC * (1 - level)}
    />
  </svg>
  <span class="mic__core">
    <AppIcon name={denied ? "micOff" : "mic"} size={Math.round(size * 0.36)} />
  </span>
</button>

<style>
  .mic {
    width: var(--size);
    height: var(--size);
    padding: 0;
    position: relative;
    flex-shrink: 0;
    border: none;
    background: none;
    cursor: pointer;
    transition: transform 0.14s var(--ease-out), opacity 0.2s;
  }

  .mic:disabled {
    opacity: 0.45;
    cursor: default;
  }

  .mic:active:not(:disabled) {
    transform: scale(0.95);
  }

  .mic:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 4px;
    border-radius: 50%;
  }

  .mic__ring {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    transform: rotate(-90deg);
    pointer-events: none;
  }

  .mic__track {
    fill: none;
    stroke: var(--border-strong);
    stroke-width: 2.5;
    transition: stroke 0.25s;
  }

  .mic__level {
    fill: none;
    stroke: url(#mic-ring-grad);
    stroke-width: 3.5;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.12s linear, opacity 0.3s;
    opacity: 0;
  }

  .mic.is-listening .mic__level {
    opacity: 1;
  }

  .mic.is-error .mic__track {
    stroke: rgba(240, 107, 112, 0.45);
  }

  /* Диск: слои стекла и металла, как у всей панели Mark V. */
  .mic__core {
    position: absolute;
    inset: 9%;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-2);
    border: 1px solid rgba(75, 203, 255, 0.25);
    background:
      radial-gradient(120% 120% at 50% 0%, rgba(75, 203, 255, 0.14), rgba(75, 203, 255, 0) 55%),
      linear-gradient(180deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0) 45%),
      var(--surface-2);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
    transition: background 0.22s, color 0.22s, border-color 0.22s, box-shadow 0.28s;
  }

  .mic:hover:not(:disabled) .mic__core {
    color: var(--cyan);
    border-color: rgba(75, 203, 255, 0.5);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.08),
      0 0 14px rgba(75, 203, 255, 0.18);
  }

  /* Слушает: залит энергией, светится на уровень звука. */
  .mic.is-listening .mic__core {
    color: var(--on-accent);
    border-color: rgba(160, 226, 255, 0.75);
    background:
      radial-gradient(120% 120% at 50% 0%, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0) 52%),
      linear-gradient(180deg, #59d3ff, var(--cyan));
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.32),
      0 0 22px rgba(75, 203, 255, 0.42);
  }

  /* Пауза: отжатый прибор */
  .mic.is-off .mic__core {
    color: var(--text-3);
    background: var(--surface);
    border-color: var(--border-strong);
    box-shadow: none;
  }

  /* Нет доступа: красный прибор */
  .mic.is-error .mic__core {
    color: var(--danger);
    border-color: rgba(240, 107, 112, 0.5);
    box-shadow: 0 0 12px rgba(240, 107, 112, 0.16);
  }
</style>
