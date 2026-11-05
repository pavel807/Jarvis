<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { store } from "../store.svelte";
  import AppIcon from "./AppIcon.svelte";

  /**
   * Кнопка микрофона: включает и выключает прослушивание движка.
   * Это реальное действие над реальным пайплайном: set_paused.
   */

  let busy = $state(false);

  const running = $derived(store.status.engineRunning);
  const paused = $derived(store.status.paused);
  /** Ждём разрешения macOS — кнопка честно показывает ошибку доступа. */
  const denied = $derived(store.micPermission === false);

  const label = $derived(denied
    ? "Нет доступа к микрофону"
    : !running
      ? "Движок запускается"
      : paused
        ? "Включить прослушивание"
        : "Выключить прослушивание");

  const disabled = $derived(busy || !running);

  async function toggle() {
    if (busy || !running) return;
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
  class="mic-btn"
  class:is-listening={!paused && running}
  class:is-off={paused && running && !denied}
  class:is-error={denied}
  {disabled}
  aria-pressed={!paused && running}
  aria-label={label}
  title={label}
  onclick={toggle}
>
  <AppIcon name={denied ? "micOff" : "mic"} size={24} />
</button>
