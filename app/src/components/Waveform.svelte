<script lang="ts">
  import { store } from "../store.svelte";

  /**
   * Осциллограмма голоса: реальные уровни звука от движка.
   * Источник различается: речь пользователя и озвучка Джарвиса.
   * Нет звука — спокойная полоска ожидания, никакого фейка.
   */

  const BARS = 7;
  const WEIGHTS = [0.5, 0.72, 0.9, 1, 0.9, 0.72, 0.5];

  const active = $derived(
    !store.status.paused &&
      (store.audioLevel > 0.03 || store.speaking),
  );

  const heights = $derived.by(() => {
    if (!active) return WEIGHTS.map(() => 0);
    const level = Math.max(0, Math.min(1, store.audioLevel));
    return WEIGHTS.map((w, i) => {
      const wave = 0.62 + 0.38 * Math.sin(level * 36 + i * 1.55);
      return Math.max(0.06, Math.min(1, level * w * wave + (store.speaking ? 0.12 : 0)));
    });
  });
</script>

<div class="wave" class:is-idle={!active} aria-hidden="true">
  {#each heights as h, i (i)}
    <span style:--h={h}></span>
  {/each}
</div>
