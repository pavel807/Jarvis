<script lang="ts">
  import { store } from "../store.svelte";

  /**
   * Осциллограмма голоса: реальные уровни звука от движка.
   * Источник различается: речь пользователя и озвучка Джарвиса.
   * Нет звука — спокойная полоска ожидания, никакого фейка.
   */

  const BARS = 13;
  const WEIGHTS = [0.42, 0.55, 0.68, 0.8, 0.9, 0.97, 1, 0.97, 0.9, 0.8, 0.68, 0.55, 0.42];

  const active = $derived(
    !store.status.paused &&
      (store.audioLevel > 0.03 || store.speaking),
  );

  const heights = $derived.by(() => {
    const t = performance.now() / 1000;
    if (!active) return WEIGHTS.map(() => 0);
    const level = Math.max(0, Math.min(1, store.audioLevel));
    return WEIGHTS.map((w, i) => {
      // Две волны с разными скоростями: спектр живой, а не механический.
      const wave =
        0.58 + 0.28 * Math.sin(level * 34 + i * 0.95) + 0.14 * Math.sin(t * 5.2 + i * 2.1);
      return Math.max(0.06, Math.min(1, level * w * wave + (store.speaking ? 0.14 : 0)));
    });
  });
</script>

<div class="wave" class:is-idle={!active} aria-hidden="true">
  {#each heights as h, i (i)}
    <span style:--h={h}></span>
  {/each}
</div>
