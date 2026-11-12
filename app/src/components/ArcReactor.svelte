<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { fade } from "svelte/transition";
  import { getOrbState } from "../assistant";
  import { INTRO_DURATION } from "../lib/reactor/intro";
  import { store } from "../store.svelte";
  import type { ReactorScene, ReactorState } from "../lib/reactor/scene";

  /**
   * Центральный реактор Mark V: Three.js-сцена в канвасе.
   *
   * Состояние приходит от реального движка (assistant.ts), уровень звука —
   * от реальных событий audio_level (микрофон и озвучка). Рендер
   * останавливается, когда вкладка скрыта; ресурсы освобождаются при
   * размонтировании. Three.js грузится лениво: остальным экранам он не нужен.
   */

  let canvas = $state<HTMLCanvasElement | null>(null);
  let scene: ReactorScene | null = null;
  let observer: ResizeObserver | null = null;

  const reactorState = $derived(getOrbState() as ReactorState);

  $effect(() => {
    // Уровень звука уходит в сцену на каждое событие audio_level.
    void store.audioLevel;
    scene?.setLevel(store.audioLevel);
  });

  $effect(() => {
    scene?.setState(reactorState);
  });

  /** Диагностика рендера — в консоль DevTools: журнал из интерфейса убран. */
  function report(message: string) {
    console.log(message);
  }

  /** Розжиг снимает оба флага: интерфейс открывается после сборки. */
  function finishIntro() {
    store.introRequest = false;
    store.booting = false;
  }

  function runIntro() {
    if (!scene) return;
    scene.playIntro(finishIntro);
  }

  $effect(() => {
    // «Проиграй анимацию реактора» в работающем интерфейсе — та же заставка.
    // Пока сцена не создана, просьба остаётся висеть: её подхватит onMount.
    if (store.introRequest && scene) runIntro();
  });

  onMount(() => {
    let visibilityHandler: (() => void) | null = null;

    void (async () => {
      if (!canvas) return;
      try {
        const { ReactorScene: SceneCtor } = await import("../lib/reactor/scene");
        if (!canvas) return;
        scene = new SceneCtor(canvas, {
          reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
        });
        // WebGPU-контракт: асинхронная инициализация обязательна до первого
        // кадра. WebGPURenderer сам падает на WebGL2, если WebGPU недоступен.
        const backend = await scene.init();
        scene.onError = report;
        console.info(`Реактор: рендер через ${backend}`);
        // Сцена создана позже первых $effect: синхронизируем актуальное
        // состояние и уровень звука, иначе реактор стартует в idle.
        scene.setState(reactorState);
        scene.setLevel(store.audioLevel);
        // Сборка на запуске: слои выходят из плоскости грани вдоль
        // собственной высоты реактора, захлопываются обратно, и сборка одним
        // рывком доворачивается во фронт — только тогда открывается интерфейс.
        // Просьба «проиграй анимацию» сюда же, если пришлась на время, пока
        // сцена ещё создавалась.
        if (store.booting || store.introRequest) runIntro();
      } catch (e) {
        // Ни WebGPU, ни WebGL2 недоступны: интерфейс остаётся рабочим без реактора.
        console.error("Рендер реактора недоступен:", e);
        report(`Реактор недоступен: ${e instanceof Error ? e.message : String(e)}`);
        if (canvas) canvas.style.display = "none";
        finishIntro();
        return;
      }
      observer = new ResizeObserver(() => scene?.resize());
      // Холст крупнее сцены и ограничен окном: следить надо за ним самим,
      // иначе изменение размера окна не дойдёт до рендерера.
      observer.observe(canvas);
      // Диагностическая ручка: состояние и бэкенд из консоли.
      (globalThis as Record<string, unknown>).__reactor = scene;

      // Скрытое окно не должно тратить GPU: цикл ставится на паузу.
      visibilityHandler = () => {
        if (document.hidden) scene?.pause();
        else scene?.resume();
      };
      document.addEventListener("visibilitychange", visibilityHandler);
    })();

    return () => {
      visibilityHandler && document.removeEventListener("visibilitychange", visibilityHandler);
      observer?.disconnect();
      observer = null;
      scene?.dispose();
      scene = null;
    };
  });

  onDestroy(() => {
    scene?.dispose();
    scene = null;
  });
</script>

<div class="reactor-stage" style:--intro-dur={`${INTRO_DURATION}s`}>
  <canvas bind:this={canvas} aria-label="Треугольный реактор Mark V"></canvas>
  {#if store.booting}
    <div class="reactor-boot" transition:fade={{ duration: 260 }} aria-hidden="true">
      <span class="reactor-boot__label">Пуск реактора</span>
      <span class="reactor-boot__track"><span class="reactor-boot__fill"></span></span>
    </div>
  {/if}
</div>
