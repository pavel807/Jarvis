<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import AppIcon from "./components/AppIcon.svelte";

  /** Окно «Утилиты»: таймеры, секундомер, календарь. Открывается голосом
   * («поставь таймер на 5 минут», «открой календарь») или кнопкой вручную. */

  const TABS = [
    { key: "timer", title: "Таймер", icon: "stop" },
    { key: "stopwatch", title: "Секундомер", icon: "play" },
    { key: "calendar", title: "Календарь", icon: "grid" },
  ] as const;

  let tab = $state<string>("timer");

  // ---------- Таймеры ----------
  interface UtilTimer {
    id: number;
    label: string;
    endsAt: number;
    total: number;
    kind: "timer" | "alarm";
    fired?: boolean;
  }
  let timers = $state<UtilTimer[]>([]);
  let now = $state(Date.now());
  let ticker: number | undefined;

  let manualLabel = $state("");
  let manualMinutes = $state(5);
  let manualBusy = $state(false);

  async function closeWindow() {
    try {
      await invoke("utils_close");
    } catch (e) {
      console.error(e);
    }
  }

  async function startManual() {
    if (manualBusy || manualMinutes <= 0) return;
    manualBusy = true;
    try {
      await invoke("timer_start", {
        label: manualLabel,
        seconds: Math.round(manualMinutes * 60),
      });
      manualLabel = "";
    } catch (e) {
      console.error(e);
    } finally {
      manualBusy = false;
    }
  }

  async function stopTimer(id: number) {
    try {
      await invoke("timer_cancel", { id });
    } catch {
      // уже сработал и сам ушёл из списка
      timers = timers.filter((t) => t.id !== id);
    }
  }

  function fmtRemaining(endsAt: number): string {
    const secs = Math.max(0, Math.ceil((endsAt - now) / 1000));
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function pctDone(t: UtilTimer): number {
    const passed = now - (t.endsAt - t.total * 1000);
    return Math.min(100, Math.max(0, (passed / (t.total * 1000)) * 100));
  }

  // ---------- Секундомер ----------
  let swRunning = $state(false);
  let swStartedAt = $state(0);
  let swAccumulated = $state(0);
  let swNow = $state(Date.now());
  const swElapsed = $derived.by(() => {
    const live = swRunning ? swNow - swStartedAt : 0;
    return swAccumulated + live;
  });
  function swToggle() {
    if (swRunning) {
      swAccumulated += Date.now() - swStartedAt;
      swRunning = false;
    } else {
      swStartedAt = Date.now();
      swRunning = true;
    }
  }
  function swReset() {
    swRunning = false;
    swAccumulated = 0;
  }
  function fmtMs(ms: number): string {
    const total = Math.floor(ms / 10);
    const cs = total % 100;
    const s = Math.floor(total / 100) % 60;
    const m = Math.floor(total / 6000);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
  }

  // ---------- Календарь ----------
  const WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
  const MONTHS = [
    "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
  ];
  const today = new Date();
  let viewYear = $state(today.getFullYear());
  let viewMonth = $state(today.getMonth());

  const calendarCells = $derived.by(() => {
    const first = new Date(viewYear, viewMonth, 1);
    const offset = (first.getDay() + 6) % 7; // неделя с понедельника
    const days = new Date(viewYear, viewMonth + 1, 0).getDate();
    const cells: { day: number; current: boolean }[] = [];
    for (let i = 0; i < offset; i++) cells.push({ day: 0, current: false });
    for (let d = 1; d <= days; d++) cells.push({ day: d, current: true });
    while (cells.length % 7 !== 0) cells.push({ day: 0, current: false });
    return cells;
  });
  const isToday = (day: number) =>
    day === today.getDate() &&
    viewMonth === today.getMonth() &&
    viewYear === today.getFullYear();
  function shiftMonth(delta: number) {
    const m = viewMonth + delta;
    viewMonth = ((m % 12) + 12) % 12;
    viewYear += Math.floor(m / 12);
  }

  let unlisteners: (() => void)[] = [];

  onMount(async () => {
    unlisteners.push(
      await listen("engine-event", (e: { payload: any }) => {
        const ev = e.payload ?? {};
        if (ev.type === "timer_started") {
          timers.push({
            id: ev.id as number,
            label: (ev.label as string) ?? "",
            endsAt: Date.now() + (ev.seconds as number) * 1000,
            total: ev.seconds as number,
            kind: (ev.kind as "timer" | "alarm") ?? "timer",
          });
          tab = "timer";
        } else if (ev.type === "timer_fired") {
          // Сигнал звенит, пока не остановят голосом или кнопкой — из списка
          // не убираем, чтобы кнопка «Остановить сигнал» осталась доступной.
          const existing = timers.find((x) => x.id === ev.id);
          if (existing) {
            existing.fired = true;
            existing.endsAt = Date.now();
          } else {
            timers.push({
              id: ev.id as number,
              label: (ev.label as string) ?? "",
              endsAt: Date.now(),
              total: (ev.seconds as number) ?? 0,
              kind: (ev.kind as "timer" | "alarm") ?? "timer",
              fired: true,
            });
          }
          tab = "timer";
        } else if (ev.type === "timer_stopped") {
          timers = timers.filter((x) => x.id !== ev.id);
        }
      }),
    );
    unlisteners.push(
      await listen("utils-tab", (e: { payload: string }) => {
        tab = e.payload;
      }),
    );
    ticker = window.setInterval(() => {
      now = Date.now();
      swNow = Date.now();
    }, 200);
  });

  onDestroy(() => {
    if (ticker) clearInterval(ticker);
    unlisteners.forEach((u) => u());
    unlisteners = [];
  });
</script>

<div class="utils">
  <div class="utils__head">
    <span class="utils__title">Утилиты</span>
    <button class="utils__close" title="Закрыть" onclick={closeWindow}>×</button>
  </div>
  <div class="utils__tabs">
    {#each TABS as t (t.key)}
      <button
        class="utils__tab"
        class:is-active={tab === t.key}
        onclick={() => (tab = t.key)}
      >
        <AppIcon name={t.icon} size={15} />
        {t.title}
      </button>
    {/each}
  </div>

  <!-- Таймеры -->
  {#if tab === "timer"}
    <div class="utils__body">
      {#if timers.length === 0}
        <div class="utils__empty">
          Активных таймеров нет. Скажи «поставь таймер на 5 минут» или поставь вручную.
        </div>
      {/if}
      {#each timers as t (t.id)}
        <div class="util-timer" class:is-fired={t.fired}>
          <div class="util-timer__time">
            {t.fired ? (t.kind === "alarm" ? "Будильник!" : "Время вышло!") : fmtRemaining(t.endsAt)}
          </div>
          {#if t.label}
            <div class="util-timer__label">{t.label}</div>
          {/if}
          <div class="util-timer__bar"><span style={`width: ${pctDone(t)}%`}></span></div>
          <button class="util-timer__stop" onclick={() => stopTimer(t.id)}>
            {t.fired ? "Остановить сигнал" : "Остановить"}
          </button>
        </div>
      {/each}

      <div class="util-manual">
        <input
          bind:value={manualLabel}
          class="util-manual__label"
          placeholder="Что напомнить (не обязательно)"
        />
        <div class="util-manual__row">
          <input bind:value={manualMinutes} type="number" min="1" max="600" />
          <span>мин</span>
          <button class="btn btn--primary" disabled={manualBusy} onclick={startManual}>
            Поставить
          </button>
        </div>
      </div>
    </div>

    <!-- Секундомер -->
  {:else if tab === "stopwatch"}
    <div class="utils__body utils__body--center">
      <div class="util-stopwatch" class:is-running={swRunning}>
        {fmtMs(swElapsed)}
      </div>
      <div class="util-stopwatch__controls">
        <button class="btn btn--primary" onclick={swToggle}>
          {swRunning ? "Пауза" : "Старт"}
        </button>
        <button class="btn btn--ghost" onclick={swReset}>Сброс</button>
      </div>
    </div>

    <!-- Календарь -->
  {:else}
    <div class="utils__body">
      <div class="util-cal__head">
        <button class="util-cal__nav" onclick={() => shiftMonth(-1)}>‹</button>
        <span class="util-cal__month">{MONTHS[viewMonth]} {viewYear}</span>
        <button class="util-cal__nav" onclick={() => shiftMonth(1)}>›</button>
      </div>
      <div class="util-cal__grid">
        {#each WEEKDAYS as w (w)}
          <span class="util-cal__weekday">{w}</span>
        {/each}
        {#each calendarCells as c, i (i)}
          <span
            class="util-cal__day"
            class:is-today={c.current && isToday(c.day)}
            class:is-empty={!c.current}
          >
            {c.current ? c.day : ""}
          </span>
        {/each}
      </div>
    </div>
  {/if}
</div>
