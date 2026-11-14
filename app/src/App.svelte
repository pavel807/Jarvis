<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { store, go, initEvents, toast, settings, NAV_PAGES, SUB_PAGES } from "./store.svelte";
  import { INTRO_DURATION } from "./lib/reactor/intro";
  import Header from "./components/Header.svelte";
  import AppIcon from "./components/AppIcon.svelte";
  import Home from "./pages/Home.svelte";
  import Memory from "./pages/Memory.svelte";
  import SettingsPage from "./pages/SettingsPage.svelte";
  import Commands from "./pages/Commands.svelte";
  import Voice from "./pages/Voice.svelte";
  import VoiceTrain from "./pages/VoiceTrain.svelte";
  import Onboarding from "./pages/Onboarding.svelte";
  import WakeWord from "./pages/WakeWord.svelte";
  import Help from "./pages/Help.svelte";

  /**
   * Каркас Mark V: шапка, системный статус, тело и нижняя навигация
   * из четырёх разделов. Вторичные экраны (команды, голос, обучение,
   * помощь) открываются поверх тела с кнопкой «назад».
   */

  const NAV = [
    { key: "home", label: "Главная", icon: "core" },
    { key: "memory", label: "Память", icon: "memory" },
    { key: "settings", label: "Настройки", icon: "gear" },
  ] as const;

  const isSubPage = $derived(SUB_PAGES.includes(store.page));
  const activeIndex = $derived.by(() => {
    if (isSubPage) return -1;
    return NAV_PAGES.indexOf(store.page);
  });

  /** Горячие клавиши: разделы по Cmd+1..4, Esc — назад со вторичного экрана. */
  function onKey(event: KeyboardEvent) {
    if ((event.metaKey || event.ctrlKey) && !event.altKey) {
      const index = Number(event.key);
      if (Number.isInteger(index) && index >= 1 && index <= NAV_PAGES.length) {
        event.preventDefault();
        go(NAV_PAGES[index - 1]);
      }
      return;
    }
    if (event.key === "Escape" && isSubPage) {
      go(store.pageBack);
    }
  }

  // Платформа для текстов доступа к микрофону: вебвью сообщает её в UA.
  const IS_MAC = /Mac/.test(navigator.userAgent);

  /** Кнопка в меню доступа к микрофону — сразу системная панель (macOS);
      на Linux доступа-диалога нет, бэкенд честно вернёт подсказку. */
  async function openMicSettings() {
    try {
      await invoke("open_mic_settings");
    } catch (e) {
      toast(`${e}` || "Не удалось открыть системные настройки", "err");
    }
  }

  const toastIcon = { ok: "check", err: "alert", info: "info" } as const;

  onMount(() => {
    void initEvents();
  });

  $effect(() => {
    // Страховка: интерфейс обязан открыться даже без отчёта реактора
    // (нет GPU, страница успела смениться, заставка повторяется).
    if (!store.booting) return;
    const fallback = setTimeout(
      () => {
        store.booting = false;
        store.introRequest = false;
      },
      (INTRO_DURATION + 2) * 1000,
    );
    return () => clearTimeout(fallback);
  });
</script>


<svelte:window onkeydown={onKey} />

<div class="app" data-boot={store.booting ? "true" : "false"}>
  <Header />

  <div class="app-body">
    {#if store.page === "home"}
      <Home />
    {:else if store.page === "memory"}
      <Memory />
    {:else if store.page === "settings"}
      <SettingsPage />
    {:else if store.page === "commands"}
      <Commands />
    {:else if store.page === "voice"}
      <Voice />
    {:else if store.page === "voicetrain"}
      <VoiceTrain />
    {:else if store.page === "wakeword"}
      <WakeWord />
    {:else if store.page === "help"}
      <Help />
    {/if}
  </div>

  <nav class="nav" style:--i={Math.max(0, activeIndex)} aria-label="Разделы">
    {#each NAV as item, i (item.key)}
      <button
        class="nav-btn"
        class:is-active={activeIndex === i}
        aria-current={activeIndex === i ? "page" : undefined}
        onclick={() => go(item.key)}
      >
        <span class="nav-btn__icon"><AppIcon name={item.icon} size={19} /></span>
        {item.label}
      </button>
    {/each}
  </nav>
</div>

<div class="toasts" role="status" aria-live="polite">
  {#each store.toasts as t (t.id)}
    <div class={`toast toast--${t.kind}`}>
      <span class="toast__icon"><AppIcon name={toastIcon[t.kind]} size={15} /></span>
      <span>{t.text}</span>
    </div>
  {/each}
</div>

<!-- Меню доступа к микрофону: движок ждёт разрешения и подключится сам. -->
{#if store.micPermission === false}
  <div class="perm-overlay">
    <div class="perm-card">
      <span class="perm-card__icon"><AppIcon name="mic" size={24} /></span>
      <h2 class="perm-card__title">Нужен доступ к микрофону</h2>
      <p class="perm-card__text">
        {IS_MAC
          ? "macOS не даёт Джарвису слышать вас. Подтвердите системный запрос (он мог открыться под окном) или выдайте доступ вручную — как только разрешение появится, микрофон подключится сам."
          : "Джарвис не слышит микрофон. Проверьте, что устройство ввода подключено и выбрано в настройках звука — как только микрофон появится, он подключится сам."}
      </p>
      <button class="btn btn--primary btn--block" onclick={openMicSettings}>
        {IS_MAC ? "Открыть Системные настройки" : "Открыть настройки звука"}
      </button>
      <button class="btn btn--ghost btn--block" onclick={() => (store.micPermission = null)}>
        Продолжить без микрофона
      </button>
    </div>
  </div>
{/if}

<!-- Мастер первого запуска: после бут-анимации, пока setup_done = false. -->
{#if !store.booting && !settings.setup_done}
  <Onboarding />
{/if}

