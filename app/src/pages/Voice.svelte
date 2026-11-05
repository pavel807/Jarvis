<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { settings, patchSettings, toast, go, store } from "../store.svelte";
  import { voiceGroups, voiceName } from "../voices";
  import AppIcon from "../components/AppIcon.svelte";

  /**
   * Голос: выбор диктора с прослушиванием примера, скорость и проверка.
   * Превью играет выбранный диктор, но не меняет голос приложения —
   * применение голоса отдельным подтверждением.
   */

  let voices = $state<[string, string][]>([]);
  let testText = $state("Привет! Я Джарвис. Говорю спокойно и понятно.");

  /** Кандидат на применение: подсвечен, но ещё не сохранён. */
  let candidate = $state("");
  let query = $state("");
  let groupFilter = $state("all");

  onMount(async () => {
    voices = await invoke<[string, string][]>("tts_voices");
  });

  const groups = $derived(voiceGroups(voices));

  const filteredGroups = $derived.by(() => {
    const q = query.trim().toLowerCase();
    const result = groups.map((group) => ({
      title: group.title,
      items: group.items.filter((voice) => {
        if (groupFilter !== "all") {
          const isMale = group.title === "Мужские";
          if (groupFilter === "male" && !isMale) return false;
          if (groupFilter === "female" && isMale) return false;
        }
        if (!q) return true;
        return voice.name.toLowerCase().includes(q) || voice.id.includes(q);
      }),
    }));
    return result.filter((g) => g.items.length > 0);
  });

  const currentName = $derived(voiceName(settings.tts_voice));

  const speedLabel = $derived.by(() => {
    const s = settings.tts_speed;
    if (s < 0.9) return `Спокойная · ${s.toFixed(2)}×`;
    if (s > 1.05) return `Быстрая · ${s.toFixed(2)}×`;
    return `Обычная · ${s.toFixed(2)}×`;
  });

  async function applyVoice(id: string) {
    await patchSettings({ tts_voice: id });
    candidate = "";
    toast(`Голос: ${voiceName(id)}`, "ok");
  }

  function pickCandidate(id: string) {
    if (id === settings.tts_voice) {
      candidate = "";
      return;
    }
    candidate = id;
  }

  async function commitSpeed() {
    await patchSettings({ tts_speed: settings.tts_speed });
  }

  /** Озвучить фразу голосом приложения. */
  function playTest() {
    if (!testText.trim()) return;
    void invoke("tts_test", { text: testText });
    toast("Озвучиваю…", "info", 1500);
  }

  /** Пример голоса без применения — движок умеет синтез любым диктором. */
  function previewVoice(id: string) {
    const text = testText.trim() || "Привет! Я Джарвис.";
    void invoke("tts_test", { text, voice: id });
    toast(`Пример: ${voiceName(id)} — голос приложения не меняется`, "info", 2000);
  }
</script>

<div class="page">
  <div class="page-head">
    <button class="icon-btn" style="align-self: flex-start" title="Назад (Esc)" aria-label="Назад" onclick={() => go(store.pageBack)}>
      <AppIcon name="chevronLeft" size={17} />
    </button>
    <div class="page-head__body">
      <h1 class="page-title">Голос</h1>
      <p class="page-sub">
        Каким голосом Джарвис отвечает. Пример можно послушать без смены голоса —
        выбор подтверждается отдельно.
      </p>
    </div>
  </div>

  <div class="voice-current card card--outline">
    <div class="voice-current__body">
      <span class="voice-current__label">Сейчас говорит</span>
      <span class="voice-current__name">{currentName}</span>
      <span class="voice-current__note">
        {voices.find(([id]) => id === settings.tts_voice)?.[1]?.split(",")[0] ?? "Silero"}
        · скорость {settings.tts_speed.toFixed(2)}×
      </span>
    </div>
    <button class="btn btn--secondary" title="Озвучить проверочную фразу" onclick={playTest}>
      <AppIcon name="play" size={15} />
      Прослушать
    </button>
  </div>

  <div class="card card--outline">
    <div class="card__title">Выбрать голос</div>
    <p class="card__hint">
      Нажмите на голос, чтобы подсветить его как кандидат, и прослушайте пример
      кнопкой справа — пример не меняет голос приложения.
    </p>

    <div class="toolbar">
      <div class="search-box">
        <span class="search-box__icon"><AppIcon name="search" size={15} /></span>
        <input
          bind:value={query}
          class="input"
          type="search"
          placeholder="Поиск голоса…"
          aria-label="Поиск по голосам"
        />
      </div>
      <div class="seg" role="tablist" aria-label="Категория голосов">
        <button
          class="seg__btn"
          class:is-active={groupFilter === "all"}
          role="tab"
          aria-selected={groupFilter === "all"}
          onclick={() => (groupFilter = "all")}
        >
          Все
        </button>
        <button
          class="seg__btn"
          class:is-active={groupFilter === "male"}
          role="tab"
          aria-selected={groupFilter === "male"}
          onclick={() => (groupFilter = "male")}
        >
          Мужские
        </button>
        <button
          class="seg__btn"
          class:is-active={groupFilter === "female"}
          role="tab"
          aria-selected={groupFilter === "female"}
          onclick={() => (groupFilter = "female")}
        >
          Женские
        </button>
      </div>
    </div>

    {#if !voices.length}
      <div class="empty"><span class="spinner"></span>Загружаю голоса…</div>
    {:else}
      {#each filteredGroups as group (group.title)}
        <div class="section-title">{group.title}</div>
        <div class="voice-grid">
          {#each group.items as voice (voice.id)}
            <div
              class="voice-row"
              class:is-candidate={candidate === voice.id}
              role="button"
              tabindex="0"
              aria-label={`Голос ${voice.name}${voice.note ? `, ${voice.note}` : ""}`}
              onclick={() => pickCandidate(voice.id)}
              onkeydown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  pickCandidate(voice.id);
                }
              }}
            >
              <span
                class="voice-row__mark"
                class:is-current={settings.tts_voice === voice.id}
                class:is-candidate={candidate === voice.id}
                title={settings.tts_voice === voice.id ? "Текущий голос" : ""}
              >
                {#if settings.tts_voice === voice.id}
                  <AppIcon name="check" size={11} />
                {/if}
              </span>
              <span class="voice-row__text">
                <span class="voice-row__name">{voice.name}</span>
                {#if voice.note}
                  <span class="voice-row__note">{voice.note}</span>
                {/if}
              </span>
              <button
                class="icon-btn"
                title="Прослушать пример"
                aria-label={`Прослушать пример голоса ${voice.name}`}
                onclick={(event) => {
                  event.stopPropagation();
                  previewVoice(voice.id);
                }}
              >
                <AppIcon name="play" size={14} />
              </button>
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty">По запросу «{query}» голосов нет.</div>
      {/each}
    {/if}

    {#if candidate}
      <div class="confirm-bar" role="alert">
        <span class="confirm-bar__text">Выбрать голос «{voiceName(candidate)}»?</span>
        <button class="btn btn--ghost btn--sm" onclick={() => (candidate = "")}>Отмена</button>
        <button class="btn btn--primary btn--sm" onclick={() => void applyVoice(candidate)}>
          Применить
        </button>
      </div>
    {/if}
  </div>

  <div class="grid-2">
    <div class="card card--outline">
      <div class="card__title">Скорость речи</div>
      <p class="card__hint">{speedLabel}</p>
      <input
        type="range"
        min="0.7"
        max="1.2"
        step="0.05"
        bind:value={settings.tts_speed}
        aria-label="Скорость речи"
        onchange={commitSpeed}
      />
      <div class="speed-scale">
        <span>Спокойнее</span>
        <span>Быстрее</span>
      </div>
    </div>

    <div class="card card--outline">
      <div class="card__title">Проверочная фраза</div>
      <p class="card__hint">Ей же озвучиваются примеры голосов.</p>
      <textarea bind:value={testText} class="textarea" rows="2" aria-label="Проверочная фраза"></textarea>
      <button class="btn btn--secondary btn--sm" style="margin-top: 10px" onclick={playTest}>
        <AppIcon name="play" size={14} />
        Озвучить текущим голосом
      </button>
    </div>
  </div>

  <div class="card card--outline">
    <button class="row-item" style="border-top: none" onclick={() => go("wakeword")}>
      <span class="row-item__text">
        <span class="row-item__title">Джарвис не всегда слышит обращение?</span>
        <span class="row-item__hint">
          Обучите распознавание на свой голос — несколько записей и проверка
        </span>
      </span>
      <span class="row-item__chevron"><AppIcon name="chevronRight" size={16} /></span>
    </button>
  </div>
</div>

<style>
  .voice-current {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .voice-current__body {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .voice-current__label {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-3);
  }

  .voice-current__name {
    font-size: 16px;
    font-weight: 650;
    letter-spacing: -0.01em;
  }

  .voice-current__note {
    font-size: 12.5px;
    color: var(--text-2);
  }

  .voice-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 4px;
  }

  .voice-row__text {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }

  .voice-row__name {
    font-size: 13.5px;
    font-weight: 550;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .voice-row__note {
    font-size: 11.5px;
    color: var(--text-3);
    white-space: nowrap;
  }

  .speed-scale {
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    color: var(--text-3);
    margin-top: 2px;
  }
</style>
