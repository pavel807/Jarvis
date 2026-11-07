<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { settings, patchSettings, toast, store, type Settings } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";
  import VoiceTraining from "../components/VoiceTraining.svelte";

  /**
   * Мастер первого запуска: (1) ждём загрузку основных моделей,
   * (2) выбор мозга — локальная модель или облако с ключом,
   * (3) знакомство с голосом — диктант фразы (можно пропустить).
   * Показывается, пока setup_done = false; закрывается финальным шагом.
   */

  let step = $state(0);
  let busy = $state(false);

  // ---------- Шаг «Автоматизация» ----------
  let installing = $state(false);
  let automationInstalled = $state(false);

  async function installAutomation() {
    if (installing) return;
    installing = true;
    try {
      await invoke<string>("uv_install");
      automationInstalled = true;
      // Включаем плагины и кладём пример-плагин.
      await patchSettings({ plugins_enabled: true });
      toast("Автоматизация установлена", "ok");
    } catch (e) {
      toast(`Не получилось: ${e}`, "err", 6000);
    } finally {
      installing = false;
    }
  }

  // ---------- Шаг 1: загрузка ----------

  const engineReady = $derived(store.status.engineRunning);
  const llmReady = $derived(store.status.llmStatus === "ready");
  const ttsReady = $derived(store.status.ttsReady);

  // ---------- Шаг 2: мозг ----------

  const llmChoices = [
    {
      backend: "uzu",
      model: "qwen3.5-2b-mirai-l",
      title: "Умная — локальная, Qwen3.5 2B",
      hint: "Работает на этом Mac, ~2,2 ГБ. Точные веса, реже ошибается.",
    },
    {
      backend: "uzu",
      model: "qwen3.5-4b-mirai-l",
      title: "Эксперт — локальная, Qwen3.5 4B",
      hint: "Старшая локальная модель, ~4,5 ГБ. Точнее, отвечает чуть медленнее.",
    },
    {
      backend: "uzu",
      model: "qwen3.5-0.8b",
      title: "Быстрая — локальная, Qwen3.5 0.8B",
      hint: "~0,5 ГБ. Самая быстрая, но чаще ошибается.",
    },
    {
      backend: "candle",
      model: "qwen3.5-0.8b",
      title: "Классическая — локальная, candle",
      hint: "Собственная реализация без Uzu. Модель из папки models/.",
    },
  ];

  function llmPicked(backend: string, model: string): boolean {
    return settings.llm_backend === backend && settings.llm_model === model;
  }

  async function pickLlm(backend: string, model: string) {
    if (llmPicked(backend, model)) return;
    await patchSettings({ llm_backend: backend, llm_model: model });
    toast("Модель включится после перезапуска Джарвиса", "ok");
  }

  const apiActive = $derived(settings.llm_backend === "api");
  const yandexActive = $derived(apiActive && settings.api_preset === "yandex");

  async function pickApi(preset: "yandex" | "openai") {
    if (apiActive && settings.api_preset === preset) return;
    const base =
      preset === "yandex"
        ? "https://llm.api.cloud.yandex.net/v1"
        : "https://api.openai.com/v1";
    const model = preset === "yandex" ? "yandexgpt-lite" : "gpt-4o-mini";
    await patchSettings({ llm_backend: "api", api_preset: preset, api_base_url: base, api_model: model });
  }

  async function saveApiField(patch: Partial<Settings>) {
    await patchSettings(patch);
  }

  let apiChecking = $state(false);
  async function checkApi() {
    if (apiChecking) return;
    apiChecking = true;
    try {
      const answer = await invoke<string>("llm_api_test");
      toast(`Модель отвечает: «${answer}»`, "ok", 4000);
    } catch (e) {
      toast(`Не получилось: ${e}`, "err", 6000);
    } finally {
      apiChecking = false;
    }
  }

  // Список моделей провайдера: запрашивается у API, выбор — тапом.
  let apiModels = $state<string[] | null>(null);
  let apiModelsLoading = $state(false);
  async function loadApiModels() {
    if (apiModelsLoading) return;
    apiModelsLoading = true;
    try {
      apiModels = await invoke<string[]>("llm_api_models");
    } catch (e) {
      apiModels = null;
      toast(`Не получилось: ${e}`, "err", 6000);
    } finally {
      apiModelsLoading = false;
    }
  }

  // ---------- Завершение ----------

  async function finish() {
    if (busy) return;
    busy = true;
    await patchSettings({ setup_done: true });
    busy = false;
  }
</script>

{#snippet api_model_picker()}
  <button class="btn btn--secondary btn--sm" disabled={apiModelsLoading} onclick={loadApiModels}>
    {apiModelsLoading
      ? "Запрашиваю список…"
      : apiModels
        ? "Обновить список моделей"
        : "Показать модели из API"}
  </button>
  {#if apiModels}
    <div class="ob-models">
      {#each apiModels as m (m)}
        <button class="row-item" onclick={() => void saveApiField({ api_model: m })}>
          <span class="row-item__text">
            <span class="row-item__title">{m}</span>
          </span>
          {#if settings.api_model === m}
            <span class="row-item__value"><AppIcon name="check" size={15} /></span>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
{/snippet}

<div class="ob-overlay" role="dialog" aria-label="Первый запуск Джарвиса">
  <div class="ob-card">
    <div class="ob-steps" aria-hidden="true">
      {#each ["Загрузка", "Модель", "Автоматизация", "Голос"] as label, i (label)}
        <span class="ob-step" class:is-active={step === i} class:is-done={step > i}>
          {i + 1}. {label}
        </span>
      {/each}
    </div>

    {#if step === 0}
      <h1 class="ob-title">Джарвис просыпается</h1>
      <p class="ob-sub">Загружаются основные модели — это занимает до минуты.</p>
      <div class="card card--flush ob-status">
        <div class="ob-row">
          <span class="ob-dot" class:is-ok={engineReady}></span>
          Голосовой движок
          <span class="ob-val">{engineReady ? "готов" : "загружается…"}</span>
        </div>
        <div class="ob-row">
          <span class="ob-dot" class:is-ok={llmReady} class:is-warn={!llmReady}></span>
          Модель ответов
          <span class="ob-val">{llmReady ? "готова" : "загружается…"}</span>
        </div>
        <div class="ob-row">
          <span class="ob-dot" class:is-ok={ttsReady} class:is-warn={!ttsReady}></span>
          Синтез речи
          <span class="ob-val">{ttsReady ? "готов" : "загружается…"}</span>
        </div>
      </div>
      <button class="btn btn--primary btn--block" disabled={!engineReady} onclick={() => (step = 1)}>
        {engineReady ? "Дальше" : "Жду готовности…"}
      </button>
    {:else if step === 1}
      <h1 class="ob-title">Кто будет отвечать?</h1>
      <p class="ob-sub">
        Локальная модель работает без интернета. Облачная — умнее, но нужен
        API-ключ, и фразы уходят в сеть.
      </p>
      <div class="card card--flush ob-picker">
        {#each llmChoices as choice (choice.backend + choice.model)}
          <button class="row-item" onclick={() => pickLlm(choice.backend, choice.model)}>
            <span class="row-item__text">
              <span class="row-item__title">{choice.title}</span>
              <span class="row-item__hint">{choice.hint}</span>
            </span>
            {#if llmPicked(choice.backend, choice.model)}
              <span class="row-item__value"><AppIcon name="check" size={15} /></span>
            {/if}
          </button>
        {/each}
        <button class="row-item" onclick={() => pickApi("yandex")}>
          <span class="row-item__text">
            <span class="row-item__title">Яндекс GPT — облако</span>
            <span class="row-item__hint">Нужен API-ключ Яндекс.Облака и идентификатор каталога</span>
          </span>
          {#if apiActive && yandexActive}
            <span class="row-item__value"><AppIcon name="check" size={15} /></span>
          {/if}
        </button>
        <button class="row-item" onclick={() => pickApi("openai")}>
          <span class="row-item__text">
            <span class="row-item__title">Своя модель (API) — облако</span>
            <span class="row-item__hint">Любой OpenAI-совместимый сервис: ключ, адрес и модель ниже</span>
          </span>
          {#if apiActive && !yandexActive}
            <span class="row-item__value"><AppIcon name="check" size={15} /></span>
          {/if}
        </button>
        {#if apiActive && yandexActive}
          <div class="ob-api">
            <input
              class="input"
              type="password"
              value={settings.api_key}
              placeholder="API-ключ (AQVN… или IAM-токен)"
              onchange={(e) => void saveApiField({ api_key: e.currentTarget.value })}
            />
            <input
              class="input"
              type="text"
              value={settings.api_folder_id}
              placeholder="Каталог (folder id, b1g…)"
              onchange={(e) => void saveApiField({ api_folder_id: e.currentTarget.value })}
            />
            <input
              class="input"
              type="text"
              value={settings.api_model}
              placeholder="Модель (yandexgpt-lite)"
              onchange={(e) => void saveApiField({ api_model: e.currentTarget.value })}
            />
            <button class="btn btn--secondary btn--sm" disabled={apiChecking} onclick={checkApi}>
              {apiChecking ? "Проверяю…" : "Проверить ключ"}
            </button>
            {@render api_model_picker()}
          </div>
        {:else if apiActive}
          <div class="ob-api">
            <input
              class="input"
              type="password"
              value={settings.api_key}
              placeholder="API-ключ (sk-…)"
              onchange={(e) => void saveApiField({ api_key: e.currentTarget.value })}
            />
            <input
              class="input"
              type="text"
              value={settings.api_base_url}
              placeholder="Адрес API (https://api.openai.com/v1)"
              onchange={(e) => void saveApiField({ api_base_url: e.currentTarget.value })}
            />
            <input
              class="input"
              type="text"
              value={settings.api_model}
              placeholder="Модель (gpt-4o-mini)"
              onchange={(e) => void saveApiField({ api_model: e.currentTarget.value })}
            />
            <button class="btn btn--secondary btn--sm" disabled={apiChecking} onclick={checkApi}>
              {apiChecking ? "Проверяю…" : "Проверить ключ"}
            </button>
            {@render api_model_picker()}
          </div>
        {/if}
      </div>
      <button class="btn btn--primary btn--block" onclick={() => (step = 2)}>Дальше</button>
    {:else if step === 2}
      <h1 class="ob-title">Автоматизация делами</h1>
      <p class="ob-sub">
        Плагины на Python: свои команды для Джарвиса — цитаты, отчёты, что
        угодно. Установка скачает движок запуска (uv, ~20 МБ) в папку
        программы и покажет процесс в консоли. Можно пропустить и включить
        позже в настройках.
      </p>
      {#if !automationInstalled}
        <button class="btn btn--primary btn--block" disabled={installing} onclick={() => void installAutomation()}>
          {installing ? "Устанавливаю…" : "Установить автоматизацию"}
        </button>
      {:else}
        <div class="card card--flush ob-status">
          <div class="ob-row">
            <span class="ob-dot is-ok"></span>
            Движок запуска (uv)
            <span class="ob-val">установлен</span>
          </div>
          <div class="ob-row">
            <span class="ob-dot is-ok"></span>
            Пример плагина «Цитаты»
            <span class="ob-val">готов</span>
          </div>
        </div>
      {/if}
      {#if store.consoleLines.length}
        <div class="ob-console">
          {#each store.consoleLines as line, i (i)}
            <div>{line}</div>
          {/each}
        </div>
      {/if}
      <div class="ob-skip-row">
        <button class="btn btn--primary btn--block" disabled={installing} onclick={() => (step = 3)}>
          {automationInstalled ? "Дальше" : "Пропустить"}
        </button>
      </div>
    {:else if step === 3}
      <h1 class="ob-title">Познакомься с моим голосом</h1>
      <p class="ob-sub">
        Прочитай фразу с экрана — Джарвис запомнит, как звучит твой голос, и
        будет отсекать посторонних. Можно пропустить: он честно предупредит,
        что фильтр пока не работает.
      </p>
      <VoiceTraining />
      <div class="ob-skip-row">
        <button class="btn btn--primary btn--block" onclick={finish}>Готово</button>
      </div>
    {/if}

    {#if step > 0}
      <button class="btn btn--ghost ob-back" onclick={() => (step -= 1)}>Назад</button>
    {/if}
  </div>
</div>

<style>
  .ob-overlay {
    position: fixed;
    inset: 0;
    z-index: 60;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg, #0a0d12);
    padding: 24px;
  }

  .ob-card {
    width: 100%;
    max-width: 460px;
    max-height: 100%;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 4px;
  }

  .ob-steps {
    display: flex;
    gap: 10px;
    font-size: 12px;
    opacity: 0.75;
  }

  .ob-step.is-active {
    opacity: 1;
    font-weight: 600;
  }

  .ob-step.is-done {
    opacity: 1;
  }

  .ob-title {
    margin: 0;
    font-size: 22px;
  }

  .ob-sub {
    margin: 0;
    font-size: 13.5px;
    line-height: 1.55;
    opacity: 0.75;
  }

  .ob-status {
    display: flex;
    flex-direction: column;
  }

  .ob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 14px;
    font-size: 14px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.12);
  }

  .ob-row:last-child {
    border-bottom: none;
  }

  .ob-val {
    margin-left: auto;
    font-size: 12.5px;
    opacity: 0.7;
  }

  .ob-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(128, 128, 128, 0.45);
    animation: ob-pulse 1.2s ease-in-out infinite;
  }

  .ob-dot.is-ok {
    background: #30b46e;
    animation: none;
  }

  .ob-dot.is-warn {
    background: rgba(230, 179, 47, 0.8);
  }

  @keyframes ob-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
  }

  .ob-picker {
    max-height: 300px;
    overflow-y: auto;
  }

  .ob-api {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px 14px;
    border-top: 1px solid rgba(128, 128, 128, 0.12);
  }

  .ob-skip-row {
    margin-top: auto;
  }

  .ob-console {
    max-height: 160px;
    overflow-y: auto;
    background: rgba(10, 14, 20, 0.85);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 10px;
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 11.5px;
    line-height: 1.6;
    opacity: 0.9;
    user-select: text;
  }

  .ob-models {
    display: flex;
    flex-direction: column;
    max-height: 220px;
    overflow-y: auto;
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 10px;
  }

  .ob-back {
    align-self: flex-start;
  }
</style>
