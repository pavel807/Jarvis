<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { settings, patchSettings, toast, go, store, type Settings } from "../store.svelte";
  import { voiceName } from "../voices";
  import SwitchRow from "../components/SwitchRow.svelte";
  import AppIcon from "../components/AppIcon.svelte";

  /**
   * Настройки Mark V: горизонтальная навигация по категориям и переходы
   * во вторичные экраны (команды, голос, обучение). Изменения применяются
   * сразу; тонкие параметры — в settings.toml, файл перечитывается на лету.
   */

  const SECTIONS = [
    { key: "modules", title: "Разделы" },
    { key: "ai", title: "Модель" },
    { key: "general", title: "Общие" },
    { key: "mic", title: "Микрофон" },
    { key: "system", title: "Система" },
] as const;

  type SectionKey = (typeof SECTIONS)[number]["key"];

  let activeSection = $state<SectionKey>("modules");
  let query = $state("");
  let sectionEls: Partial<Record<SectionKey, HTMLElement>> = {};
  let observer: IntersectionObserver | null = null;

  function hit(...texts: (string | undefined)[]): boolean {
    const q = query.trim().toLowerCase();
    if (!q) return true;
    return texts.some((t) => t?.toLowerCase().includes(q));
  }

  function scrollTo(key: SectionKey) {
    activeSection = key;
    sectionEls[key]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- Общие ----------

  let autostart = $state(false);
  let autostartSupported = $state(true);

  // ---------- Микрофон ----------

  type MicStatus = "authorized" | "denied" | "pending" | "unknown";
  let micStatus = $state<MicStatus>("unknown");
  let micBusy = $state(false);
  // Сколько образцов голоса владельца записано (для статуса профиля).
  let voiceSamples = $state<number | null>(null);

  async function refreshVoiceSamples() {
    try {
      const samples = await invoke<{ train: [string, number][] }>("ww_samples");
      // В профиле участвуют только записи с меткой владельца ([jarvis] в имени);
      // записи фона (none) в фильтре не используются.
      voiceSamples = samples.train.filter((entry) => entry[0].includes("[jarvis]")).length;
    } catch {
      voiceSamples = null;
    }
  }

  const MIC_STATUS_TEXT: Record<MicStatus, string> = {
    authorized: "Разрешён — Джарвис слышит вас",
    denied: "Запрещён macOS — включите вручную в Системных настройках",
    pending: "Ждёт ответа на системный запрос",
    unknown: "Не удалось определить",
  };

  const micStatusText = $derived(MIC_STATUS_TEXT[micStatus]);

  async function refreshMicStatus() {
    try {
      micStatus = await invoke<MicStatus>("mic_status");
    } catch {
      // движок ещё не поднят
    }
  }

  async function requestMic() {
    if (micBusy) return;
    micBusy = true;
    try {
      const status = await invoke<MicStatus>("mic_request");
      micStatus = status;
      if (status === "pending") {
        toast("Системный запрос показан — подтвердите его", "info");
      } else if (status === "authorized") {
        toast("Доступ к микрофону уже разрешён", "ok");
      } else if (status === "denied") {
        toast("Повторный запрос macOS не показывает — включите в настройках", "err");
      } else {
        toast("Статус определить не удалось", "err");
      }
    } catch (e) {
      toast(`Не получилось: ${e}`, "err");
    } finally {
      micBusy = false;
    }
  }

  async function openMicSettings() {
    try {
      await invoke("open_mic_settings");
    } catch (e) {
      toast(`Не получилось: ${e}`, "err");
    }
  }

  // --- Облачная модель: статус не нужен, поля задаются в разделе «AI-модель» ---

  $effect(() => {
    const granted = store.micPermission;
    if (granted === true) micStatus = "authorized";
    else if (granted === false) micStatus = "pending";
  });

  // ---------- AI-модель ----------

  /**
   * Модели ответов: mirai-m — 4-битные веса, mirai-l — 8-битные: та же
   * сеть, вдвое меньше квантизационных повреждений — меньше выдумывает.
   * 4B — старшая модель, точнее на фактах. Смена — после перезапуска.
   */
  const llmChoices = [
    {
      backend: "uzu",
      model: "qwen3.5-2b-mirai-l",
      title: "Умная — Qwen3.5 2B (8-бит)",
      hint: "Точные веса, ~2,2 ГБ. Рекомендуется: заметно меньше выдумывает, чем 4-бит.",
    },
    {
      backend: "uzu",
      model: "qwen3.5-4b-mirai-l",
      title: "Эксперт — Qwen3.5 4B (8-бит)",
      hint: "Старшая модель, ~4,5 ГБ. Максимальная точность, отвечает чуть медленнее.",
    },
    {
      backend: "uzu",
      model: "qwen3.5-0.8b",
      title: "Быстрая — Qwen3.5 0.8B (4-бит)",
      hint: "~0,5 ГБ. Самая быстрая, но чаще ошибается.",
    },
    {
      backend: "candle",
      model: "qwen3.5-0.8b",
      title: "Классическая — candle 0.8B",
      hint: "Собственная реализация без Uzu. Модель из папки models/.",
    },
  ];

  /** Облачные варианты: отдельная логика выбора (пресет + поля ключа). */
  const apiChoices = [
    {
      preset: "yandex" as const,
      title: "Яндекс GPT — облако",
      hint: "Ответы в интернете через Яндекс. Нужен API-ключ Яндекс.Облака и идентификатор каталога.",
    },
    {
      preset: "openai" as const,
      title: "Своя модель (API) — облако",
      hint: "Любой OpenAI-совместимый сервис: OpenAI, OpenRouter, LM Studio и другие. Ключ, адрес и модель — ниже.",
    },
  ];

  function apiPicked(preset: string): boolean {
    return settings.llm_backend === "api" && settings.api_preset === preset;
  }

  function llmPicked(choice: (typeof llmChoices)[number]): boolean {
    return settings.llm_backend === choice.backend && settings.llm_model === choice.model;
  }

  async function pickLlm(choice: (typeof llmChoices)[number]) {
    if (llmPicked(choice)) return;
    await patchSettings({ llm_backend: choice.backend, llm_model: choice.model });
    toast("Модель сменится после перезапуска Джарвиса", "ok");
  }

  // ---------- Облачная модель ----------

  /** Облако выбрано? */
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
    toast(
      settings.api_key
        ? "Облако включится после перезапуска Джарвиса"
        : "Впиши API-ключ ниже — он сохранится сам",
      "ok",
    );
  }

  async function saveApiField(patch: Partial<Settings>) {
    await patchSettings(patch);
  }

  // Проверка облачного ключа минимальным запросом.
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

  // ---------- Система ----------

  onMount(() => {
    void refreshMicStatus();
    void refreshVoiceSamples();
    void (async () => {
      try {
        const { isEnabled } = await import("@tauri-apps/plugin-autostart");
        autostart = await isEnabled();
      } catch {
        autostartSupported = false;
      }
    })();

    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const key = (entry.target as HTMLElement).dataset.section as SectionKey;
            if (key) activeSection = key;
          }
        }
      },
      { rootMargin: "-16% 0px -74% 0px" },
    );
    for (const el of Object.values(sectionEls)) el && observer.observe(el);
    return () => observer?.disconnect();
  });

  async function toggleAutostart(value: boolean) {
    autostart = value;
    try {
      const { enable, disable } = await import("@tauri-apps/plugin-autostart");
      await (value ? enable() : disable());
      toast(value ? "Будет запускаться при входе" : "Автозапуск выключен", "ok");
    } catch (e) {
      autostart = !value;
      toast(`Не получилось: ${e}`, "err");
    }
  }
</script>

<div class="page">
  <div class="toolbar">
    <div class="search-box">
      <span class="search-box__icon"><AppIcon name="search" size={14} /></span>
      <input
        bind:value={query}
        class="input"
        type="search"
        placeholder="Поиск по настройкам…"
        aria-label="Поиск по настройкам"
      />
    </div>
  </div>

  <nav class="settings-nav" aria-label="Категории настроек">
    {#each SECTIONS as section (section.key)}
      <button
        class="settings-nav__item"
        class:is-active={activeSection === section.key}
        onclick={() => scrollTo(section.key)}
      >
        {section.title}
      </button>
    {/each}
  </nav>

  <!-- Переходы во вторичные экраны -->
  <section class="settings-section" data-section="modules" bind:this={sectionEls.modules}>
    <h2 class="section-title">Разделы</h2>
    <div class="card card--flush">
      {#if hit("команды", "триггеры", "голосовые фразы")}
        <button class="row-item" onclick={() => go("commands")}>
          <span class="row-item__icon"><AppIcon name="commands" size={16} /></span>
          <span class="row-item__text">
            <span class="row-item__title">Команды</span>
            <span class="row-item__hint">
              Голосовые фразы и действия: поиск, приоритет, включение и отключение
            </span>
          </span>
          <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
        </button>
      {/if}
      {#if hit("голос", "диктор", "синтез", "скорость")}
        <button class="row-item" onclick={() => go("voice")}>
          <span class="row-item__icon"><AppIcon name="speaker" size={16} /></span>
          <span class="row-item__text">
            <span class="row-item__title">Голос и скорость речи</span>
            <span class="row-item__hint">
              Сейчас выбран {voiceName(settings.tts_voice)} · {settings.tts_speed.toFixed(2)}×
            </span>
          </span>
          <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
        </button>
      {/if}
      {#if hit("обучение", "wake word", "джарвис", "распознавание")}
        <button class="row-item" onclick={() => go("wakeword")}>
          <span class="row-item__icon"><AppIcon name="mic" size={16} /></span>
          <span class="row-item__text">
            <span class="row-item__title">Обучение на свой голос</span>
            <span class="row-item__hint">
              Джарвис не всегда слышит обращение — запишите образцы
            </span>
          </span>
          <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
        </button>
      {/if}
      {#if hit("помощь", "проблемы", "журнал", "диагностика")}
        <button class="row-item" onclick={() => go("help")}>
          <span class="row-item__icon"><AppIcon name="help" size={16} /></span>
          <span class="row-item__text">
            <span class="row-item__title">Помощь</span>
            <span class="row-item__hint">Как пользоваться и что делать, если не работает</span>
          </span>
          <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
        </button>
      {/if}
    </div>
  </section>

  <!-- Общие -->
  <section class="settings-section" data-section="general" bind:this={sectionEls.general}>
    <h2 class="section-title">Общие</h2>
    <div class="card card--flush">
      {#if hit("отвечать", "вопросы", "поведение")}
        <SwitchRow
          checked={settings.llm_enabled}
          title="Отвечать на любые вопросы"
          hint="Если команда не распознана, Джарвис отвечает сам — без интернета."
          onchange={(value) => void patchSettings({ llm_enabled: value })}
        />
      {/if}
      {#if hit("темы", "память", "забыли")}
        <SwitchRow
          checked={settings.topic_memory}
          title="Помнить незакрытые темы"
          hint="«Мы это забыли» — вернёт разговор к теме, от которой отвлеклись."
          onchange={(value) => void patchSettings({ topic_memory: value })}
        />
      {/if}
      {#if hit("интернет", "поиск", "погода", "новости")}
        <SwitchRow
          checked={settings.web_search_auto}
          title="Искать свежие данные в интернете"
          hint="Погода, курсы, новости. Требуется подключение к сети."
          onchange={(value) => void patchSettings({ web_search_auto: value })}
        />
      {/if}
      {#if hit("числа", "сокращения", "нормализация")}
        <SwitchRow
          checked={settings.llm_normalize}
          title="Проговаривать числа и сокращения"
          hint="«1,5 ГБ» прозвучит как «полтора гигабайта»."
          onchange={(value) => void patchSettings({ llm_normalize: value })}
        />
      {/if}
      {#if hit("звук", "сигнал", "подтверждение")}
        <SwitchRow
          checked={settings.ack_before_llm}
          title="Звук перед ответом"
          hint="Короткий сигнал, что вопрос услышан. Выключено — ответ быстрее."
          onchange={(value) => void patchSettings({ ack_before_llm: value })}
        />
      {/if}
    </div>
  </section>

  <!-- Микрофон -->
  <section class="settings-section" data-section="mic" bind:this={sectionEls.mic}>
    <h2 class="section-title">Микрофон</h2>
    <div class="card card--flush">
      {#if hit("доступ", "микрофон", "разрешение")}
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title">Доступ к микрофону</span>
            <span class="row-item__hint">{micStatusText}</span>
          </span>
          <span class="badge" class:badge--ok={micStatus === "authorized"}>
            {micStatus === "authorized" ? "OK" : micStatus.toUpperCase()}
          </span>
        </div>
        <button class="row-item" disabled={micBusy} onclick={requestMic}>
          <span class="row-item__text">
            <span class="row-item__title">Запросить доступ</span>
            <span class="row-item__hint">Показать системный запрос macOS ещё раз</span>
          </span>
          <span class="row-item__chevron"><AppIcon name="mic" size={15} /></span>
        </button>
        <button class="row-item" onclick={openMicSettings}>
          <span class="row-item__text">
            <span class="row-item__title">Открыть Системные настройки</span>
            <span class="row-item__hint">Конфиденциальность и безопасность → Микрофон</span>
          </span>
          <span class="row-item__chevron"><AppIcon name="gear" size={15} /></span>
        </button>
      {/if}
      {#if hit("посторонние", "голоса", "фильтр", "мой голос", "speakers", "диаризация")}
        <SwitchRow
          checked={settings.speaker_filter}
          title="Только мой голос"
          hint="Посторонние голоса на фоне отсекаются до распознавания и не доходят до ответов. Профиль строится из записей в «Обучении» и обновляется сам после новых записей."
          onchange={(value) => void patchSettings({ speaker_filter: value })}
        />
        {#if voiceSamples !== null}
          <button
            class="row-item"
            onclick={() => go("wakeword")}
            aria-label="Профиль голоса — перейти к обучению"
          >
            <span class="row-item__text">
              <span class="row-item__title">Профиль голоса</span>
              <span class="row-item__hint">
                {voiceSamples > 0
                  ? `Готов: ${voiceSamples} ${voiceSamples === 1 ? "образец" : (voiceSamples < 5 ? "образца" : "образцов")} голоса`
                  : "Нет образцов — фильтр не работает. Запиши их в «Обучении»"}
              </span>
            </span>
            <span class="badge" class:badge--ok={voiceSamples > 0}>
              {voiceSamples > 0 ? "OK" : "ПУСТО"}
            </span>
          </button>
        {/if}
      {/if}
    </div>
  </section>

  <!-- Модель -->
  <section class="settings-section" data-section="ai" bind:this={sectionEls.ai}>
    <h2 class="section-title">Модель</h2>
    <div class="card card--flush">
      {#each llmChoices as choice (choice.backend + choice.model)}
        {#if hit(choice.title, choice.hint, "модель")}
          <button class="row-item" onclick={() => pickLlm(choice)}>
            <span class="row-item__text">
              <span class="row-item__title">{choice.title}</span>
              <span class="row-item__hint">{choice.hint}</span>
            </span>
            {#if llmPicked(choice)}
              <span class="row-item__value">
                <AppIcon name="check" size={15} />
              </span>
            {/if}
          </button>
        {/if}
      {/each}
      {#each apiChoices as choice (choice.preset)}
        {#if hit(choice.title, choice.hint, "модель", "облако", "api")}
          <button class="row-item" onclick={() => pickApi(choice.preset)}>
            <span class="row-item__text">
              <span class="row-item__title">{choice.title}</span>
              <span class="row-item__hint">{choice.hint}</span>
            </span>
            {#if apiPicked(choice.preset)}
              <span class="row-item__value">
                <AppIcon name="check" size={15} />
              </span>
            {/if}
          </button>
        {/if}
      {/each}
      {#if apiActive && yandexActive}
        <div class="api-fields">
          <label class="api-field">
            <span class="api-field__label">API-ключ</span>
            <input
              class="input"
              type="password"
              value={settings.api_key}
              placeholder="AQVN… или IAM-токен"
              onchange={(e) => void saveApiField({ api_key: e.currentTarget.value })}
            />
          </label>
          <label class="api-field">
            <span class="api-field__label">Каталог (folder id)</span>
            <input
              class="input"
              type="text"
              value={settings.api_folder_id}
              placeholder="b1g…"
              onchange={(e) => void saveApiField({ api_folder_id: e.currentTarget.value })}
            />
          </label>
          <label class="api-field">
            <span class="api-field__label">Модель</span>
            <input
              class="input"
              type="text"
              value={settings.api_model}
              placeholder="yandexgpt-lite"
              onchange={(e) => void saveApiField({ api_model: e.currentTarget.value })}
            />
          </label>
          <button class="btn btn--secondary btn--sm" disabled={apiChecking} onclick={checkApi}>
            {apiChecking ? "Проверяю…" : "Проверить ключ"}
          </button>
        </div>
      {:else if apiActive}
        <div class="api-fields">
          <label class="api-field">
            <span class="api-field__label">API-ключ</span>
            <input
              class="input"
              type="password"
              value={settings.api_key}
              placeholder="sk-…"
              onchange={(e) => void saveApiField({ api_key: e.currentTarget.value })}
            />
          </label>
          <label class="api-field">
            <span class="api-field__label">Адрес API</span>
            <input
              class="input"
              type="text"
              value={settings.api_base_url}
              placeholder="https://api.openai.com/v1"
              onchange={(e) => void saveApiField({ api_base_url: e.currentTarget.value })}
            />
          </label>
          <label class="api-field">
            <span class="api-field__label">Модель</span>
            <input
              class="input"
              type="text"
              value={settings.api_model}
              placeholder="gpt-4o-mini"
              onchange={(e) => void saveApiField({ api_model: e.currentTarget.value })}
            />
          </label>
          <button class="btn btn--secondary btn--sm" disabled={apiChecking} onclick={checkApi}>
            {apiChecking ? "Проверяю…" : "Проверить ключ"}
          </button>
        </div>
      {/if}
    </div>
  </section>

  <!-- Система -->
  <section class="settings-section" data-section="system" bind:this={sectionEls.system}>
    <h2 class="section-title">Система</h2>
    {#if autostartSupported}
      <div class="card card--flush">
        {#if hit("автозапуск", "вход", "запуск")}
          <SwitchRow
            checked={autostart}
            title="Запускать при входе в систему"
            hint="Джарвис будет ждать в строке меню сразу после включения Mac."
            onchange={toggleAutostart}
          />
        {/if}
      </div>
    {/if}
  </section>
</div>

<style>
  .api-fields {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px 14px 14px;
    border-top: 1px solid var(--border-soft, rgba(128, 128, 128, 0.15));
  }

  .api-field {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .api-field__label {
    font-size: 12px;
    opacity: 0.7;
  }
</style>
