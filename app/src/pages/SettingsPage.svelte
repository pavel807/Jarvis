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
    { key: "telegram", title: "Telegram" },
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
      // В профиле участвуют записи владельца: образцы «Джарвис» ([jarvis]) и
      // диктанты ([voice], приходят в общем списке); фоновые (none) не считаются.
      voiceSamples = samples.train.filter(
        (entry) => entry[0].includes("[jarvis]") || entry[0].includes("[voice]"),
      ).length;
    } catch {
      voiceSamples = null;
    }
  }

  // Вебвью сообщает платформу в UA: «Mac» — macOS, иначе Linux/Windows.
  const IS_MAC = /Mac/.test(navigator.userAgent);

  const MIC_STATUS_TEXT: Record<MicStatus, string> = {
    authorized: "Разрешён — Джарвис слышит вас",
    denied: IS_MAC
      ? "Запрещён macOS — включите вручную в Системных настройках"
      : "Заблокирован — проверьте настройки звука и устройство ввода",
    pending: IS_MAC
      ? "Ждёт ответа на системный запрос"
      : "Микрофон занят другим приложением",
    unknown: IS_MAC
      ? "Не удалось определить"
      : "Микрофон не найден — проверьте подключение устройства ввода",
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
        toast(
          IS_MAC
            ? "Повторный запрос macOS не показывает — включите в настройках"
            : "Запрос не удался — проверьте настройки звука",
          "err",
        );
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

  // ---------- Модели на диске: что установлено и что можно скачать ----------

  /** Список моделей из бэкенда: установлена ли, размер, кнопка «Скачать». */
  let models = $state<{ name: string; installed: boolean; size_mb: number | null; downloadable: boolean }[]>([]);

  async function loadModels() {
    try {
      models = await invoke("llm_models_status");
    } catch (e) {
      console.warn("llm_models_status:", e);
    }
  }

  async function downloadModel(name: string) {
    try {
      await invoke("llm_download_model", { name });
      toast(`Скачиваю ${name} — прогресс внизу списка`, "ok");
    } catch (e) {
      toast(`Не удалось начать загрузку: ${e}`, "err");
    }
  }

  // ---------- Облачная модель ----------

  /** Облако выбрано? */
  const apiActive = $derived(settings.llm_backend === "api");
  const yandexActive = $derived(apiActive && settings.api_preset === "yandex");

  /** Выбран ли этот облачный пресет. */
  function apiPicked(preset: string): boolean {
    return settings.llm_backend === "api" && settings.api_preset === preset;
  }

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
        ? "Переключаюсь на облако — модель загрузится на лету"
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

  // ---------- AI-модель ----------

  /**
   * Три варианта мозга: Яндекс GPT (облако), своя модель через API (облако)
   * и Qwen 3.5 локально. Внутри «Qwen 3.5» — выбор версии. На macOS версии
   * считает движок Mirai (uzu): mirai-l — 8-битные веса, вдвое меньше
   * квантизационных повреждений — меньше выдумывает. На Linux — candle,
   * и локальная модель одна: 0.8B из релиза. Смена — после перезапуска.
   */
  const isMac = /Mac/.test(navigator.userAgent);

  const qwenChoices = isMac
    ? [
        {
          model: "qwen3.5-2b-mirai-l",
          title: "Умная — 2B (8-бит)",
          hint: "Точные веса, ~2,2 ГБ. Рекомендуется: заметно меньше выдумывает.",
        },
        {
          model: "qwen3.5-4b-mirai-l",
          title: "Эксперт — 4B (8-бит)",
          hint: "Старшая модель, ~4,5 ГБ. Максимальная точность, отвечает чуть медленнее.",
        },
        {
          model: "qwen3.5-0.8b",
          title: "Быстрая — 0.8B (4-бит)",
          hint: "~0,5 ГБ. Самая быстрая, но чаще ошибается.",
        },
      ]
    : [
        {
          model: "qwen3.5-0.8b",
          title: "Qwen 3.5 0.8B",
          hint: "Локальная модель, ~0,5 ГБ. Качается из релиза при первом запуске.",
        },
      ];

  /** Ручные каталоги из models/ (candle-совместимые): показываем как есть. */
  const manualModels = $derived(
    models.filter((m) => !qwenChoices.some((c) => c.model === m.name)),
  );

  /** Три больших варианта выбора мозга. */
  const llmVariants = [
    {
      key: "yandex",
      title: "Яндекс GPT — облако",
      hint: "Ответы через Яндекс. Нужен API-ключ Яндекс.Облака и идентификатор каталога.",
    },
    {
      key: "openai",
      title: "Своя модель (API) — облако",
      hint: "Любой OpenAI-совместимый сервис: OpenAI, OpenRouter, LM Studio и другие. Ключ, адрес и модель — после выбора.",
    },
    {
      key: "qwen",
      title: "Qwen 3.5 — локально",
      hint: "Считается на этом компьютере, без интернета. Внутри — выбрать и скачать версию.",
    },
  ] as const;

  type VariantKey = (typeof llmVariants)[number]["key"];

  function variantPicked(key: VariantKey): boolean {
    if (key === "yandex") return apiPicked("yandex");
    if (key === "openai") return apiPicked("openai");
    return settings.llm_backend === "uzu" || settings.llm_backend === "candle";
  }

  /** Раскрыт ли список версий Qwen. */
  let qwenOpen = $state(false);

  async function pickVariant(key: VariantKey) {
    if (variantPicked(key)) {
      // Повторный клик по «Qwen локально» раскрывает или сворачивает версии.
      qwenOpen = key === "qwen" ? !qwenOpen : qwenOpen;
      return;
    }
    if (key === "qwen") {
      qwenOpen = true;
      await pickQwen(qwenChoices[0]);
      return;
    }
    qwenOpen = false;
    await pickApi(key);
  }

  function qwenPicked(choice: (typeof qwenChoices)[number]): boolean {
    const backend = isMac ? "uzu" : "candle";
    return settings.llm_backend === backend && settings.llm_model === choice.model;
  }

  async function pickQwen(choice: (typeof qwenChoices)[number]) {
    if (qwenPicked(choice)) return;
    await patchSettings({ llm_backend: isMac ? "uzu" : "candle", llm_model: choice.model });
    toast("Модель сменится после перезапуска Джарвиса", "ok");
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
    void loadModels();
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
      {#if hit("wake word", "джарвис", "распознавание", "диктант", "профиль голоса", "дополнительно")}
        <button class="row-item" onclick={() => go("wakeword")}>
          <span class="row-item__icon"><AppIcon name="mic" size={16} /></span>
          <span class="row-item__text">
            <span class="row-item__title">Wake word и голос — дополнительно</span>
            <span class="row-item__hint">
              Джарвис слышит фразу от любого человека; здесь — проверка фразы
              и записи для профиля голоса
            </span>
          </span>
          <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
        </button>
      {/if}
      {#if hit("помощь", "проблемы", "диагностика")}
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
      {#if hit("имя", "ассистент", "название", "как называть")}
        <div class="row-item" style="flex-direction: column; align-items: stretch; gap: 6px">
          <span class="row-item__title">Имя ассистента</span>
          <span class="row-item__hint">
            Как он представляется в разговоре. То же слово — фраза-активация:
            скажите имя, и ассистент откликнется.
          </span>
          <input
            class="input"
            type="text"
            value={settings.assistant_name}
            placeholder="Джарвис"
            onchange={(e) => {
              const v = e.currentTarget.value.trim() || "Джарвис";
              void patchSettings({ assistant_name: v, wake_word: v });
            }}
          />
        </div>
      {/if}
      {#if hit("звук", "сигнал", "подтверждение")}
        <SwitchRow
          checked={settings.ack_before_llm}
          title="Звук перед ответом"
          hint="Короткий сигнал, что вопрос услышан. Выключено — ответ быстрее."
          onchange={(value) => void patchSettings({ ack_before_llm: value })}
        />
      {/if}
      {#if hit("автозапуск", "вход", "запуск") && autostartSupported}
        <SwitchRow
          checked={autostart}
          title="Запускать при входе в систему"
          hint="Джарвис будет ждать в строке меню сразу после включения."
          onchange={toggleAutostart}
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
            <span class="row-item__hint">{IS_MAC
              ? "Показать системный запрос macOS ещё раз"
              : "На Linux отдельного разрешения не нужно — проверим микрофон"}</span>
          </span>
          <span class="row-item__chevron"><AppIcon name="mic" size={15} /></span>
        </button>
        <button class="row-item" onclick={openMicSettings}>
          <span class="row-item__text">
            <span class="row-item__title">{IS_MAC ? "Открыть Системные настройки" : "Открыть настройки звука"}</span>
            <span class="row-item__hint">{IS_MAC
              ? "Конфиденциальность и безопасность → Микрофон"
              : "Уровни ввода и выбор устройства"}</span>
          </span>
          <span class="row-item__chevron"><AppIcon name="gear" size={15} /></span>
        </button>
      {/if}
      {#if hit("посторонние", "голоса", "фильтр", "мой голос", "speakers", "диаризация")}
        {#if voiceSamples !== null}
          <button
            class="row-item"
            onclick={() => go("voicetrain")}
            aria-label="Профиль голоса — обучение диктантом"
          >
            <span class="row-item__text">
              <span class="row-item__title">Профиль голоса</span>
              <span class="row-item__hint">
                {voiceSamples > 0
                  ? `Готов: ${voiceSamples} ${voiceSamples === 1 ? "запись" : (voiceSamples < 5 ? "записи" : "записей")}. Обучить ещё`
                  : "Нет записей — фильтр не работает. Прочитай фразу в обучении"}
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
      {#each llmVariants as variant (variant.key)}
        {#if hit(variant.title, variant.hint, "модель")}
          <button class="row-item" onclick={() => pickVariant(variant.key)}>
            <span class="row-item__text">
              <span class="row-item__title">{variant.title}</span>
              <span class="row-item__hint">{variant.hint}</span>
            </span>
            {#if variantPicked(variant.key)}
              <span class="row-item__value">
                <AppIcon name={variant.key === "qwen" ? (qwenOpen ? "chevronUp" : "chevronDown") : "check"} size={15} />
              </span>
            {/if}
          </button>
        {/if}
      {/each}
      {#if qwenOpen && variantPicked("qwen")}
        {#each qwenChoices as choice (choice.model)}
          {#if hit(choice.title, choice.hint, "модель", "qwen")}
            <div class="row-item qwen-row">
              <button class="qwen-row__pick" onclick={() => void pickQwen(choice)}>
                <span class="row-item__text">
                  <span class="row-item__title">{choice.title}</span>
                  <span class="row-item__hint">{choice.hint}</span>
                </span>
                {#if qwenPicked(choice)}
                  <span class="row-item__value"><AppIcon name="check" size={15} /></span>
                {/if}
              </button>
              {#if !qwenPicked(choice)}
                <button class="btn btn--secondary btn--sm qwen-row__dl" disabled={!!store.download} onclick={() => downloadModel(choice.model)}>
                  Скачать
                </button>
              {/if}
            </div>
          {/if}
        {/each}
        {#each manualModels.filter((m) => hit(m.name, "модель")) as m (m.name)}
          <div class="row-item qwen-row">
            <span class="row-item__text">
              <span class="row-item__title">{m.name}</span>
              <span class="row-item__hint">
                {m.installed ? `Установлена${m.size_mb ? ` — ${m.size_mb} МБ` : ""}` : "Не скачана"}
              </span>
            </span>
            {#if m.installed}
              <span class="row-item__value"><AppIcon name="check" size={15} /></span>
            {:else if m.downloadable}
              <button class="btn btn--secondary btn--sm" disabled={!!store.download} onclick={() => downloadModel(m.name)}>
                Скачать
              </button>
            {/if}
          </div>
        {/each}
      {/if}
      {#if store.download}
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title">Загрузка: {store.download.what}</span>
            <span class="row-item__hint">
              <progress value={store.download.percent} max="100"></progress> {store.download.percent}%
            </span>
          </span>
        </div>
      {/if}
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

  <!-- Telegram -->
  <section class="settings-section" data-section="telegram" bind:this={sectionEls.telegram}>
    <h2 class="section-title">Telegram</h2>
    <div class="card card--flush">
      {#if hit("телеграм", "telegram", "бот", "чат", "канал", "команды")}
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title">Канал в Telegram</span>
            <span class="row-item__hint">
              Токен бота от BotFather и id чатов через запятую. Пусто — канал выключен.
            </span>
          </span>
        </div>
        <SwitchRow
          checked={settings.telegram_execute}
          title="Выполнять команды из Telegram"
          hint="Бот исполняет тот же список команд, что и голосом: «открой ютуб», таймеры и другие. Выключено — только разговор."
          onchange={(value) => void patchSettings({ telegram_execute: value })}
        />
        <div class="api-fields">
          <label class="api-field">
            <span class="api-field__label">Токен бота</span>
            <input
              class="input"
              type="password"
              value={settings.telegram_bot_token}
              placeholder="123456789:AAH..."
              onchange={(e) => void patchSettings({ telegram_bot_token: e.currentTarget.value })}
            />
          </label>
          <label class="api-field">
            <span class="api-field__label">Чаты</span>
            <input
              class="input"
              type="text"
              value={settings.telegram_chat_ids}
              placeholder="123456789, -1001234567890"
              onchange={(e) => void patchSettings({ telegram_chat_ids: e.currentTarget.value })}
            />
          </label>
        </div>
      {/if}
    </div>
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

  /* Версии Qwen внутри «Qwen 3.5»: сдвиг вправо — вложенность читается. */
  .qwen-row {
    padding-left: 18px;
    border-top: 1px solid var(--border-soft, rgba(128, 128, 128, 0.12));
  }

  .qwen-row__pick {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    background: none;
    border: none;
    padding: 0;
    text-align: left;
    color: inherit;
    cursor: pointer;
  }
</style>
