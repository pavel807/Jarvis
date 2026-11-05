<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { settings, patchSettings, toast, go, store } from "../store.svelte";
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
    { key: "general", title: "Общие" },
    { key: "stt", title: "Распознавание" },
    { key: "tts", title: "Синтез речи" },
    { key: "mic", title: "Микрофон" },
    { key: "ai", title: "AI-модель" },
    { key: "web", title: "Интернет" },
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

  function llmPicked(choice: (typeof llmChoices)[number]): boolean {
    return settings.llm_backend === choice.backend && settings.llm_model === choice.model;
  }

  async function pickLlm(choice: (typeof llmChoices)[number]) {
    if (llmPicked(choice)) return;
    await patchSettings({ llm_backend: choice.backend, llm_model: choice.model });
    toast("Модель сменится после перезапуска Джарвиса", "ok");
  }

  const maxTokenChoices = [
    { value: 0, label: "Без лимита" },
    { value: 128, label: "128 токенов" },
    { value: 256, label: "256 токенов" },
    { value: 512, label: "512 токенов" },
    { value: 1024, label: "1024 токена" },
  ];

  async function pickMaxTokens(value: number) {
    await patchSettings({ llm_max_tokens: value });
  }

  // ---------- Система ----------

  onMount(() => {
    void refreshMicStatus();
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
      <span class="icon"><AppIcon name="search" size={14} /></span>
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
    </div>
  </section>

  <!-- Распознавание -->
  <section class="settings-section" data-section="stt" bind:this={sectionEls.stt}>
    <h2 class="section-title">Распознавание речи</h2>
    <div class="card">
      {#if hit("чувствительность", "порог", "wake", "джарвис")}
        <p class="card__title">Чувствительность к слову «Джарвис»</p>
        <p class="card__hint">
          Ниже порог — срабатывает чаще, но и на постороннюю речь тоже.
          Сейчас: {Math.round(settings.wake_threshold * 100)}%
        </p>
        <input
          type="range"
          min="0.4"
          max="0.85"
          step="0.05"
          bind:value={settings.wake_threshold}
          aria-label="Чувствительность к слову Джарвис"
          onchange={() => void patchSettings({ wake_threshold: settings.wake_threshold })}
        />
      {/if}
          {#if hit("диалог", "продолжение", "followup", "уточнение")}
            <div class="row-item" style="margin-top: 8px">
              <span class="row-item__text">
                <span class="row-item__title">Продолжение диалога</span>
                <span class="row-item__hint">
                  После ответа Джарвис слушает уточнение {settings.followup_window_secs} секунд —
                  можно не повторять «Джарвис»
                </span>
              </span>
              <button
                class="switch"
                class:is-on={settings.followup_window_secs > 0}
                role="switch"
                aria-checked={settings.followup_window_secs > 0}
                aria-label="Продолжение диалога"
                onclick={(event) => {
                  const value = !(event.currentTarget as HTMLElement).classList.contains("is-on");
                  void patchSettings({ followup_window_secs: value ? 8 : 0 });
                }}
              ></button>
            </div>
          {/if}
          {#if hit("посторонние", "голоса", "фильтр", "мой голос", "speakers", "диаризация")}
            <SwitchRow
              checked={settings.speaker_filter}
              title="Только мой голос"
              hint="Посторонние голоса на фоне отсекаются до распознавания и не доходят до ответов. Профиль строится из записей в «Обучении» и обновляется сам после новых записей."
              onchange={(value) => void patchSettings({ speaker_filter: value })}
            />
          {/if}
    </div>
  </section>

  <!-- Синтез речи -->
  <section class="settings-section" data-section="tts" bind:this={sectionEls.tts}>
    <h2 class="section-title">Синтез речи</h2>
    <div class="card card--flush">
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
    </div>
  </section>

  <!-- AI-модель -->
  <section class="settings-section" data-section="ai" bind:this={sectionEls.ai}>
    <h2 class="section-title">AI-модель</h2>
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
      {#if hit("вариативность", "температура", "temperature")}
        <div class="row-item" style="flex-direction: column; align-items: stretch; gap: 6px">
          <div style="display: flex; justify-content: space-between; align-items: baseline">
            <span class="row-item__title">Вариативность ответов</span>
            <span class="row-item__value">{settings.llm_temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1.2"
            step="0.05"
            bind:value={settings.llm_temperature}
            aria-label="Температура модели"
            onchange={() => void patchSettings({ llm_temperature: settings.llm_temperature })}
          />
          <span class="row-item__hint">
            Ниже — точнее и суше, выше — свободнее. 0.6 подходит большинству.
          </span>
        </div>
      {/if}
      {#if hit("длина", "лимит", "токены")}
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title">Максимальная длина ответа</span>
            <span class="row-item__hint">Страховка от слишком долгих монологов</span>
          </span>
          <select
            class="select"
            style="width: auto"
            aria-label="Максимальная длина ответа"
            bind:value={settings.llm_max_tokens}
            onchange={() => void pickMaxTokens(settings.llm_max_tokens)}
          >
            {#each maxTokenChoices as choice (choice.value)}
              <option value={choice.value}>{choice.label}</option>
            {/each}
          </select>
        </div>
      {/if}
    </div>
  </section>

  <!-- Интернет -->
  <section class="settings-section" data-section="web" bind:this={sectionEls.web}>
    <h2 class="section-title">Интернет и поиск</h2>
    <div class="card card--flush">
      {#if hit("интернет", "поиск", "погода", "новости")}
        <SwitchRow
          checked={settings.web_search_auto}
          title="Искать свежие данные в интернете"
          hint="Погода, курсы, новости. Требуется подключение к сети."
          onchange={(value) => void patchSettings({ web_search_auto: value })}
        />
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
    {#if hit("файлы", "toml", "данные")}
      <div class="card">
        <p class="dim" style="font-size: 12.5px; line-height: 1.55">
          Команды и тонкая настройка хранятся в
          <span class="mono">commands.toml</span> и <span class="mono">settings.toml</span>.
          Их можно править вручную — Джарвис подхватит изменения без перезапуска.
        </p>
      </div>
    {/if}
  </section>
</div>
