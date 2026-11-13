<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { listen, type UnlistenFn } from "@tauri-apps/api/event";
  import { store, toast, go, settings, patchSettings } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";

  interface Sample {
    path: string;
    duration: number;
  }

  /** Столько записей хватает для устойчивого распознавания. */
  const RECOMMENDED = 8;
  const MINIMUM = 3;

  let train = $state<Sample[]>([]);
  let test = $state<Sample[]>([]);
  let building = $state(false);
  let built = $state(false);
  let testFile = $state("");
  let testScore = $state<number | null>(null);
  let notice = $state("");
  interface PhraseReport {
    phrase: string;
    total: number;
    detected: number;
    worst_score: number;
    window_ms: number;
    samples_dir: string;
  }

  let phraseBusy = $state(false);
  let phraseReport = $state<PhraseReport | null>(null);

  const name = $derived(settings.assistant_name || "Джарвис");

  /**
   * Сгенерировать фразу-активацию разными дикторами и проверить, что
   * модель её слышит: сколько вариантов произнесения распознано буквально.
   */
  async function generatePhrase() {
    const phrase = (settings.wake_word ?? "").trim();
    if (!phrase) {
      toast("Впишите фразу-активацию", "err");
      return;
    }
    phraseBusy = true;
    try {
      const report = await invoke<PhraseReport>("wake_phrase_generate", { phrase });
      phraseReport = report;
      await patchSettings({ wake_window_ms: report.window_ms });
      toast(
        `Фраза услышана в ${report.detected} из ${report.total} вариантов`,
        report.detected === report.total ? "ok" : "err",
      );
    } catch (e) {
      toast(`Не удалось проверить фразу: ${e}`, "err");
    } finally {
      phraseBusy = false;
    }
  }

  const isNoise = (s: Sample) => s.path.includes("none");
  const phraseSamples = $derived(train.filter((s) => !isNoise(s)));
  const noiseSamples = $derived(train.filter(isNoise));

  const ready = $derived(phraseSamples.length >= MINIMUM);
  const progress = $derived(
    Math.min(100, Math.round((phraseSamples.length / RECOMMENDED) * 100)),
  );

  /** Оценку модели переводим в понятный вывод, а не в голое число. */
  const verdict = $derived.by(() => {
    const score = testScore;
    if (score === null) return null;
    const percent = Math.round(score * 100);
    if (score >= 0.75) return { tone: "ok", text: `Совпадение ${percent}% — Джарвис узнаёт вас` };
    if (score >= 0.5) return { tone: "warn", text: `Совпадение ${percent}%. Уже неплохо — добавьте ещё записей` };
    return { tone: "err", text: `Совпадение ${percent}%. Запишите образцы ближе к микрофону` };
  });

  async function refresh() {
    const s = await invoke<{ train: [string, number][]; test: [string, number][] }>("ww_samples");
    train = s.train.map(([path, duration]) => ({ path, duration }));
    test = s.test.map(([path, duration]) => ({ path, duration }));
  }

  async function record(target: string, kind: string) {
    if (store.recording) return;
    store.recording = true;
    notice = "";
    try {
      await invoke("ww_record", { target, kind });
    } catch (e) {
      store.recording = false;
      toast(String(e), "err");
    }
  }

  async function remove(path: string) {
    await invoke("ww_delete", { path });
    if (testFile === path) testFile = "";
    await refresh();
  }

  async function buildModel() {
    building = true;
    notice = "";
    try {
      await invoke("ww_build");
    } catch (e) {
      building = false;
      toast(String(e), "err");
    }
  }

  async function applyModel() {
    try {
      await invoke("ww_apply", { path: null });
      toast("Готово — Джарвис будет слышать вас лучше", "ok");
      notice = "Модель включена. Если что-то пойдёт не так, верните прежнюю настройку в settings.toml.";
    } catch (e) {
      toast(String(e), "err");
    }
  }

  async function runTest() {
    if (!testFile) return;
    try {
      testScore = await invoke<number>("ww_test", { path: testFile });
    } catch (e) {
      toast(String(e), "err");
    }
  }

  let unlisten: UnlistenFn | null = null;

  onMount(async () => {
    await refresh();
    if (test.length) testFile = test[0].path;
    unlisten = await listen<{ type: string }>("engine-event", (event) => {
      if (event.payload.type === "record_saved") {
        store.recording = false;
        refresh().then(() => {
          if (!testFile && test.length) testFile = test[0].path;
        });
      } else if (event.payload.type === "train_done") {
        building = false;
        built = true;
        notice = "Модель готова. Проверьте её и включите.";
      } else if (event.payload.type === "error") {
        building = false;
        store.recording = false;
      }
    });
  });

  onDestroy(() => {
    // Без отписки каждый вход на страницу вешал бы ещё один обработчик.
    unlisten?.();
    unlisten = null;
  });
</script>

<div class="page">
  <div class="page-head">
    <button class="icon-btn" style="align-self: flex-start" title="Назад (Esc)" aria-label="Назад" onclick={() => go(store.pageBack)}>
      <AppIcon name="chevronLeft" size={17} />
    </button>
    <div class="page-head__body">
      <h1 class="page-title">Wake word и голос — дополнительно</h1>
      <p class="page-sub">
        Ассистент срабатывает на фразу «{name}». Если он путает её с другой речью
        или не откликается — запишите несколько образцов своим голосом.
      </p>
    </div>
  </div>

  <div class="card card--outline" style="margin-bottom: 12px">
    <div class="card__title">Активационная фраза</div>
    <p class="card__hint">
      Любая фраза: «Джарвис», «Компьютер», «Робот». Детектор слушает
      ASR-модель и срабатывает, когда она услышала фразу буквально, — голос
      значения не имеет. Фраза — то же слово, что и имя ассистента, и
      применяется сразу после сохранения.
    </p>
    <label class="api-field">
      <span class="api-field__label">Фраза-активация</span>
      <input
        class="input"
        type="text"
        value={settings.wake_word}
        placeholder="Джарвис"
        onchange={(e) => {
          // Имя и фраза — одна переменная: сменил фразу, сменилось имя.
          const v = e.currentTarget.value.trim() || settings.assistant_name || "Джарвис";
          void patchSettings({ wake_word: v, assistant_name: v });
        }}
      />
    </label>
    <button class="btn btn--primary btn--block" disabled={phraseBusy} onclick={generatePhrase}>
      {#if phraseBusy}
        <span class="spinner"></span>
      {/if}
      {phraseBusy ? "Синтезирую…" : "Сгенерировать и проверить фразу"}
    </button>
    {#if phraseReport}
      <p class="card__hint">
        Услышано {phraseReport.detected} из {phraseReport.total} вариантов
        (худший счёт {phraseReport.worst_score.toFixed(3)}), окно детектора —
        {phraseReport.window_ms} мс.
      </p>
    {/if}
  </div>

  <div class="grid-2">
    <div>
      <div class="card card--outline">
        <div class="card__title">1 · Запишите обращение</div>
        <p class="card__hint">
          Скажите «Джарвис» естественно, как обычно. Запись остановится сама.
          Достаточно {MINIMUM} записей, лучше — {RECOMMENDED}.
        </p>

        <button
          class="btn btn--primary btn--block"
          disabled={store.recording}
          onclick={() => record("train", "wake")}
        >
          {#if store.recording}
            <span class="spinner"></span>
          {:else}
            <AppIcon name="mic" size={16} />
          {/if}
          {store.recording ? "Говорите…" : "Записать «Джарвис»"}
        </button>

        <div class="progress" style="margin-top: 12px">
          <div class="progress__fill" style={`width: ${progress}%`}></div>
        </div>
        <p class="hint">Записано {phraseSamples.length} из {RECOMMENDED}</p>

        {#if phraseSamples.length}
          <div class="samples">
            {#each phraseSamples as s, i (s.path)}
              <div class="samples__item">
                <span class="samples__name">Запись {i + 1}</span>
                <span class="samples__time">{s.duration.toFixed(1)} с</span>
                <button class="icon-btn" title="Удалить" aria-label={`Удалить запись ${i + 1}`} onclick={() => remove(s.path)}>
                  <AppIcon name="trash" size={15} />
                </button>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <div class="card card--outline">
        <div class="card__title">Фоновый шум <span class="dim" style="font-weight: 400">— необязательно</span></div>
        <p class="card__hint">
          Запишите пару фрагментов обычной обстановки: разговоры, телевизор, музыка.
          Так Джарвис реже срабатывает на чужую речь.
        </p>
        <button
          class="btn btn--secondary btn--block"
          disabled={store.recording}
          onclick={() => record("train", "none")}
        >
          <AppIcon name="waves" size={16} />
          Записать фон
        </button>

        {#if noiseSamples.length}
          <div class="samples">
            {#each noiseSamples as s, i (s.path)}
              <div class="samples__item">
                <span class="samples__name">Фон {i + 1}</span>
                <span class="samples__time">{s.duration.toFixed(1)} с</span>
                <button class="icon-btn" title="Удалить" aria-label={`Удалить фон ${i + 1}`} onclick={() => remove(s.path)}>
                  <AppIcon name="trash" size={15} />
                </button>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <div>
      <div class="card card--outline">
        <div class="card__title">2 · Соберите модель</div>
        <p class="card__hint">
          Джарвис сравнит ваши записи между собой и запомнит, как звучит именно
          ваше «Джарвис». Это занимает пару секунд.
        </p>
        <button class="btn btn--primary btn--block" disabled={building || !ready} onclick={buildModel}>
          {#if building}
            <span class="spinner"></span>
          {:else}
            <AppIcon name="sparkle" size={16} />
          {/if}
          {building ? "Собираю…" : built ? "Собрать заново" : "Собрать модель"}
        </button>
        {#if !ready}
          <p class="hint">Нужно ещё {MINIMUM - phraseSamples.length} записей.</p>
        {/if}
      </div>

      <div class="card card--outline">
        <div class="card__title">3 · Проверьте и включите</div>
        <p class="card__hint">
          Запишите проверочное «Джарвис» и посмотрите, насколько оно похоже на ваши образцы.
        </p>

        <div class="btn-row">
          <button
            class="btn btn--secondary"
            disabled={store.recording}
            onclick={() => record("test", "wake")}
          >
            <AppIcon name="mic" size={16} />
            Проверочная запись
          </button>
          <button class="btn btn--ghost" disabled={!testFile} onclick={runTest}>Проверить</button>
        </div>

        {#if test.length}
          <div class="samples">
            {#each test as s, i (s.path)}
              <div class="samples__item">
                <button
                  class="samples__name samples__pick"
                  class:is-active={testFile === s.path}
                  onclick={() => (testFile = s.path)}
                >
                  Проверка {i + 1}
                </button>
                <span class="samples__time">{s.duration.toFixed(1)} с</span>
                <button class="icon-btn" title="Удалить" aria-label={`Удалить проверку ${i + 1}`} onclick={() => remove(s.path)}>
                  <AppIcon name="trash" size={15} />
                </button>
              </div>
            {/each}
          </div>
        {/if}

        {#if verdict}
          <p class={`verdict verdict--${verdict.tone}`} role="status">{verdict.text}</p>
        {/if}

        <button class="btn btn--primary btn--block" style="margin-top: 12px" onclick={applyModel}>
          <AppIcon name="check" size={16} />
          Включить модель
        </button>
        {#if notice}
          <p class="hint">{notice}</p>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .samples {
    margin-top: 10px;
    border-top: 1px solid var(--border);
  }

  .samples__item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }

  .samples__item:last-child {
    border-bottom: none;
  }

  .samples__name {
    flex: 1;
    min-width: 0;
  }

  .samples__pick {
    background: none;
    border: none;
    text-align: left;
    font: inherit;
    color: var(--text-2);
    cursor: pointer;
    padding: 4px 0;
    border-radius: 6px;
  }

  .samples__pick.is-active {
    color: var(--accent-text);
    font-weight: 600;
  }

  .samples__time {
    color: var(--text-3);
    font-size: 12px;
  }

  .verdict {
    margin-top: 12px;
    padding: 9px 12px;
    border-radius: var(--r-md);
    font-size: 12.5px;
    line-height: 1.45;
  }

  .verdict--ok {
    background: var(--ok-soft);
    color: var(--ok);
  }

  .verdict--warn {
    background: var(--warn-soft);
    color: var(--warn);
  }

  .verdict--err {
    background: var(--danger-soft);
    color: var(--accent-text);
  }
</style>
