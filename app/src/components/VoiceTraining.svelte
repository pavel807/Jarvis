<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { store, toast } from "../store.svelte";
  import AppIcon from "./AppIcon.svelte";

  /**
   * Диктант для профиля голоса: на экране фраза, покрывающая все звуки
   * русской речи. Пользователь читает её вслух и нажимает «Стоп» — запись
   * проверяется FastConformer'ом (прочитано ли то, что на экране) и уходит
   * в профиль фильтра «Только мой голос».
   */

  const PHRASE = `За окном светит солнце, и щебечут воробьи.
Ты слышишь этот шум?
Запиши в задачник: купить хлеб, молоко и шесть спелых яблок.
Открой жёлтую папку, найди чужой документ и проверь цифры: восемьсот сорок семь.
Маленькая Юля едет на юг в июле.
Что ты видишь на экране?
Повтори всё чётко и с выражением — громко, тихо, быстро и медленно.`;

  const state = $derived(store.voiceTrain.state);
  const lastOk = $derived(store.voiceTrain.lastOk);
  const overlap = $derived(store.voiceTrain.overlap);

  let busy = $state(false);

  async function start() {
    if (busy) return;
    busy = true;
    store.voiceTrain = { state: "recording", lastOk: null, overlap: 0, transcript: "" };
    try {
      await invoke("voice_record_start");
    } catch (e) {
      store.voiceTrain.state = "idle";
      toast(`Не получилось: ${e}`, "err");
    } finally {
      busy = false;
    }
  }

  async function stop() {
    if (busy) return;
    busy = true;
    store.voiceTrain.state = "checking";
    try {
      await invoke("voice_record_stop");
    } catch (e) {
      store.voiceTrain.state = "idle";
      toast(`Не получилось: ${e}`, "err");
    } finally {
      busy = false;
    }
  }
</script>

<div class="voice-training">
  <p class="vt-hint">
    Читай текст вслух в обычном темпе — это запись твоего голоса для фильтра
    «Только мой голос». Паузы между предложениями не страшны. Дочитал — нажми
    «Стоп».
  </p>

  <div class="card vt-phrase-card">
    <p class="vt-phrase">{PHRASE}</p>
  </div>

  <div class="vt-actions">
    {#if state === "recording"}
      <button class="btn btn--secondary" onclick={stop}>
        <span class="vt-dot"></span>
        Стоп
      </button>
      <span class="vt-live">Идёт запись…</span>
    {:else if state === "checking"}
      <span class="vt-live"><span class="spinner"></span> Проверяю запись…</span>
    {:else}
      <button class="btn btn--primary" onclick={start}>
        <AppIcon name="mic" size={15} />
        {lastOk === null ? "Начать чтение" : "Перечитать"}
      </button>
    {/if}
  </div>

  {#if lastOk !== null && state === "idle"}
    <div class="card vt-result" class:is-ok={lastOk} class:is-bad={!lastOk}>
      {#if lastOk}
        <p class="vt-result__title">Голос записан — совпадение {Math.round(overlap * 100)}%</p>
        <p class="vt-result__hint">
          Профиль фильтра «Только мой голос» обновится сам. Можно прочитать ещё
          раз — каждая запись делает фильтр точнее.
        </p>
      {:else}
        <p class="vt-result__title">Похоже, прочитано не то ({Math.round(overlap * 100)}% совпадение)</p>
        <p class="vt-result__hint">
          Читай именно текст с экрана, не спеша и погромче. Запись не сохранена —
          начни заново.
        </p>
        {#if store.voiceTrain.transcript}
          <p class="vt-transcript">Распознал: «{store.voiceTrain.transcript}»</p>
        {/if}
      {/if}
    </div>
  {/if}
</div>

<style>
  .voice-training {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .vt-hint {
    font-size: 13px;
    line-height: 1.5;
    opacity: 0.75;
    margin: 0;
  }

  .vt-phrase-card {
    padding: 16px 18px;
  }

  .vt-phrase {
    margin: 0;
    font-size: 16.5px;
    line-height: 1.65;
    white-space: pre-line;
    user-select: text;
  }

  .vt-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .vt-live {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    opacity: 0.8;
  }

  .vt-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #e5484d;
    animation: vt-pulse 1.2s ease-in-out infinite;
  }

  @keyframes vt-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
  }

  .vt-result {
    padding: 12px 14px;
  }

  .vt-result.is-ok {
    border-color: rgba(48, 180, 110, 0.4);
  }

  .vt-result.is-bad {
    border-color: rgba(229, 72, 77, 0.45);
  }

  .vt-result__title {
    margin: 0 0 4px;
    font-size: 14px;
    font-weight: 600;
  }

  .vt-result__hint {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.5;
    opacity: 0.75;
  }

  .vt-transcript {
    margin: 8px 0 0;
    font-size: 12px;
    line-height: 1.5;
    opacity: 0.6;
    font-style: italic;
  }
</style>
