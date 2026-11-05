<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { store, pushTurn, toast } from "../store.svelte";
  import AppIcon from "./AppIcon.svelte";

  /**
   * Командный ввод: вторичный способ общения после голоса.
   * Отправляет запрос в тот же llm_ask, что и голосовой путь.
   */

  let text = $state("");

  const busy = $derived(store.chat.busy);

  async function send() {
    const prompt = text.trim();
    if (!prompt || busy) return;
    pushTurn({ role: "user", text: prompt, source: "text" });
    store.chat.question = prompt;
    store.chat.done = "";
    store.chat.streaming = "";
    store.chat.error = "";
    store.chat.busy = true;
    text = "";
    try {
      await invoke("llm_ask", { prompt });
    } catch (e) {
      store.chat.busy = false;
      pushTurn({ role: "assistant", text: `Не получилось отправить запрос: ${e}`, source: "text", error: true });
    }
  }

  function stop() {
    void invoke("llm_cancel");
  }
</script>

<div class="cmd-input">
  <input
    bind:value={text}
    class="cmd-input__field"
    type="text"
    placeholder={busy ? "Джарвис отвечает…" : "Команда или вопрос…"}
    aria-label="Текстовая команда Джарвису"
    disabled={!store.status.engineRunning}
    onkeydown={(event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void send();
      }
    }}
  />
  {#if busy}
    <button
      class="icon-btn icon-btn--danger cmd-input__send"
      title="Остановить (Esc)"
      aria-label="Остановить ответ"
      onclick={stop}
    >
      <AppIcon name="stop" size={15} />
    </button>
  {:else}
    <button
      class="btn btn--primary cmd-input__send"
      disabled={!text.trim() || !store.status.engineRunning}
      title="Отправить (Enter)"
      aria-label="Отправить"
      onclick={send}
    >
      <AppIcon name="send" size={15} />
    </button>
  {/if}
</div>
