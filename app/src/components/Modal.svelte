<script lang="ts">
  import type { Snippet } from "svelte";
  import AppIcon from "./AppIcon.svelte";

  /**
   * Модальное окно — центральный диалог для рабочего стола.
   * Портал в body: внутри страницы оно попало бы под скролл-контейнер.
   * Закрытие: Esc, клик по фону, крестик.
   */
  let {
    title,
    onclose,
    wide = false,
    children,
    headActions,
    foot,
  }: {
    title: string;
    onclose: () => void;
    wide?: boolean;
    children?: Snippet;
    headActions?: Snippet;
    foot?: Snippet;
  } = $props();

  let dialog = $state<HTMLElement | null>(null);

  $effect(() => {
    // Фокус внутрь: Esc и Tab работают сразу после открытия.
    dialog?.focus();
  });

  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape") onclose();
  }
</script>

<svelte:window onkeydown={onKey} />

<div
  class="modal-backdrop"
  role="presentation"
  onclick={(event) => {
    if (event.target === event.currentTarget) onclose();
  }}
>
  <div
    bind:this={dialog}
    class="modal"
    class:modal--wide={wide}
    role="dialog"
    aria-modal="true"
    aria-label={title}
    tabindex="-1"
  >
    <div class="modal__head">
      <div class="modal__title">{title}</div>
      {@render headActions?.()}
      <button class="icon-btn" title="Закрыть" aria-label="Закрыть" onclick={onclose}>
        <AppIcon name="close" size={17} />
      </button>
    </div>
    <div class="modal__body">
      {@render children?.()}
    </div>
    {#if foot}
      <div class="modal__foot">
        {@render foot()}
      </div>
    {/if}
  </div>
</div>
