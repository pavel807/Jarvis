<script lang="ts">
  /** Строка настройки с переключателем: подпись, пояснение и свитч. */
  let {
    title,
    hint,
    checked,
    disabled = false,
    onchange,
  }: {
    title: string;
    hint?: string;
    checked: boolean;
    disabled?: boolean;
    onchange?: (value: boolean) => void;
  } = $props();
</script>

<div class="row-item">
  <span class="row-item__text">
    <span class="row-item__title">{title}</span>
    {#if hint}
      <span class="row-item__hint">{hint}</span>
    {/if}
  </span>
  <button
    class="switch"
    class:is-on={checked}
    role="switch"
    aria-checked={checked}
    aria-label={title}
    {disabled}
    onclick={() => onchange?.(!checked)}
  ></button>
</div>
