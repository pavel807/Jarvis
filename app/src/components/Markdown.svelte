<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { renderMarkdown } from "../lib/markdown";

  /**
   * Ответ ассистента в Markdown. Ссылки открываются через open_external.
   * Пока ответ печатается (streaming), перерисовка не чаще STEPS_MS:
   * разбор всего текста на каждый токен замораживал интерфейс.
   */
  let {
    text = "",
    streaming = false,
  }: { text?: string; streaming?: boolean } = $props();

  const STEPS_MS = 120;

  let html = $state("");
  let lastAt = 0;
  let pending: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function renderNow(value: string) {
    html = renderMarkdown(value);
    lastAt = Date.now();
    pending = null;
  }

  $effect(() => {
    const current = text;
    if (!streaming) {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      pending = null;
      renderNow(current);
      return;
    }
    pending = current;
    if (timer) return;
    const since = Date.now() - lastAt;
    if (since >= STEPS_MS) {
      renderNow(pending);
    } else {
      timer = setTimeout(() => {
        timer = null;
        if (pending !== null) renderNow(pending);
      }, STEPS_MS - since);
    }
    return () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };
  });

  /**
   * Делегирование кликов по ссылкам через action: содержимое приходит
   * из {@html}, поэтому слушатель вешаем программно, а не в разметке.
   */
  function linkOpener(node: HTMLElement) {
    function onClick(event: MouseEvent) {
      const anchor = (event.target as HTMLElement).closest("a");
      if (!anchor) return;
      event.preventDefault();
      const href = anchor.getAttribute("href") ?? "";
      if (href.startsWith("http://") || href.startsWith("https://")) {
        void invoke("open_external", { url: href });
      }
    }
    node.addEventListener("click", onClick);
    return {
      destroy() {
        node.removeEventListener("click", onClick);
      },
    };
  }
</script>

<!-- eslint-disable-next-line svelte/no-at-html-tags -- содержимое экранируется в renderMarkdown -->
<div class="md" use:linkOpener>{@html html}</div>
