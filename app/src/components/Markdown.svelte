<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { renderMarkdown } from "../lib/markdown";

  /** Ответ ассистента в Markdown. Ссылки открываются через open_external. */
  let { text = "" }: { text?: string } = $props();

  const html = $derived(renderMarkdown(text));

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
