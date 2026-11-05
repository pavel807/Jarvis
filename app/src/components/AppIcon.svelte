<script lang="ts">
  /**
   * Статичные иконки интерфейса в духе Lucide: сетка 24×24, штрих 1.7,
   * скруглённые концы. Наследуют цвет текста, поэтому не требуют
   * отдельных токенов для активного состояния.
   */
  let {
    name,
    size = 20,
    class: extraClass = "",
  }: { name: string; size?: number; class?: string } = $props();

  const ICONS: Record<string, string> = {
    home: '<path d="M4 10.6 12 4.2l8 6.4V19a1.6 1.6 0 0 1-1.6 1.6H5.6A1.6 1.6 0 0 1 4 19z"/><path d="M9.6 20.6v-5.4h4.8v5.4"/>',
    commands:
      '<path d="M4 6.5h16M4 12h16M4 17.5h9"/><circle cx="17.5" cy="17.5" r="2.2" fill="currentColor" stroke="none"/>',
    speaker:
      '<path d="M11.2 5.2 6.8 8.9H4.2v6.2h2.6l4.4 3.7z"/><path d="M15.2 9.6a3.6 3.6 0 0 1 0 4.8M18 7a7.2 7.2 0 0 1 0 10"/>',
    gear: '<path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 0 0 2.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 0 0 1.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 0 0-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 0 0-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 0 0-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 0 0-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 0 0 1.066-2.573c-.94-1.543.826-3.31 2.37-2.37c1-.608 2.296-.07 2.572 1.065z"/><circle cx="12" cy="12" r="3"/>',
    more: '<circle cx="5.5" cy="12" r="1.7" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.7" fill="currentColor" stroke="none"/><circle cx="18.5" cy="12" r="1.7" fill="currentColor" stroke="none"/>',
    grid: '<rect x="4" y="4" width="7" height="7" rx="1.8"/><rect x="13" y="4" width="7" height="7" rx="1.8"/><rect x="4" y="13" width="7" height="7" rx="1.8"/><rect x="13" y="13" width="7" height="7" rx="1.8"/>',
    chat: '<path d="M20 11.6a3.6 3.6 0 0 1-3.6 3.6H10l-4.6 4.2v-4.2A3.6 3.6 0 0 1 4 11.6v-4A3.6 3.6 0 0 1 7.6 4h8.8A3.6 3.6 0 0 1 20 7.6z"/>',
    help: '<circle cx="12" cy="12" r="8.8"/><path d="M9.4 9.4a2.7 2.7 0 1 1 3.4 2.9c-.7.2-1 .8-1 1.5v.4"/><circle cx="12" cy="17.1" r="1.05" fill="currentColor" stroke="none"/>',
    mic: '<path d="M12 15.4a3 3 0 0 0 3-3V6.2a3 3 0 0 0-6 0v6.2a3 3 0 0 0 3 3z"/><path d="M6.6 11.4a5.4 5.4 0 0 0 10.8 0M12 16.8v3.6"/>',
    micOff:
      '<path d="M15.2 9.6V6.2a3 3 0 0 0-5.7-1.3M9 9v3.2a3 3 0 0 0 4.6 2.5"/><path d="M6.6 11.4a5.4 5.4 0 0 0 8.5 4.4M12 16.8v3.6"/><path d="m4.5 4.5 15 15"/>',
    panelLeft:
      '<rect x="4" y="4.4" width="16" height="15.2" rx="2.4"/><path d="M9.6 4.4v15.2"/>',
    chevronRight: '<path d="m9.5 5.5 6.5 6.5-6.5 6.5"/>',
    chevronLeft: '<path d="m14.5 5.5-6.5 6.5 6.5 6.5"/>',
    chevronDown: '<path d="m5.5 9.5 6.5 6.5 6.5-6.5"/>',
    check: '<path d="m4.8 12.4 4.9 4.9L19.2 7"/>',
    close: '<path d="M6.4 6.4l11.2 11.2M17.6 6.4 6.4 17.6"/>',
    trash:
      '<path d="M4.2 7h15.6M9.6 7V5.6A1.6 1.6 0 0 1 11.2 4h1.6a1.6 1.6 0 0 1 1.6 1.6V7"/><path d="m6.6 7 .8 12.1a1.6 1.6 0 0 0 1.6 1.5h6a1.6 1.6 0 0 0 1.6-1.5L17.4 7"/>',
    pencil:
      '<path d="M14.2 5.6 18.4 9.8 8.2 20H4v-4.2z"/><path d="m12.6 7.2 4.2 4.2"/>',
    plus: '<path d="M12 5.2v13.6M5.2 12h13.6"/>',
    play: '<path d="M8.4 5.6v12.8L19 12z" fill="currentColor" stroke-linejoin="round"/>',
    pause: '<path d="M9.2 5.4v13.2M14.8 5.4v13.2"/>',
    stop: '<rect x="6.4" y="6.4" width="11.2" height="11.2" rx="2.2"/>',
    globe:
      '<circle cx="12" cy="12" r="8.6"/><path d="M3.4 12h17.2"/><path d="M12 3.4c2.2 2.4 3.3 5.3 3.3 8.6S14.2 18.2 12 20.6c-2.2-2.4-3.3-5.3-3.3-8.6S9.8 5.8 12 3.4z"/>',
    link: '<path d="M10.2 13.8a3.9 3.9 0 0 0 5.6 0l2.6-2.6a3.9 3.9 0 1 0-5.6-5.6l-1.4 1.4"/><path d="M13.8 10.2a3.9 3.9 0 0 0-5.6 0l-2.6 2.6a3.9 3.9 0 1 0 5.6 5.6l1.4-1.4"/>',
    folder:
      '<path d="M4 7.6A2.1 2.1 0 0 1 6.1 5.5h3.1a2 2 0 0 1 1.6.8l.9 1.2h6.2a2.1 2.1 0 0 1 2.1 2.1v7A2.1 2.1 0 0 1 17.9 19H6.1A2.1 2.1 0 0 1 4 16.9z"/>',
    power: '<path d="M12 4v7.6"/><path d="M7.4 7.2a6.8 6.8 0 1 0 9.2 0"/>',
    terminal: '<rect x="4" y="4.6" width="16" height="14.8" rx="2.4"/><path d="m7.6 9.6 3.4 3.2-3.4 3.2M13.2 16h4"/>',
    search: '<circle cx="11" cy="11" r="6.4"/><path d="m20.4 20.4-4.7-4.7"/>',
    sparkle:
      '<path d="M12 3.4c.7 4.6 3.4 7.3 8 8-4.6.7-7.3 3.4-8 8-.7-4.6-3.4-7.3-8-8 4.6-.7 7.3-3.4 8-8z"/><path d="M19 3.4c.2 1.2.9 1.9 2.1 2.1-1.2.2-1.9.9-2.1 2.1-.2-1.2-.9-1.9-2.1-2.1 1.2-.2 1.9-.9 2.1-2.1z"/>',
    info: '<circle cx="12" cy="12" r="8.6"/><path d="M12 11.2v5.2"/><circle cx="12" cy="7.9" r="1.05" fill="currentColor" stroke="none"/>',
    alert:
      '<path d="M12 4.6 20.8 19.4H3.2z"/><path d="M12 10.2v4"/><circle cx="12" cy="16.9" r="1" fill="currentColor" stroke="none"/>',
    waves: '<path d="M4 10v4M8 7.2v9.6M12 4.6v14.8M16 7.2v9.6M20 10v4"/>',
    arrowUp: '<path d="M12 19V5.4M6.4 11 12 5.4 17.6 11"/>',
    arrowDown: '<path d="M12 5v13.6M17.6 13 12 18.6 6.4 13"/>',
    copy: '<rect x="8.6" y="8.6" width="11.4" height="11.4" rx="2.4"/><path d="M5.4 15.4A2.4 2.4 0 0 1 4 13.2V6.4A2.4 2.4 0 0 1 6.4 4h6.8a2.4 2.4 0 0 1 2.2 1.4"/>',
    brain:
      '<path d="M9.4 4.4A2.6 2.6 0 0 0 6.8 7v.5a2.5 2.5 0 0 0-1.3 4.4 2.6 2.6 0 0 0 1.5 4.2 2.5 2.5 0 0 0 3.7 2.1V6.1a2.3 2.3 0 0 0-1.3-1.7z"/><path d="M14.6 4.4A2.6 2.6 0 0 1 17.2 7v.5a2.5 2.5 0 0 1 1.3 4.4 2.6 2.6 0 0 1-1.5 4.2 2.5 2.5 0 0 1-3.7 2.1"/>',
    grip: '<circle cx="9" cy="6" r="1.4" fill="currentColor" stroke="none"/><circle cx="15" cy="6" r="1.4" fill="currentColor" stroke="none"/><circle cx="9" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="15" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="9" cy="18" r="1.4" fill="currentColor" stroke="none"/><circle cx="15" cy="18" r="1.4" fill="currentColor" stroke="none"/>',
    sliders: '<path d="M5 5.5v5M5 15v3.5M12 5.5v2M12 11.5v7M19 5.5v8.5M19 17.5v1.5"/><circle cx="5" cy="13" r="2.1"/><circle cx="12" cy="9.5" r="2.1"/><circle cx="19" cy="16" r="2.1"/>',
    book: '<path d="M5 5.6A2.1 2.1 0 0 1 7.1 3.5H19v15.2H7.1A2.1 2.1 0 0 0 5 20.8z"/><path d="M5 18.8a2.1 2.1 0 0 1 2.1-2.1H19"/>',
    activity: '<path d="M3.6 12h3.2l2.4-6.4 4 12.8 2.4-6.4h4.8"/>',
    cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4" rx="1"/><path d="M9.5 3v3M14.5 3v3M9.5 18v3M14.5 18v3M3 9.5h3M3 14.5h3M18 9.5h3M18 14.5h3"/>',
    eraser: '<path d="m7.6 20-3.4-3.4a1.8 1.8 0 0 1 0-2.6l9-9a1.8 1.8 0 0 1 2.6 0l3.2 3.2a1.8 1.8 0 0 1 0 2.6l-9.2 9.2z"/><path d="m6.5 9.5 8 8M7.6 20h12"/>',
    core: '<path d="M12 3.6 20.6 18.6H3.4z"/><path d="M12 9.2 15.8 15.8H8.2z"/>',
    memory:
      '<ellipse cx="12" cy="6" rx="7.4" ry="2.8"/><path d="M4.6 6v6c0 1.55 3.31 2.8 7.4 2.8s7.4-1.25 7.4-2.8V6"/><path d="M4.6 12v6c0 1.55 3.31 2.8 7.4 2.8s7.4-1.25 7.4-2.8v-6"/>',
    send: '<path d="M20 4 10.4 13.6M20 4l-6.4 16-3.2-6.4L4 10.4z"/>',
  };
</script>

<svg
  class="icon {extraClass}"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.7"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
>
  {@html ICONS[name] ?? ""}
</svg>

<style>
  .icon {
    display: block;
    flex-shrink: 0;
  }
</style>
