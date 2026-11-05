<script lang="ts">
  import { store, go } from "../store.svelte";
  import AppIcon from "./AppIcon.svelte";

  /** Шапка Mark V: слово системы, производитель, статус связи с движком. */


  const linkOn = $derived(store.status.engineRunning);
  const linkLabel = $derived(linkOn ? "ONLINE" : "LINK…");
</script>

<header class="header">
  <div class="header__id">
    <span class="header__wordmark">J.A.R.V.I.S</span>
    <span class="header__corp">SOSHIMI SOFTWARE</span>
  </div>
  <div class="header__spacer"></div>
  <span class="header__link" class:is-on={linkOn} title={linkOn ? "Движок работает" : "Движок запускается"}>
    <span class="dot"></span>
    {linkLabel}
  </span>
  <button
    class="icon-btn"
    title="Настройки"
    aria-label="Открыть настройки"
    onclick={() => go("settings")}
  >
    <AppIcon name="gear" size={17} />
  </button>
</header>
