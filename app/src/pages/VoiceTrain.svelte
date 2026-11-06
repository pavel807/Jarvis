<script lang="ts">
  import { store, go } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";
  import VoiceTraining from "../components/VoiceTraining.svelte";

  /**
   * Обучение голоса: диктант фразы для профиля фильтра «Только мой голос».
   * Отдельно от обучения wake word — здесь важен весь диапазон голоса,
   * а не только слово «Джарвис».
   */
</script>

<div class="page">
  <div class="page-head">
    <button class="icon-btn" style="align-self: flex-start" title="Назад (Esc)" aria-label="Назад" onclick={() => go(store.pageBack)}>
      <AppIcon name="chevronLeft" size={17} />
    </button>
    <div class="page-head__body">
      <h1 class="page-title">Обучение голоса</h1>
      <p class="page-sub">
        Профиль фильтра «Только мой голос»: Джарвис учится отличать тебя от
        посторонних. Одна-две записи достаточно; больше — точнее.
      </p>
    </div>
  </div>

  <VoiceTraining />
</div>
