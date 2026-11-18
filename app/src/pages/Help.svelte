<script lang="ts">
  import { go, settings, store } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";

  /** Помощь: как пользоваться и что делать, если что-то не работает. */

  const wake = $derived(settings.wake_word || "Джарвис");

  const steps = $derived([
    { icon: "mic", title: `Скажите «${wake}»` },
    { icon: "commands", title: "Назовите команду или задайте вопрос" },
    { icon: "stop", title: "Скажите «замолчи», чтобы прервать ответ" },
  ]);

  const problems = $derived([
    {
      q: "Не реагирует на фразу-активацию",
      a: "Дайте доступ к микрофону в системных настройках",
    },
    {
      q: "Первый запуск идёт долго",
      a: "Скачиваются модели — это занимает до минуты",
    },
    {
      q: "Отвечает, но не слышно голоса",
      a: "Голос загружается в первую минуту после старта",
    },
    {
      q: "Не понимает нужную фразу",
      a: "Добавьте слова-триггеры в разделе «Команды»",
    },
    {
      q: "Срабатывает на чужую речь",
      a: "Проверьте микрофон и чувствительность в настройках",
    },
  ]);

  const examples = [
    "открой ютуб",
    "запусти монтаж",
    "какая погода в москве",
    "найди рецепт борща",
    "замолчи",
  ];
</script>

<div class="page">
  <div class="page-head">
    <button class="icon-btn" style="align-self: flex-start" title="Назад (Esc)" aria-label="Назад" onclick={() => go(store.pageBack)}>
      <AppIcon name="chevronLeft" size={17} />
    </button>
    <div class="page-head__body">
      <h1 class="page-title">Помощь</h1>
    </div>
  </div>

  <div class="grid-2">
    <div>
      <div class="section-title">Как пользоваться</div>
      <div class="card card--flush card--outline">
        {#each steps as step (step.title)}
          <div class="row-item">
            <span class="step-icon"><AppIcon name={step.icon} size={16} /></span>
            <span class="row-item__text">
              <span class="row-item__title">{step.title}</span>
            </span>
          </div>
        {/each}
      </div>

      <div class="section-title">Примеры фраз</div>
      <div class="card card--outline">
        <div class="chips">
          {#each examples as phrase (phrase)}
            <span class="chip">{phrase}</span>
          {/each}
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Если что-то не работает</div>
      <div class="card card--outline">
        {#each problems as item, i (item.q)}
          <div class="problem" class:is-first={i === 0}>
            <div class="problem__q">{item.q}</div>
            <div class="problem__a">{item.a}</div>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<style>
  .step-icon {
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: var(--cyan-soft);
    color: var(--cyan);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .problem {
    padding: 11px 0;
    border-top: 1px solid var(--border);
  }

  .problem.is-first {
    border-top: none;
    padding-top: 2px;
  }

  .problem__q {
    font-size: 13.5px;
    font-weight: 550;
    margin-bottom: 3px;
  }

  .problem__a {
    font-size: 12.5px;
    color: var(--text-3);
    line-height: 1.5;
  }
</style>
