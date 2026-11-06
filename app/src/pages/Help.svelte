<script lang="ts">
  import { go, store } from "../store.svelte";
  import AppIcon from "../components/AppIcon.svelte";

  /** Помощь: как пользоваться, типовые проблемы, файлы данных. */

  const steps = [
    {
      icon: "mic",
      title: "Скажите «Джарвис»",
      body: "Джарвис откликнется, а индикатор оживёт — значит, он слушает.",
    },
    {
      icon: "commands",
      title: "Назовите команду или спросите",
      body: "«Открой ютуб», «запусти монтаж», «какая сегодня погода». Список команд — в разделе «Команды».",
    },
    {
      icon: "stop",
      title: "Скажите «замолчи», если нужно прервать",
      body: "Ответ остановится сразу, и Джарвис снова начнёт ждать обращения. Той же кнопкой «Остановить» на главной.",
    },
  ];

  const problems = [
    {
      q: "Не реагирует на «Джарвис»",
      a: "Проверьте, что приложению разрешён доступ к микрофону в системных настройках. Затем пройдите обучение: «Голос» → «Обучение на свой голос» — он подстроится под ваш голос.",
    },
    {
      q: "Первый запуск идёт долго",
      a: "При первом запуске скачиваются модели — около 4 ГБ. Дальше Джарвис стартует за несколько секунд.",
    },
    {
      q: "Отвечает, но не слышно голоса",
      a: "Голос загружается в первую минуту после старта. Состояние видно в «Ещё» → «Состояние подсистем».",
    },
    {
      q: "Не понимает нужную фразу",
      a: "Откройте раздел «Команды» и добавьте свои слова-триггеры к нужной команде — они могут быть любыми.",
    },
    {
      q: "Срабатывает на чужую речь",
      a: "Запишите несколько фрагментов фонового шума в разделе обучения — Джарвис станет разборчивее. Там же поднимите порог чувствительности в настройках.",
    },
  ];

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
      <p class="page-sub">
        Коротко о том, как всё устроено и что делать, если что-то не работает.
      </p>
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
              <span class="row-item__hint">{step.body}</span>
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

      <div class="section-title">Где что лежит</div>
      <div class="card card--flush card--outline">
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title"><span class="mono">commands.toml</span> — команды</span>
            <span class="row-item__hint">
              Слова-триггеры и действия. Удобнее править на странице «Команды».
            </span>
          </span>
        </div>
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title"><span class="mono">settings.toml</span> — тонкая настройка</span>
            <span class="row-item__hint">
              Основные настройки есть в разделе «Настройки», остальное можно поправить
              в файле — изменения подхватятся на лету.
            </span>
          </span>
        </div>
        <div class="row-item">
          <span class="row-item__text">
            <span class="row-item__title"><span class="mono">models/</span> — модели</span>
            <span class="row-item__hint">
              Распознавание речи, синтез голоса и модель ответов. Всё считается на этом компьютере.
            </span>
          </span>
        </div>
      </div>
    </div>
  </div>

  <div class="section-title">Диагностика</div>
  <div class="card card--flush">
    <button class="row-item" onclick={() => go("system")}>
      <span class="row-item__icon"><AppIcon name="activity" size={16} /></span>
      <span class="row-item__text">
        <span class="row-item__title">Состояние подсистем</span>
        <span class="row-item__hint">
          Статусы движка, речи и микрофона — в разделе «Система»
        </span>
      </span>
      <span class="row-item__chevron"><AppIcon name="chevronRight" size={15} /></span>
    </button>
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
