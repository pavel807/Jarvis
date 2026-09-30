<script setup lang="ts">
import PageIcon from "../components/PageIcon.vue";

const steps = [
  {
    title: "Активация",
    body: "Скажите «Джарвис» — реактор ускорится и станет ярче: идёт окно прямого слушания (~8 секунд).",
  },
  {
    title: "Команды",
    body: "Говорите команду из списка на странице «Команды»: «открой ютуб», «запусти монтажа», «найди погоду в Москве»… Порядок в списке задаёт приоритет.",
  },
  {
    title: "Любые вопросы",
    body: "Если команда не распознана, вопрос уходит в локальную Qwen3-1.7B — интернет не нужен. Ответ озвучивается по предложениям.",
  },
  {
    title: "Прервать",
    body: "Скажите «замолчи» / «стоп» — остановит озвучку и генерацию, вернёт к режиму wake word.",
  },
  {
    title: "Своё wake word",
    body: "Страница «Wake Word»: запишите 6–10 сэмплов фразы и пару фоновых шумов, столько же в проверочные, обучите модель и нажмите «Применить».",
  },
  {
    title: "Горячие конфиги",
    body: "commands.toml и settings.toml перечитываются на лету — правки применяются без перезапуска.",
  },
];
</script>

<template>
  <h1><PageIcon name="help" /> Помощь</h1>

  <div class="card">
    <h2>Как пользоваться</h2>
    <table>
      <tbody>
        <tr v-for="(s, i) in steps" :key="i">
          <td style="width: 170px"><span class="badge ok">{{ s.title }}</span></td>
          <td>{{ s.body }}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Подсистемы</h2>
    <div class="row" style="gap: 16px">
      <span class="badge">Wake word: RustPotter</span>
      <span class="badge">STT: GigaAM v3</span>
      <span class="badge">TTS: Qwen3-TTS</span>
      <span class="badge">LLM: Qwen3-1.7B (Metal)</span>
    </div>
    <p class="dim" style="margin-top: 12px">
      Всё работает локально. Модели лежат в <span class="mono">models/</span>,
      конфиги — <span class="mono">commands.toml</span> и <span class="mono">settings.toml</span>.
    </p>
  </div>

  <div class="card">
    <h2>Если что-то не так</h2>
    <table>
      <tbody>
        <tr>
          <td style="width: 260px">Не слышит «Джарвис»</td>
          <td class="dim">
            Проверьте микрофон в системных настройках; на странице «Wake Word»
            обучите модель на своём голосе и снизьте порог в «Настройках».
          </td>
        </tr>
        <tr>
          <td>LLM: loading слишком долго</td>
          <td class="dim">Первый запуск качает ~3.4 ГБ весов. Дальше загрузка — секунды.</td>
        </tr>
        <tr>
          <td>Нет звука ответов</td>
          <td class="dim">TTS-модель грузится первой минутой после старта — статус виден на дашборде.</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
