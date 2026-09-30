<script setup lang="ts">
// Анимированные SVG-иконки для заголовков страниц.
// name: mic | commands | speaker | chat | gear | help | wave
const props = defineProps<{ name: string }>();
</script>

<template>
  <!-- Микрофон с пульсирующими кольцами — wake word -->
  <svg v-if="props.name === 'mic'" class="picon" viewBox="0 0 24 24" fill="none">
    <circle class="pulse p1" cx="12" cy="12" r="6" />
    <circle class="pulse p2" cx="12" cy="12" r="6" />
    <rect x="9.5" y="6" width="5" height="9" rx="2.5" class="stroke" />
    <path d="M6.5 12a5.5 5.5 0 0 0 11 0M12 17.5V20" class="stroke" />
  </svg>

  <!-- Команды: строки списка, плавно пробегают -->
  <svg v-else-if="props.name === 'commands'" class="picon" viewBox="0 0 24 24" fill="none">
    <path d="M5 6h9M5 12h14M5 18h7" class="stroke thick" />
    <circle cx="19.5" cy="6" r="1.6" class="dot d1" />
    <circle cx="5" cy="12" r="1.6" class="dot d2" />
    <circle cx="16.5" cy="18" r="1.6" class="dot d3" />
  </svg>

  <!-- Динамик с волнами — голос -->
  <svg v-else-if="props.name === 'speaker'" class="picon" viewBox="0 0 24 24" fill="none">
    <path d="M4 9.5v5h3l4.5 4v-13L7 9.5H4z" class="fill" />
    <path class="wave w1" d="M15 9.5a4 4 0 0 1 0 5" />
    <path class="wave w2" d="M17.5 7.5a7 7 0 0 1 0 9" />
    <path class="wave w3" d="M20 5.5a10 10 0 0 1 0 13" />
  </svg>

  <!-- Чат: пузырь с печатающимися точками -->
  <svg v-else-if="props.name === 'chat'" class="picon" viewBox="0 0 24 24" fill="none">
    <path d="M4 6.5A3.5 3.5 0 0 1 7.5 3h9A3.5 3.5 0 0 1 20 6.5v6a3.5 3.5 0 0 1-3.5 3.5H10l-4.5 4v-4H7.5A3.5 3.5 0 0 1 4 12.5v-6z" class="stroke" />
    <circle class="tdot t1" cx="9" cy="9.5" r="1.2" />
    <circle class="tdot t2" cx="12.5" cy="9.5" r="1.2" />
    <circle class="tdot t3" cx="16" cy="9.5" r="1.2" />
  </svg>

  <!-- Шестерня — настройки -->
  <svg v-else-if="props.name === 'gear'" class="picon" viewBox="0 0 24 24" fill="none">
    <g class="gear">
      <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z" class="stroke" />
      <path
        d="M12 2.8v2.4M12 18.8v2.4M21.2 12h-2.4M5.2 12H2.8M18.5 5.5l-1.7 1.7M7.2 16.8l-1.7 1.7M18.5 18.5l-1.7-1.7M7.2 7.2L5.5 5.5"
        class="stroke thick"
      />
    </g>
  </svg>

  <!-- Вопрос — помощь -->
  <svg v-else-if="props.name === 'help'" class="picon" viewBox="0 0 24 24" fill="none">
    <circle class="stroke halo" cx="12" cy="12" r="9" />
    <path d="M9.5 9.2a2.6 2.6 0 1 1 3.6 2.4c-.8.35-1.1.9-1.1 1.9" class="stroke" />
    <circle cx="12" cy="16.6" r="1.1" class="fill" />
  </svg>

  <!-- Эквалайзер — дашборд / LLM -->
  <svg v-else class="picon" viewBox="0 0 24 24" fill="none">
    <path class="eq b1" d="M4 10v4" />
    <path class="eq b2" d="M8 7v10" />
    <path class="eq b3" d="M12 4v16" />
    <path class="eq b4" d="M16 7v10" />
    <path class="eq b5" d="M20 10v4" />
  </svg>
</template>

<style scoped>
.picon {
  width: 22px;
  height: 22px;
  vertical-align: -5px;
  margin-right: 8px;
  overflow: visible;
}

.stroke {
  stroke: var(--cyan);
  stroke-width: 1.6;
  stroke-linecap: round;
  stroke-linejoin: round;
  fill: none;
}
.thick { stroke-width: 2; }
.fill { fill: var(--cyan); }
.halo { opacity: 0.5; }

/* микрофон: пульсирующие кольца */
.pulse {
  stroke: var(--cyan);
  stroke-width: 1;
  fill: none;
  opacity: 0;
  transform-origin: 12px 12px;
  animation: mic-pulse 2.2s ease-out infinite;
}
.p2 { animation-delay: 1.1s; }
@keyframes mic-pulse {
  0% { transform: scale(0.5); opacity: 0.7; }
  100% { transform: scale(2); opacity: 0; }
}

/* команды: точки-маркеры */
.dot {
  fill: var(--green);
  animation: dot-blink 1.8s ease-in-out infinite;
}
.d2 { animation-delay: 0.6s; }
.d3 { animation-delay: 1.2s; }
@keyframes dot-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.2; }
}

/* динамик: волны с задержкой */
.wave {
  stroke: var(--cyan);
  stroke-width: 1.6;
  stroke-linecap: round;
  fill: none;
  opacity: 0.15;
  animation: wave-fade 1.6s ease-in-out infinite;
}
.w2 { animation-delay: 0.25s; }
.w3 { animation-delay: 0.5s; }
@keyframes wave-fade {
  0%, 100% { opacity: 0.15; }
  50% { opacity: 1; }
}

/* чат: печатающиеся точки */
.tdot {
  fill: var(--cyan);
  animation: typing 1.4s ease-in-out infinite;
}
.t2 { animation-delay: 0.2s; }
.t3 { animation-delay: 0.4s; }
@keyframes typing {
  0%, 100% { opacity: 0.2; }
  50% { opacity: 1; }
}

/* шестерня */
.gear {
  transform-origin: 12px 12px;
  animation: spin 9s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* эквалайзер */
.eq {
  stroke: var(--cyan);
  stroke-width: 2;
  stroke-linecap: round;
  animation: eq 1.1s ease-in-out infinite;
}
.b1, .b5 { animation-delay: 0.15s; }
.b2, .b4 { animation-delay: 0.3s; }
.b3 { animation-delay: 0.45s; }
@keyframes eq {
  0%, 100% { transform: scaleY(0.55); }
  50% { transform: scaleY(1.25); }
}
.eq { transform-origin: center; transform-box: fill-box; }
</style>
