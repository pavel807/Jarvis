<script setup lang="ts">
// Порт arc-реактора (исходник: Svelte/SCSS). Состояния:
// disconnected | idle | active — управляются пропсом state.
const props = withDefaults(
  defineProps<{
    state?: "disconnected" | "idle" | "active";
    theme?: "cyan" | "white" | "blue" | "green" | "red";
  }>(),
  { state: "idle", theme: "cyan" },
);

const marks = Array.from({ length: 60 }, (_, i) => ({
  style: {
    transform: `rotate(${i * 6}deg) translateY(-44.333cqw)`,
  },
}));

const coils = Array.from({ length: 8 }, (_, i) => ({
  cls: `coil-${i + 1}`,
  style: { transform: `rotate(${i * 45}deg)` },
}));
</script>

<template>
  <div class="reactor-container" :class="[props.state, `arc-${props.theme}`]">
    <div class="reactor-container-inner circle abs-center">
      <ul class="marks">
        <li v-for="(m, i) in marks" :key="i" :style="m.style" />
      </ul>
      <div class="e7">
        <div class="semi_arc_3 e5_1">
          <div class="semi_arc_3 e5_1_ghost" />
          <div class="semi_arc_3 e5_2">
            <div class="semi_arc_3 e5_2_ghost" />
            <div class="semi_arc_3 e5_3">
              <div class="semi_arc_3 e5_3_ghost" />
              <div class="semi_arc_3 e5_4">
                <div class="semi_arc_3 e5_4_ghost" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="tunnel circle abs-center" />
    <div class="core-wrapper circle abs-center" />
    <div class="core-outer circle abs-center" />
    <div class="core-inner circle abs-center" />
    <div class="coil-container">
      <div v-for="c in coils" :key="c.cls" class="coil circle" :class="c.cls" :style="c.style" />
    </div>
  </div>
</template>

<style scoped>
/* [ ПЕРЕМЕННЫЕ РЕАКТОРА ] */
.reactor-container {
  --arc-color: 2, 254, 255;
  --arc-glow: #52fefe;
  --arc-core-border: #1b4e5f;
  --arc-core-bg: #073c4b;

  width: 100cqw;
  height: 100cqw;
  margin: auto;
  position: relative;
  border-radius: 50%;
  transition: transform 0.5s ease, opacity 0.5s ease, filter 0.5s ease;
  transform: scale(0.95);
  opacity: 0.9;
}

.reactor-container.arc-cyan {
  --arc-color: 2, 254, 255;
  --arc-glow: #52fefe;
  --arc-core-border: #1b4e5f;
  --arc-core-bg: #073c4b;
}
.reactor-container.arc-white {
  --arc-color: 255, 255, 255;
  --arc-glow: #ffffff;
  --arc-core-border: #4a4a4a;
  --arc-core-bg: #2a2a2a;
}
.reactor-container.arc-blue {
  --arc-color: 50, 150, 255;
  --arc-glow: #3296ff;
  --arc-core-border: #1b3c5f;
  --arc-core-bg: #072a4b;
}
.reactor-container.arc-green {
  --arc-color: 50, 255, 100;
  --arc-glow: #32ff64;
  --arc-core-border: #1b5f2a;
  --arc-core-bg: #074b15;
}
.reactor-container.arc-red {
  --arc-color: 255, 50, 50;
  --arc-glow: #ff5050;
  --arc-core-border: #5f1b1b;
  --arc-core-bg: #4b0707;
}

.circle { border-radius: 50%; }
.abs-center {
  position: absolute;
  top: 0; right: 0; bottom: 0; left: 0;
  margin: auto;
}
ul { list-style: none; margin: 0; padding: 0; }

/* [ ФОНОВОЕ СВЕЧЕНИЕ ] */
.reactor-container::before {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 333.333cqw;
  height: 333.333cqw;
  border-radius: 50%;
  background: radial-gradient(
    circle,
    rgba(var(--arc-color), 0.2) 0%,
    rgba(var(--arc-color), 0.15) 30%,
    rgba(var(--arc-color), 0.1) 40%,
    rgba(var(--arc-color), 0.04) 50%,
    transparent 70%
  );
  z-index: -1;
  pointer-events: none;
  transition: opacity 0.5s ease, transform 0.5s ease;
}

/* [ ЯДРО ] */
.reactor-container-inner {
  height: 79.333cqw;
  width: 79.333cqw;
  background-color: #161a1b;
  box-shadow: 0 0 50px 15px rgba(var(--arc-color), 0.3),
    inset 0 0 50px 15px rgba(var(--arc-color), 0.3);
  transition: box-shadow 0.5s ease;
}

.core-inner {
  width: 23.333cqw;
  height: 23.333cqw;
  border: 1.667cqw solid var(--arc-core-border);
  background-color: #ffffff;
  box-shadow: 0 0 7px 5px var(--arc-glow), 0 0 10px 10px var(--arc-glow) inset;
  transition: box-shadow 0.5s ease;
}

.core-outer {
  width: 40cqw;
  height: 40cqw;
  border: 0.3333cqw solid var(--arc-glow);
  background-color: #ffffff;
  box-shadow: 0 0 2px 1px var(--arc-glow), 0 0 10px 5px var(--arc-glow) inset;
  transition: box-shadow 0.5s ease;
}

.core-wrapper {
  width: 60cqw;
  height: 60cqw;
  background-color: var(--arc-core-bg);
  box-shadow: 0 0 5px 4px var(--arc-glow), 0 0 6px 2px var(--arc-glow) inset;
  transition: box-shadow 0.5s ease;
}

.tunnel {
  width: 73.333cqw;
  height: 73.333cqw;
  background-color: #ffffff;
  box-shadow: 0 0 5px 1px var(--arc-glow), 0 0 5px 4px var(--arc-glow) inset;
  transition: box-shadow 0.5s ease;
}

/* [ КАТУШКИ ] */
.coil-container {
  position: relative;
  width: 100%;
  height: 100%;
  animation: reactor-anim 10s infinite linear;
}

.coil {
  position: absolute;
  width: 10cqw;
  height: 6.667cqw;
  top: calc(50% - 36.667cqw);
  left: calc(50% - 5cqw);
  transform-origin: 5cqw 36.667cqw;
  background-color: var(--arc-core-bg);
  box-shadow: 0 0 5px var(--arc-glow) inset;
}

@keyframes reactor-anim {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* [ ДУГИ ] */
.e7 {
  position: absolute;
  z-index: 1;
  width: 195%;
  height: 195%;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  border: 2cqw solid transparent;
  border-radius: 50%;
  transition: opacity 0.5s ease;
  opacity: 0.5;
}

.semi_arc_3 {
  position: absolute;
  width: 93%;
  height: 93%;
  left: calc((100% - 93%) / 2);
  top: calc((100% - 93%) / 2);
  border-style: solid;
  border-color: transparent;
  border-radius: 50%;
  box-sizing: border-box;
  animation: rotate 6s linear infinite;
  overflow: hidden;
}

.e5_1 {
  border-width: 0.3333cqw;
  border-top-color: rgba(var(--arc-color), 0.25);
  border-left-color: rgba(var(--arc-color), 0.25);
  animation: rotate 8s linear infinite;
}
.e5_2 {
  border-width: 0.6667cqw;
  border-right-color: rgba(var(--arc-color), 0.4);
  border-bottom-color: rgba(var(--arc-color), 0.4);
  animation: rotate_anti 6s linear infinite;
}
.e5_3 {
  border-width: 1cqw;
  border-top-color: rgba(var(--arc-color), 0.6);
  animation: rotate 4s linear infinite;
}
.e5_4 {
  border-width: 1.3333cqw;
  border-bottom-color: rgba(var(--arc-color), 0.8);
  animation: rotate_anti 5s linear infinite;
}

.e5_1_ghost {
  border-width: 0.3333cqw;
  border-top-color: rgba(var(--arc-color), 0.08);
  border-right-color: rgba(var(--arc-color), 0.08);
  animation: rotate_anti 12s linear infinite;
}
.e5_2_ghost {
  border-width: 0.6667cqw;
  border-top-color: rgba(var(--arc-color), 0.12);
  border-left-color: rgba(var(--arc-color), 0.12);
  animation: rotate 9s linear infinite;
}
.e5_3_ghost {
  border-width: 1cqw;
  border-right-color: rgba(var(--arc-color), 0.18);
  border-bottom-color: rgba(var(--arc-color), 0.18);
  animation: rotate_anti 6s linear infinite;
}
.e5_4_ghost {
  border-width: 1.3333cqw;
  border-top-color: rgba(var(--arc-color), 0.24);
  border-left-color: rgba(var(--arc-color), 0.24);
  animation: rotate 7.5s linear infinite;
}

@keyframes rotate {
  0% { transform: rotateZ(0deg); }
  100% { transform: rotateZ(360deg); }
}
@keyframes rotate_anti {
  0% { transform: rotateZ(0deg); }
  100% { transform: rotateZ(-360deg); }
}

/* [ РИСКИ ] */
.marks {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
}

.marks li {
  width: 0.8333cqw;
  height: 3.667cqw;
  background: rgba(var(--arc-color), 0.8);
  position: absolute;
  left: 50%;
  top: 50%;
  margin-left: -0.4167cqw;
  margin-top: -1.8333cqw;
  animation: colour_ease2 3s infinite ease-in-out;
}

@keyframes colour_ease2 {
  0% { background: rgba(var(--arc-color), 1); }
  50% { background: rgba(var(--arc-color), 0.3); }
  100% { background: rgba(var(--arc-color), 1); }
}

/* [ СОСТОЯНИЕ: DISCONNECTED ] */
.reactor-container.disconnected {
  transform: scale(0.85);
  opacity: 0.4;
  filter: grayscale(0.7) brightness(0.6);
}
.reactor-container.disconnected .coil-container { animation-duration: 20s; }
.reactor-container.disconnected .e7 { opacity: 0.3; }
.reactor-container.disconnected .e5_1 {
  border-top-color: rgba(var(--arc-color), 0.1);
  border-left-color: rgba(var(--arc-color), 0.1);
  animation: rotate 35s linear infinite;
}
.reactor-container.disconnected .e5_2 {
  border-right-color: rgba(var(--arc-color), 0.15);
  border-bottom-color: rgba(var(--arc-color), 0.15);
  animation: rotate_anti 30s linear infinite;
}
.reactor-container.disconnected .e5_3 {
  border-top-color: rgba(var(--arc-color), 0.2);
  animation: rotate 25s linear infinite;
}
.reactor-container.disconnected .e5_4 {
  border-bottom-color: rgba(var(--arc-color), 0.25);
  animation: rotate_anti 28s linear infinite;
}
.reactor-container.disconnected::before {
  opacity: 0.3;
  transform: translate(-50%, -50%) scale(0.7);
}

/* [ СОСТОЯНИЕ: IDLE ] */
.reactor-container.idle .e7 { opacity: 0.6; }
.reactor-container.idle .e5_1 {
  border-top-color: rgba(var(--arc-color), 0.2);
  border-left-color: rgba(var(--arc-color), 0.2);
  animation: rotate 9s linear infinite;
}
.reactor-container.idle .e5_2 {
  border-right-color: rgba(var(--arc-color), 0.35);
  border-bottom-color: rgba(var(--arc-color), 0.35);
  animation: rotate_anti 15s linear infinite;
}
.reactor-container.idle .e5_3 {
  border-top-color: rgba(var(--arc-color), 0.5);
  animation: rotate 12s linear infinite;
}
.reactor-container.idle .e5_4 {
  border-bottom-color: rgba(var(--arc-color), 0.65);
  animation: rotate_anti 14s linear infinite;
}

/* [ СОСТОЯНИЕ: ACTIVE ] */
.reactor-container.active {
  transform: scale(1.05);
  opacity: 1;
}
.reactor-container.active .coil-container { animation-duration: 3s; }
.reactor-container.active .e7 { opacity: 1; }
.reactor-container.active .e5_1 {
  border-top-color: rgba(var(--arc-color), 0.3);
  border-left-color: rgba(var(--arc-color), 0.3);
  animation: rotate 4s linear infinite;
}
.reactor-container.active .e5_2 {
  border-right-color: rgba(var(--arc-color), 0.5);
  border-bottom-color: rgba(var(--arc-color), 0.5);
  animation: rotate_anti 3s linear infinite;
}
.reactor-container.active .e5_3 {
  border-top-color: rgba(var(--arc-color), 0.7);
  animation: rotate 2s linear infinite;
}
.reactor-container.active .e5_4 {
  border-bottom-color: rgba(var(--arc-color), 0.9);
  animation: rotate_anti 2.5s linear infinite;
}
.reactor-container.active .reactor-container-inner {
  box-shadow: 0 0 70px 25px rgba(var(--arc-color), 0.3),
    inset 0 0 70px 25px rgba(var(--arc-color), 0.3);
}
.reactor-container.active .core-inner {
  box-shadow: 0 0 15px 10px var(--arc-glow), 0 0 20px 15px var(--arc-glow) inset;
}
.reactor-container.active .core-outer {
  box-shadow: 0 0 5px 3px var(--arc-glow), 0 0 15px 10px var(--arc-glow) inset;
}
.reactor-container.active .core-wrapper {
  box-shadow: 0 0 10px 8px var(--arc-glow), 0 0 10px 5px var(--arc-glow) inset;
}
.reactor-container.active .tunnel {
  box-shadow: 0 0 10px 3px var(--arc-glow), 0 0 10px 8px var(--arc-glow) inset;
}
.reactor-container.active .marks li {
  animation: colour_ease2_active 1s infinite ease-in-out;
}
.reactor-container.active::before {
  opacity: 1;
  transform: translate(-50%, -50%) scale(1.1);
  animation: bg-pulse 3s ease-in-out infinite;
}

@keyframes bg-pulse {
  0%, 100% { opacity: 1; transform: translate(-50%, -50%) scale(1.1); }
  50% { opacity: 0.85; transform: translate(-50%, -50%) scale(1.05); }
}
@keyframes colour_ease2_active {
  0% { background: rgba(var(--arc-color), 1); box-shadow: 0 0 5px rgba(var(--arc-color), 1); }
  50% { background: rgba(var(--arc-color), 0.3); box-shadow: none; }
  100% { background: rgba(var(--arc-color), 1); box-shadow: 0 0 5px rgba(var(--arc-color), 1); }
}
</style>
