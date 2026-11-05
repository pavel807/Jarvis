/**
 * Снимок интерфейса в точном размере окна Tauri (400×820).
 *
 * Обычный `chrome --screenshot` подставляет свои размеры окна, поэтому
 * здесь размер задаётся принудительно через DevTools Protocol.
 *
 * Запуск (нужен `bun run dev` в app/):
 *   node dev/screenshot.mjs <файл.png> "<url>" [<js до снимка>]
 *
 * Пауза перед снимком задаётся переменной SHOT_DELAY (мс, по умолчанию 1800) —
 * маленькая пауза позволяет поймать анимацию появления в процессе.
 *
 * Пример:
 *   node dev/screenshot.mjs /tmp/home.png "http://localhost:1420/dev/preview.html?page=dashboard"
 */

import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

const PORT = process.env.CDP_PORT ?? "9222";
const SHOT_DELAY = Number(process.env.SHOT_DELAY ?? 1800);
const [, , outFile, url, preScript] = process.argv;

if (!outFile || !url) {
  console.error("Использование: node dev/screenshot.mjs <файл.png> <url> [js]");
  process.exit(1);
}

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = targets.find((t) => t.type === "page");
if (!page) throw new Error(`Нет открытой страницы. Запущен ли Chrome с --remote-debugging-port=${PORT}?`);

const ws = new WebSocket(page.webSocketDebuggerUrl);
let nextId = 0;
const pending = new Map();

function send(method, params = {}) {
  const id = ++nextId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
}

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  const slot = pending.get(msg.id);
  if (!slot) return;
  pending.delete(msg.id);
  if (msg.error) slot.reject(new Error(msg.error.message));
  else slot.resolve(msg.result);
};

await new Promise((resolve) => (ws.onopen = resolve));

await send("Page.enable");
// Жёстко фиксируем размер окна приложения, иначе Chrome подставит свой.
await send("Emulation.setDeviceMetricsOverride", {
  width: 400,
  height: 820,
  deviceScaleFactor: 2,
  mobile: false,
});
await send("Page.navigate", { url });
await new Promise((r) => setTimeout(r, SHOT_DELAY));

if (preScript) {
  await send("Runtime.evaluate", { expression: preScript, awaitPromise: true });
  await new Promise((r) => setTimeout(r, 600));
}

const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
await mkdir(dirname(outFile), { recursive: true });
await writeFile(outFile, Buffer.from(shot.data, "base64"));
console.log(`${outFile} — 400×820@2x`);
ws.close();
