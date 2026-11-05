/**
 * Выполняет JS на открытой странице через DevTools Protocol и печатает
 * результат. Нужен для проверки состояния интерфейса без скриншотов.
 *
 * Запуск (нужен Chrome с --remote-debugging-port=9222):
 *   node dev/eval.mjs "<выражение>"
 */

const PORT = process.env.CDP_PORT ?? "9222";
const [, , expression] = process.argv;

if (!expression) {
  console.error("Использование: node dev/eval.mjs \"<js-выражение>\"");
  process.exit(1);
}

const targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
const page = targets.find((t) => t.type === "page");
if (!page) throw new Error(`Нет открытой страницы на порту ${PORT}`);

const ws = new WebSocket(page.webSocketDebuggerUrl);
let nextId = 0;
const pending = new Map();

ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  }
});

function send(method, params = {}) {
  const id = ++nextId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve) => pending.set(id, resolve));
}

await new Promise((resolve) => ws.addEventListener("open", resolve));

const result = await send("Runtime.evaluate", {
  expression,
  awaitPromise: true,
  returnByValue: true,
});

if (result.result?.exceptionDetails) {
  console.error("Ошибка:", result.result.exceptionDetails.text);
  process.exit(1);
}
console.log(JSON.stringify(result.result?.result?.value, null, 2));
ws.close();
