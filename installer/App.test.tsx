// `nsh test` runs this file in a headless runner it builds with the crate.
// Только то, что не запускает установку: режимы, тумблеры, пути.

import { expect, open, test } from "nsh/test";

test("открывается и показывает пути", async () => {
  const app = await open("./App.tsx");
  expect(app.texts()).toContain("Jarvis");
  expect(app.texts()).toContain("Установить Jarvis");
  expect(app.texts().join("\n")).toContain("jarvis");
});

test("переключение на удаление и обратно", async () => {
  const app = await open("./App.tsx");
  expect(app.texts()).toContain("Собрать из исходников (если релиз недоступен)");
  await app.press("Удаление");
  expect(app.texts()).toContain("Удалить Jarvis");
  expect(app.texts()).toContain("Стереть данные, модели и плагины (там голосовые сэмплы)");
  await app.press("Установка");
  expect(app.texts()).toContain("Установить Jarvis");
});
