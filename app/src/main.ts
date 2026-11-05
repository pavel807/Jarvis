import { mount } from "svelte";
import { getCurrentWindow } from "@tauri-apps/api/window";
import App from "./App.svelte";
import UtilsWindow from "./UtilsWindow.svelte";
import "./style.css";

// Окно «Утилиты» (таймер/секундомер/календарь) грузит тот же бандл:
// различаем по метке окна Tauri. В браузере метки нет — обычное приложение.
function boot() {
  const target = document.getElementById("app");
  if (!target) return;

  try {
    if (getCurrentWindow().label === "utils") {
      mount(UtilsWindow, { target });
      return;
    }
  } catch {
    // вне Tauri — главный интерфейс
  }
  mount(App, { target });
}
boot();
