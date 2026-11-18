import { Scroll, Text, View, mount, signal } from "nsh";
import type { TextStyle, ViewStyle } from "nsh";
import { currentWindow } from "nsh/desktop";
import { Installer } from "jarvis-installer/native";

// ── Состояние ──────────────────────────────────────────────────────────────

type Mode = "install" | "uninstall";
type StepState = "idle" | "run" | "ok" | "fail" | "skip";
interface Step {
  label: string;
  hint?: string;
  state: StepState;
}
interface Paths {
  repo: string;
  bin: string;
  cli: string;
  data: string;
  desktop: string;
}
interface Preflight {
  platform: string;
  asset: string;
  curl: boolean;
  bun: boolean;
  cargo: boolean;
  linux: boolean;
  deps: { name: string; ok: boolean }[];
  missing: string[];
}
interface Reply {
  ok: boolean;
  log: string;
  dir?: string;
}

const mode = signal<Mode>("install");
const running = signal(false);
const done = signal<null | "ok" | "fail">(null);
const purge = signal(false);
const buildFallback = signal(false);
const steps = signal<Step[]>([]);
const log = signal<string[]>([]);
const paths = signal<Paths | null>(null);

Installer.paths().then((r) => paths.set(JSON.parse(r)));

const say = (t: string) => log.update((l) => [...l, t]);
const setStep = (i: number, patch: Partial<Step>) =>
  steps.update((s) => s.map((x, j) => (j === i ? { ...x, ...patch } : x)));
const failRunning = () =>
  steps.update((s) => s.map((x) => (x.state === "run" ? { ...x, state: "fail" as StepState } : x)));

// ── Сценарии ───────────────────────────────────────────────────────────────

async function runInstall() {
  const fromSource = buildFallback();
  const list: Step[] = [
    { label: "Проверяю систему", state: "idle" },
    { label: "Ставлю системные библиотеки", state: "idle" },
    fromSource
      ? { label: "Собираю из исходников", hint: "bun tauri build — это долго", state: "idle" }
      : { label: "Скачиваю сборку", state: "idle" },
    { label: "Устанавливаю Jarvis", state: "idle" },
  ];
  steps.set(list);
  running.set(true);
  try {
    setStep(0, { state: "run" });
    const pf: Preflight = JSON.parse(await Installer.preflight());
    say(`${pf.platform} · ${pf.asset || "нет готовой сборки"}`);
    if (pf.linux && pf.deps.length) {
      say(pf.deps.map((d) => `  ${d.ok ? "✓" : "✗"} ${d.name}`).join("\n"));
    }
    if (fromSource && (!pf.bun || !pf.cargo)) {
      throw new Error("нет bun или cargo — для сборки из исходников нужны оба");
    }
    if (!fromSource && !pf.curl) throw new Error("curl не найден — без него не скачать сборку");
    const need = pf.missing ?? [];
    if (!pf.linux || need.length === 0) {
      setStep(0, { state: "ok" });
      setStep(1, { state: "skip" });
    } else {
      setStep(0, { state: "ok" });
      setStep(1, { state: "run", hint: need.join(", ") });
      const r: Reply = JSON.parse(await Installer.installPackages(need.join(" ")));
      say(r.log);
      if (!r.ok) throw new Error(r.log);
      setStep(1, { state: "ok" });
    }
    let src = "";
    if (fromSource) {
      setStep(2, { state: "run" });
      const b: Reply = JSON.parse(await Installer.build());
      say(b.log);
      if (!b.ok) throw new Error(b.log);
      setStep(2, { state: "ok" });
    } else {
      setStep(2, { state: "run", hint: pf.asset });
      const d: Reply = JSON.parse(await Installer.download());
      say(d.log);
      if (!d.ok) throw new Error(d.log);
      src = d.dir ?? "";
      setStep(2, { state: "ok" });
    }
    setStep(3, { state: "run" });
    const f: Reply = JSON.parse(await Installer.installFiles(src));
    say(f.log);
    if (!f.ok) throw new Error(f.log);
    setStep(3, { state: "ok" });
    done.set("ok");
  } catch (e) {
    say(String((e as Error)?.message ?? e));
    failRunning();
    done.set("fail");
  } finally {
    running.set(false);
  }
}

async function runUninstall() {
  const list: Step[] = [{ label: "Убираю бинарь и ярлык", state: "idle" }];
  if (purge()) list.push({ label: "Стираю данные, модели и плагины", state: "idle" });
  steps.set(list);
  running.set(true);
  try {
    setStep(0, { state: "run" });
    const r: Reply = JSON.parse(await Installer.uninstall(purge()));
    say(r.log);
    if (!r.ok) throw new Error(r.log);
    setStep(0, { state: "ok" });
    if (purge()) setStep(1, { state: "ok" });
    done.set("ok");
  } catch (e) {
    say(String((e as Error)?.message ?? e));
    failRunning();
    done.set("fail");
  } finally {
    running.set(false);
  }
}

function start() {
  if (running()) return;
  done.set(null);
  log.set([]);
  if (mode() === "install") runInstall();
  else runUninstall();
}

function switchMode(m: Mode) {
  if (running()) return;
  if (mode() === m) return;
  mode.set(m);
  steps.set([]);
  done.set(null);
  log.set([]);
}

// ── Оформление ─────────────────────────────────────────────────────────────
// Мир — ночной рабочий стол pro-приложений: глубокий чёрный, стекло, свет
// изнутри объекта. Один яркий момент — стеклянный орб-монограмма; хром
// тихий, нейтральные световые слои, никакого цветного неона.

const C = {
  bg: "#0a0a0c",
  surface: "rgba(255,255,255,0.05)",
  surface2: "rgba(255,255,255,0.08)",
  hairline: "rgba(255,255,255,0.1)",
  text: "#f5f5f7",
  secondary: "rgba(235,235,245,0.6)",
  muted: "rgba(235,235,245,0.3)",
  accent: "#0a84ff",
  ok: "#30d158",
  bad: "#ff453a",
  badHover: "#e83b31",
  btnLight: "#f5f5f7",
  btnLightHover: "#ffffff",
};

const V: Record<string, ViewStyle> = {
  root: {
    flex: 1,
    backgroundColor: C.bg,
    paddingTop: 40,
    paddingBottom: 28,
    paddingHorizontal: 32,
    justifyContent: "center",
    gap: 24,
  },
  hero: { alignItems: "center", gap: 16 },
  orbGlow: {
    width: 128,
    height: 128,
    alignItems: "center",
    justifyContent: "center",
    backgroundImage: "radial-gradient(circle at 50% 42%, rgba(255,255,255,0.08), transparent 68%)",
  },
  orb: {
    width: 84,
    height: 84,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.16)",
    backgroundColor: "rgba(255,255,255,0.05)",
    backgroundImage: "linear-gradient(145deg, rgba(255,255,255,0.11), rgba(255,255,255,0.02))",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: [
      { x: 0, y: 12, blur: 36, color: "#ffffff14" },
      { x: 0, y: 3, blur: 10, color: "#00000066" },
    ],
  },
  heroText: { alignItems: "center", gap: 5 },
  seg: {
    flexDirection: "row",
    backgroundColor: C.surface,
    borderRadius: 12,
    padding: 2,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.06)",
  },
  segItem: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 10,
    alignItems: "center",
    cursor: "pointer",
    transitionProperty: "background-color",
    transitionDuration: 180,
  },
  segItemOn: { backgroundColor: C.surface2 },
  panel: {
    gap: 10,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.06)",
  },
  pathRow: { flexDirection: "row", gap: 12, alignItems: "center" },
  pathKey: { width: 64 },
  pathVal: { flex: 1 },
  toggles: { gap: 14 },
  toggleRow: { flexDirection: "row", alignItems: "center", gap: 12, cursor: "pointer" },
  toggleTrack: {
    width: 44,
    height: 26,
    borderRadius: 13,
    backgroundColor: C.surface2,
    padding: 2,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.06)",
    transitionProperty: "background-color",
    transitionDuration: 180,
  },
  toggleTrackOn: { backgroundColor: C.ok },
  toggleKnob: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: "#ffffff",
    boxShadow: [{ x: 0, y: 1, blur: 4, color: "#00000055" }],
    transitionProperty: "transform",
    transitionDuration: 180,
    transitionTimingFunction: "ease-out",
  },
  toggleKnobOn: { transform: [{ translateX: 18 }] },
  stepRows: { gap: 2 },
  stepRow: { flexDirection: "row", alignItems: "center", gap: 14, paddingVertical: 7 },
  stepBody: { flex: 1 },
  stepIcon: {
    width: 26,
    height: 26,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.14)",
    alignItems: "center",
    justifyContent: "center",
  },
  stepIconRun: { borderColor: "rgba(10,132,255,0.5)" },
  stepIconOk: { borderColor: "rgba(48,209,88,0.45)", backgroundColor: "rgba(48,209,88,0.1)" },
  stepIconFail: { borderColor: "rgba(255,69,58,0.45)", backgroundColor: "rgba(255,69,58,0.1)" },
  spinner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: C.accent,
    animationName: "pulse",
    animationDuration: "1.1s",
    animationIterationCount: "infinite",
  },
  progressTrack: {
    height: 3,
    borderRadius: 1.5,
    backgroundColor: "rgba(255,255,255,0.08)",
    overflow: "hidden",
  },
  progressFill: {
    height: 3,
    borderRadius: 1.5,
    backgroundColor: C.btnLight,
    transitionProperty: "width",
    transitionDuration: 400,
    transitionTimingFunction: "ease-out",
  },
  button: {
    borderRadius: 14,
    paddingVertical: 15,
    alignItems: "center",
    cursor: "pointer",
    transitionProperty: "background-color, opacity",
    transitionDuration: 150,
  },
  buttonOff: { opacity: 0.35 },
  bannerOk: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(48,209,88,0.3)",
    backgroundColor: "rgba(48,209,88,0.08)",
    padding: 14,
    gap: 3,
  },
  bannerFail: {
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255,69,58,0.3)",
    backgroundColor: "rgba(255,69,58,0.07)",
    padding: 14,
    gap: 3,
  },
  consoleBox: {
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.06)",
    maxHeight: 140,
    overflow: "hidden",
  },
  footer: { alignItems: "center" },
};

const T: Record<string, TextStyle> = {
  orbText: { fontSize: 30, fontWeight: "600", color: C.text },
  title: { fontSize: 28, fontWeight: "600", color: C.text, letterSpacing: -0.4 },
  subtitle: { fontSize: 15, color: C.secondary },
  segText: { fontSize: 14, fontWeight: "500", color: C.secondary },
  segTextOn: { color: C.text },
  pathKey: { fontSize: 13, color: C.muted },
  pathVal: { fontSize: 12, color: C.secondary, fontFamily: "monospace" },
  stepsTitle: { fontSize: 12, color: C.muted },
  stepNum: { fontSize: 11, color: C.secondary, fontWeight: "500" },
  stepGlyph: { fontSize: 12, fontWeight: "600" },
  stepLabel: { fontSize: 15, color: C.text },
  stepLabelDim: { color: C.muted },
  stepHint: { fontSize: 12, color: C.secondary, marginTop: 1 },
  buttonText: { fontSize: 17, fontWeight: "600", color: "#0a0a0c" },
  buttonTextDanger: { color: "#ffffff" },
  bannerTitle: { fontSize: 15, fontWeight: "600", color: C.ok },
  bannerTitleFail: { color: C.bad },
  bannerNote: { fontSize: 13, color: C.secondary },
  toggleLabel: { fontSize: 14, color: C.text, flex: 1 },
  consoleText: {
    fontSize: 11,
    fontFamily: "monospace",
    color: C.secondary,
    lineHeight: 17,
    padding: 12,
  },
  footerText: { fontSize: 12, color: C.muted, textAlign: "center", lineHeight: 18 },
};

// ── Куски интерфейса ───────────────────────────────────────────────────────

function Segmented() {
  const items: [Mode, string][] = [
    ["install", "Установка"],
    ["uninstall", "Удаление"],
  ];
  return (
    <View style={V.seg}>
      {items.map(([m, label]) => (
        <View
          testID={`mode-${m}`}
          style={[V.segItem, mode() === m && V.segItemOn]}
          activeStyle={{ opacity: 0.7 }}
          onPress={() => switchMode(m)}
        >
          <Text style={[T.segText, mode() === m && T.segTextOn]}>{label}</Text>
        </View>
      ))}
    </View>
  );
}

function Toggle(props: { on: () => boolean; flip: () => void; label: string }) {
  return (
    <View style={V.toggleRow} onPress={props.flip} activeStyle={{ opacity: 0.85 }}>
      <View style={[V.toggleTrack, props.on() && V.toggleTrackOn]}>
        <View style={[V.toggleKnob, props.on() && V.toggleKnobOn]} />
      </View>
      <Text style={T.toggleLabel}>{props.label}</Text>
    </View>
  );
}

function StepIcon(props: { step: Step; index: number }) {
  const s = props.step;
  if (s.state === "run")
    return (
      <View style={[V.stepIcon, V.stepIconRun]}>
        <View style={V.spinner} />
      </View>
    );
  if (s.state === "ok")
    return (
      <View style={[V.stepIcon, V.stepIconOk]}>
        <Text style={[T.stepGlyph, { color: C.ok }]}>✓</Text>
      </View>
    );
  if (s.state === "fail")
    return (
      <View style={[V.stepIcon, V.stepIconFail]}>
        <Text style={[T.stepGlyph, { color: C.bad }]}>✕</Text>
      </View>
    );
  if (s.state === "skip")
    return (
      <View style={V.stepIcon}>
        <Text style={[T.stepGlyph, { color: C.muted }]}>—</Text>
      </View>
    );
  return (
    <View style={V.stepIcon}>
      <Text style={T.stepNum}>{props.index + 1}</Text>
    </View>
  );
}

function Steps() {
  return (
    <View style={{ gap: 12 }}>
      <Text style={T.stepsTitle}>Ход установки</Text>
      <View style={V.stepRows}>
        <Each of={steps()} key="index">
          {(s, i) => (
            <View style={V.stepRow}>
              <StepIcon step={s()} index={i()} />
              <View style={V.stepBody}>
                <Text
                  style={[
                    T.stepLabel,
                    (s().state === "idle" || s().state === "skip") && T.stepLabelDim,
                  ]}
                >
                  {s().label}
                </Text>
                <If when={s().hint}>
                  <Text style={T.stepHint}>{s().hint}</Text>
                </If>
              </View>
            </View>
          )}
        </Each>
      </View>
      <View style={V.progressTrack}>
        <View
          style={[
            V.progressFill,
            {
              get width() {
                const list = steps();
                const passed = list.filter((s) => s.state === "ok" || s.state === "skip").length;
                return `${Math.round((passed / (list.length || 1)) * 100)}%` as `${number}%`;
              },
            },
          ]}
        />
      </View>
    </View>
  );
}

function Banner() {
  return (
    <If when={done()}>
      <View style={done() === "ok" ? V.bannerOk : V.bannerFail}>
        <Text style={[T.bannerTitle, done() !== "ok" && T.bannerTitleFail]}>
          {done() === "ok" ? "Готово" : "Не получилось"}
        </Text>
        <Text style={T.bannerNote}>
          {done() === "ok"
            ? mode() === "install"
              ? "Jarvis в меню приложений и в PATH — команда jarvis"
              : "Что именно убрано — в журнале ниже"
            : "Подробности в журнале ниже"}
        </Text>
      </View>
    </If>
  );
}

function Console() {
  return (
    <If when={log().length}>
      <View style={V.consoleBox}>
        <Scroll>
          <Text style={T.consoleText}>{log().join("\n")}</Text>
        </Scroll>
      </View>
    </If>
  );
}

function App() {
  try {
    const win = currentWindow();
    win.set({ title: "Jarvis", resizable: false });
    win.setBounds({ size: { width: 500, height: 800 } });
    win.center();
  } catch {
    // не десктоп — окно уже настроено хостом
  }
  const installing = () => mode() === "install";
  return (
    <View style={V.root}>
      <View style={V.hero}>
        <View style={V.orbGlow}>
          <View style={V.orb}>
            <Text style={T.orbText}>J</Text>
          </View>
        </View>
        <View style={V.heroText}>
          <Text style={T.title}>Jarvis</Text>
          <Text style={T.subtitle}>Голосовой ассистент для десктопа</Text>
        </View>
      </View>

      <Segmented />

      <If when={paths()}>
        {(p) => (
          <View style={V.panel}>
            <View style={V.pathRow}>
              <Text style={T.pathKey}>Бинарь</Text>
              <Text style={T.pathVal}>{p().bin}</Text>
            </View>
            <View style={V.pathRow}>
              <Text style={T.pathKey}>Данные</Text>
              <Text style={T.pathVal}>{p().data}</Text>
            </View>
          </View>
        )}
      </If>

      <If when={installing()}>
        <View style={V.toggles}>
          <Toggle
            on={buildFallback}
            flip={() => buildFallback.update((v) => !v)}
            label="Собрать из исходников (если релиз недоступен)"
          />
        </View>
      </If>
      <If when={!installing()}>
        <View style={V.toggles}>
          <Toggle
            on={purge}
            flip={() => purge.update((v) => !v)}
            label="Стереть данные, модели и плагины (там голосовые сэмплы)"
          />
        </View>
      </If>

      <If when={steps().length}>
        <Steps />
      </If>

      <Banner />

      <View
        style={[
          V.button,
          running() && V.buttonOff,
          {
            get backgroundColor() {
              return installing() ? C.btnLight : C.bad;
            },
          },
        ]}
        hoverStyle={{
          get backgroundColor() {
            return installing() ? C.btnLightHover : C.badHover;
          },
        }}
        activeStyle={{ opacity: 0.8 }}
        onPress={start}
      >
        <Text
          style={[T.buttonText, !installing() && T.buttonTextDanger]}
        >
          {running()
            ? installing()
              ? "Устанавливаю…"
              : "Удаляю…"
            : installing()
              ? "Установить Jarvis"
              : "Удалить Jarvis"}
        </Text>
      </View>

      <Console />

      <View style={V.footer}>
        <Text style={T.footerText}>
          {installing()
            ? buildFallback()
              ? "Сборка из исходников — bun и cargo должны быть в PATH\nМодели скачаются при первом запуске"
              : "Готовый бинарник из GitHub Releases\nМодели скачаются при первом запуске"
            : "Без галочки выше останутся модели, плагины и настройки"}
        </Text>
      </View>
    </View>
  );
}

mount(App);
