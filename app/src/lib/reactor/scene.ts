import * as THREE from "three/webgpu";
import { WebGPURenderer } from "three/webgpu";
import {
  chamferCenters,
  channelGeometry,
  cornerDirections,
  frameGeometry,
  longEdgeAngle,
  longEdgeMidpoints,
  plateGeometry,
  polygonCurve,
  truncatedTrianglePoints,
} from "./geometry";
import { createMaterials, glowTexture, type ReactorMaterials } from "./materials";
import {
  INTRO,
  INTRO_DURATION,
  clamp01,
  easeInOutCubic,
  easeOutBack,
  easeOutSnap,
  phase,
} from "./intro";

/**
 * Сцена реактора Mark V на WebGPU (three/webgpu, фоллбэк на WebGL2 внутри
 * WebGPURenderer).
 *
 * Композиция — фронтальная, слои уходят ВГЛУБЬ по Z:
 *
 *   корпус (шасси с фасками, вент-пазами со свечением, угловыми узлами,
 *           энергетическим швом по контуру)                 z ≈ 0
 *   утопленная тёмная ступень                               z −0.07
 *   внутреннее кольцо на кронштейнах                        z −0.11
 *   три лазерных излучателя внутри проёма кольца, лучи в ядро  z −0.085
 *   дно колодца → ядро (кристалл + внутренний контр-октаэдр +
 *   гироскопические дуги) → стекло                          z −0.21…−0.035
 *
 * Анимации: левитация, контр-вращение гироскопических дуг, искры-октаэдры
 * по каналам, быстрые импульсы по основному контуру, заряд и лучи трёх
 * излучателей, пульс вент-тепла, мерцание ядра, дыхание кромки. Состояния —
 * от реального движка, уровень звука — от audio_level.
 *
 * Контракты (по скиллу WebGPU): фоллбэк-бэкенд определяется автоматически,
 * инициализация асинхронная и обязательная до первого кадра, пауза рендера
 * при скрытом окне, создание ресурсов — один раз, полная утилизация.
 */

export type ReactorState = "off" | "idle" | "listening" | "processing" | "speaking" | "error";

/** Описанный радиус корпуса равностороннего треугольника (вершина вниз). */
const S = 1.06;
/** Срез угла (пропорционален масштабу слоя). */
const CUT = 0.27;
/** Смещение внутренней сборки по Y: 0 — по центру корпуса. */
const INNER_LIFT = 0;

/** Контур корпуса и его производные: самоподобные копии с равными срезами. */
const OUTLINE = truncatedTrianglePoints(S, CUT);
const CHASSIS_INNER = truncatedTrianglePoints(S * 0.79, CUT * 0.79);
const RING_OUTER = truncatedTrianglePoints(S * 0.57, CUT * 0.57);
const RING_INNER = truncatedTrianglePoints(S * 0.44, CUT * 0.44);
const RING_MID = truncatedTrianglePoints(S * 0.505, CUT * 0.505);
const FLOOR_LOOP = truncatedTrianglePoints(S * 0.755, CUT * 0.755);

/** Целевая энергетика состояния: 0..1. */
const STATE_ENERGY: Record<ReactorState, number> = {
  off: 0.16,
  idle: 0.55,
  listening: 0.85,
  processing: 0.95,
  speaking: 0.8,
  error: 0.5,
};

/** Скорость механики и частиц (отн. единицы). */
const STATE_FLOW: Record<ReactorState, number> = {
  off: 0.15,
  idle: 0.4,
  listening: 1.1,
  processing: 2.2,
  speaking: 0.8,
  error: 0.25,
};

/**
 * Кадрирование: реактор вписан в окно по максимуму и крупно.
 *
 * Дистанция рабочего кадра считается из габарита модели и пропорций канваса
 * (см. workZ), поэтому увеличение не режет грани даже на очень узком окне.
 * «Схема» с раскрытым веером слоёв стоит в DIAGRAM_PULL раз дальше рабочей.
 */
/** Половина высоты и половины ширины внешнего контура корпуса (измерено). */
const REACTOR_FIT = { y: 0.68, x: 0.78 };
/** Долю видимой половины кадра, которую разрешено занять модели. 0.88 — с
 * запасом на поворот сборки в заставке: повёрнутый корпус ближе к камере. */
const REACTOR_MARGIN = 0.88;
/** Во сколько раз «схема» стоит дальше рабочего кадра. Равносторонний корпус
 * у основания уже прежнего, ближний угол в повороте к объективу сильно не
 * приближается: проекция контура в NDC по всему таймлайну заставки не выходит
 * за 0.887 при aspect ≥ 1.2 и за 0.942 на самом узком окне (aspect 0.8), то
 * есть за край ничего не вылезает. */
const DIAGRAM_PULL = 1.45;
/**
 * Рабочая точка обзора — центр габарита треугольника, а не ноль сцены. Она же
 * точка обзора «схемы»: веер слоёв раскрывается вдоль собственной высоты
 * реактора, то есть вбок, а не вверх, поэтому поднимать камеру и задирать
 * цель обзора больше не за чем — схема отличается только расстоянием.
 */
const CAM_WORK = { y: -0.15, look: -0.15 };

/** Описатель узла для стыковки: откуда прилетает и как закручен. */
interface IntroLayer {
  obj: THREE.Object3D;
  /** Откуда узел прилетает по Z до стыковки (из глубины). */
  fromZ: number;
  /** Масштаб до стыковки. */
  fromScale: number;
  /** Разворот вокруг своей оси до стыковки, радианы. */
  spin: number;
}

/**
 * Деталь развалки: базовое положение из сборки и вынос ВДОЛЬ СОБСТВЕННОЙ
 * ВЫСОТЫ РЕАКТОРА (локальная Z сборки — ось, перпендикулярная грани).
 * Координаты X и Y у детали не меняются никогда: слой остаётся ровно там же
 * в плоскости грани и только выходит из неё наружу — как страница, снятая с
 * корешка лежащей на столе книги. Поэтому сборка не выглядит собранной из
 * разлетевшихся кусков.
 */
interface ExplodedPart {
  obj: THREE.Object3D;
  base: THREE.Vector3;
  baseRot: THREE.Euler;
  baseScale: THREE.Vector3;
  /** Насколько деталь выходит из плоскости грани над своим местом, единицы сцены. */
  travel: number;
  /** Небольшой крен в полёте, радианы: слой покачивается, а не плывёт. */
  spin: number;
  /** 0 — ядро (выходит первым и остаётся в глубине), 1 — корпус (вершина веера). */
  outDelay: number;
  inDelay: number;
}

const CYAN = new THREE.Color(0x4bcbff);
const ENERGY = new THREE.Color(0xbdf2ff);
const RED = new THREE.Color(0xf06b70);
const RED_BRIGHT = new THREE.Color(0xe2505c);
const OFF = new THREE.Color(0x3c5568);
const OFF_BRIGHT = new THREE.Color(0x51788f);
const DEEP_BLUE_GLOW = new THREE.Color(0x0b4a68);

const tmpColor = new THREE.Color();
const tmpObject = new THREE.Object3D();

interface ParticleSpec {
  lane: THREE.Curve<THREE.Vector3>;
  offset: number;
  speed: number;
  outward: boolean;
}

/** Лазерный излучатель: ствол смотрит в ядро, луч сходится в кристалле. */
interface LaserEmitter {
  lens: THREE.Mesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>;
  flash: THREE.Sprite;
  /** Бегущий по лучу импульс. */
  bolt: THREE.Sprite;
  beamStart: number;
  beamEnd: number;
  rings: THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>[];
  phase: number;
}

/** Орбита осколка: наклон задаётся группой, радиус и фаза — здесь. */
interface OrbitSpec {
  radius: number;
  speed: number;
  phase: number;
}

export class ReactorScene {
  private canvas: HTMLCanvasElement;
  private renderer: WebGPURenderer | null = null;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private m: ReactorMaterials;

  /**
   * Вся сборка в одном узле: интро доворачивает её и отъезжает камерой,
   * не трогая свет (иначе поплыли бы блики на металле).
   */
  private rig = new THREE.Group();
  /** Корпус: стоит на месте. */
  private housing = new THREE.Group();
  /** Внутреннее кольцо: по центру корпуса. */
  private innerRing = new THREE.Group();
  /** Камера ядра: кристалл, дуги, стекло, ореол. */
  private chamber = new THREE.Group();

  private crystal!: THREE.Mesh<THREE.IcosahedronGeometry, THREE.MeshPhysicalMaterial>;
  private crystalEdges!: THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>;
  private innerCore!: THREE.Mesh<THREE.OctahedronGeometry, THREE.MeshBasicMaterial>;
  private heart!: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  private halo!: THREE.Sprite;
  private haloOuter!: THREE.Sprite;
  /** Объёмный контейнмент-каркас вокруг кристалла. */
  private coreCage!: THREE.Group;
  private gyroA: THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>[] = [];
  private conduits: { mesh: THREE.Mesh; dim: boolean }[] = [];
  private emitters: LaserEmitter[] = [];
  private ventGlows: THREE.Mesh<THREE.BoxGeometry, THREE.MeshBasicMaterial>[] = [];
  private edgeLoop: THREE.LineSegments | null = null;

  /** Концентрические кольца и риски на дне колодца. */
  private wellField!: THREE.Group;
  private wellTicks!: THREE.Group;

  /** Искры по каналам: инстансированные октаэдры. */
  private particles: THREE.InstancedMesh | null = null;
  private particleSpecs: ParticleSpec[] = [];
  private particleLanes: THREE.Curve<THREE.Vector3>[] = [];
  /** Быстрые импульсы по основному контуру. */
  private pulses: THREE.InstancedMesh | null = null;
  private pulseLane: THREE.Curve<THREE.Vector3> | null = null;
  private pulseOffsets: number[] = [];
  /** Орбитальные осколки: группа задаёт наклон орбиты, меш летит по ней. */
  private orbitShards: THREE.Mesh[] = [];
  private orbitSpecs: OrbitSpec[] = [];

  private coreLight!: THREE.PointLight;

  private state: ReactorState = "idle";
  private level = 0;
  private targetLevel = 0;
  private energy = 0.5;
  private clock = new THREE.Clock();
  private reducedMotion = false;
  private disposed = false;
  private paused = false;
  private inited = false;

  /** Узлы интро-сборки: заполняются после постройки сцены. */
  private introLayers: IntroLayer[] = [];
  /** Детали развалки: прямые дети трёх узлов, расходятся вдоль оси сборки. */
  private parts: ExplodedPart[] = [];
  /** Осевая линия «схемы»: проступает на развалке, гаснет на сборке. */
  private introAxis:
    | THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
    | null = null;
  /** Время с начала интро; считается по моно-часам, а не суммой dt. */
  private introTime = 0;
  /** Моно-время начала интро: таймлайн живёт по часам, а не по накопленному dt. */
  private introStartMs = 0;
  /** Суммарно простой цик отрисовки, мс. */
  private pausedMs = 0;
  /** Момент остановки цикла; 0 — цикл идёт. */
  private pauseStartMs = 0;
  private introActive = false;
  /** Состояние, запрошенное во время сборки — применяется на стыке. */
  private pendingState: ReactorState = "idle";
  private onIntroDone: (() => void) | null = null;
  /** Ошибка кадра уже отправлена наружу: повторять каждую сотню кадров нельзя. */
  private frameFailed = false;
  /** Сообщение о сорванной отрисовке — в журнал интерфейса. */
  onError: ((message: string) => void) | null = null;
  /** Сколько кадров отрисовано: диагностика «цикл стоит» против «анимации нет». */
  private frames = 0;
  /** Амплитуда заставки: 1 обычно, мягче при уменьшенном движении. */
  private calm = 1;
  /** Кадры на момент пуска заставки — по ним видно, идёт ли цикл. */
  private framesAtStart = 0;
  /** Страж зависания: снимает заставку, если кадры не идут. */
  private introWatchdog: number | null = null;

  private textures: THREE.Texture[] = [];

  constructor(canvas: HTMLCanvasElement, options: { reducedMotion?: boolean } = {}) {
    this.canvas = canvas;
    this.reducedMotion = options.reducedMotion ?? false;
    this.m = createMaterials();

    this.camera = new THREE.PerspectiveCamera(24, 1, 0.1, 40);
    this.camera.position.set(0, CAM_WORK.y, 3.5);
    this.camera.lookAt(0, CAM_WORK.look, 0);

    this.buildLights();
    this.buildHousing();
    this.buildInnerRing();
    this.buildEmitters();
    this.buildChamber();
    this.buildParticles();

    // Внутренняя сборка — единый узел, по центру корпуса.
    this.innerRing.position.y = INNER_LIFT;
    this.chamber.position.y = INNER_LIFT;

    this.rig.add(this.housing, this.innerRing, this.chamber);
    this.scene.add(this.rig);

    // Базовые позиции деталей снимаются до первого кадра: дальше их
    // двигает анимация, а «как собрано» должно остаться в памяти.
    this.collectParts();

    // Щелчок узлов на место в конце захлопа: приседание маленькое, иначе
    // реактор выглядит собранным из прилетевших кусков, а не расслоившимся.
    // Закрутка у всех узлов в одну сторону — как у всего веера деталей.
    this.introLayers = [
      { obj: this.chamber, fromZ: -0.12, fromScale: 0.94, spin: 0.09 },
      { obj: this.innerRing, fromZ: -0.09, fromScale: 0.955, spin: 0.11 },
      { obj: this.housing, fromZ: -0.07, fromScale: 0.97, spin: 0.07 },
    ];
  }

  // ---------- Инициализация рендера (асинхронная по контракту WebGPU) ----------

  async init(): Promise<string> {
    const renderer = new WebGPURenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    await renderer.init();
    this.renderer = renderer;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;

    this.scene.environment = this.buildEnvironment();
    this.scene.environmentIntensity = 0.9;

    this.resize();
    this.clock.getDelta();
    renderer.setAnimationLoop((time) => void this.frame(time));
    this.inited = true;

    const backend = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend
      ? "WebGPU"
      : "WebGL2 (фоллбэк)";
    return backend;
  }

  /**
   * Окружение для отражений металла: процедурный equirect-градиент с
   * холодным горизонтом и ярким пятном ключевого света. WebGPURenderer
   * строит PMREM из equirect-текстуры сам.
   */
  private buildEnvironment(): THREE.CanvasTexture {
    const w = 1024;
    const h = 512;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;

    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, "#16242f");
    sky.addColorStop(0.45, "#1d3d52");
    sky.addColorStop(0.55, "#0e2231");
    sky.addColorStop(1, "#04070b");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    // Яркое пятно «ключевого света» — даёт живые блики на фасках титана.
    const spot = ctx.createRadialGradient(w * 0.3, h * 0.3, 0, w * 0.3, h * 0.3, w * 0.22);
    spot.addColorStop(0, "rgba(214, 238, 252, 0.9)");
    spot.addColorStop(1, "rgba(214, 238, 252, 0)");
    ctx.fillStyle = spot;
    ctx.fillRect(0, 0, w, h);

    // Холодный циановый от отсвет снизу-справа — контр-блик.
    const cool = ctx.createRadialGradient(w * 0.72, h * 0.62, 0, w * 0.72, h * 0.62, w * 0.16);
    cool.addColorStop(0, "rgba(75, 203, 255, 0.5)");
    cool.addColorStop(1, "rgba(75, 203, 255, 0)");
    ctx.fillStyle = cool;
    ctx.fillRect(0, 0, w, h);

    const texture = new THREE.CanvasTexture(canvas);
    texture.mapping = THREE.EquirectangularReflectionMapping;
    texture.colorSpace = THREE.SRGBColorSpace;
    this.textures.push(texture);
    return texture;
  }

  // ---------- Свет ----------

  private buildLights() {
    this.scene.add(new THREE.AmbientLight(0x1a2733, 1.05));

    // Мягкий ключевой свет спереди-сверху: титан читается объёмом.
    const key = new THREE.DirectionalLight(0xe4f1fc, 1.15);
    key.position.set(-1.6, 2.2, 3.4);
    this.scene.add(key);

    // Холодная синяя заливка справа-снизу.
    const fill = new THREE.DirectionalLight(0x3f9fdd, 0.6);
    fill.position.set(2.0, -1.2, 2.2);
    this.scene.add(fill);

    // Контровой: кромки и внутренние стенки через зазоры слоёв.
    const rim = new THREE.DirectionalLight(0x9fd8ff, 0.85);
    rim.position.set(0.4, 1.0, -3.2);
    this.scene.add(rim);

    // Энергия ядра: основной свет + глубокая подсветка колодца.
    this.coreLight = this.m.coreLight;
    this.coreLight.position.set(0, INNER_LIFT, -0.1);
    this.scene.add(this.coreLight);

    const wellLight = new THREE.PointLight(0x11608c, 0.45, 3, 1.8);
    wellLight.position.set(0, INNER_LIFT, -0.3);
    this.scene.add(wellLight);
  }

  // ---------- 1. Корпус (фронтальная грань z = 0) ----------

  private buildHousing() {
    // Задняя пластина: дно всей конструкции, заподлицо с тылом шасси.
    const backplate = new THREE.Mesh(plateGeometry(S * 0.97, CUT * 0.97, 0.05), this.m.darkInk);
    backplate.position.z = -0.285;
    this.housing.add(backplate);

    // Толстое титановое шасси: фронтальная грань на z = 0.
    const chassis = new THREE.Mesh(frameGeometry(S, CUT, S * 0.79, CUT * 0.79, 0.26), this.m.titanium);
    chassis.position.z = -0.13;
    this.housing.add(chassis);

    // Утопленная тёмная ступень сразу за шасси.
    const step = new THREE.Mesh(
      frameGeometry(S * 0.79, CUT * 0.79, S * 0.745, CUT * 0.745, 0.2),
      this.m.darkMetal,
    );
    step.position.z = -0.17;
    this.housing.add(step);

    // Вент-пазы на трёх длинных рёбрах + тёплое свечение внутри каждого.
    const slotGeo = new THREE.BoxGeometry(0.1, 0.016, 0.016);
    const glowGeo = new THREE.BoxGeometry(0.085, 0.007, 0.006);
    for (let edge = 0; edge < 3; edge++) {
      const angle = longEdgeAngle(OUTLINE, edge);
      const mid = longEdgeMidpoints(OUTLINE)[edge];
      for (const t of [-0.3, -0.1, 0.1, 0.3]) {
        const slot = new THREE.Mesh(slotGeo, this.m.darkInk);
        slot.position.set(mid.x + Math.cos(angle) * t, mid.y + Math.sin(angle) * t, 0.006);
        slot.rotation.z = angle;
        this.housing.add(slot);

        const glow = new THREE.Mesh(glowGeo, this.m.ventGlow);
        glow.position.set(mid.x + Math.cos(angle) * t, mid.y + Math.sin(angle) * t, 0.0);
        glow.rotation.z = angle;
        this.housing.add(glow);
        this.ventGlows.push(glow);
      }
      // Крепёжный винт в середине длинного ребра.
      const screw = new THREE.Mesh(
        new THREE.CylinderGeometry(0.022, 0.022, 0.045, 6),
        this.m.darkMetal,
      );
      screw.position.set(mid.x, mid.y, 0.012);
      screw.rotation.x = Math.PI / 2;
      // Осевое вращение шестигранника: rotation.z после rotation.x наклонило
      // бы цилиндр, а rotation.y крутит его вокруг собственной оси.
      screw.rotation.y = angle;
      this.housing.add(screw);
    }

    // Панельный шов на каждой длинной грани: разбивает плоский титан
    // и читается как разъём корпуса.
    const grooveGeo = new THREE.BoxGeometry(0.62, 0.01, 0.012);
    for (let edge = 0; edge < 3; edge++) {
      const angle = longEdgeAngle(OUTLINE, edge);
      const mid = longEdgeMidpoints(OUTLINE)[edge];
      const inward = mid.clone().normalize().multiplyScalar(-0.072);
      const groove = new THREE.Mesh(grooveGeo, this.m.darkInk);
      groove.position.set(mid.x + inward.x, mid.y + inward.y, 0.004);
      groove.rotation.z = angle;
      this.housing.add(groove);
    }

    // Угловые узлы на трёх срезах: накладка + шестигранный болт.
    // Накладка должна целиком лежать на скосе: сдвигаем узел внутрь на её
    // радиус, иначе край накладки выходит за силуэт корпуса и сверху справа
    // читается как торчащий прямоугольник.
    const padRadius = 0.055;
    const padGeo = new THREE.CylinderGeometry(0.052, padRadius, 0.04, 6);
    const boltGeo = new THREE.CylinderGeometry(0.026, 0.028, 0.042, 6);
    const centers = chamferCenters(OUTLINE);
    for (const [i, rawCenter] of centers.entries()) {
      const center = rawCenter.clone();
      center.addScaledVector(
        center.clone().normalize(),
        -(padRadius + 0.006),
      );
      const pad = new THREE.Mesh(padGeo, this.m.titaniumBrushed);
      pad.position.set(center.x, center.y, 0.012);
      pad.rotation.x = Math.PI / 2;
      // Вращаем вокруг оси цилиндра (Y до наклона), иначе rotation.z
      // заваливает накладку набок — это и портило верхний правый болт.
      pad.rotation.y = (i * Math.PI) / 3 + 0.35;
      this.housing.add(pad);

      const bolt = new THREE.Mesh(boltGeo, this.m.darkMetal);
      bolt.position.set(center.x, center.y, 0.038);
      bolt.rotation.x = Math.PI / 2;
      bolt.rotation.y = (i * Math.PI) / 3 + 0.6;
      this.housing.add(bolt);
    }

    // Энергетический шов: тонкая линия по контуру фронтальной грани,
    // дышит вместе с ядром.
    const seamPoints = OUTLINE.map(
      (p) => new THREE.Vector3(p.x * 0.995, p.y * 0.995, 0.008),
    );
    // LineSegments вместо LineLoop: WebGPURenderer не поддерживает LineLoop
    // и молча роняет весь кадр. Замыкаем контур явно.
    const seamGeo = new THREE.BufferGeometry().setFromPoints([
      ...seamPoints,
      seamPoints[0],
    ]);
    this.edgeLoop = new THREE.LineSegments(seamGeo, this.m.edgeLine);
    this.housing.add(this.edgeLoop);
  }

  // ---------- 2. Внутреннее кольцо ----------

  private buildInnerRing() {
    // Кольцо: вторичная рама, физически отделена зазором от корпуса.
    const ring = new THREE.Mesh(
      frameGeometry(S * 0.57, CUT * 0.57, S * 0.44, CUT * 0.44, 0.12),
      this.m.titaniumBrushed,
    );
    ring.position.z = -0.17;
    this.innerRing.add(ring);

    // Кронштейны-башмаки на тыльной стороне углов кольца.
    const bracketGeo = new THREE.BoxGeometry(0.07, 0.11, 0.06);
    for (const center of chamferCenters(RING_OUTER)) {
      const dir = Math.atan2(center.y, center.x);
      const bracket = new THREE.Mesh(bracketGeo, this.m.darkMetal);
      bracket.position.set(center.x, center.y, -0.165);
      bracket.rotation.z = dir + Math.PI / 2;
      this.innerRing.add(bracket);
    }

    // Основной энергетический канал по фронтальной грани кольца.
    const main = new THREE.Mesh(channelGeometry(RING_MID, 0.014), this.m.conduit);
    main.position.z = -0.105;
    this.innerRing.add(main);
    this.conduits.push({ mesh: main, dim: false });

    // Тусклый контур по ступени (система корпуса).
    const floor = new THREE.Mesh(channelGeometry(FLOOR_LOOP, 0.008), this.m.conduitDim);
    floor.position.z = -0.075;
    this.housing.add(floor);
    this.conduits.push({ mesh: floor, dim: true });
  }

  // ---------- 3. Лазерные излучатели ----------

  private buildEmitters() {
    const dirs = cornerDirections(RING_OUTER);

    // Три лазерных излучателя: ствол смотрит в ядро, лучи сходятся
    // в кристалле. Корпус — тёмный графит, рёбра охлаждения — титан,
    // дульная линза и кольца заряда — энергия.
    const emitterZ = -0.085;
    // Излучатели стоят внутри проёма внутреннего кольца, а не на его полке:
    // просвет кольца по направлению скоса — 0.343·S, а внешний край
    // кронштейна при baseR = 0.30·S доходит только до 0.325·S.
    const muzzleR = 0.195 * S;
    const baseR = 0.30 * S;
    const barrelLen = baseR - muzzleR;
    const barrelMid = (muzzleR + baseR) / 2;
    const beamStartR = muzzleR - 0.018;
    // Луч доходит до ядра и кончается ВНУТРИ его огранки: край кристалла в
    // самом тесном положении (дыхание + оффлайн) — 0.083, лучи заканчиваются
    // на 0.05, поэтому зазора между дулом и ядром не бывает ни на одном состоянии.
    const beamEndR = 0.05 * S;
    const beamLen = beamStartR - beamEndR;
    const beamMidR = beamEndR + beamLen / 2;

    const barrelGeo = new THREE.CylinderGeometry(0.024, 0.036, barrelLen, 18);
    const finGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.007, 16);
    const collarGeo = new THREE.CylinderGeometry(0.036, 0.04, 0.028, 12);
    const mountGeo = new THREE.BoxGeometry(0.07, 0.1, 0.07);
    const ringGeo = new THREE.TorusGeometry(0.045, 0.005, 6, 24);
    const muzzleRingGeo = new THREE.TorusGeometry(0.028, 0.006, 8, 24);
    const lensGeo = new THREE.CylinderGeometry(0.019, 0.019, 0.012, 18);
    const beamCoreGeo = new THREE.CylinderGeometry(0.006, 0.006, beamLen, 6, 1, true);
    const beamGlowGeo = new THREE.CylinderGeometry(0.019, 0.019, beamLen, 8, 1, true);
    const boltGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.03, 6);

    for (const [i, angle] of dirs.entries()) {
      const group = new THREE.Group();
      group.position.z = emitterZ;
      group.rotation.z = angle;

      // Силовой кронштейн у внутреннего кольца.
      const mount = new THREE.Mesh(mountGeo, this.m.darkMetal);
      mount.position.x = baseR - 0.008;
      group.add(mount);

      // Ствол: узкое дуло смотрит к центру (−X).
      const barrel = new THREE.Mesh(barrelGeo, this.m.darkMetal);
      barrel.rotation.z = Math.PI / 2;
      barrel.position.x = barrelMid;
      group.add(barrel);

      // Охлаждающие рёбра вдоль ствола.
      for (let f = 0; f < 3; f++) {
        const fin = new THREE.Mesh(finGeo, this.m.titaniumBrushed);
        fin.rotation.z = Math.PI / 2;
        fin.position.x = muzzleR + 0.024 + f * 0.022;
        group.add(fin);
      }

      // Задняя обойма и дульный коллектор.
      const collar = new THREE.Mesh(collarGeo, this.m.titaniumBrushed);
      collar.rotation.z = Math.PI / 2;
      collar.position.x = baseR - 0.035;
      group.add(collar);

      const muzzleCollar = new THREE.Mesh(collarGeo, this.m.titanium);
      muzzleCollar.rotation.z = Math.PI / 2;
      muzzleCollar.position.x = muzzleR + 0.005;
      muzzleCollar.scale.set(0.72, 0.9, 0.72);
      group.add(muzzleCollar);

      // Дульное кольцо и линза.
      const muzzleRing = new THREE.Mesh(muzzleRingGeo, this.m.charge);
      muzzleRing.rotation.y = Math.PI / 2;
      muzzleRing.position.x = muzzleR - 0.008;
      group.add(muzzleRing);

      const lens = new THREE.Mesh(lensGeo, this.m.lens);
      lens.rotation.z = Math.PI / 2;
      lens.position.x = muzzleR - 0.012;
      group.add(lens);

      // Бегущее кольцо заряда на стволе.
      const chargeRing = new THREE.Mesh(ringGeo, this.m.charge);
      chargeRing.rotation.y = Math.PI / 2;
      chargeRing.position.x = muzzleR + 0.025;
      group.add(chargeRing);

      // Луч: тонкое ядро + мягкий ореол, сходится в центре кристалла.
      const beamCore = new THREE.Mesh(beamCoreGeo, this.m.beamCore);
      beamCore.rotation.z = Math.PI / 2;
      beamCore.position.x = beamMidR;
      group.add(beamCore);

      const beamGlow = new THREE.Mesh(beamGlowGeo, this.m.beamGlow);
      beamGlow.rotation.z = Math.PI / 2;
      beamGlow.position.x = beamMidR;
      group.add(beamGlow);

      // Вспышка на дуле и бегущий по лучу импульс.
      const flash = new THREE.Sprite(this.m.flash);
      flash.position.x = muzzleR - 0.015;
      flash.scale.setScalar(0.09);
      group.add(flash);

      const bolt = new THREE.Sprite(this.m.flash);
      bolt.position.x = beamStartR;
      bolt.scale.setScalar(0.035);
      group.add(bolt);

      // Крепёж кронштейна.
      for (const bx of [baseR - 0.038, baseR + 0.022]) {
        const screw = new THREE.Mesh(boltGeo, this.m.titaniumBrushed);
        screw.position.set(bx, 0.042, 0.04);
        screw.rotation.x = Math.PI / 2;
        group.add(screw);
      }

      // Излучатели — неподвижная турель внутри проёма внутреннего кольца.
      this.innerRing.add(group);
      this.emitters.push({
        lens,
        flash,
        bolt,
        beamStart: beamStartR,
        beamEnd: beamEndR,
        rings: [chargeRing, muzzleRing],
        phase: (i * Math.PI * 2) / 3,
      });
    }
  }

  // ---------- 4. Энергоотсек ----------

  private buildChamber() {
    // Дно колодца: глубокий синий со свечением.
    const well = new THREE.Mesh(plateGeometry(S * 0.42, CUT * 0.42, 0.04), this.m.deepBlue);
    well.position.z = -0.21;
    this.chamber.add(well);

    // Концентрические кольца поля: задают масштаб и глубину вокруг ядра.
    this.wellField = new THREE.Group();
    this.wellField.position.z = -0.2;
    for (const radius of [0.19 * S, 0.29 * S]) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(radius - 0.0018, radius + 0.0018, 64),
        this.m.wellRing,
      );
      this.wellField.add(ring);
    }

    // Риски по внешнему кольцу: медленно сканируют колодец.
    this.wellTicks = new THREE.Group();
    const tickGeo = new THREE.BoxGeometry(0.02, 0.003, 0.003);
    const tickLongGeo = new THREE.BoxGeometry(0.034, 0.0035, 0.003);
    const tickR = 0.31 * S;
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const tick = new THREE.Mesh(i % 6 === 0 ? tickLongGeo : tickGeo, this.m.wellRing);
      tick.position.set(Math.cos(a) * tickR, Math.sin(a) * tickR, 0);
      tick.rotation.z = a;
      this.wellTicks.add(tick);
    }
    this.wellField.add(this.wellTicks);
    this.chamber.add(this.wellField);

    // Два слоя ореола: внешний — мягкая дымка, внутренний — свет ядра.
    const haloMap = glowTexture("rgba(214, 244, 255, 0.9)", "rgba(75, 203, 255, 0)");
    this.textures.push(haloMap);
    this.m.halo.map = haloMap;
    this.m.haloOuter.map = haloMap;
    // Дульные вспышки используют ту же мягкую текстуру.
    this.m.flash.map = haloMap;
    this.m.flash.needsUpdate = true;

    this.haloOuter = new THREE.Sprite(this.m.haloOuter);
    this.haloOuter.scale.setScalar(0.82);
    this.haloOuter.position.z = -0.19;
    this.chamber.add(this.haloOuter);

    this.halo = new THREE.Sprite(this.m.halo);
    this.halo.scale.setScalar(0.44);
    this.halo.position.z = -0.175;
    this.chamber.add(this.halo);

    // Кристалл ядра: фасеточный, вращается, светится изнутри.
    this.crystal = new THREE.Mesh(new THREE.IcosahedronGeometry(0.125 * S, 0), this.m.crystal);
    this.crystal.position.z = -0.15;
    this.chamber.add(this.crystal);

    // Рёбра кристалла: без них на маленьком размере огранка читается
    // как белое пятно, а не как кристалл.
    this.crystalEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(this.crystal.geometry, 1),
      this.m.crystalEdge,
    );
    this.crystalEdges.position.copy(this.crystal.position);
    this.chamber.add(this.crystalEdges);

    // Внутренний контр-октаэдр: вращается навстречу кристаллу,
    // виден сквозь прозрачные грани.
    this.innerCore = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.058 * S, 0),
      new THREE.MeshBasicMaterial({
        color: 0xbdf2ff,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    this.innerCore.position.z = -0.15;
    this.chamber.add(this.innerCore);

    // Сердце: маленькая яркая сфера в центре — точка, за которую
    // цепляется взгляд, когда кристалл вспыхивает.
    this.heart = new THREE.Mesh(
      new THREE.SphereGeometry(0.022 * S, 12, 12),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    this.heart.position.z = -0.142;
    this.chamber.add(this.heart);

    // Контейнмент-каркас: икосаэдр из тонких линий и светящихся узлов.
    this.coreCage = new THREE.Group();
    this.coreCage.position.z = -0.15;
    const cageRadius = 0.185 * S;
    const cage = new THREE.LineSegments(
      new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(cageRadius, 0)),
      this.m.field,
    );
    this.coreCage.add(cage);

    const phi = (1 + Math.sqrt(5)) / 2;
    const cageVerts: [number, number, number][] = [
      [-1, phi, 0],
      [1, phi, 0],
      [-1, -phi, 0],
      [1, -phi, 0],
      [0, -1, phi],
      [0, 1, phi],
      [0, -1, -phi],
      [0, 1, -phi],
      [phi, 0, -1],
      [phi, 0, 1],
      [-phi, 0, -1],
      [-phi, 0, 1],
    ];
    const nodeGeo = new THREE.OctahedronGeometry(0.009 * S, 0);
    const nodes = new THREE.InstancedMesh(nodeGeo, this.m.fieldNode, cageVerts.length);
    for (const [i, v] of cageVerts.entries()) {
      tmpObject.position.set(v[0], v[1], v[2]).normalize().multiplyScalar(cageRadius);
      tmpObject.rotation.set(0, 0, 0);
      tmpObject.scale.setScalar(1);
      tmpObject.updateMatrix();
      nodes.setMatrixAt(i, tmpObject.matrix);
    }
    nodes.instanceMatrix.needsUpdate = true;
    this.coreCage.add(nodes);
    this.chamber.add(this.coreCage);

    // Гироскопические дуги: три тонкие дуги в разных плоскостях.
    const arcDefs = [
      { radius: 0.19 * S, tube: 0.0045, arc: 2.1, rx: 0.35, ry: 0, rz: 0 },
      { radius: 0.235 * S, tube: 0.004, arc: 1.3, rx: -0.2, ry: 0.5, rz: 0 },
      { radius: 0.158 * S, tube: 0.0035, arc: 2.8, rx: 0.1, ry: -0.4, rz: 0.9 },
    ];
    for (const def of arcDefs) {
      const arc = new THREE.Mesh(
        new THREE.TorusGeometry(def.radius, def.tube, 8, 64, def.arc),
        this.m.arc,
      );
      arc.position.z = -0.145;
      arc.rotation.set(def.rx, def.ry, def.rz);
      // Светящиеся наконечники: дуга читается как сегмент прибора.
      const tipGeo = new THREE.OctahedronGeometry(0.008 * S, 0);
      for (const angle of [0, def.arc]) {
        const tip = new THREE.Mesh(tipGeo, this.m.fieldNode);
        tip.position.set(Math.cos(angle) * def.radius, Math.sin(angle) * def.radius, 0);
        arc.add(tip);
      }
      this.chamber.add(arc);
      this.gyroA.push(arc);
    }

    // Три орбитальных осколка: разные плоскости, скорости и фазы.
    const orbitDefs = [
      { radius: 0.19 * S, tiltX: 0.5, tiltY: 0.2, tiltZ: 0.3, speed: 1.15, phase: 0 },
      { radius: 0.215 * S, tiltX: -0.35, tiltY: 0.9, tiltZ: -0.2, speed: 0.8, phase: 2.1 },
      { radius: 0.172 * S, tiltX: 0.95, tiltY: -0.5, tiltZ: 0.6, speed: 1.45, phase: 4.2 },
    ];
    const shardGeo = new THREE.OctahedronGeometry(0.014 * S, 0);
    for (const def of orbitDefs) {
      const group = new THREE.Group();
      group.position.z = -0.15;
      group.rotation.set(def.tiltX, def.tiltY, def.tiltZ);

      // Едва заметный след орбиты: помогает читать движение осколка.
      const orbit = new THREE.Mesh(
        new THREE.TorusGeometry(def.radius, 0.0016, 4, 64),
        this.m.wellRing,
      );
      group.add(orbit);

      const shard = new THREE.Mesh(shardGeo, this.m.shard);
      const shardGlow = new THREE.Sprite(this.m.flash);
      shardGlow.scale.setScalar(0.05);
      shard.add(shardGlow);
      group.add(shard);

      this.chamber.add(group);
      this.orbitShards.push(shard);
      this.orbitSpecs.push({ radius: def.radius, speed: def.speed, phase: def.phase });
    }

    // Стекло камеры над колодцем: за фронтальной гранью корпуса.
    const glass = new THREE.Mesh(plateGeometry(S * 0.43, CUT * 0.43, 0.03), this.m.chamberGlass);
    glass.position.z = -0.035;
    this.chamber.add(glass);
  }

  // ---------- 5. Искры и импульсы (инстансированные октаэдры) ----------

  private buildParticles() {
    const lanePaths: THREE.Curve<THREE.Vector3>[] = [];
    const outward: boolean[] = [];

    // Контур основного канала: подъём внутренней сборки — по Y, глубина — Z.
    const ringLane = RING_MID.map((p) => new THREE.Vector2(p.x, p.y + INNER_LIFT));
    for (let i = 0; i < 3; i++) {
      lanePaths.push(polygonCurve(ringLane, -0.105));
      outward.push(false);
    }
    // Радиальные лучи от ядра к углам колодца.
    for (const angle of cornerDirections(RING_OUTER)) {
      const dir = new THREE.Vector2(Math.cos(angle), Math.sin(angle));
      lanePaths.push(
        new THREE.LineCurve3(
          new THREE.Vector3(dir.x * 0.12 * S, dir.y * 0.12 * S + INNER_LIFT, -0.13),
          new THREE.Vector3(dir.x * 0.4 * S, dir.y * 0.4 * S + INNER_LIFT, -0.13),
        ),
      );
      outward.push(true);
    }
    this.particleLanes = lanePaths;

    // Искры: фасеточные октаэдры инстансингом — один draw call.
    const COUNT = 56;
    const sparkGeo = new THREE.OctahedronGeometry(0.011 * S, 0);
    this.particles = new THREE.InstancedMesh(sparkGeo, this.m.particles, COUNT);
    this.particles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for (let i = 0; i < COUNT; i++) {
      this.particleSpecs.push({
        lane: lanePaths[i % lanePaths.length],
        offset: Math.random(),
        speed: 0.6 + Math.random() * 0.8,
        outward: outward[i % lanePaths.length],
      });
      tmpObject.position.set(0, 0, 10); // за сценой до первого тика
      tmpObject.updateMatrix();
      this.particles.setMatrixAt(i, tmpObject.matrix);
    }
    this.rig.add(this.particles);

    // Быстрые импульсы по основному контуру: три ярких бегунка.
    this.pulseLane = polygonCurve(ringLane, -0.1);
    this.pulses = new THREE.InstancedMesh(
      new THREE.OctahedronGeometry(0.016 * S, 0),
      new THREE.MeshBasicMaterial({
        color: 0xd6f4ff,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false,
      }),
      3,
    );
    this.pulses.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.pulseOffsets = [0, 1 / 3, 2 / 3];
    this.rig.add(this.pulses);
  }

  /**
   * Инвентарь деталей развалки: берём прямые дети трёх узлов — это те
   * единицы, на которые реактор и расслаивается (плита, рамка, кронштейн,
   * дуга, картер).
   *
   * Каждый узел занимает свою «станцию» вдоль собственной высоты реактора —
   * оси, перпендикулярной треугольной грани (локальная Z сборки). Ядро почти
   * не сдвигается, кольцо уходит глубже, корпус — глубже всех. Внутри станции
   * детали расходятся мелким веером по той же оси. X и Y не трогаются нигде:
   * слой стоит ровно там же в плоскости грани, поэтому картина читается как
   * расслоение, а не как разбросанные куски.
   */
  private collectParts() {
    /** Вынос станции вдоль собственной оси высоты реактора (локальная Z), в
     * обратную от зрителя сторону: слои уходят вглубь, а не на зрителя. */
    const STATIONS: [THREE.Group, number][] = [
      [this.chamber, 0],
      [this.innerRing, -0.5],
      [this.housing, -0.95],
    ];
    /** Мелкий веер деталей внутри одной станции: только глубже, никогда вбок. */
    const WITHIN = 0.22;

    const total = STATIONS.reduce((n, [group]) => n + group.children.length, 0);
    const last = Math.max(1, total - 1);

    const parts: ExplodedPart[] = [];
    let index = 0;
    for (const [group, station] of STATIONS) {
      // Порядок внутри узла — от ближней к зрителю детали к дальней: веер
      // расходится в ту же сторону, что и станции, слои не пересекаются.
      const children = [...group.children].sort((a, b) => b.position.z - a.position.z);
      const span = Math.max(1, children.length - 1);
      for (const [j, obj] of children.entries()) {
        const rank = index / last;
        parts.push({
          obj,
          base: obj.position.clone(),
          baseRot: obj.rotation.clone(),
          baseScale: obj.scale.clone(),
          // Одна сторона для всех: ни один винтик не едет навстречу слою.
          travel: station - (children.length > 1 ? (j / span) * WITHIN : 0),
          // Покачивание в одну сторону: весь веер «сыплется» одинаково.
          spin: 0.07 + 0.07 * ((index % 4) / 3),
          outDelay: rank,
          inDelay: 1 - rank,
        });
        index++;
      }
    }
    this.parts = parts;
    this.buildIntroAxis(STATIONS, WITHIN);
  }

  /**
   * Осевая нить схемы: стержень вдоль собственной высоты реактора (локальная
   * Z) с муфтами на станциях узлов. Без него вынутые из грани плоские детали
   * читаются как набор, а не как последовательность слоёв, нанизанных на одну
   * ось.
   */
  private buildIntroAxis(stations: [THREE.Group, number][], within: number) {
    const pts: number[] = [];
    // Границы нити — по реальному размаху веера, независимо от того, в какую
    // сторону он раскрыт.
    const offs = stations.flatMap(([, z]) => [z - within / 2, z + within / 2]);
    const lo = Math.min(...offs);
    const hi = Math.max(...offs);
    pts.push(0, 0, lo, 0, 0, hi);
    const R = 0.055;
    for (const [, z] of stations) {
      // Муфта станции: крестик в плоскости, перпендикулярной оси.
      pts.push(-R, 0, z, R, 0, z);
      pts.push(0, -R, z, 0, R, z);
      // Засечки границ веера узла.
      pts.push(0, 0, z - within / 2, R * 0.6, 0, z - within / 2);
      pts.push(0, 0, z + within / 2, R * 0.6, 0, z + within / 2);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    this.introAxis = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        color: CYAN,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    this.rig.add(this.introAxis);
  }

  // ---------- Публичное API ----------

  setState(state: ReactorState) {
    // На сборке реактор сознательно «не горит»: состояние ждёт своей очереди.
    if (this.introActive) {
      this.pendingState = state;
      return;
    }
    this.state = state;
  }

  /**
   * Заставка реактора: сборка проявляется на своём месте, слои выходят из
   * плоскости грани вдоль собственной высоты реактора — все в одну сторону,
   * ни один винтик не едет навстречу, — так же целиком захлопываются, узлы
   * щёлкаются, и сборка одним рывком доворачивается во фронт, после чего
   * реактор загорается. Никуда не улетает и не собирается из кусков: каждый
   * слой стоит ровно там же в плоскости грани.
   *
   * Уменьшенное движение не отменяет сборку, только смягчает амплитуды.
   * Таймлайн живёт по часам, а не по сумме dt, поэтому совпадает с полосой
   * запуска в интерфейсе при любом числе кадров в секунду.
   *
   * Анимация воспроизводится по запросу — и на старте, и по команде
   * «проиграй анимацию реактора». `onDone` вызывается ровно один раз: на
   * розжиге либо при отмене сборки (сцены уже нет, кадр сорвался, цикл встал).
   */
  playIntro(onDone: () => void) {
    if (this.disposed) {
      onDone();
      return;
    }
    // Повторный вызов во время игры — просто меняем получателя отчёта.
    if (this.introActive) {
      this.onIntroDone = onDone;
      return;
    }
    // Пресет уменьшенного движения не отменяет сборку: она запрошена явно и
    // без неё интерфейс выглядит сломанным. Амплитуды просто мягче.
    this.calm = this.reducedMotion ? 0.45 : 1;
    this.resetIntro();
    this.introActive = true;
    this.introTime = 0;
    // 0 = «часы ещё не пущены»: отсчёт стартует на первом реальном кадре,
    // иначе тяжёлый первый кадр (компиляция шейдеров) съест половину заставки.
    this.introStartMs = 0;
    this.pausedMs = 0;
    this.framesAtStart = this.frames;
    this.pendingState = this.state;
    this.state = "off";
    this.onIntroDone = onDone;
    this.applyIntro(0);
    // Страж от зависания: цикл отрисовки может не дать ни кадра (окно под
    // WKWebView, сбой бэкенда). Заставка обязана закончиться сама.
    this.introWatchdog = window.setTimeout(
      () => this.checkStall(),
      INTRO_DURATION * 1000 + 1200,
    );
  }

  /** Текущее состояние — для диагностики интеграции. */
  get currentState(): ReactorState {
    return this.state;
  }

  /** Заставка ещё идёт — для диагностики интеграции. */
  get introRunning(): boolean {
    return this.introActive;
  }

  /** Сколько кадров действительно отрисовано: 0 или стоящий счётчик = цикл не идёт. */
  get frameCount(): number {
    return this.frames;
  }

  /** Активный бэкенд рендера. */
  get backendName(): string {
    const backend = this.renderer?.backend as { isWebGPUBackend?: boolean } | undefined;
    return backend?.isWebGPUBackend ? "WebGPU" : "WebGL2 (фоллбэк)";
  }

  /** Реальный уровень звука 0..1 (микрофон или озвучка). */
  setLevel(level: number) {
    const norm = Math.max(0, Math.min(1, (level - 0.03) * 9));
    this.targetLevel = this.reducedMotion ? norm * 0.4 : norm;
  }

  resize() {
    const width = this.canvas.clientWidth || 300;
    const height = this.canvas.clientHeight || 300;
    this.renderer?.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer?.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    // Пропорции окна поменялись — кадрирование пересчитывается следом.
    if (this.introActive) this.applyIntro(this.introTime);
    else this.frameWork();
  }

  /**
   * Дистанция рабочего кадра: настолько крупно, насколько пускает окно.
   * Ограничитель — та ось, где модель упирается в край (обычно высота, но на
   * узком окне — ширина), плюс запас REACTOR_MARGIN до края.
   */
  private workZ(): number {
    const tan = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const aspect = Math.max(0.4, this.camera.aspect);
    const half = Math.max(REACTOR_FIT.y, REACTOR_FIT.x / aspect);
    return half / (tan * REACTOR_MARGIN);
  }

  /** Камера в рабочем положении: им пользуется и простои, и конец заставки. */
  private frameWork() {
    this.camera.position.set(0, CAM_WORK.y, this.workZ());
    this.camera.lookAt(0, CAM_WORK.look, 0);
  }

  pause() {
    if (this.paused || this.disposed || !this.inited) return;
    this.paused = true;
    this.pauseStartMs = performance.now();
    this.renderer?.setAnimationLoop(null);
  }

  resume() {
    if (!this.paused || this.disposed || !this.inited) return;
    this.paused = false;
    // Заставка живёт по часам: время стоящего окна в таймлайн не входит.
    if (this.pauseStartMs) {
      this.pausedMs += performance.now() - this.pauseStartMs;
      this.pauseStartMs = 0;
    }
    this.clock.getDelta();
    this.renderer?.setAnimationLoop((time) => void this.frame(time));
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.clearIntroWatchdog();
    // Оборванная сборка не должна оставлять интерфейс в режиме загрузки.
    if (this.introActive) {
      this.introActive = false;
      const done = this.onIntroDone;
      this.onIntroDone = null;
      done?.();
    }
    this.renderer?.setAnimationLoop(null);
    this.scene.traverse((obj) => {
      // Спрайт делит геометрию между всеми экземплярами — её не трогаем.
      if ((obj as THREE.Sprite).isSprite) {
        (obj as THREE.Sprite).material.dispose();
        return;
      }
      const mesh = obj as THREE.Mesh;
      mesh.geometry?.dispose();
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(material)) material.forEach((m) => m.dispose());
      else material?.dispose();
    });
    for (const t of this.textures) t.dispose();
    this.renderer?.dispose();
  }

  // ---------- Анимация ----------

  /**
   * Кадр заставки. Фазы заданы в intro.ts; здесь только то, что они значат
   * для сцены: проявление на месте, расслоение вверх по оси ядра, захлоп,
   * щелчок узлов и один рывок доворота в рабочий вид.
   */
  private applyIntro(time: number) {
    // 1. Проявление: сборка вырастает из темноты ровно там, где стоит.
    const appear = easeOutBack(phase(time, INTRO.APPEAR.at, INTRO.APPEAR.dur));

    // 2. Расслоение: каждый слой выходит из плоскости грани наружу вдоль
    // собственной высоты реактора и так же возвращается назад. X и Y не
    // сдвигаются нигде.
    let open = 0;
    for (const part of this.parts) {
      const up = easeOutSnap(
        phase(time, INTRO.SPREAD.at + part.outDelay * INTRO.SPREAD.stagger, INTRO.SPREAD.dur),
      );
      const down = easeInOutCubic(
        phase(time, INTRO.GATHER.at + part.inDelay * INTRO.GATHER.stagger, INTRO.GATHER.dur),
      );
      const k = up * (1 - down);
      part.obj.position.set(
        part.base.x,
        part.base.y,
        part.base.z + part.travel * k * this.calm,
      );
      // Покачивание и надутость в полёте: слой проживает перелёт, а не плывёт
      // по линейке. На границах фазы k=0/1 обе поправки равны нулю, поэтому
      // поза всегда возвращается ровно в собранную.
      const swing = Math.sin(Math.PI * k);
      part.obj.rotation.copy(part.baseRot);
      part.obj.rotation.z += part.spin * swing * this.calm;
      part.obj.rotation.x += part.spin * 0.6 * swing * this.calm;
      const puff = 1 + 0.05 * swing;
      part.obj.scale.set(part.baseScale.x * puff, part.baseScale.y * puff, part.baseScale.z);
      open += k;
    }
    open /= Math.max(1, this.parts.length);

    // 3. Доворот: «хоп» — и сборка встаёт лицом к зрителю с перелётом.
    const turn = easeOutBack(phase(time, INTRO.TURN.at, INTRO.TURN.dur));
    // past уходит за ноль на перелёте: сборка проскакивает фронт и возвращается.
    // Для кадра и нити схемы перелёт обрезаем — там отрицательного нет смысла.
    const past = 1 - turn;
    const tilt = clamp01(past);
    const fan = open * tilt * this.calm;

    // Осевая нить схемы: видна, только пока слои раскрыты.
    if (this.introAxis) this.introAxis.material.opacity = 0.45 * fan;
    // На схеме реактор «не горит», но детали должны читаться: энергия
    // поднимается вместе с веером и гаснет к захлопу.
    this.energy = Math.max(this.energy, 0.32 * fan);

    // 4. Щелчок узлов: лёгкое приседание к своему месту, без дальнего прилёта
    // — реактор не должен выглядеть собранным из разлетевшихся кусков.
    for (const [i, layer] of this.introLayers.entries()) {
      const p = easeOutBack(
        phase(time, INTRO.DOCK.at + i * INTRO.DOCK.stagger, INTRO.DOCK.dur),
      );
      layer.obj.position.z = (1 - p) * layer.fromZ;
      layer.obj.scale.setScalar(1 + (1 - p) * (layer.fromScale - 1));
      layer.obj.rotation.z = (1 - p) * layer.spin;
    }

    // 5. Поза и кадр: в веере сборка чуть повёрнута, чтобы слои читались
    // пластами; на довороте — строго во фронт. Крена вокруг Z нет: он
    // наклоняет саму ось расслоения. Поворот в −Y подобран под направление
    // веера (он раскрывается вдоль −Z): слои уходят вбок и в глубину, то есть
    // ровно в ту сторону, в которую у реактора его собственная высота.
    this.rig.rotation.set(-0.16 * past * this.calm, -0.95 * past, 0);
    this.rig.scale.setScalar(appear * (1 - 0.24 * fan));

    // Луч тот же, что и в рабочем виде: отъезд только по расстоянию. Оно
    // считается по max(веер, поворот): повёрнутая к нам боком сборка ближе к
    // объективу, чем фронтальная, и без запаса её ближний угол лезет за край.
    const out = Math.max(fan, tilt);
    this.camera.position.set(0, CAM_WORK.y, this.workZ() * (1 + (DIAGRAM_PULL - 1) * out));
    this.camera.lookAt(0, CAM_WORK.look, 0);

    // Искры и импульсы загораются вместе с проявлением сборки.
    const fade = 0.2 + appear * 0.8;
    if (this.particles) this.m.particles.opacity *= fade;
    if (this.pulses) (this.pulses.material as THREE.MeshBasicMaterial).opacity *= fade;
  }

  /** Поза «как собрано»: нужна для повтора заставки и для её отмены. */
  private resetIntro() {
    for (const part of this.parts) {
      part.obj.position.copy(part.base);
      part.obj.rotation.copy(part.baseRot);
      part.obj.scale.copy(part.baseScale);
    }
    for (const layer of this.introLayers) {
      layer.obj.position.z = 0;
      layer.obj.rotation.z = 0;
      layer.obj.scale.setScalar(1);
    }
    if (this.introAxis) this.introAxis.material.opacity = 0;
    this.rig.rotation.set(0, 0, 0);
    this.rig.position.set(0, 0, 0);
    this.rig.scale.setScalar(1);
    this.frameWork();
  }

  /** Розжиг: реактор остаётся в рабочем виде и живёт по состоянию движка. */
  private finishIntro() {
    this.clearIntroWatchdog();
    this.introActive = false;
    this.resetIntro();

    this.state = this.pendingState;
    // Розжиг: вспышка уровня звучит как запуск реактора.
    this.energy = Math.max(this.energy, 0.25);
    this.targetLevel = this.reducedMotion ? 0.3 : 0.85;

    const done = this.onIntroDone;
    this.onIntroDone = null;
    done?.();
  }

  private clearIntroWatchdog() {
    if (this.introWatchdog !== null) {
      window.clearTimeout(this.introWatchdog);
      this.introWatchdog = null;
    }
  }

  /**
   * Страж зависания: если к концу таймлайна цикл не дал ни одного кадра
   * (окно скрыто, бэкенд отрисовки встал), заставка всё равно заканчивается —
   * интерфейс не имеет права оставаться в режиме загрузки навсегда.
   */
  private checkStall() {
    this.introWatchdog = null;
    if (!this.introActive) return;
    if (this.introStartMs === 0 || this.frames === this.framesAtStart) {
      const reason = this.paused
        ? "окно скрыто, кадры не идут"
        : "цикл отрисовки не дал ни одного кадра";
      console.error(`Реактор: ${reason}`);
      this.onError?.(`Реактор: ${reason} — сборка остановлена`);
    }
    this.finishIntro();
  }

  private cornerPoint(edge: number, radius: number): THREE.Vector2 {
    const angle = -Math.PI / 2 + (edge * Math.PI * 2) / 3;
    return new THREE.Vector2(Math.cos(angle) * radius, Math.sin(angle) * radius);
  }

  /**
   * Точка входа цикла отрисовки. Кадр может сорваться на одном кадре из
   * сотен — молча терять из-за этого всю анимацию нельзя: ошибка считается
   * один раз, уходит наружу через onError и обязательно снимает заставку,
   * чтобы интерфейс не остался висеть в режиме загрузки.
   */
  private async frame(time: number) {
    if (this.disposed || this.paused || !this.renderer) return;
    try {
      await this.draw(time);
    } catch (e) {
      if (!this.frameFailed) {
        this.frameFailed = true;
        const message = e instanceof Error ? e.message : String(e);
        console.error("Реактор: кадр отрисовки сорвался", e);
        this.onError?.(`Реактор остановлен: ${message}`);
      }
      if (this.introActive) this.finishIntro();
      this.renderer?.setAnimationLoop(null);
    }
  }

  private async draw(time: number) {
    const renderer = this.renderer;
    if (!renderer) return;
    this.frames++;
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    // Плавный переход энергетики и спад уровня звука.
    this.energy += (STATE_ENERGY[this.state] - this.energy) * Math.min(1, dt * 3);
    this.targetLevel = Math.max(this.targetLevel * (1 - dt * 4), 0);
    this.level += (this.targetLevel - this.level) * Math.min(1, dt * 10);

    // На уменьшенном движении реактор не должен выглядеть выключенным:
    // 0.55 — спокойнее нормы, но жизнь заметна глазом.
    const motion = this.reducedMotion ? 0.55 : 1;
    const isError = this.state === "error";
    const isOff = this.state === "off";

    // Реактор «висит» в тёмной среде; внутренняя сборка по центру корпуса.
    const float = Math.sin(t * 0.45) * 0.01 * motion;
    this.housing.position.y = float;
    this.innerRing.position.y = INNER_LIFT + float;
    this.chamber.position.y = INNER_LIFT + float;
    this.particles?.position.setY(float);
    this.pulses?.position.setY(float);

    // Скорость энергии для импульсов по контуру.
    const spin = STATE_FLOW[this.state] * motion;
    this.innerRing.rotation.z = Math.sin(t * 0.14) * 0.01 * motion;

    // Цвета энергии: циан / красный аварии / приглушённый оффлайн.
    const targetMain = isError ? RED : isOff ? OFF : CYAN;
    const targetBright = isError ? RED_BRIGHT : isOff ? OFF_BRIGHT : ENERGY;

    // Пульс каналов: волна бежит по контуру.
    const wave = Math.sin(t * (2.2 + STATE_FLOW[this.state]) * motion);
    for (const { mesh, dim } of this.conduits) {
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.color.lerp(
        dim ? targetMain : tmpColor.copy(targetMain).lerp(targetBright, 0.55),
        Math.min(1, dt * 5),
      );
      const base = dim ? 0.35 : 0.75;
      mat.opacity = isOff
        ? 0.12
        : Math.min(1, base * (0.5 + this.energy * 0.7) + this.level * 0.3 + (dim ? 0 : wave * 0.12));
    }

    // Лазерные излучатели: заряд бежит по кольцам, лучи сходятся в ядре.
    const beamBase = isOff
      ? 0
      : isError
        ? 0.38
        : this.state === "processing"
          ? 0.72
          : this.state === "listening"
            ? 0.24 + this.level * 0.5
            : this.state === "speaking"
              ? 0.2 + this.level * 0.45
              : 0.1 + this.energy * 0.08;
    const beamPulse = 0.85 + Math.sin(t * (isError ? 8.5 : 3.2) * motion) * 0.15;
    const beamPower = Math.min(1, beamBase * beamPulse + this.level * 0.25);

    const beamCoreMat = this.m.beamCore;
    beamCoreMat.color.lerp(targetBright, Math.min(1, dt * 5));
    beamCoreMat.opacity = isOff ? 0 : Math.min(0.95, 0.18 + beamPower * 0.9);
    const beamGlowMat = this.m.beamGlow;
    beamGlowMat.color.lerp(targetMain, Math.min(1, dt * 5));
    beamGlowMat.opacity = isOff ? 0 : Math.min(0.7, 0.1 + beamPower * 0.6);

    this.m.lens.color.lerp(targetBright, Math.min(1, dt * 5));
    this.m.lens.opacity = isOff ? 0.1 : Math.min(1, 0.35 + this.energy * 0.4 + beamPower * 0.3);
    this.m.charge.color.lerp(targetMain, Math.min(1, dt * 5));
    this.m.charge.opacity = isOff ? 0.08 : Math.min(0.9, 0.18 + this.energy * 0.3 + beamPower * 0.45);
    this.m.flash.color.lerp(targetBright, Math.min(1, dt * 5));
    this.m.flash.opacity = isOff ? 0 : Math.min(0.8, 0.1 + beamPower * 0.6 + this.level * 0.15);

    const emitterSpeed = 1.5 + STATE_FLOW[this.state] * 0.7;
    for (const emitter of this.emitters) {
      const phase = t * emitterSpeed * motion + emitter.phase;
      const charge = Math.max(0, Math.sin(phase));
      emitter.lens.scale.setScalar(0.9 + charge * 0.16 + beamPower * 0.06);
      for (const ring of emitter.rings) {
        ring.scale.setScalar(0.88 + charge * 0.22 + beamPower * 0.05);
      }
      emitter.flash.scale.setScalar(0.04 + beamPower * 0.07 + charge * 0.012);

      // Импульс бежит от дула к поверхности ядра.
      const travel = (t * (0.35 + beamPower * 1.1) * motion + emitter.phase / (Math.PI * 2)) % 1;
      emitter.bolt.position.x =
        emitter.beamEnd + (emitter.beamStart - emitter.beamEnd) * (1 - travel);
      emitter.bolt.scale.setScalar(0.018 + beamPower * 0.03 + charge * 0.006);
    }

    // Гироскопические дуги: контр-вращение, в обработке ускоряются.
    const gyroSpeed = 0.5 + STATE_FLOW[this.state] * 0.9;
    for (const [i, arc] of this.gyroA.entries()) {
      const direction = i % 2 === 0 ? 1 : -0.7;
      arc.rotation.z += dt * gyroSpeed * direction * (1 + i * 0.1) * motion;
      const mat = arc.material as THREE.MeshBasicMaterial;
      mat.color.lerp(targetMain, Math.min(1, dt * 5));
      mat.opacity = isOff ? 0.06 : Math.min(0.7, 0.2 + this.energy * 0.35 + this.level * 0.25);
    }

    // Контейнмент-каркас: медленно вращается навстречу кристаллу и
    // сжимается, когда энергия растёт — ядро «держат» силовые линии.
    this.coreCage.rotation.y += dt * (isOff ? 0.04 : 0.16 + this.energy * 0.14) * motion;
    this.coreCage.rotation.x = Math.sin(t * 0.35) * 0.2 * motion;
    this.coreCage.scale.setScalar(0.94 + this.energy * 0.08 + this.level * 0.08);
    this.m.field.color.lerp(targetMain, Math.min(1, dt * 5));
    this.m.field.opacity = isOff
      ? 0.05
      : Math.min(0.45, 0.1 + this.energy * 0.16 + this.level * 0.12);
    this.m.fieldNode.color.lerp(targetBright, Math.min(1, dt * 5));
    this.m.fieldNode.opacity = isOff
      ? 0.08
      : Math.min(0.9, 0.35 + this.energy * 0.3 + this.level * 0.2);

    // Кристалл: дыхание + аудио + покачивание; контр-октаэдр навстречу.
    const breathe = 0.94 + Math.sin(t * (isError ? 2.6 : 1.5) * motion) * 0.05;
    this.crystal.rotation.y += dt * (isOff ? 0.08 : 0.4) * motion;
    this.crystal.rotation.x = Math.sin(t * 0.4) * 0.15 * motion;
    this.crystal.position.y = Math.sin(t * 1.1) * 0.008 * motion;
    const pulse = isOff ? breathe : breathe + this.level * 0.3;
    const s = pulse * (0.9 + this.energy * 0.18);
    this.crystal.scale.set(s, s, s * 0.62);

    // Рёбра кристалла живут вместе с ним и подчёркивают огранку.
    this.crystalEdges.position.copy(this.crystal.position);
    this.crystalEdges.rotation.copy(this.crystal.rotation);
    this.crystalEdges.scale.copy(this.crystal.scale);
    this.m.crystalEdge.color.lerp(targetBright, Math.min(1, dt * 5));
    this.m.crystalEdge.opacity = isOff
      ? 0.08
      : Math.min(0.8, 0.25 + this.energy * 0.35 + this.level * 0.25);

    this.innerCore.rotation.y -= dt * (isOff ? 0.1 : 0.9) * motion;
    this.innerCore.rotation.z += dt * 0.5 * motion;
    this.innerCore.scale.setScalar(s * 0.9);
    const innerMat = this.innerCore.material as THREE.MeshBasicMaterial;
    innerMat.color.lerp(targetBright, Math.min(1, dt * 5));
    innerMat.opacity = isOff ? 0.2 : Math.min(1, 0.45 + this.energy * 0.5 + this.level * 0.5);

    // Орбитальные осколки: группа задаёт плоскость орбиты, по ней летит меш.
    for (const [i, shard] of this.orbitShards.entries()) {
      const spec = this.orbitSpecs[i];
      const angle = t * spec.speed * motion + spec.phase;
      const radius = spec.radius + Math.sin(t * 0.8 + i) * 0.008;
      shard.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
      shard.rotation.set(t * 1.6 + i, t * 1.2 + i, 0);
    }
    // Материал на все три осколка общий — настраиваем его один раз за кадр.
    this.m.shard.color.lerp(targetBright, Math.min(1, dt * 5));
    this.m.shard.opacity = isOff ? 0.15 : 0.5 + this.energy * 0.4;

    this.m.crystal.emissive.lerp(targetBright, Math.min(1, dt * 5));
    this.m.crystal.emissiveIntensity = isOff ? 0.15 : 0.3 + this.energy * 0.55 + this.level * 0.7;

    this.m.deepBlue.emissive.lerp(isError ? RED_BRIGHT : DEEP_BLUE_GLOW, Math.min(1, dt * 4));
    this.m.deepBlue.emissiveIntensity = isOff ? 0.12 : 0.22 + this.energy * 0.4;

    this.m.halo.color.lerp(targetMain, Math.min(1, dt * 5));
    this.m.halo.opacity = isOff ? 0.08 : Math.min(0.6, 0.12 + this.energy * 0.32 + this.level * 0.22);
    this.halo.scale.setScalar(
      0.38 + this.energy * 0.28 + this.level * 0.2 + Math.sin(t * 1.2) * 0.025 * motion,
    );

    this.coreLight.color.lerp(targetMain, Math.min(1, dt * 5));
    // Мерцание ядра: два несоизмеримых синуса поверх дыхания.
    const flicker = (Math.sin(t * 13.7) * 0.5 + Math.sin(t * 7.3) * 0.5) * 0.05 * this.energy * motion;
    this.coreLight.intensity = isOff
      ? 0.15
      : 0.45 + this.energy * 1.0 + this.level * 0.9 + flicker;

    // Тепло вент-пазов: медленное дыхание, краснеет при аварии.
    const ventMat = this.m.ventGlow;
    ventMat.color.lerp(isError ? RED : targetMain, Math.min(1, dt * 4));
    ventMat.opacity = isOff
      ? 0.06
      : Math.min(0.6, 0.12 + this.energy * 0.3 + Math.max(0, Math.sin(t * 0.9)) * 0.12);

    // Энергетический шов корпуса дышит в такт каналу.
    const seamMat = this.m.edgeLine;
    seamMat.color.lerp(targetMain, Math.min(1, dt * 5));
    seamMat.opacity = isOff ? 0.05 : Math.min(0.4, 0.1 + this.energy * 0.2 + Math.abs(wave) * 0.08);

    // Искры по каналам: инстанс-матрицы обновляются одной пачкой.
    const flow = STATE_FLOW[this.state] * motion * (0.35 + this.energy * 0.8);
    if (this.particles) {
      for (const [i, p] of this.particleSpecs.entries()) {
        p.offset += dt * flow * p.speed * (p.outward ? 1.6 : 1);
        if (p.offset >= 1) p.offset %= 1;
        const point = p.lane.getPointAt(Math.min(0.999, p.offset));
        tmpObject.position.copy(point);
        const sparkScale = 0.7 + Math.sin(t * 6 + i * 1.3) * 0.3;
        tmpObject.scale.setScalar(sparkScale);
        tmpObject.rotation.set(t * 2 + i, t * 1.4 + i, 0);
        tmpObject.updateMatrix();
        this.particles.setMatrixAt(i, tmpObject.matrix);
      }
      this.particles.instanceMatrix.needsUpdate = true;
      this.m.particles.color.lerp(targetMain, Math.min(1, dt * 5));
      this.m.particles.opacity = isOff ? 0.1 : Math.min(0.9, 0.3 + this.energy * 0.55);
    }

    // Быстрые импульсы по основному контуру.
    if (this.pulses && this.pulseLane) {
      for (const [i, offset] of this.pulseOffsets.entries()) {
        this.pulseOffsets[i] = (offset + dt * 0.45 * (0.4 + spin) * motion) % 1;
        const point = this.pulseLane.getPointAt(Math.min(0.999, this.pulseOffsets[i]));
        tmpObject.position.copy(point);
        tmpObject.scale.setScalar(1 + Math.sin(t * 9 + i * 2) * 0.25);
        tmpObject.rotation.set(t * 3, t * 2, 0);
        tmpObject.updateMatrix();
        this.pulses.setMatrixAt(i, tmpObject.matrix);
      }
      this.pulses.instanceMatrix.needsUpdate = true;
      const pulseMat = this.pulses.material as THREE.MeshBasicMaterial;
      pulseMat.color.lerp(targetBright, Math.min(1, dt * 5));
      pulseMat.opacity = isOff ? 0.1 : Math.min(1, 0.5 + this.energy * 0.45);
    }

    // Заставка поверх рабочего кадра: развалка, сборка, доворот, камера.
    if (this.introActive) {
      const now = time || performance.now();
      if (this.introStartMs === 0) {
        this.introStartMs = now;
        this.pausedMs = 0;
      }
      this.introTime = Math.min(
        Math.max(0, (now - this.introStartMs - this.pausedMs) / 1000),
        INTRO_DURATION,
      );
      this.applyIntro(this.introTime);
      if (this.introTime >= INTRO_DURATION) this.finishIntro();
    }

    await renderer.renderAsync(this.scene, this.camera);
  }
}
