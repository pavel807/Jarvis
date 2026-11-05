import * as THREE from "three/webgpu";

/**
 * Материалы реактора Mark V.
 *
 * Отдельный модуль: геометрия не знает о материалах, сцена собирает их
 * один раз и анимирует только энергетические (emissive/basic) слоты.
 *
 * Палитра: титан с холодным отблеском, тёмный графит для утопленных
 * деталей, циановая энергия для каналов и ядра.
 */

export interface ReactorMaterials {
  /** Внешнее шасси: полированный титан. */
  titanium: THREE.MeshStandardMaterial;
  /** Внутреннее кольцо: шлифованный титан, грубее внешнего. */
  titaniumBrushed: THREE.MeshStandardMaterial;
  /** Утопленные компоненты: тёмный графит. */
  darkMetal: THREE.MeshStandardMaterial;
  /** Задняя пластина: почти чёрная сталь. */
  darkInk: THREE.MeshStandardMaterial;
  /** Дно энергоотсека: глубокий синий с лёгким свечением. */
  deepBlue: THREE.MeshStandardMaterial;
  /** Энергетические каналы: яркие, без tone mapping. */
  conduit: THREE.MeshBasicMaterial;
  conduitDim: THREE.MeshBasicMaterial;
  /** Кристалл ядра: фасеточный, полупрозрачный, светится изнутри. */
  crystal: THREE.MeshPhysicalMaterial;
  /** Стекло камеры ядра. */
  chamberGlass: THREE.MeshPhysicalMaterial;
  /** Ореол ядра (аддитивный спрайт) — цвет анимируется. */
  halo: THREE.SpriteMaterial;
  /** Внешний мягкий ореол: второй слой глубины вокруг ядра. */
  haloOuter: THREE.SpriteMaterial;
  /** Искры-частицы: фасеточные октаэдры, аддитивные. */
  particles: THREE.MeshBasicMaterial;
  /** Гироскопические дуги вокруг ядра. */
  arc: THREE.MeshBasicMaterial;
  /** Контейнмент-каркас ядра: объёмная сетка вокруг кристалла. */
  field: THREE.LineBasicMaterial;
  /** Узлы каркаса и наконечники дуг. */
  fieldNode: THREE.MeshBasicMaterial;
  /** Концентрические кольца и риски колодца. */
  wellRing: THREE.MeshBasicMaterial;
  /** Орбитальные осколки вокруг ядра. */
  shard: THREE.MeshBasicMaterial;
  /** Линза лазерного излучателя. */
  lens: THREE.MeshBasicMaterial;
  /** Тонкое ядро луча. */
  beamCore: THREE.MeshBasicMaterial;
  /** Мягкий ореол луча. */
  beamGlow: THREE.MeshBasicMaterial;
  /** Кольца заряда на стволе излучателя. */
  charge: THREE.MeshBasicMaterial;
  /** Вспышка на дуле. */
  flash: THREE.SpriteMaterial;
  /** Фасеточные рёбра кристалла: читают огранку на свету. */
  crystalEdge: THREE.LineBasicMaterial;
  /** Свечение вент-пазов (тепло). */
  ventGlow: THREE.MeshBasicMaterial;
  /** Тонкая кромка по контуру шасси. */
  edgeLine: THREE.LineBasicMaterial;
  /** Свет ядра. */
  coreLight: THREE.PointLight;
}

export function createMaterials(): ReactorMaterials {
  const titanium = new THREE.MeshStandardMaterial({
    color: 0xb9c6d0,
    metalness: 0.94,
    roughness: 0.3,
    envMapIntensity: 0.9,
  });

  const titaniumBrushed = new THREE.MeshStandardMaterial({
    color: 0x8497a4,
    metalness: 0.9,
    roughness: 0.46,
    envMapIntensity: 0.7,
  });

  const darkMetal = new THREE.MeshStandardMaterial({
    color: 0x25323d,
    metalness: 0.86,
    roughness: 0.54,
    flatShading: true,
    envMapIntensity: 0.5,
  });

  const darkInk = new THREE.MeshStandardMaterial({
    color: 0x111a22,
    metalness: 0.72,
    roughness: 0.62,
    envMapIntensity: 0.4,
  });

  const deepBlue = new THREE.MeshStandardMaterial({
    color: 0x0d2b3f,
    metalness: 0.55,
    roughness: 0.38,
    emissive: 0x0b4a68,
    emissiveIntensity: 0.4,
  });

  const conduit = new THREE.MeshBasicMaterial({
    color: 0x4bcbff,
    toneMapped: false,
    transparent: true,
    opacity: 0.95,
  });

  const conduitDim = new THREE.MeshBasicMaterial({
    color: 0x1e7fae,
    toneMapped: false,
    transparent: true,
    opacity: 0.55,
  });

  const crystal = new THREE.MeshPhysicalMaterial({
    color: 0x9fe8ff,
    metalness: 0,
    roughness: 0.1,
    transparent: true,
    opacity: 0.78,
    transmission: 0.62,
    thickness: 0.42,
    ior: 1.45,
    emissive: 0x9fe8ff,
    emissiveIntensity: 0.4,
    flatShading: true,
    toneMapped: false,
  });

  const chamberGlass = new THREE.MeshPhysicalMaterial({
    color: 0x9adfff,
    metalness: 0,
    roughness: 0.06,
    transparent: true,
    opacity: 0.1,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const halo = new THREE.SpriteMaterial({
    color: 0x4bcbff,
    transparent: true,
    opacity: 0.28,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const haloOuter = new THREE.SpriteMaterial({
    color: 0x4bcbff,
    transparent: true,
    opacity: 0.1,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const particles = new THREE.MeshBasicMaterial({
    color: 0x9ce4ff,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const arc = new THREE.MeshBasicMaterial({
    color: 0x4bcbff,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const field = new THREE.LineBasicMaterial({
    color: 0x7fdcff,
    transparent: true,
    opacity: 0.22,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const fieldNode = new THREE.MeshBasicMaterial({
    color: 0xd6f4ff,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const wellRing = new THREE.MeshBasicMaterial({
    color: 0x2ea8dd,
    transparent: true,
    opacity: 0.22,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
    side: THREE.DoubleSide,
  });

  const shard = new THREE.MeshBasicMaterial({
    color: 0xd6f4ff,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const lens = new THREE.MeshBasicMaterial({
    color: 0xd6f4ff,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const beamCore = new THREE.MeshBasicMaterial({
    color: 0xf4fdff,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const beamGlow = new THREE.MeshBasicMaterial({
    color: 0x4bcbff,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const charge = new THREE.MeshBasicMaterial({
    color: 0x4bcbff,
    transparent: true,
    opacity: 0.4,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const flash = new THREE.SpriteMaterial({
    color: 0xbdf2ff,
    transparent: true,
    opacity: 0.4,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const crystalEdge = new THREE.LineBasicMaterial({
    color: 0xd6f4ff,
    transparent: true,
    opacity: 0.45,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  });

  const ventGlow = new THREE.MeshBasicMaterial({
    color: 0x2ea8dd,
    transparent: true,
    opacity: 0.4,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
  });

  const edgeLine = new THREE.LineBasicMaterial({
    color: 0x4bcbff,
    transparent: true,
    opacity: 0.3,
    toneMapped: false,
  });

  const coreLight = new THREE.PointLight(0x4bcbff, 1.6, 7, 1.7);

  return {
    titanium,
    titaniumBrushed,
    darkMetal,
    darkInk,
    deepBlue,
    conduit,
    conduitDim,
    crystal,
    chamberGlass,
    halo,
    haloOuter,
    particles,
    arc,
    field,
    fieldNode,
    wellRing,
    shard,
    lens,
    beamCore,
    beamGlow,
    charge,
    flash,
    crystalEdge,
    ventGlow,
    edgeLine,
    coreLight,
  };
}

/** Текстура мягкого свечения для ореола и частиц. */
export function glowTexture(inner: string, outer: string): THREE.CanvasTexture {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, inner);
  gradient.addColorStop(0.3, inner);
  gradient.addColorStop(1, outer);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
