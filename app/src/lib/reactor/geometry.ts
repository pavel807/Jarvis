import * as THREE from "three/webgpu";

/**
 * Геометрия реактора Mark V.
 *
 * Основа — РАВНОСТОРОННИЙ треугольник (вершина вниз, основание сверху).
 * Вершины лежат на описанной ОКРУЖНОСТИ радиуса R в параметрических углах
 * −90°, 30°, 150°: нижняя на оси Y, верхние симметричны относительно неё.
 * Все три стороны равны √3·R, основания углов симметричны оси.
 *
 * Каждый угол срезан на одинаковое расстояние `cut` вдоль обеих смежных
 * сторон — контур из шести рёбер: три равных стороны по (√3·R − 2·cut) и три
 * равных скоса по `cut`. Срезы — настоящая геометрия экструзии, а не линии.
 *
 * Все слои реактора — масштабированные копии этого контура (R и cut растут
 * вместе, поэтому слои самоподобны); внутренняя сборка поднимается над
 * центром силами сцены (INNER_LIFT).
 * Плоскость XY, глубина по Z.
 */

/**
 * Параметрические углы вершин: низ, право, лево. Это не углы треугольника,
 * а точки на окружности — реальные направления сторон задаёт контур,
 * который из них построен.
 */
export function cornerAngles(): number[] {
  return [-Math.PI / 2, Math.PI / 6, (Math.PI * 5) / 6];
}

/** Вершины равностороннего треугольника (вершина вниз) на описанной окружности. */
export function trianglePoints(circumradius: number): THREE.Vector2[] {
  return cornerAngles().map(
    (a) => new THREE.Vector2(Math.cos(a) * circumradius, Math.sin(a) * circumradius),
  );
}

/**
 * Шесть точек контура: углы равностороннего треугольника, срезанные
 * на `cut` от каждой вершины вдоль смежных сторон.
 */
export function truncatedTrianglePoints(circumradius: number, cut: number): THREE.Vector2[] {
  const verts = trianglePoints(circumradius);
  const out: THREE.Vector2[] = [];
  for (let i = 0; i < 3; i++) {
    const prev = verts[(i + 2) % 3];
    const v = verts[i];
    const next = verts[(i + 1) % 3];
    const toPrev = prev.clone().sub(v).normalize().multiplyScalar(cut);
    const toNext = next.clone().sub(v).normalize().multiplyScalar(cut);
    out.push(v.clone().add(toPrev), v.clone().add(toNext));
  }
  return out;
}

/** Контур со срезанными углами как Shape. */
export function truncatedTriangleShape(circumradius: number, cut: number): THREE.Shape {
  const shape = new THREE.Shape();
  const points = truncatedTrianglePoints(circumradius, cut);
  shape.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) shape.lineTo(points[i].x, points[i].y);
  shape.closePath();
  return shape;
}

/** Кольцо между двумя срезанными контурами (рама). Дыра — обратный обход. */
export function truncatedRingShape(
  outerRadius: number,
  outerCut: number,
  innerRadius: number,
  innerCut: number,
): THREE.Shape {
  const shape = truncatedTriangleShape(outerRadius, outerCut);
  const hole = new THREE.Path();
  const reversed = [...truncatedTrianglePoints(innerRadius, innerCut)].reverse();
  hole.moveTo(reversed[0].x, reversed[0].y);
  for (let i = 1; i < reversed.length; i++) hole.lineTo(reversed[i].x, reversed[i].y);
  hole.closePath();
  shape.holes.push(hole);
  return shape;
}

/** Экструдированная рама с фаской: основной профиль корпуса. */
export function frameGeometry(
  outerRadius: number,
  outerCut: number,
  innerRadius: number,
  innerCut: number,
  depth: number,
): THREE.ExtrudeGeometry {
  const geo = new THREE.ExtrudeGeometry(
    truncatedRingShape(outerRadius, outerCut, innerRadius, innerCut),
    {
      depth,
      bevelEnabled: true,
      bevelThickness: depth * 0.14,
      bevelSize: depth * 0.1,
      bevelSegments: 2,
      curveSegments: 1,
    },
  );
  geo.translate(0, 0, -depth / 2);
  return geo;
}

/** Сплошная пластина со срезанными углами. */
export function plateGeometry(circumradius: number, cut: number, depth: number): THREE.ExtrudeGeometry {
  const geo = new THREE.ExtrudeGeometry(truncatedTriangleShape(circumradius, cut), {
    depth,
    bevelEnabled: true,
    bevelThickness: depth * 0.2,
    bevelSize: depth * 0.16,
    bevelSegments: 1,
    curveSegments: 1,
  });
  geo.translate(0, 0, -depth / 2);
  return geo;
}

/** Радиальная спица-кронштейн: трапеция вдоль направления `angle`. */
export function spokeGeometry(
  outerRadius: number,
  innerRadius: number,
  outerWidth: number,
  innerWidth: number,
  depth: number,
): THREE.ExtrudeGeometry {
  const shape = new THREE.Shape();
  shape.moveTo(-outerWidth / 2, outerRadius);
  shape.lineTo(outerWidth / 2, outerRadius);
  shape.lineTo(innerWidth / 2, innerRadius);
  shape.lineTo(-innerWidth / 2, innerRadius);
  shape.closePath();
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelThickness: depth * 0.18,
    bevelSize: depth * 0.12,
    bevelSegments: 1,
    curveSegments: 1,
  });
  geo.translate(0, 0, -depth / 2);
  return geo;
}

/** Замкнутый путь по готовому набору точек (каналы, частицы). */
export function polygonCurve(points: THREE.Vector2[], z = 0): THREE.CurvePath<THREE.Vector3> {
  const path = new THREE.CurvePath<THREE.Vector3>();
  const pts = points.map((p) => new THREE.Vector3(p.x, p.y, z));
  for (let i = 0; i < pts.length; i++) {
    path.add(new THREE.LineCurve3(pts[i], pts[(i + 1) % pts.length]));
  }
  return path;
}

/** Трубка-канал вдоль полигонального контура. */
export function channelGeometry(points: THREE.Vector2[], radius: number): THREE.TubeGeometry {
  return new THREE.TubeGeometry(polygonCurve(points), 18, radius, 10, true);
}

/** Радиальный луч от `from` к `to` вдоль направления угла `angle`. */
export function radialCurve(angle: number, from: number, to: number, z = 0): THREE.LineCurve3 {
  const dir = new THREE.Vector2(Math.cos(angle), Math.sin(angle));
  return new THREE.LineCurve3(
    new THREE.Vector3(dir.x * from, dir.y * from, z),
    new THREE.Vector3(dir.x * to, dir.y * to, z),
  );
}

/**
 * Центры трёх срезанных углов контура `points` — места угловых узлов.
 * Скос вершины i — ребро между точками (2i) и (2i+1).
 */
export function chamferCenters(points: THREE.Vector2[]): THREE.Vector2[] {
  const centers: THREE.Vector2[] = [];
  for (let i = 0; i < 3; i++) {
    const a = points[i * 2];
    const b = points[i * 2 + 1];
    centers.push(new THREE.Vector2((a.x + b.x) / 2, (a.y + b.y) / 2));
  }
  return centers;
}

/** Направления трёх срезанных углов (углы для спиц и катушек). */
export function cornerDirections(points: THREE.Vector2[]): number[] {
  return chamferCenters(points).map((c) => Math.atan2(c.y, c.x));
}

/** Середины трёх длинных рёбер (2i+1 → 2i+2) — крепёж граней. */
export function longEdgeMidpoints(points: THREE.Vector2[]): THREE.Vector2[] {
  const mids: THREE.Vector2[] = [];
  for (let i = 0; i < 3; i++) {
    const a = points[(i * 2 + 1) % points.length];
    const b = points[(i * 2 + 2) % points.length];
    mids.push(new THREE.Vector2((a.x + b.x) / 2, (a.y + b.y) / 2));
  }
  return mids;
}

/** Угол наклона длинного ребра i — для ориентации пазов и винтов. */
export function longEdgeAngle(points: THREE.Vector2[], i: number): number {
  const a = points[(i * 2 + 1) % points.length];
  const b = points[(i * 2 + 2) % points.length];
  return Math.atan2(b.y - a.y, b.x - a.x);
}
