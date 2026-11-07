/**
 * Three.js view of a hole: the green as a small floating diorama, the ball,
 * the aim arrow, and the playback of a shot.
 *
 * Course coordinates (x, y) in metres map to three.js (x, h(x, y), −y), so the
 * course keeps its orientation seen from above.
 */
import * as THREE from "three";
import { height, gradient, PHYSICS, type Hole, type Playback, type Vec2 } from "../golf";

export interface Aim {
  /** Radians, counter-clockwise from +x. */
  angle: number;
  /** In [0, 1]. */
  power: number;
}

export interface RenderStats {
  fps: number;
  drawCalls: number;
  triangles: number;
  pixelRatio: number;
}

const RAIL_HEIGHT = 0.07;
const RAIL_WIDTH = 0.08;
const BLOCK_HEIGHT = 0.08;
const HEDGE_HEIGHT = 0.16;
const BASE_DEPTH = 0.18;
const CELL = 0.1;
const DRAG_THRESHOLD_PX = 6;
const ORBIT_SPEED = 0.006;
const MIN_PITCH = 0.25;
const MAX_PITCH = 1.45;
/** Pulling back this fraction of the camera distance gives full power. */
const FULL_PULL = 0.22;

const COLORS = {
  green: new THREE.Color("#4d9a4f"),
  stripe: new THREE.Color("#58a95a"),
  tee: new THREE.Color("#3d7f40"),
  water: "#3b82c4",
  rail: "#c4935f",
  block: "#b5835a",
  hedge: "#2f6b3a",
  base: "#6b5a4a",
  bumper: "#e2463b",
  bumperTop: "#f6f6f6",
  cup: "#141414",
  ball: "#fafafa",
  flag: "#e2463b",
};

const toWorld = (hole: Hole, x: number, y: number, lift = 0) =>
  new THREE.Vector3(x, height(hole.terrain, x, y) + lift, -y);

/** Sutherland–Hodgman: `poly` clipped to the convex counter-clockwise `clip`. */
function clipConvex(poly: Vec2[], clip: Vec2[]): Vec2[] {
  let out = poly;
  for (let i = 0; i < clip.length && out.length; i++) {
    const a = clip[i];
    const b = clip[(i + 1) % clip.length];
    const side = (p: Vec2) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const p = input[j];
      const q = input[(j + 1) % input.length];
      const sp = side(p);
      const sq = side(q);
      if (sp >= 0) out.push(p);
      if (sp >= 0 !== sq >= 0) {
        const f = sp / (sp - sq);
        out.push([p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f]);
      }
    }
  }
  return out;
}

/** The convex polygon pushed outward by `d` (counter-clockwise input). */
function offsetConvex(p: Vec2[], d: number): Vec2[] {
  const n = p.length;
  const normals = p.map((a, i) => {
    const b = p[(i + 1) % n];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    return [(b[1] - a[1]) / len, -(b[0] - a[0]) / len] as Vec2;
  });
  return p.map((v, i) => {
    const n0 = normals[(i + n - 1) % n];
    const n1 = normals[i];
    const k = d / (1 + n0[0] * n1[0] + n0[1] * n1[1]);
    return [v[0] + (n0[0] + n1[0]) * k, v[1] + (n0[1] + n1[1]) * k];
  });
}

/** Points along the closed polygon, at most `step` apart. */
function subdivide(p: Vec2[], step: number): Vec2[] {
  const out: Vec2[] = [];
  p.forEach((a, i) => {
    const b = p[(i + 1) % p.length];
    const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
    for (let k = 0; k < n; k++) out.push([a[0] + ((b[0] - a[0]) * k) / n, a[1] + ((b[1] - a[1]) * k) / n]);
  });
  return out;
}

function bounds(p: Vec2[]) {
  const xs = p.map((v) => v[0]);
  const ys = p.map((v) => v[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
}

class GeometryBuilder {
  positions: number[] = [];
  normals: number[] = [];
  colors: number[] = [];

  vertex(p: THREE.Vector3, n: THREE.Vector3, c?: THREE.Color) {
    this.positions.push(p.x, p.y, p.z);
    this.normals.push(n.x, n.y, n.z);
    if (c) this.colors.push(c.r, c.g, c.b);
  }

  /** A flat-shaded quad a b c d (counter-clockwise seen from outside). */
  quad(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3) {
    const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(d, a)).normalize();
    for (const p of [a, b, c, a, c, d]) this.vertex(p, n);
  }

  build(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.positions, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.normals, 3));
    if (this.colors.length) g.setAttribute("color", new THREE.Float32BufferAttribute(this.colors, 3));
    return g;
  }
}

/** The terrain over a convex region, on a grid, with smooth analytic normals. */
function surface(hole: Hole, region: Vec2[], lift: number, color?: (x: number, y: number) => THREE.Color) {
  const g = new GeometryBuilder();
  const { x0, y0, x1, y1 } = bounds(region);
  const normal = (x: number, y: number) => {
    const [hx, hy] = gradient(hole.terrain, x, y);
    return new THREE.Vector3(-hx, 1, hy).normalize();
  };
  for (let x = x0; x < x1; x += CELL)
    for (let y = y0; y < y1; y += CELL) {
      const cell = clipConvex(
        [
          [x, y],
          [x + CELL, y],
          [x + CELL, y + CELL],
          [x, y + CELL],
        ],
        region,
      );
      if (cell.length < 3) continue;
      const c = color?.(x + CELL / 2, y + CELL / 2);
      // Course counter-clockwise is three.js counter-clockwise seen from above.
      for (let i = 1; i + 1 < cell.length; i++)
        for (const [px, py] of [cell[0], cell[i], cell[i + 1]]) g.vertex(toWorld(hole, px, py, lift), normal(px, py), c);
    }
  return g.build();
}

/** A wall around a closed polygon: top following the terrain, from `bottom` up. */
function prism(hole: Hole, outline: Vec2[], top: number, bottom: (x: number, y: number) => number, g: GeometryBuilder) {
  const pts = subdivide(outline, CELL);
  const up = (p: Vec2) => toWorld(hole, p[0], p[1], top);
  const down = (p: Vec2) => new THREE.Vector3(p[0], bottom(p[0], p[1]), -p[1]);
  pts.forEach((a, i) => {
    const b = pts[(i + 1) % pts.length];
    g.quad(down(a), down(b), up(b), up(a));
  });
  return pts;
}

/** Height between two contour lines on the green, m. */
const CONTOUR = 0.02;

/** Thin darker lines at heights (k + ½)·CONTOUR, to read the slopes (none on flat ground at 0). */
function contoured(material: THREE.MeshStandardMaterial) {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vHeight;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvHeight = (modelMatrix * vec4(transformed, 1.0)).y;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vHeight;")
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
        float level = vHeight / ${CONTOUR.toFixed(4)};
        float dist = abs(fract(level) - 0.5);
        float line = 1.0 - smoothstep(0.0, 1.2 * fwidth(level), dist);
        diffuseColor.rgb *= 1.0 - 0.3 * line;`,
      );
  };
  return material;
}

export class CourseView extends EventTarget {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(38, 1, 0.05, 200);
  private readonly world = new THREE.Group();
  private readonly courseGroup = new THREE.Group();
  private readonly ball: THREE.Mesh;
  private readonly arrow: THREE.Mesh;
  private readonly arrowMaterial: THREE.MeshBasicMaterial;
  private readonly sun: THREE.DirectionalLight;
  private readonly raycaster = new THREE.Raycaster();
  private readonly resizeObserver: ResizeObserver;

  private hole: Hole | null = null;
  private ballPos: Vec2 = [0, 0];
  private aim: Aim | null = null;
  private aimEnabled = false;
  private target = new THREE.Vector3();
  /** The course direction the camera looks along, radians. */
  private heading = Math.PI / 2;
  private pitch = 0.75;
  private distance = 8;
  /** The user zoomed: keep their distance on resize. */
  private userZoomed = false;
  private pointer:
    | { mode: "orbit"; id: number; x: number; y: number; moved: boolean }
    | { mode: "pan"; id: number; grab: THREE.Vector3 }
    | { mode: "aim"; id: number }
    | null = null;
  private playing: {
    pb: Playback;
    start: number;
    index: number;
    sink: number;
    resolve: () => void;
  } | null = null;

  /** Keyboard panning: the velocity the held keys ask for, screen-relative, and the one reached. */
  private panIntent: [number, number] = [0, 0];
  private panVelocity: [number, number] = [0, 0];
  private lastFrame: number | null = null;
  private frames = 0;
  private fpsWindowStart = performance.now();
  private fps = 0;

  constructor(private readonly container: HTMLElement) {
    super();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    const canvas = this.renderer.domElement;
    canvas.style.touchAction = "none";
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    container.appendChild(canvas);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7a66, 1.3));
    this.sun = new THREE.DirectionalLight(0xffffff, 2.8);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.bias = -0.0005;
    this.sun.shadow.normalBias = 0.01;
    this.scene.add(this.sun, this.sun.target);
    this.scene.add(this.world);
    this.world.add(this.courseGroup);

    this.ball = new THREE.Mesh(
      new THREE.SphereGeometry(PHYSICS.ballRadius, 32, 16),
      new THREE.MeshStandardMaterial({ color: COLORS.ball, roughness: 0.35 }),
    );
    this.ball.castShadow = true;
    this.world.add(this.ball);

    // A flat arrow along +x, from the ball outwards, scaled by power.
    const shape = new THREE.Shape();
    shape.moveTo(0.06, -0.012);
    shape.lineTo(0.84, -0.012);
    shape.lineTo(0.84, -0.04);
    shape.lineTo(1, 0);
    shape.lineTo(0.84, 0.04);
    shape.lineTo(0.84, 0.012);
    shape.lineTo(0.06, 0.012);
    const arrowGeometry = new THREE.ShapeGeometry(shape);
    arrowGeometry.rotateX(-Math.PI / 2);
    this.arrowMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9, depthTest: false });
    this.arrow = new THREE.Mesh(arrowGeometry, this.arrowMaterial);
    this.arrow.renderOrder = 10;
    this.arrow.visible = false;
    this.world.add(this.arrow);

    canvas.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    canvas.addEventListener("pointerup", (e) => this.onPointerUp(e));
    canvas.addEventListener("pointercancel", () => this.cancelPointer());
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.renderer.setAnimationLoop((t) => this.frame(t));
  }

  get busy() {
    return this.playing !== null;
  }

  /** Shows a hole, the ball at `ball`, and frames the camera on it. */
  setHole(hole: Hole, ball: Vec2) {
    this.finishPlayback();
    for (const child of [...this.courseGroup.children]) {
      this.courseGroup.remove(child);
      child.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
        }
      });
    }
    this.hole = hole;
    this.buildCourse(hole);
    this.frameCamera(hole);
    this.setBall(ball);
  }

  setBall(at: Vec2, visible = true) {
    this.ballPos = at;
    this.ball.visible = visible;
    if (this.hole) this.ball.position.copy(toWorld(this.hole, at[0], at[1], PHYSICS.ballRadius));
    this.updateArrow();
  }

  /** Whether dragging from the ball aims a shot. */
  setAimEnabled(enabled: boolean) {
    this.aimEnabled = enabled;
    if (!enabled && this.pointer?.mode === "aim") this.pointer = null;
    this.updateArrow();
  }

  setAim(aim: Aim | null) {
    this.aim = aim;
    this.updateArrow();
  }

  getAim(): Aim | null {
    return this.aim;
  }

  /** Animates a shot; resolves once the ball has stopped (or dropped). */
  play(pb: Playback): Promise<void> {
    this.finishPlayback();
    return new Promise((resolve) => {
      this.playing = { pb, start: performance.now(), index: 0, sink: 0, resolve };
      this.setAim(null);
    });
  }

  stats(): RenderStats {
    const info = this.renderer.info.render;
    return {
      fps: this.fps,
      drawCalls: info.calls,
      triangles: info.triangles,
      pixelRatio: this.renderer.getPixelRatio(),
    };
  }

  // --- Scene ------------------------------------------------------------------------

  private buildCourse(hole: Hole) {
    const spec = hole.spec;
    const add = (geometry: THREE.BufferGeometry, material: THREE.Material, shadows = true) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.receiveShadow = true;
      mesh.castShadow = shadows;
      this.courseGroup.add(mesh);
      return mesh;
    };

    // Lowest point of the playing area, for the base.
    const { x0, y0, x1, y1 } = bounds(spec.boundary);
    let low = Infinity;
    for (let x = x0; x <= x1; x += CELL) for (let y = y0; y <= y1; y += CELL) low = Math.min(low, height(hole.terrain, x, y));
    for (const [x, y] of spec.boundary) low = Math.min(low, height(hole.terrain, x, y));
    const baseY = low - BASE_DEPTH;

    // The green, mown in stripes, and the tee mat.
    const [tx, ty] = spec.tee;
    const green = surface(hole, spec.boundary, 0, (x, y) => {
      if (Math.abs(x - tx) < 0.18 && Math.abs(y - ty) < 0.18) return COLORS.tee;
      const along = Math.floor(((x - x0) * 0 + (y - y0)) / 0.5);
      return along % 2 ? COLORS.stripe : COLORS.green;
    });
    add(green, contoured(new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95 })), false);

    for (const pond of spec.water ?? []) {
      const region = clipConvex(pond, spec.boundary);
      if (region.length < 3) continue;
      add(
        surface(hole, region, 0.004),
        new THREE.MeshStandardMaterial({ color: COLORS.water, roughness: 0.15, metalness: 0.1, transparent: true, opacity: 0.88 }),
        false,
      );
    }

    // Rails: a ring between the boundary and its outward offset, plus the base below.
    const wood = new THREE.MeshStandardMaterial({ color: COLORS.rail, roughness: 0.7 });
    const rail = new GeometryBuilder();
    const inner = spec.boundary;
    const outer = offsetConvex(inner, RAIL_WIDTH);
    const n = inner.length;
    for (let i = 0; i < n; i++) {
      const a = inner[i];
      const b = inner[(i + 1) % n];
      const a2 = outer[i];
      const b2 = outer[(i + 1) % n];
      const steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / CELL));
      for (let k = 0; k < steps; k++) {
        const lerp = (p: Vec2, q: Vec2, f: number): Vec2 => [p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f];
        const p0 = lerp(a, b, k / steps);
        const p1 = lerp(a, b, (k + 1) / steps);
        const q0 = lerp(a2, b2, k / steps);
        const q1 = lerp(a2, b2, (k + 1) / steps);
        // Rail height follows the terrain along the inner edge.
        const top = (p: Vec2, ref: Vec2) => new THREE.Vector3(p[0], height(hole.terrain, ref[0], ref[1]) + RAIL_HEIGHT, -p[1]);
        const foot = (p: Vec2) => toWorld(hole, p[0], p[1], -0.01);
        const bottom = (p: Vec2) => new THREE.Vector3(p[0], baseY, -p[1]);
        rail.quad(top(p0, p0), top(q0, p0), top(q1, p1), top(p1, p1));
        rail.quad(foot(p0), top(p0, p0), top(p1, p1), foot(p1));
        rail.quad(bottom(q0), bottom(q1), top(q1, p1), top(q0, p0));
      }
    }
    add(rail.build(), wood);

    // Underside of the diorama.
    const base = new GeometryBuilder();
    const down = new THREE.Vector3(0, -1, 0);
    for (let i = 1; i + 1 < n; i++)
      for (const p of [outer[0], outer[i + 1], outer[i]]) base.vertex(new THREE.Vector3(p[0], baseY, -p[1]), down);
    add(base.build(), new THREE.MeshStandardMaterial({ color: COLORS.base, roughness: 1 }), false);

    // Blocks, hedges and bumpers.
    const materials = {
      block: new THREE.MeshStandardMaterial({ color: COLORS.block, roughness: 0.75 }),
      hedge: new THREE.MeshStandardMaterial({ color: COLORS.hedge, roughness: 1 }),
      bumper: new THREE.MeshStandardMaterial({ color: COLORS.bumper, roughness: 0.45 }),
      bumperTop: new THREE.MeshStandardMaterial({ color: COLORS.bumperTop, roughness: 0.5 }),
    };
    const up = new THREE.Vector3(0, 1, 0);
    /** A flat polygon at `lift` above the terrain, fanned from its centroid. */
    const lid = (pts: Vec2[], lift: number, g: GeometryBuilder) => {
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      const center = toWorld(hole, cx, cy, lift);
      pts.forEach((a, i) => {
        const b = pts[(i + 1) % pts.length];
        for (const p of [center, toWorld(hole, a[0], a[1], lift), toWorld(hole, b[0], b[1], lift)]) g.vertex(p, up);
      });
    };
    for (const o of hole.obstacles) {
      if (o.kind !== "block") continue;
      const style = o.style ?? "block";
      const top = style === "hedge" ? HEDGE_HEIGHT : style === "bumper" ? BLOCK_HEIGHT + 0.03 : BLOCK_HEIGHT;
      const g = new GeometryBuilder();
      lid(prism(hole, o.points, top, (x, y) => height(hole.terrain, x, y) - 0.02, g), top, g);
      add(g.build(), materials[style]);
      if (style === "bumper") {
        // A white octagon on top, half the size.
        const cx = o.points.reduce((s, p) => s + p[0], 0) / o.points.length;
        const cy = o.points.reduce((s, p) => s + p[1], 0) / o.points.length;
        const cap = new GeometryBuilder();
        lid(o.points.map(([x, y]) => [cx + (x - cx) * 0.55, cy + (y - cy) * 0.55] as Vec2), top + 0.002, cap);
        add(cap.build(), materials.bumperTop, false);
      }
    }

    // The cup and its flag.
    const [cx, cy] = spec.cup;
    const cupY = height(hole.terrain, cx, cy);
    const [gx, gy] = gradient(hole.terrain, cx, cy);
    const cup = new THREE.Mesh(
      new THREE.CircleGeometry(PHYSICS.cupRadius, 40),
      new THREE.MeshStandardMaterial({ color: COLORS.cup, roughness: 1, polygonOffset: true, polygonOffsetFactor: -2 }),
    );
    cup.lookAt(new THREE.Vector3(-gx, 1, gy));
    cup.position.set(cx, cupY + 0.001, -cy);
    this.courseGroup.add(cup);
    const pole = new THREE.Mesh(
      new THREE.CylinderGeometry(0.005, 0.005, 0.55, 8),
      new THREE.MeshStandardMaterial({ color: "#e7e5e4", roughness: 0.4 }),
    );
    pole.position.set(cx, cupY + 0.275, -cy);
    pole.castShadow = true;
    const flagShape = new THREE.Shape();
    flagShape.moveTo(0, 0);
    flagShape.lineTo(0.2, -0.06);
    flagShape.lineTo(0, -0.12);
    const flag = new THREE.Mesh(
      new THREE.ShapeGeometry(flagShape),
      new THREE.MeshStandardMaterial({ color: COLORS.flag, side: THREE.DoubleSide, roughness: 0.6 }),
    );
    flag.position.set(cx, cupY + 0.55, -cy);
    flag.castShadow = true;
    this.courseGroup.add(pole, flag);

    // Shadow camera around the hole.
    const size = Math.max(x1 - x0, y1 - y0) / 2 + 0.5;
    const mid = new THREE.Vector3((x0 + x1) / 2, 0, -(y0 + y1) / 2);
    this.sun.position.copy(mid).add(new THREE.Vector3(-3, 8, 4));
    this.sun.target.position.copy(mid);
    const cam = this.sun.shadow.camera;
    cam.left = cam.bottom = -size * 1.3;
    cam.right = cam.top = size * 1.3;
    cam.near = 0.1;
    cam.far = 30;
    cam.updateProjectionMatrix();
  }

  /** Looks down the hole from behind the tee. */
  private frameCamera(hole: Hole) {
    const { x0, y0, x1, y1 } = bounds(hole.spec.boundary);
    this.target.set((x0 + x1) / 2, height(hole.terrain, (x0 + x1) / 2, (y0 + y1) / 2), -(y0 + y1) / 2);
    const [tx, ty] = hole.spec.tee;
    const [cx, cy] = hole.spec.cup;
    // Behind the tee, looking towards the cup.
    this.heading = Math.atan2(cy - ty, cx - tx);
    this.pitch = 0.8;
    this.userZoomed = false;
    this.fitDistance(hole);
  }

  /** The closest distance at which the whole hole (rails included) is in view. */
  private fitDistance(hole: Hole) {
    const corners = offsetConvex(hole.spec.boundary, RAIL_WIDTH).flatMap(([x, y]) => [
      toWorld(hole, x, y, RAIL_HEIGHT),
      toWorld(hole, x, y, -BASE_DEPTH),
    ]);
    const fits = (d: number) => {
      this.distance = d;
      this.updateCamera();
      this.camera.updateMatrixWorld();
      return corners.every((c) => {
        const p = c.clone().project(this.camera);
        return Math.abs(p.x) < 0.94 && Math.abs(p.y) < 0.9 && p.z < 1;
      });
    };
    let lo = 0.5;
    let hi = 60;
    for (let i = 0; i < 30; i++) {
      const mid = (lo + hi) / 2;
      if (fits(mid)) hi = mid;
      else lo = mid;
    }
    this.distance = hi;
    this.updateCamera();
  }

  private updateCamera() {
    const cp = Math.cos(this.pitch);
    // Course direction (cos, sin) is world (cos, 0, −sin); the camera sits behind it.
    this.camera.position.set(
      this.target.x - this.distance * cp * Math.cos(this.heading),
      this.target.y + this.distance * Math.sin(this.pitch),
      this.target.z + this.distance * cp * Math.sin(this.heading),
    );
    this.camera.lookAt(this.target);
  }

  private updateArrow() {
    const show = this.aimEnabled && this.aim !== null && this.aim.power > 0.01 && this.hole !== null;
    this.arrow.visible = show;
    if (!show) return;
    const { angle, power } = this.aim!;
    // Length: the distance the ball would roll on flat ground, capped for display.
    const length = Math.max(0.15, 0.25 + power * 1.6);
    this.arrow.position.copy(this.ball.position).setY(this.ball.position.y - PHYSICS.ballRadius + 0.012);
    this.arrow.rotation.set(0, angle, 0);
    this.arrow.scale.set(length, 1, 1 + power * 0.8);
    this.arrowMaterial.color.setHSL((1 - power) * 0.33, 0.85, 0.5);
  }

  // --- Input ------------------------------------------------------------------------

  private ndc(e: PointerEvent | WheelEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  }

  /** The course point under the pointer, on the horizontal plane through the ball. */
  private groundPoint(e: PointerEvent): Vec2 | null {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -this.ball.position.y);
    const hit = this.raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    return hit ? [hit.x, -hit.z] : null;
  }

  private nearBall(e: PointerEvent): boolean {
    const p = this.ball.position.clone().project(this.camera);
    const rect = this.renderer.domElement.getBoundingClientRect();
    const sx = ((p.x + 1) / 2) * rect.width + rect.left;
    const sy = ((1 - p.y) / 2) * rect.height + rect.top;
    return Math.hypot(e.clientX - sx, e.clientY - sy) < 36;
  }

  /** The point under the pointer on the horizontal plane through the view's target. */
  private targetPlanePoint(e: PointerEvent): THREE.Vector3 | null {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -this.target.y);
    return this.raycaster.ray.intersectPlane(plane, new THREE.Vector3());
  }

  /**
   * Pans continuously while keys are held: `dx` right and `dy` forward as seen
   * on screen, each in [−1, 1], at a speed that follows the zoom.
   */
  setPanIntent(dx: number, dy: number) {
    const norm = Math.hypot(dx, dy);
    this.panIntent = norm > 1 ? [dx / norm, dy / norm] : [dx, dy];
  }

  /** Eases the panning velocity towards the held keys' and moves the view by it. */
  private stepPan(dt: number) {
    const speed = 0.6 * this.distance;
    // Reaches the held keys' velocity, or a stop, in about a tenth of a second.
    const ease = 1 - Math.exp(-dt / 0.08);
    const v = this.panVelocity;
    for (const i of [0, 1]) {
      v[i] += (this.panIntent[i] * speed - v[i]) * ease;
      if (this.panIntent[i] === 0 && Math.abs(v[i]) < 1e-3) v[i] = 0;
    }
    if (v[0] || v[1]) this.pan(v[0] * dt, v[1] * dt);
  }

  /** Moves the view across the course, `dx` right and `dy` forward as seen on screen, in metres. */
  pan(dx: number, dy: number) {
    const [c, s] = [Math.cos(this.heading), Math.sin(this.heading)];
    // Forward is the heading (course (c, s) → world (c, −s)); right is a quarter turn clockwise.
    this.moveTarget(dy * c + dx * s, -(dy * s) + dx * c);
  }

  private moveTarget(wx: number, wz: number) {
    if (!this.hole) return;
    const { x0, y0, x1, y1 } = bounds(this.hole.spec.boundary);
    const margin = 1;
    this.target.x = Math.min(x1 + margin, Math.max(x0 - margin, this.target.x + wx));
    this.target.z = Math.min(-y0 + margin, Math.max(-y1 - margin, this.target.z + wz));
    this.updateCamera();
  }

  private onPointerDown(e: PointerEvent) {
    if (this.pointer) return;
    // Right or middle button, or shift with the left one: pan.
    const panning = e.pointerType === "mouse" && (e.button === 1 || e.button === 2 || (e.button === 0 && e.shiftKey));
    if (e.pointerType === "mouse" && e.button !== 0 && !panning) return;
    this.renderer.domElement.setPointerCapture(e.pointerId);
    const grab = panning ? this.targetPlanePoint(e) : null;
    if (grab) {
      this.pointer = { mode: "pan", id: e.pointerId, grab };
      this.renderer.domElement.style.cursor = "move";
    } else if (this.aimEnabled && this.ball.visible && this.nearBall(e)) {
      this.pointer = { mode: "aim", id: e.pointerId };
      this.renderer.domElement.style.cursor = "grabbing";
    } else {
      this.pointer = { mode: "orbit", id: e.pointerId, x: e.clientX, y: e.clientY, moved: false };
    }
  }

  private onPointerMove(e: PointerEvent) {
    if (!this.pointer) {
      this.renderer.domElement.style.cursor = this.aimEnabled && this.ball.visible && this.nearBall(e) ? "grab" : "";
      return;
    }
    if (this.pointer.id !== e.pointerId) return;
    if (this.pointer.mode === "pan") {
      // Keep the grabbed point under the pointer.
      const p = this.targetPlanePoint(e);
      if (p) this.moveTarget(this.pointer.grab.x - p.x, this.pointer.grab.z - p.z);
      return;
    }
    if (this.pointer.mode === "orbit") {
      const dx = e.clientX - this.pointer.x;
      const dy = e.clientY - this.pointer.y;
      if (!this.pointer.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
      this.pointer.moved = true;
      this.pointer.x = e.clientX;
      this.pointer.y = e.clientY;
      this.heading += dx * ORBIT_SPEED;
      this.pitch = Math.min(MAX_PITCH, Math.max(MIN_PITCH, this.pitch + dy * ORBIT_SPEED));
      this.updateCamera();
      return;
    }
    const p = this.groundPoint(e);
    if (!p) return;
    // Pull back from the ball: the shot goes the other way.
    const dx = this.ballPos[0] - p[0];
    const dy = this.ballPos[1] - p[1];
    const pull = Math.hypot(dx, dy);
    const power = Math.min(1, pull / (FULL_PULL * this.distance));
    this.setAim({ angle: Math.atan2(dy, dx), power });
    this.dispatchEvent(new CustomEvent<Aim>("aim", { detail: this.aim! }));
  }

  private onPointerUp(e: PointerEvent) {
    if (!this.pointer || this.pointer.id !== e.pointerId) return;
    const mode = this.pointer.mode;
    this.pointer = null;
    this.renderer.domElement.style.cursor = "";
    if (mode === "aim" && this.aim && this.aim.power >= 0.02)
      this.dispatchEvent(new CustomEvent<Aim>("shoot", { detail: this.aim }));
  }

  private cancelPointer() {
    this.pointer = null;
  }

  /** Cancels an aim in progress (keeps the arrow). */
  cancelDrag() {
    if (this.pointer?.mode === "aim") this.pointer = null;
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault();
    this.userZoomed = true;
    this.distance = Math.min(40, Math.max(1.5, this.distance * Math.exp(e.deltaY * 0.001)));
    this.updateCamera();
  }

  // --- Frame ------------------------------------------------------------------------

  private finishPlayback() {
    if (!this.playing) return;
    const { pb, resolve } = this.playing;
    this.playing = null;
    this.setBall(pb.final, pb.outcome === "rest");
    resolve();
  }

  private animate(now: number) {
    const p = this.playing!;
    const hole = this.hole!;
    const { samples } = p.pb;
    const t = (now - p.start) / 1000;
    while (p.index + 1 < samples.length && samples[p.index + 1].t <= t) p.index++;
    const a = samples[p.index];
    const b = samples[Math.min(p.index + 1, samples.length - 1)];
    const f = b.t > a.t ? Math.min(1, (t - a.t) / (b.t - a.t)) : 1;
    const x = a.x + (b.x - a.x) * f;
    const y = a.y + (b.y - a.y) * f;
    const prev = this.ball.position.clone();
    this.ball.position.copy(toWorld(hole, x, y, PHYSICS.ballRadius));
    // Roll: rotate about the horizontal axis perpendicular to the motion.
    const move = this.ball.position.clone().sub(prev);
    const dist = move.length();
    if (dist > 1e-6) {
      const axis = new THREE.Vector3(0, 1, 0).cross(move).normalize();
      this.ball.rotateOnWorldAxis(axis, dist / PHYSICS.ballRadius);
    }
    const last = samples[samples.length - 1];
    if (t < last.t) return;
    // Ended: drop into the cup or sink in the water, then stop.
    if (p.pb.outcome === "rest") return this.finishPlayback();
    p.sink = Math.min(1, (t - last.t) / 0.35);
    const [fx, fy] = p.pb.outcome === "cup" ? p.pb.final : [last.x, last.y];
    this.ball.position.copy(toWorld(hole, fx, fy, PHYSICS.ballRadius * (1 - 3 * p.sink)));
    if (p.sink >= 1) this.finishPlayback();
  }

  private frame(now: number) {
    // Capped, so that a frame after the tab was hidden does not leap.
    const dt = this.lastFrame === null ? 0 : Math.min(0.05, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    this.stepPan(dt);
    if (this.playing) this.animate(now);
    this.renderer.render(this.scene, this.camera);
    this.frames++;
    if (now - this.fpsWindowStart >= 500) {
      this.fps = (this.frames * 1000) / (now - this.fpsWindowStart);
      this.frames = 0;
      this.fpsWindowStart = now;
    }
  }

  private resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.hole && !this.userZoomed) this.fitDistance(this.hole);
  }
}
