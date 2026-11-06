/**
 * The course: public physics constants and the 18 holes, as authored in
 * `course.json`, and their compiled form (wall faces, terrain gradient) that
 * the simulation and the renderer use.
 *
 * A hole is a convex boundary of rails, convex blocks and round bumpers inside
 * it, water ponds (which end the shot), and one quadratic terrain. Everything
 * the encrypted simulation needs is linear or quadratic in the ball's state.
 */
import courseJson from "./course.json";

export type Vec2 = [number, number];

export interface PhysicsSpec {
  /** m/s². */
  gravity: number;
  /** Fraction of gravity that drives a rolling ball (5/7 for a solid sphere). */
  rollFactor: number;
  /** Linear drag λ, 1/s: a ball on flat ground slows as e^{-λt}. */
  drag: number;
  ballRadius: number;
  /** Time grid of the simulation, s. */
  dt: number;
  /** How long each segment between two contacts is followed, s. */
  horizon: number;
  /** Contacts the simulation resolves per shot. */
  maxContacts: number;
  /** Fastest shot, m/s. */
  maxSpeed: number;
  /** A ball slower than this comes to rest, unless the slope drives it on. */
  stopSpeed: number;
  /** A ball bouncing off a wall slower than this comes to rest against it. */
  wallRestSpeed: number;
  cupRadius: number;
  /** A ball over the cup slower than this drops in. */
  captureSpeed: number;
  /** Strokes after which a hole is given up. */
  maxStrokes: number;
  restitution: { rail: number; block: number; bumper: number };
}

export interface TerrainSpec {
  /** Height h(q) = ½ dᵀHd + s·d with d = q − center. */
  center: Vec2;
  /** H = [[xx, xy], [xy, yy]], 1/m. */
  hessian: [number, number, number];
  /** Gradient at the center. */
  slope: Vec2;
}

export type ObstacleSpec =
  /** `hedge` only changes the look: a block that reads as the edge of the course. */
  | { kind: "block"; points: Vec2[]; restitution?: number; style?: "hedge" }
  | { kind: "bumper"; center: Vec2; radius: number; restitution?: number };

export interface HoleSpec {
  name: string;
  par: number;
  tee: Vec2;
  cup: Vec2;
  /** Counter-clockwise, convex. */
  boundary: Vec2[];
  terrain?: TerrainSpec;
  obstacles?: ObstacleSpec[];
  /** Convex ponds. */
  water?: Vec2[][];
}

export interface CourseSpec {
  physics: PhysicsSpec;
  holes: HoleSpec[];
}

/** A straight face: `clearance(q) = n·q − b` is the ball center's distance to it, minus the ball radius. */
export interface Face {
  n: Vec2;
  b: number;
}

export type Obstacle =
  | { kind: "rail"; faces: Face[]; restitution: number }
  | { kind: "block"; points: Vec2[]; faces: Face[]; restitution: number; style?: "hedge" }
  | { kind: "bumper"; center: Vec2; radius: number; reach: number; restitution: number };

export interface Hole {
  index: number;
  spec: HoleSpec;
  terrain: TerrainSpec;
  /** The rails first (inward normals), then blocks and bumpers (outward normals). */
  obstacles: Obstacle[];
}

export const COURSE = courseJson as CourseSpec;
export const PHYSICS = COURSE.physics;

const FLAT: TerrainSpec = { center: [0, 0], hessian: [0, 0, 0], slope: [0, 0] };

export function height(t: TerrainSpec, x: number, y: number): number {
  const dx = x - t.center[0];
  const dy = y - t.center[1];
  const [hxx, hxy, hyy] = t.hessian;
  return 0.5 * (hxx * dx * dx + 2 * hxy * dx * dy + hyy * dy * dy) + t.slope[0] * dx + t.slope[1] * dy;
}

export function gradient(t: TerrainSpec, x: number, y: number): Vec2 {
  const dx = x - t.center[0];
  const dy = y - t.center[1];
  const [hxx, hxy, hyy] = t.hessian;
  return [hxx * dx + hxy * dy + t.slope[0], hxy * dx + hyy * dy + t.slope[1]];
}

/** Twice the signed area: positive for counter-clockwise polygons. */
function signedArea2(p: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const [x0, y0] = p[i];
    const [x1, y1] = p[(i + 1) % p.length];
    s += x0 * y1 - x1 * y0;
  }
  return s;
}

function isConvexCcw(p: Vec2[]): boolean {
  if (p.length < 3 || signedArea2(p) <= 0) return false;
  for (let i = 0; i < p.length; i++) {
    const a = p[i];
    const b = p[(i + 1) % p.length];
    const c = p[(i + 2) % p.length];
    if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0) return false;
  }
  return true;
}

/**
 * The faces of a convex counter-clockwise polygon, offset by the ball radius:
 * normals point inward if `inward`, else outward.
 */
function polygonFaces(p: Vec2[], inward: boolean, radius: number): Face[] {
  return p.map((a, i) => {
    const b = p[(i + 1) % p.length];
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const len = Math.hypot(ex, ey);
    // Left of a counter-clockwise edge is inside.
    const sign = inward ? 1 : -1;
    const n: Vec2 = [(-ey / len) * sign, (ex / len) * sign];
    return { n, b: n[0] * a[0] + n[1] * a[1] + radius };
  });
}

export const clearance = (f: Face, x: number, y: number) => f.n[0] * x + f.n[1] * y - f.b;

/** Whether the point is strictly inside a convex counter-clockwise polygon. */
export function insidePolygon(p: Vec2[], x: number, y: number): boolean {
  for (let i = 0; i < p.length; i++) {
    const a = p[i];
    const b = p[(i + 1) % p.length];
    if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) <= 0) return false;
  }
  return true;
}

/** Whether a ball centered at (x, y) overlaps the obstacle. */
export function touches(o: Obstacle, x: number, y: number): boolean {
  switch (o.kind) {
    case "rail":
      return o.faces.some((f) => clearance(f, x, y) < 0);
    case "block":
      return o.faces.every((f) => clearance(f, x, y) < 0);
    case "bumper":
      return (x - o.center[0]) ** 2 + (y - o.center[1]) ** 2 < o.reach ** 2;
  }
}

export function compileHole(spec: HoleSpec, index: number, physics: PhysicsSpec = PHYSICS): Hole {
  const r = physics.ballRadius;
  const where = `hole ${index + 1} (${spec.name})`;
  if (!isConvexCcw(spec.boundary)) throw new Error(`${where}: the boundary is not convex counter-clockwise`);
  const obstacles: Obstacle[] = [
    { kind: "rail", faces: polygonFaces(spec.boundary, true, r), restitution: physics.restitution.rail },
  ];
  for (const o of spec.obstacles ?? []) {
    if (o.kind === "block") {
      if (!isConvexCcw(o.points)) throw new Error(`${where}: a block is not convex counter-clockwise`);
      obstacles.push({
        kind: "block",
        points: o.points,
        faces: polygonFaces(o.points, false, r),
        restitution: o.restitution ?? physics.restitution.block,
        style: o.style,
      });
    } else {
      obstacles.push({
        kind: "bumper",
        center: o.center,
        radius: o.radius,
        reach: o.radius + r,
        restitution: o.restitution ?? physics.restitution.bumper,
      });
    }
  }
  for (const w of spec.water ?? [])
    if (!isConvexCcw(w)) throw new Error(`${where}: a pond is not convex counter-clockwise`);
  for (const [label, [x, y]] of [
    ["tee", spec.tee],
    ["cup", spec.cup],
  ] as const) {
    if (obstacles.some((o) => touches(o, x, y))) throw new Error(`${where}: the ${label} is not on the green`);
    if ((spec.water ?? []).some((w) => insidePolygon(w, x, y))) throw new Error(`${where}: the ${label} is in water`);
  }
  return { index, spec, terrain: spec.terrain ?? FLAT, obstacles };
}

export const HOLES: Hole[] = COURSE.holes.map((h, i) => compileHole(h, i));
