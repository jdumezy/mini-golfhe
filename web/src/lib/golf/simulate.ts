/**
 * The shot simulation, written as the circuit the server will evaluate.
 *
 * On a quadratic terrain with linear drag, the ball's motion between two
 * contacts solves a linear ODE: its state at time τ is an affine map of the
 * state at τ = 0, with public coefficients (a matrix exponential). The
 * simulation therefore never integrates step by step. For each segment
 * between two contacts it
 *
 *   1. evaluates the state at every time τ_k = k·dt of a public grid at once
 *      (one affine map per grid point: under CKKS, one slot per grid point);
 *   2. tests every (grid point, obstacle face) pair for contact, all in
 *      parallel (one sign per slot);
 *   3. keeps the first grid point in contact and applies the bounce there.
 *
 * Its sequential depth is the number of contacts, a public constant, not the
 * number of time steps. Everything after the ball stops, drops in the cup or
 * lands in water is decided by the player once the segments are decrypted
 * (see playback.ts): the server only finds contacts.
 *
 * Bounce responses use only arithmetic that CKKS evaluates exactly (a fixed
 * Newton step replaces the bumper normal's square root), so this is the
 * plaintext reference of the encrypted circuit, not merely a similar physics.
 */
import { expm } from "./linalg";
import { clearance, PHYSICS, type Hole, type Obstacle, type PhysicsSpec, type TerrainSpec } from "./course";

/** x, y, vx, vy. */
export type State = [number, number, number, number];

/** The affine maps s(k·dt) = A_k s(0) + c_k of a terrain, for k = 0..steps. */
export interface Flow {
  dt: number;
  steps: number;
  /** For each k, the 4×5 matrix [A_k | c_k], row-major. */
  maps: Float64Array;
}

export function computeFlow(terrain: TerrainSpec, physics: PhysicsSpec = PHYSICS): Flow {
  const { dt, horizon, drag } = physics;
  const steps = Math.round(horizon / dt);
  const k = physics.gravity * physics.rollFactor;
  const [hxx, hxy, hyy] = terrain.hessian;
  const [cx, cy] = terrain.center;
  // a = −k∇h(q) − λv, with ∇h(q) = H(q − c) + s.
  const ux = -k * (terrain.slope[0] - hxx * cx - hxy * cy);
  const uy = -k * (terrain.slope[1] - hxy * cx - hyy * cy);
  // Augmented generator on (x, y, vx, vy, 1).
  // prettier-ignore
  const m = Float64Array.from([
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    -k * hxx, -k * hxy, -drag, 0, ux,
    -k * hxy, -k * hyy, 0, -drag, uy,
    0, 0, 0, 0, 0,
  ].map((v) => v * dt));
  const step = expm(m, 5);
  const maps = new Float64Array((steps + 1) * 20);
  let a = Float64Array.from([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]);
  for (let i = 0; i <= steps; i++) {
    maps.set(a.subarray(0, 20), i * 20);
    const next = new Float64Array(25);
    for (let r = 0; r < 5; r++)
      for (let j = 0; j < 5; j++) {
        let s = 0;
        for (let t = 0; t < 5; t++) s += a[r * 5 + t] * step[t * 5 + j];
        next[r * 5 + j] = s;
      }
    a = next;
  }
  return { dt, steps, maps };
}

/** The state `k` grid steps after `s`. */
export function advance(flow: Flow, k: number, s: State): State {
  const m = flow.maps;
  const o = k * 20;
  const out: State = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    const i = o + r * 5;
    out[r] = m[i] * s[0] + m[i + 1] * s[1] + m[i + 2] * s[2] + m[i + 3] * s[3] + m[i + 4];
  }
  return out;
}

const flows = new WeakMap<Hole, Flow>();

export function flowOf(hole: Hole): Flow {
  let f = flows.get(hole);
  if (!f) flows.set(hole, (f = computeFlow(hole.terrain)));
  return f;
}

/** A change of velocity and position. */
type Delta = [number, number, number, number];

/**
 * The bounce off `o` of a ball at state `s` (and `prev` one grid step before),
 * or null if it is not in contact and approaching.
 */
function bounce(o: Obstacle, s: State, prev: State): Delta | null {
  const [x, y, vx, vy] = s;
  const d: Delta = [0, 0, 0, 0];
  let hit = false;
  // Reflect the normal velocity and mirror the penetration, both scaled by 1 + e.
  const add = (nx: number, ny: number, depth: number, e: number) => {
    const vn = nx * vx + ny * vy;
    d[0] -= (1 + e) * depth * nx;
    d[1] -= (1 + e) * depth * ny;
    d[2] -= (1 + e) * vn * nx;
    d[3] -= (1 + e) * vn * ny;
    hit = true;
  };
  switch (o.kind) {
    case "rail":
      for (const f of o.faces) {
        const c = clearance(f, x, y);
        if (c < 0 && f.n[0] * vx + f.n[1] * vy < 0) add(f.n[0], f.n[1], c, o.restitution);
      }
      break;
    case "block":
      if (!o.faces.every((f) => clearance(f, x, y) < 0)) break;
      // The faces the ball came through: those it was still outside of a step before.
      for (const f of o.faces)
        if (clearance(f, prev[0], prev[1]) >= 0 && f.n[0] * vx + f.n[1] * vy < 0)
          add(f.n[0], f.n[1], clearance(f, x, y), o.restitution);
      break;
    case "bumper": {
      const ux = (x - o.center[0]) / o.reach;
      const uy = (y - o.center[1]) / o.reach;
      const m = ux * ux + uy * uy;
      if (m >= 1 || ux * vx + uy * vy >= 0) break;
      // 1/|u| by two Newton steps from 1 (|u| is close to 1 at contact).
      const y1 = (3 - m) / 2;
      const inv = (y1 * (3 - m * y1 * y1)) / 2;
      add(ux * inv, uy * inv, o.reach * (m * inv - 1), o.restitution);
      break;
    }
  }
  return hit ? d : null;
}

/** A piece of trajectory: the ball leaves `state` at time `t` and follows the flow. */
export interface Segment {
  t: number;
  state: State;
}

export interface Shot {
  /** The shot, then the state after each contact. */
  segments: Segment[];
  /** The obstacles hit at each contact, for diagnostics. */
  contacts: number[][];
  /** Whether all contacts were spent: the last segment was not searched. */
  exhausted: boolean;
}

/** The server's computation, in the clear. */
export function simulate(hole: Hole, shot: State, physics: PhysicsSpec = PHYSICS): Shot {
  const flow = flowOf(hole);
  const segments: Segment[] = [{ t: 0, state: shot }];
  const contacts: number[][] = [];
  for (let c = 0; c < physics.maxContacts; c++) {
    const { t, state } = segments[segments.length - 1];
    let prev = state;
    let found = false;
    for (let k = 1; k <= flow.steps && !found; k++) {
      const s = advance(flow, k, state);
      const total: Delta = [0, 0, 0, 0];
      const hit: number[] = [];
      hole.obstacles.forEach((o, i) => {
        const d = bounce(o, s, prev);
        if (!d) return;
        hit.push(i);
        for (let j = 0; j < 4; j++) total[j] += d[j];
      });
      if (hit.length) {
        segments.push({ t: t + k * flow.dt, state: [s[0] + total[0], s[1] + total[1], s[2] + total[2], s[3] + total[3]] });
        contacts.push(hit);
        found = true;
      }
      prev = s;
    }
    if (!found) return { segments, contacts, exhausted: false };
  }
  return { segments, contacts, exhausted: true };
}

/** The initial state of a shot from `at`, at angle `angle` (radians, counter-clockwise from +x) and power in [0, 1]. */
export function shotState(at: [number, number], angle: number, power: number, physics: PhysicsSpec = PHYSICS): State {
  const v = power * physics.maxSpeed;
  return [at[0], at[1], v * Math.cos(angle), v * Math.sin(angle)];
}
