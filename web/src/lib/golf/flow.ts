/**
 * The ball's motion between two contacts, for drawing the decrypted
 * trajectory.
 *
 * On a quadratic terrain with linear drag, the motion solves a linear ODE: the
 * state at time τ is an affine map of the state at τ = 0, with public
 * coefficients (a matrix exponential), tabulated on the simulation's time
 * grid. The server finds the contacts (see fhe/physics); the player follows
 * these maps between them.
 */
import { expm } from "./linalg";
import { PHYSICS, type Hole, type PhysicsSpec, type TerrainSpec } from "./course";

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

/** A piece of trajectory: the ball leaves `state` at time `t` and follows the flow. */
export interface Segment {
  t: number;
  state: State;
}

export interface Shot {
  /** The shot, then the state after each contact. */
  segments: Segment[];
  /** Whether all contacts were spent: the last segment was not searched. */
  exhausted: boolean;
}

/** The initial state of a shot from `at`, at angle `angle` (radians, counter-clockwise from +x) and power in [0, 1]. */
export function shotState(at: [number, number], angle: number, power: number, physics: PhysicsSpec = PHYSICS): State {
  const v = power * physics.maxSpeed;
  return [at[0], at[1], v * Math.cos(angle), v * Math.sin(angle)];
}
