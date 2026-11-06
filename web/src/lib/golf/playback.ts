/**
 * The player's side of a shot: from the (decrypted) segments, the path the
 * ball takes, and where it ends. Rest, the cup and water are decided here, on
 * the path, since nothing after them changes the outcome.
 */
import {
  gradient,
  insidePolygon,
  PHYSICS,
  touches,
  type Hole,
  type PhysicsSpec,
  type Vec2,
} from "./course";
import { advance, flowOf, type Shot, type State } from "./flow";

export interface Sample {
  t: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** The ball touches an obstacle here. */
  contact?: boolean;
}

export type Outcome = "rest" | "cup" | "water";

export interface Playback {
  samples: Sample[];
  outcome: Outcome;
  /** Where the ball ends up (the cup's center if holed). */
  final: Vec2;
}

const speed = (s: { vx: number; vy: number }) => Math.hypot(s.vx, s.vy);
const sample = (t: number, s: State): Sample => ({ t, x: s[0], y: s[1], vx: s[2], vy: s[3] });

function lerp(a: Sample, b: Sample, f: number): Sample {
  const m = (u: number, v: number) => u + (v - u) * f;
  return { t: m(a.t, b.t), x: m(a.x, b.x), y: m(a.y, b.y), vx: m(a.vx, b.vx), vy: m(a.vy, b.vy) };
}

/** The first point on [a, b] where the ball touches an obstacle, by bisection. */
function firstContact(hole: Hole, a: Sample, b: Sample): Sample {
  const hit = (s: Sample) => hole.obstacles.some((o) => touches(o, s.x, s.y));
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) / 2;
    if (hit(lerp(a, b, mid))) hi = mid;
    else lo = mid;
  }
  return { ...lerp(a, b, lo), contact: true };
}

export function playback(hole: Hole, shot: Shot, physics: PhysicsSpec = PHYSICS): Playback {
  const flow = flowOf(hole);
  const roll = physics.gravity * physics.rollFactor;
  const [cupX, cupY] = hole.spec.cup;
  const samples: Sample[] = [];

  const end = (outcome: Outcome, last: Sample): Playback => {
    samples.push(last);
    return { samples, outcome, final: outcome === "cup" ? [cupX, cupY] : [last.x, last.y] };
  };

  /** The outcome of moving from the last sample to `b`, if the shot ends on the way. */
  const check = (b: Sample): Playback | null => {
    const a = samples[samples.length - 1];
    if (!a) return null;
    // Closest approach to the cup on [a, b].
    const ex = b.x - a.x;
    const ey = b.y - a.y;
    const len2 = ex * ex + ey * ey;
    const f = len2 > 0 ? Math.min(1, Math.max(0, ((cupX - a.x) * ex + (cupY - a.y) * ey) / len2)) : 0;
    const c = lerp(a, b, f);
    if (Math.hypot(c.x - cupX, c.y - cupY) < physics.cupRadius && speed(c) < physics.captureSpeed) return end("cup", c);
    if ((hole.spec.water ?? []).some((w) => insidePolygon(w, b.x, b.y))) return end("water", b);
    const [gx, gy] = gradient(hole.terrain, b.x, b.y);
    // Below this drive the drag alone brings the ball under stopSpeed.
    if (speed(b) < physics.stopSpeed && roll * Math.hypot(gx, gy) < physics.drag * physics.stopSpeed)
      return end("rest", b);
    return null;
  };

  for (let j = 0; j < shot.segments.length; j++) {
    const { t, state } = shot.segments[j];
    const next = shot.segments[j + 1];
    const last = !next;
    const steps = next ? Math.round((next.t - t) / flow.dt) : flow.steps;
    for (let k = 0; k < steps; k++) {
      const s = sample(t + k * flow.dt, advance(flow, k, state));
      // Out of contacts: the ball stops before the first obstacle it would hit.
      if (last && shot.exhausted && k > 0 && hole.obstacles.some((o) => touches(o, s.x, s.y)))
        return end("rest", { ...samples[samples.length - 1], vx: 0, vy: 0 });
      const out = check(s);
      if (out) return out;
      samples.push(s);
    }
    if (!next) break;
    // No contact: the next segment carries on from where this one ends.
    if (next.contact === false) continue;
    // Show the ball touching the obstacle, between the last free sample and the bounce.
    const before = sample(next.t, advance(flow, steps, state));
    const touch = firstContact(hole, samples[samples.length - 1], before);
    const out = check(touch);
    if (out) return out;
    samples.push(touch);
    const after = sample(next.t, next.state);
    if (speed(after) < physics.wallRestSpeed) return end("rest", { ...touch, vx: 0, vy: 0 });
  }
  // The horizon ran out: the ball is all but stopped.
  const lastSample = samples[samples.length - 1];
  return { samples, outcome: "rest", final: [lastSample.x, lastSample.y] };
}
