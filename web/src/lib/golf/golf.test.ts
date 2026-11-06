import { describe, expect, it } from "vitest";
import {
  compileHole,
  gradient,
  HOLES,
  PHYSICS,
  type HoleSpec,
  type ObstacleSpec,
  type TerrainSpec,
  type Vec2,
} from "./course";
import { advance, computeFlow, simulate, shotState, type State } from "./simulate";
import { playback } from "./playback";

const box = (w: number, h: number): Vec2[] => [
  [0, 0],
  [w, 0],
  [w, h],
  [0, h],
];

function hole(extra: Partial<HoleSpec> = {}): ReturnType<typeof compileHole> {
  return compileHole({ name: "test", par: 2, tee: [1, 1], cup: [9, 9], boundary: box(10, 10), ...extra }, 0);
}

/** Classic RK4 on the same ODE, for comparison. */
function rk4(terrain: TerrainSpec, s: State, t: number, n: number): State {
  const k = PHYSICS.gravity * PHYSICS.rollFactor;
  const f = (u: State): State => {
    const [gx, gy] = gradient(terrain, u[0], u[1]);
    return [u[2], u[3], -k * gx - PHYSICS.drag * u[2], -k * gy - PHYSICS.drag * u[3]];
  };
  const h = t / n;
  for (let i = 0; i < n; i++) {
    const add = (a: State, b: State, c: number): State => a.map((v, j) => v + c * b[j]) as State;
    const k1 = f(s);
    const k2 = f(add(s, k1, h / 2));
    const k3 = f(add(s, k2, h / 2));
    const k4 = f(add(s, k3, h));
    s = s.map((v, j) => v + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])) as State;
  }
  return s;
}

describe("flow", () => {
  it("matches a numerical integration on a tilted bowl", () => {
    const terrain: TerrainSpec = { center: [2, 3], hessian: [0.08, 0.02, 0.05], slope: [0.03, -0.01] };
    const flow = computeFlow(terrain);
    const s0: State = [1, 1, 2, -1];
    for (const k of [1, 37, 250, flow.steps]) {
      const exact = rk4(terrain, s0, k * flow.dt, k * 20);
      advance(flow, k, s0).forEach((v, i) => expect(v).toBeCloseTo(exact[i], 6));
    }
  });

  it("decays exponentially on flat ground", () => {
    const flow = computeFlow({ center: [0, 0], hessian: [0, 0, 0], slope: [0, 0] });
    const [x, , vx] = advance(flow, 200, [0, 0, 3, 0]);
    const t = 200 * flow.dt;
    expect(vx).toBeCloseTo(3 * Math.exp(-PHYSICS.drag * t), 9);
    expect(x).toBeCloseTo((3 / PHYSICS.drag) * (1 - Math.exp(-PHYSICS.drag * t)), 9);
  });
});

describe("contacts", () => {
  it("bounces off a rail, keeping the tangential velocity", () => {
    const h = hole();
    const shot = simulate(h, [5, 5, 4, 1]);
    expect(shot.contacts[0]).toEqual([0]);
    const [x, , vx, vy] = shot.segments[1].state;
    expect(vx).toBeLessThan(0);
    expect(x).toBeLessThan(10 - PHYSICS.ballRadius);
    // e·|vn| out, tangential velocity unchanged by the bounce.
    const before = advance(computeFlow(h.terrain), Math.round(shot.segments[1].t / PHYSICS.dt), [5, 5, 4, 1]);
    expect(vx).toBeCloseTo(-PHYSICS.restitution.rail * before[2], 9);
    expect(vy).toBeCloseTo(before[3], 9);
  });

  it("bounces straight back off a bumper hit head-on", () => {
    const bumper: ObstacleSpec = { kind: "bumper", center: [5, 5], radius: 0.1 };
    const shot = simulate(hole({ obstacles: [bumper] }), [2, 5, 3, 0]);
    expect(shot.contacts[0]).toEqual([1]);
    const [x, y, vx, vy] = shot.segments[1].state;
    expect(vx).toBeLessThan(0);
    expect(Math.abs(vy)).toBeLessThan(1e-9);
    expect(5 - x).toBeGreaterThan(0.1 + PHYSICS.ballRadius - 1e-3);
    expect(y).toBeCloseTo(5, 9);
  });

  it("bounces off the face of a block it comes through", () => {
    const block: ObstacleSpec = {
      kind: "block",
      points: [
        [4, 4],
        [6, 4],
        [6, 6],
        [4, 6],
      ],
    };
    // Coming down onto the top face.
    const shot = simulate(hole({ obstacles: [block] }), [5.2, 8, 0.3, -2.5]);
    expect(shot.contacts[0]).toEqual([1]);
    const [, y, vx, vy] = shot.segments[1].state;
    expect(vy).toBeGreaterThan(0);
    expect(vx).toBeGreaterThan(0);
    expect(y).toBeGreaterThan(6 + PHYSICS.ballRadius - 1e-3);
  });

  it("spends at most maxContacts contacts", () => {
    // A fast ball bouncing between close rails.
    const h = compileHole({ name: "t", par: 2, tee: [0.15, 0.5], cup: [0.15, 9], boundary: box(0.3, 10) }, 0);
    const shot = simulate(h, [0.15, 0.5, 6, 0.1]);
    expect(shot.exhausted).toBe(true);
    expect(shot.segments).toHaveLength(PHYSICS.maxContacts + 1);
  });
});

describe("playback", () => {
  const putt = (h: ReturnType<typeof compileHole>, v: number) => playback(h, simulate(h, [1, 1, v, v]));

  it("holes a well-paced putt", () => {
    const h = hole({ cup: [3, 3] });
    // Reach the cup at about 0.5 m/s: v0 = 0.5 + λ·d.
    const p = putt(h, (0.5 + PHYSICS.drag * Math.hypot(2, 2)) / Math.SQRT2);
    expect(p.outcome).toBe("cup");
    expect(p.final).toEqual([3, 3]);
  });

  it("lets a fast ball run over the cup", () => {
    const p = putt(hole({ cup: [3, 3] }), 4);
    expect(p.outcome).not.toBe("cup");
  });

  it("comes to rest on flat ground after the expected distance", () => {
    const h = hole();
    const p = playback(h, simulate(h, [1, 5, 2, 0]));
    expect(p.outcome).toBe("rest");
    expect(p.final[0] - 1).toBeCloseTo((2 - PHYSICS.stopSpeed) / PHYSICS.drag, 1);
  });

  it("ends the shot in water", () => {
    const pond: Vec2[] = [
      [4, 0.5],
      [5, 0.5],
      [5, 1.5],
      [4, 1.5],
    ];
    const h = hole({ water: [pond] });
    expect(playback(h, simulate(h, [1, 1, 4, 0])).outcome).toBe("water");
  });

  it("settles at the bottom of a bowl", () => {
    const h = hole({ terrain: { center: [5, 5], hessian: [0.1, 0, 0.1], slope: [0, 0] } });
    const p = playback(h, simulate(h, [5, 5, 3, 0.5]));
    expect(p.outcome).toBe("rest");
    expect(Math.hypot(p.final[0] - 5, p.final[1] - 5)).toBeLessThan(0.3);
  });

  it("keeps samples in time order", () => {
    const h = hole({ obstacles: [{ kind: "bumper", center: [5, 5], radius: 0.3 }] });
    const p = playback(h, simulate(h, [1, 1, 4, 3.6]));
    for (let i = 1; i < p.samples.length; i++) expect(p.samples[i].t).toBeGreaterThanOrEqual(p.samples[i - 1].t);
  });
});

describe("course", () => {
  it("has 18 holes that compile", () => {
    expect(HOLES).toHaveLength(18);
  });

  it("aims shots counter-clockwise from +x", () => {
    const [, , vx, vy] = shotState([0, 0], Math.PI / 2, 0.5);
    expect(vx).toBeCloseTo(0, 12);
    expect(vy).toBeCloseTo(PHYSICS.maxSpeed / 2, 12);
  });
});
