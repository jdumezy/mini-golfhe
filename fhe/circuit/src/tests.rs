use mini_golfhe_course::{Course, Hole};
use mini_golfhe_physics::{Flow, State, simulate};

use crate::plan::{SLOTS, STEPS, slot};
use crate::{Decoded, Event, Plan, Prepared, SlotSim, decode_contact, encode_state, evaluate};

/// A pseudo-random generator, for reproducible shots.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Shots from the tee and from random spots of the green, in every direction and at every speed.
fn shots(hole: &Hole, n: usize, rng: &mut Lcg) -> Vec<State> {
    let physics = &Course::get().physics;
    let b = &hole.spec.boundary;
    let (x0, x1) = (b.iter().map(|p| p[0]).fold(f64::MAX, f64::min), b.iter().map(|p| p[0]).fold(f64::MIN, f64::max));
    let (y0, y1) = (b.iter().map(|p| p[1]).fold(f64::MAX, f64::min), b.iter().map(|p| p[1]).fold(f64::MIN, f64::max));
    let mut out = Vec::new();
    while out.len() < n {
        let at =
            if out.len() % 3 == 0 { hole.spec.tee } else { [x0 + (x1 - x0) * rng.next(), y0 + (y1 - y0) * rng.next()] };
        if hole.obstacles.iter().any(|o| o.touches(at[0], at[1])) {
            continue;
        }
        let (a, v) = (rng.next() * std::f64::consts::TAU, (0.05 + 0.95 * rng.next()) * physics.max_speed);
        out.push([at[0], at[1], v * a.cos(), v * a.sin()]);
    }
    out
}

struct Run {
    contacts: Vec<Decoded>,
    /// How far the free indicator strays outside [0, 1].
    stray: f64,
}

fn run(plan: &Prepared<SlotSim>, shot: &State) -> Run {
    let engine = SlotSim { slots: SLOTS };
    let mut stray = 0.0f64;
    let outputs = evaluate(&engine, plan, &encode_state(shot), |e| {
        if let Event::Free { free, .. } = e {
            for k in 0..STEPS {
                let v = free[slot(k, 0)];
                stray = stray.max(-v).max(v - 1.0);
            }
        }
    })
    .unwrap();
    Run { contacts: outputs.iter().map(|o| decode_contact(o)).collect(), stray }
}

/// Where the ball ends up, by the browser's rules (web/src/lib/golf/playback.ts)
/// on grid samples: holed, in water, or at rest at a point.
#[derive(Debug, PartialEq)]
enum End {
    Cup,
    Water,
    Rest([f64; 2]),
}

/// `segments`: start time, state and whether it starts at a contact.
fn play(hole: &Hole, flow: &Flow, segments: &[(f64, State, bool)], exhausted: bool) -> End {
    let p = &Course::get().physics;
    let mut last: Option<State> = None;
    for (j, (t, s0, _)) in segments.iter().enumerate() {
        let next = segments.get(j + 1);
        let steps = next.map_or(flow.steps, |n| ((n.0 - t) / p.dt).round() as usize);
        for k in 0..steps {
            let s = flow.advance(k, s0);
            if next.is_none() && exhausted && k > 0 && hole.obstacles.iter().any(|o| o.touches(s[0], s[1])) {
                let l = last.unwrap();
                return End::Rest([l[0], l[1]]);
            }
            if let Some(a) = last {
                let (ex, ey) = (s[0] - a[0], s[1] - a[1]);
                let len2 = ex * ex + ey * ey;
                let (cx, cy) = (hole.spec.cup[0], hole.spec.cup[1]);
                let f = if len2 > 0.0 { (((cx - a[0]) * ex + (cy - a[1]) * ey) / len2).clamp(0.0, 1.0) } else { 0.0 };
                let speed = (a[2] + (s[2] - a[2]) * f).hypot(a[3] + (s[3] - a[3]) * f);
                if (a[0] + ex * f - cx).hypot(a[1] + ey * f - cy) < p.cup_radius && speed < p.capture_speed {
                    return End::Cup;
                }
            }
            if hole.spec.water.iter().any(|w| mini_golfhe_course::inside_polygon(w, s[0], s[1])) {
                return End::Water;
            }
            let g = hole.terrain.gradient(s[0], s[1]);
            if s[2].hypot(s[3]) < p.stop_speed && p.roll() * g[0].hypot(g[1]) < p.drag * p.stop_speed {
                return End::Rest([s[0], s[1]]);
            }
            last = Some(s);
        }
        if let Some(n) = next
            && n.2
            && n.1[2].hypot(n.1[3]) < p.wall_rest_speed
        {
            let l = last.unwrap();
            return End::Rest([l[0], l[1]]);
        }
    }
    let l = last.unwrap();
    End::Rest([l[0], l[1]])
}

fn same_end(a: &End, b: &End) -> bool {
    match (a, b) {
        (End::Rest(p), End::Rest(q)) => (p[0] - q[0]).hypot(p[1] - q[1]) < 0.05,
        _ => a == b,
    }
}

/// Whether a decrypted contact is the reference's, up to the comparison's
/// blur: the same step or the next, positions within 2 cm, speeds within 10%.
fn close(r: &mini_golfhe_physics::Contact, g: &Decoded, t: f64, dt: f64) -> bool {
    let speed = r.state[2].hypot(r.state[3]);
    (g.hit > 0.5) == r.hit
        && (t - r.t).abs() <= dt * 1.01
        && (g.state[0] - r.state[0]).hypot(g.state[1] - r.state[1]) <= 0.02
        && (g.state[2] - r.state[2]).hypot(g.state[3] - r.state[3]) <= 0.1 * speed + 0.02
}

#[test]
fn the_circuit_finds_the_reference_contacts() {
    let course = Course::get();
    let p = &course.physics;
    let mut rng = Lcg(7);
    let (mut total, mut exact, mut near, mut ends, mut stray) = (0, 0, 0, 0, 0.0f64);
    for hole in &course.holes {
        let flow = Flow::new(&hole.terrain, p);
        let plan = Plan::new(hole, &flow, p).unwrap();
        let prepared = Prepared::new(&SlotSim { slots: SLOTS }, &plan).unwrap();
        let (mut hole_exact, mut hole_near, mut hole_ends) = (0, 0, 0);
        let shots = shots(hole, 12, &mut rng);
        for shot in &shots {
            let reference = simulate(hole, &flow, p, shot);
            let got = run(&prepared, shot);
            stray = stray.max(got.stray);
            let mut t = 0.0;
            let (mut same, mut ok) = (true, true);
            for (r, g) in reference.iter().zip(&got.contacts) {
                t += g.dt;
                let err = (0..4).map(|j| (g.state[j] - r.state[j]).abs()).fold(0.0, f64::max);
                same &= (g.hit > 0.5) == r.hit && (t - r.t).abs() < 1e-4 && err < 1e-3;
                ok &= close(r, g, t, p.dt);
                if !ok {
                    break;
                }
            }
            let show = std::env::var("SHOW").is_ok_and(|h| h.parse() == Ok(hole.index + 1));
            if (!ok || std::env::var("ALL").is_ok()) && show {
                eprintln!("  shot {shot:?}");
                let mut t = 0.0;
                for (r, g) in reference.iter().zip(&got.contacts) {
                    t += g.dt;
                    let round = |s: [f64; 4]| s.map(|v| (v * 1e3).round() / 1e3);
                    eprintln!("    ref hit {} t {:.2} {:?}", r.hit, r.t, round(r.state));
                    eprintln!("    got hit {:.3} t {:.2} {:?}", g.hit, t, round(g.state));
                }
            }
            // Where the ball ends up, from the reference's contacts and from the circuit's.
            let reference_segments: Vec<_> =
                std::iter::once((0.0, *shot, false)).chain(reference.iter().map(|c| (c.t, c.state, c.hit))).collect();
            let mut t = 0.0;
            let circuit_segments: Vec<_> = std::iter::once((0.0, *shot, false))
                .chain(got.contacts.iter().map(|c| {
                    t += c.dt;
                    // The browser rounds the time to the grid.
                    ((t / p.dt).round() * p.dt, c.state, c.hit > 0.5)
                }))
                .collect();
            let (want, have) =
                (play(hole, &flow, &reference_segments, true), play(hole, &flow, &circuit_segments, true));
            let same_ending = same_end(&want, &have);
            if !same_ending && show {
                eprintln!("    ends: reference {want:?}, circuit {have:?}");
            }
            (ends, hole_ends) = (ends + same_ending as usize, hole_ends + same_ending as usize);
            total += 1;
            (exact, hole_exact) = (exact + same as usize, hole_exact + same as usize);
            (near, hole_near) = (near + ok as usize, hole_near + ok as usize);
        }
        eprintln!(
            "hole {:>2}: {hole_exact}/{n} exact, {hole_near}/{n} within the blur, {hole_ends}/{n} same ending, band up to {:.1} mm",
            hole.index + 1,
            plan.band * 1e3,
            n = shots.len()
        );
    }
    eprintln!(
        "{exact}/{total} shots exact, {near}/{total} within the blur, {ends}/{total} end the same; free strays by {stray:.1e}"
    );
    assert!(ends as f64 >= 0.95 * total as f64, "{ends}/{total}");
    assert!(stray < 1e-3);
}
