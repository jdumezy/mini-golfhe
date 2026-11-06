//! The shot simulation, written as the circuit the server will evaluate.
//!
//! On a quadratic terrain with linear drag, the ball's motion between two
//! contacts solves a linear ODE: its state at time τ is an affine map of the
//! state at τ = 0, with public coefficients (a matrix exponential). The
//! simulation therefore never integrates step by step. For each segment
//! between two contacts it
//!
//!   1. evaluates the state at every time τ_k = k·dt of a public grid at once
//!      (one affine map per grid point: under CKKS, one slot per grid point);
//!   2. tests every (grid point, obstacle face) pair for contact, all in
//!      parallel (one sign per slot);
//!   3. keeps the first grid point in contact and applies the bounce there.
//!
//! Its sequential depth is the number of contacts, a public constant, not the
//! number of time steps. Everything after the ball stops, drops in the cup or
//! lands in water is decided by the player once the contacts are decrypted:
//! the server only finds contacts.
//!
//! Bounce responses use only arithmetic that CKKS evaluates exactly (a fixed
//! Newton step replaces the bumper normal's square root).

use mini_golfhe_course::{Detector, Hole, Physics, Terrain};

/// x, y, vx, vy.
pub type State = [f64; 4];

type Mat5 = [[f64; 5]; 5];

fn mul(a: &Mat5, b: &Mat5) -> Mat5 {
    let mut c = [[0.0; 5]; 5];
    for i in 0..5 {
        for k in 0..5 {
            for j in 0..5 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn identity() -> Mat5 {
    let mut m = [[0.0; 5]; 5];
    (0..5).for_each(|i| m[i][i] = 1.0);
    m
}

/// Matrix exponential by scaling and squaring of a Taylor series.
fn expm(a: &Mat5) -> Mat5 {
    let norm = a.iter().map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max);
    let squarings = (norm / 0.25).log2().ceil().max(0.0) as i32;
    let scale = 2f64.powi(-squarings);
    let x = a.map(|r| r.map(|v| v * scale));
    let (mut term, mut sum) = (identity(), identity());
    for k in 1..=18 {
        term = mul(&term, &x).map(|r| r.map(|v| v / k as f64));
        for i in 0..5 {
            for j in 0..5 {
                sum[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..squarings {
        sum = mul(&sum, &sum);
    }
    sum
}

/// The affine maps s(k·dt) = A_k s(0) + c_k of a terrain, for k = 0..=steps.
#[derive(Debug, Clone)]
pub struct Flow {
    pub dt: f64,
    pub steps: usize,
    /// For each k, the 4×5 matrix [A_k | c_k].
    pub maps: Vec<[[f64; 5]; 4]>,
}

impl Flow {
    pub fn new(terrain: &Terrain, physics: &Physics) -> Flow {
        let (dt, steps, k) = (physics.dt, physics.steps(), physics.roll());
        let [hxx, hxy, hyy] = terrain.hessian;
        let [cx, cy] = terrain.center;
        // a = −k∇h(q) − λv, with ∇h(q) = H(q − c) + s.
        let ux = -k * (terrain.slope[0] - hxx * cx - hxy * cy);
        let uy = -k * (terrain.slope[1] - hxy * cx - hyy * cy);
        let l = physics.drag;
        // Augmented generator on (x, y, vx, vy, 1).
        let m: Mat5 = [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [-k * hxx, -k * hxy, -l, 0.0, ux],
            [-k * hxy, -k * hyy, 0.0, -l, uy],
            [0.0; 5],
        ];
        let step = expm(&m.map(|r| r.map(|v| v * dt)));
        let mut a = identity();
        let mut maps = Vec::with_capacity(steps + 1);
        for _ in 0..=steps {
            maps.push([a[0], a[1], a[2], a[3]]);
            a = mul(&a, &step);
        }
        Flow { dt, steps, maps }
    }

    /// The state `k` grid steps after `s`.
    pub fn advance(&self, k: usize, s: &State) -> State {
        self.maps[k].map(|r| r[0] * s[0] + r[1] * s[1] + r[2] * s[2] + r[3] * s[3] + r[4])
    }
}

/// Steps back at which a block's entered faces are told (see [`bounce`]).
pub const FACE_LAG: usize = 3;

/// The change of state `s` takes bouncing off detector `d`, entered at this
/// step (`prev` is the state [`FACE_LAG`] steps before, or at the segment's
/// start): off every face the ball came through, the faces it was still in
/// front of then (a rail has one).
pub fn bounce(hole: &Hole, d: &Detector, s: &State, prev: &State) -> State {
    let [x, y, vx, vy] = *s;
    let mut delta = [0.0; 4];
    let k = 1.0 + d.restitution;
    for f in d.faces(&hole.faces) {
        if d.rail || f.clearance(prev[0], prev[1]) >= 0.0 {
            // Reflect the normal velocity and mirror the penetration, both scaled by 1 + e.
            let (depth, vn) = (f.clearance(x, y), f.n[0] * vx + f.n[1] * vy);
            delta[0] -= k * depth * f.n[0];
            delta[1] -= k * depth * f.n[1];
            delta[2] -= k * vn * f.n[0];
            delta[3] -= k * vn * f.n[1];
        }
    }
    delta
}

/// The state right after one contact, at time `t`. When no contact was found
/// within the segment, `hit` is false and the next segment continues from the
/// state at the segment's last step; every shot has the same number of contacts.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Contact {
    pub hit: bool,
    pub t: f64,
    pub state: State,
}

/// The server's computation, in the clear: `max_contacts` contacts after the
/// shot `shot`.
///
/// A segment is followed on the grid `k = 0..steps` from its start. Its
/// contact is the first step `k ≥ 1` where the ball is inside a detector it
/// was not inside at the start; it bounces off every such detector. Without
/// one, the segment ends at its last step.
pub fn simulate(hole: &Hole, flow: &Flow, physics: &Physics, shot: &State) -> Vec<Contact> {
    let mut contacts = vec![Contact::default(); physics.max_contacts];
    let (mut t, mut start) = (0.0, *shot);
    for contact in contacts.iter_mut() {
        let at_start: Vec<bool> = hole.detectors.iter().map(|d| d.inside(&hole.faces, start[0], start[1])).collect();
        let mut found = None;
        for k in 1..flow.steps {
            let s = flow.advance(k, &start);
            let prev = flow.advance(k.saturating_sub(FACE_LAG), &start);
            let mut after = s;
            for (d, &was) in hole.detectors.iter().zip(&at_start) {
                if !was && d.inside(&hole.faces, s[0], s[1]) {
                    let delta = bounce(hole, d, &s, &prev);
                    (0..4).for_each(|j| after[j] += delta[j]);
                    found = Some(k);
                }
            }
            if found.is_some() {
                start = after;
                break;
            }
        }
        let k = found.unwrap_or(flow.steps - 1);
        if found.is_none() {
            start = flow.advance(k, &start);
        }
        t += k as f64 * flow.dt;
        *contact = Contact { hit: found.is_some(), t, state: start };
    }
    contacts
}

#[cfg(test)]
mod tests {
    use super::*;
    use mini_golfhe_course::Course;

    /// A 10 m × 10 m test hole with the course physics, plus `extra` hole fields.
    fn hole(extra: &str) -> (Hole, Physics) {
        custom_hole(&format!(
            r#"{{"name": "test", "par": 2, "tee": [1, 1], "cup": [9, 9],
                "boundary": [[0, 0], [10, 0], [10, 10], [0, 10]] {extra}}}"#
        ))
    }

    fn custom_hole(spec: &str) -> (Hole, Physics) {
        let mut course: serde_json::Value = serde_json::from_str(mini_golfhe_course::COURSE_JSON).unwrap();
        course["holes"] = serde_json::Value::Array(vec![serde_json::from_str(spec).unwrap()]);
        let course = Course::parse(&course.to_string()).unwrap();
        (course.holes[0].clone(), course.physics)
    }

    fn rk4(terrain: &Terrain, physics: &Physics, mut s: State, t: f64, n: usize) -> State {
        let k = physics.roll();
        let f = |u: &State| {
            let [gx, gy] = terrain.gradient(u[0], u[1]);
            [u[2], u[3], -k * gx - physics.drag * u[2], -k * gy - physics.drag * u[3]]
        };
        let h = t / n as f64;
        let add = |a: &State, b: &State, c: f64| -> State { std::array::from_fn(|j| a[j] + c * b[j]) };
        for _ in 0..n {
            let k1 = f(&s);
            let k2 = f(&add(&s, &k1, h / 2.0));
            let k3 = f(&add(&s, &k2, h / 2.0));
            let k4 = f(&add(&s, &k3, h));
            s = std::array::from_fn(|j| s[j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]));
        }
        s
    }

    #[test]
    fn flow_matches_a_numerical_integration_on_a_tilted_bowl() {
        let physics = &Course::get().physics;
        let terrain = Terrain { center: [2.0, 3.0], hessian: [0.08, 0.02, 0.05], slope: [0.03, -0.01] };
        let flow = Flow::new(&terrain, physics);
        let s0 = [1.0, 1.0, 2.0, -1.0];
        for k in [1, 37, 250, flow.steps] {
            let exact = rk4(&terrain, physics, s0, k as f64 * flow.dt, k * 20);
            let got = flow.advance(k, &s0);
            for j in 0..4 {
                assert!((got[j] - exact[j]).abs() < 1e-6, "k = {k}: {got:?} vs {exact:?}");
            }
        }
    }

    #[test]
    fn bounces_off_a_rail_keeping_the_tangential_velocity() {
        let (hole, physics) = hole("");
        let flow = Flow::new(&hole.terrain, &physics);
        let shot = [5.0, 5.0, 5.0, 1.0];
        let c = simulate(&hole, &flow, &physics, &shot);
        assert!(c[0].hit);
        let before = flow.advance((c[0].t / flow.dt).round() as usize, &shot);
        let [x, _, vx, vy] = c[0].state;
        assert!(x < 10.0 - physics.ball_radius);
        assert!((vx + physics.restitution.rail * before[2]).abs() < 1e-9);
        assert!((vy - before[3]).abs() < 1e-9);
    }

    #[test]
    fn bounces_straight_back_off_an_octagonal_bumper_hit_head_on() {
        let (hole, physics) = hole(r#", "obstacles": [{"kind": "bumper", "center": [5, 5], "radius": 0.1}]"#);
        let flow = Flow::new(&hole.terrain, &physics);
        let c = simulate(&hole, &flow, &physics, &[2.0, 5.0, 3.0, 0.0]);
        let [x, y, vx, vy] = c[0].state;
        assert!(vx < 0.0 && vy.abs() < 1e-9 && (y - 5.0).abs() < 1e-9);
        // Off the face towards −x, at the octagon's apothem.
        let apothem = 0.1 * (std::f64::consts::PI / 8.0).cos();
        assert!(5.0 - x > apothem + physics.ball_radius - 1e-3);
    }

    #[test]
    fn bounces_off_the_face_of_a_block_it_comes_through() {
        let block = r#", "obstacles": [{"kind": "block", "points": [[4, 4], [6, 4], [6, 6], [4, 6]]}]"#;
        let (hole, physics) = hole(block);
        let flow = Flow::new(&hole.terrain, &physics);
        let c = simulate(&hole, &flow, &physics, &[5.2, 8.0, 0.3, -2.5]);
        let [_, y, vx, vy] = c[0].state;
        assert!(vy > 0.0 && vx > 0.0 && y > 6.0 + physics.ball_radius - 1e-3);
    }

    #[test]
    fn always_returns_max_contacts_slots() {
        let (hole, physics) = hole("");
        let flow = Flow::new(&hole.terrain, &physics);
        // Too slow to reach any rail.
        let c = simulate(&hole, &flow, &physics, &[5.0, 5.0, 0.5, 0.0]);
        assert_eq!(c.len(), physics.max_contacts);
        assert!(c.iter().all(|c| !c.hit));
        // Fast between close rails: every slot used.
        let (narrow, physics) = custom_hole(
            r#"{"name": "narrow", "par": 2, "tee": [0.15, 0.5], "cup": [0.15, 9],
                "boundary": [[0, 0], [0.3, 0], [0.3, 10], [0, 10]]}"#,
        );
        let flow = Flow::new(&narrow.terrain, &physics);
        let c = simulate(&narrow, &flow, &physics, &[0.15, 0.5, 6.0, 0.1]);
        assert!(c.iter().all(|c| c.hit));
        assert!(c.windows(2).all(|w| w[0].t < w[1].t));
    }
}
