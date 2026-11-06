//! The public data of a hole's circuit: slot layout, masks, the affine maps
//! from the state to every (time step, face) slot, and the polynomials.
//!
//! Layout. A ciphertext has [`SLOTS`] slots in four windows of [`WINDOW`].
//! The *grid* fills window 0: slot `k·FACES + f` is time step `k` (of
//! [`STEPS`]) and face `f` of the hole. Window 1 holds the grid again at step
//! 0 (every step's slot holds step 0), to compare each step with the start of
//! the segment. A *state* ciphertext holds x, y, vx, vy, each repeated over
//! its own window, so that rotating it by `j·WINDOW` brings component `j`
//! over window 0 (and component `j + w` over window `w`).

use anyhow::{Result, ensure};
use mini_golfhe_course::{Hole, MAX_FACES, Physics};
use mini_golfhe_physics::{FACE_LAG, Flow, State};

use crate::poly::{self, Poly};

pub const STEPS: usize = 256;
pub const LOG_STEPS: usize = 8;
pub const FACES: usize = MAX_FACES;
pub const LOG_FACES: usize = 5;
pub const WINDOW: usize = STEPS * FACES;
pub const LOG_WINDOW: usize = LOG_STEPS + LOG_FACES;
pub const SLOTS: usize = 4 * WINDOW;

/// Detectors the ball may newly be inside at one step, at most: the count is
/// turned into 0 or 1 by a polynomial exact up to it.
pub const MAX_INSIDE: usize = 7;

/// Bounds on the gaps are widened by this factor, for the CKKS error.
const NORM_MARGIN: f64 = 1.02;

/// Gap of the slots that are no detector's face: negative ("inside") and constant.
const PADDING_GAP: f64 = -0.5;

/// `value[slot] = Σ_i coef[i][slot] · rot(s, i·WINDOW)[slot] + offset[slot]`
/// for a state ciphertext `s`: in window `w`, `coef[i]` multiplies component `i + w`.
#[derive(Clone, Debug)]
pub struct Affine {
    pub coef: [Vec<f64>; 4],
    pub offset: Vec<f64>,
}

impl Affine {
    fn zero() -> Affine {
        Affine { coef: std::array::from_fn(|_| vec![0.0; SLOTS]), offset: vec![0.0; SLOTS] }
    }

    /// Sets the value at `slot` to `a · s + c`.
    fn set(&mut self, slot: usize, a: [f64; 4], c: f64) {
        let w = slot / WINDOW;
        for (j, &aj) in a.iter().enumerate() {
            self.coef[(j + 4 - w) % 4][slot] = aj;
        }
        self.offset[slot] = c;
    }

    pub fn eval(&self, slot: usize, s: &State) -> f64 {
        let w = slot / WINDOW;
        (0..4).map(|i| self.coef[i][slot] * s[(i + w) % 4]).sum::<f64>() + self.offset[slot]
    }
}

pub fn slot(k: usize, f: usize) -> usize {
    k * FACES + f
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub hole: usize,
    pub max_contacts: usize,
    pub dt: f64,
    /// Gaps closer to zero than this, in metres, may be misjudged (the worst slot).
    pub band: f64,
    /// Normalized gap of every grid slot (and of step 0 in window 1): the
    /// ball's clearance at step `k` to face `f`.
    pub gap: Affine,
    /// The state at step `k`, in every slot of the step.
    pub state: [Affine; 4],
    /// The bounce off face `f` at step `k`, for the state components.
    pub delta: [Affine; 4],
    /// Composite sign, its last factor turned into `(1 − sign)/2`: 1 behind a face.
    pub inside: Vec<Poly>,
    /// Slots of rail faces (each a detector), every step, windows 0 and 1.
    pub rails: Vec<f64>,
    /// For each block size, the first-face slots of the blocks of that size, windows 0 and 1.
    pub blocks: Vec<(usize, Vec<f64>)>,
    /// `[i]`: the slots of the `i`-th face of every block (blocks of more than `i` faces).
    pub block_faces: Vec<Vec<f64>>,
    /// Face 0 of every step of the grid.
    pub first_face: Vec<f64>,
    /// Face 0 of the grid's last step.
    pub last_step: Vec<f64>,
    /// Face 0 of every step, in every window.
    pub first_faces: Vec<f64>,
    /// `k · dt` at face 0 of step `k`.
    pub step_time: Vec<f64>,
    /// The first slot of each window.
    pub window_starts: Vec<f64>,
    pub is_positive: Poly,
}

impl Plan {
    pub fn new(hole: &Hole, flow: &Flow, physics: &Physics) -> Result<Plan> {
        ensure!(flow.steps >= STEPS && physics.steps() == STEPS, "the physics grid must have {STEPS} steps");
        ensure!(hole.faces.len() <= FACES, "hole {} has more than {FACES} faces", hole.index + 1);
        let m = &flow.maps;

        // The states a segment may start from: positions in the hole's
        // bounding box (or 0 after a missed contact), speeds up to the fastest shot.
        let b = &hole.spec.boundary;
        let span = |i: usize| {
            let lo = b.iter().map(|p| p[i]).fold(f64::MAX, f64::min);
            let hi = b.iter().map(|p| p[i]).fold(f64::MIN, f64::max);
            ((lo + hi) / 2.0, (hi - lo) / 2.0)
        };
        let ((cx, hx), (cy, hy)) = (span(0), span(1));
        // The largest |a·s + c| over those states.
        let bound = |a: [f64; 4], c: f64| {
            let positions = (a[0] * cx + a[1] * cy + c).abs() + a[0].abs() * hx + a[1].abs() * hy;
            positions.max(c.abs()) + physics.max_speed * a[2].hypot(a[3])
        };

        let mut gap = Affine::zero();
        gap.offset.fill(PADDING_GAP);
        let mut state: [Affine; 4] = std::array::from_fn(|_| Affine::zero());
        let mut delta: [Affine; 4] = std::array::from_fn(|_| Affine::zero());
        let mut band = 0.0f64;
        let mut gaps_at_start = vec![([0.0; 4], 0.0); FACES];
        for (k, maps) in m.iter().enumerate().take(STEPS) {
            let row = |r: usize| -> [f64; 4] { std::array::from_fn(|j| maps[r][j]) };
            for f in 0..FACES {
                for (r, a) in state.iter_mut().enumerate() {
                    a.set(slot(k, f), row(r), maps[r][4]);
                }
            }
            for d in &hole.detectors {
                let e = -(1.0 + d.restitution);
                for (i, face) in d.faces(&hole.faces).iter().enumerate() {
                    let f = d.first + i;
                    let s = slot(k, f);
                    let [nx, ny] = face.n;
                    // Clearance and normal speed, affine in the state.
                    let along = |a: usize, b: usize| -> ([f64; 4], f64) {
                        (std::array::from_fn(|j| nx * maps[a][j] + ny * maps[b][j]), nx * maps[a][4] + ny * maps[b][4])
                    };
                    let ((g, g0), (v, v0)) = (along(0, 1), along(2, 3));
                    let g0 = g0 - face.b;
                    let scaled = |a: [f64; 4], c: f64, x: f64| (a.map(|v| e * x * v), e * x * c);
                    for (r, (a, c)) in [scaled(g, g0, nx), scaled(g, g0, ny), scaled(v, v0, nx), scaled(v, v0, ny)]
                        .into_iter()
                        .enumerate()
                    {
                        delta[r].set(s, a, c);
                    }
                    // Each slot's gap, divided by its own bound, lies in [−1, 1]:
                    // the comparison resolves a fraction of that bound.
                    let norm = bound(g, g0) * NORM_MARGIN;
                    gap.set(s, g.map(|v| v / norm), g0 / norm);
                    band = band.max(norm * 2f64.powf(-poly::SIGN_LOG_TAU));
                    if k == 0 {
                        gaps_at_start[f] = (g.map(|v| v / norm), g0 / norm);
                    }
                }
            }
        }
        // Window 1: every step's slot holds the gap at step 0. So do the last
        // steps of window 3, which the first steps of the grid read when they
        // look back FACE_LAG steps (rotations wrap around the ring).
        for d in &hole.detectors {
            for (f, &(a, c)) in gaps_at_start.iter().enumerate().skip(d.first).take(d.count) {
                for k in 0..STEPS {
                    gap.set(WINDOW + slot(k, f), a, c);
                }
                for k in STEPS - FACE_LAG..STEPS {
                    gap.set(3 * WINDOW + slot(k, f), a, c);
                }
            }
        }

        let mask = |windows: usize, pred: &dyn Fn(usize, usize) -> bool| -> Vec<f64> {
            let mut v = vec![0.0; SLOTS];
            for w in 0..windows {
                for k in 0..STEPS {
                    for f in 0..FACES {
                        if pred(k, f) {
                            v[w * WINDOW + slot(k, f)] = 1.0;
                        }
                    }
                }
            }
            v
        };
        let rails = mask(2, &|_, f| hole.detectors.iter().any(|d| d.rail && d.first == f));
        let mut sizes: Vec<usize> = hole.detectors.iter().filter(|d| !d.rail).map(|d| d.count).collect();
        sizes.sort_unstable();
        sizes.dedup();
        ensure!(sizes.iter().all(|&n| n <= 8), "blocks have at most 8 faces");
        let blocks = sizes
            .iter()
            .map(|&n| (n, mask(2, &|_, f| hole.detectors.iter().any(|d| !d.rail && d.count == n && d.first == f))))
            .collect();
        let widest = sizes.last().copied().unwrap_or(0);
        let block_faces = (0..widest)
            .map(|i| mask(1, &|_, f| hole.detectors.iter().any(|d| !d.rail && d.count > i && d.first + i == f)))
            .collect();

        let mut step_time = vec![0.0; SLOTS];
        for k in 0..STEPS {
            step_time[slot(k, 0)] = k as f64 * flow.dt;
        }
        let mut window_starts = vec![0.0; SLOTS];
        (0..4).for_each(|w| window_starts[w * WINDOW] = 1.0);

        let mut inside = poly::sign().to_vec();
        let last = inside.len() - 1;
        inside[last] = inside[last].affine(0.5, -0.5);

        Ok(Plan {
            hole: hole.index,
            max_contacts: physics.max_contacts,
            dt: flow.dt,
            band,
            gap,
            state,
            delta,
            inside,
            rails,
            blocks,
            block_faces,
            first_face: mask(1, &|_, f| f == 0),
            last_step: mask(1, &|k, f| k == STEPS - 1 && f == 0),
            first_faces: mask(4, &|_, f| f == 0),
            step_time,
            window_starts,
            is_positive: poly::is_positive(MAX_INSIDE),
        })
    }
}

/// The slots of a state ciphertext: each component over its window.
pub fn encode_state(s: &State) -> Vec<f64> {
    (0..SLOTS).map(|i| s[i / WINDOW]).collect()
}

/// A contact as decrypted: whether there was one, the time it took, and the state after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoded {
    pub hit: f64,
    pub dt: f64,
    pub state: State,
}

/// Reads a contact's output ciphertext (see [`crate::eval`]).
pub fn decode_contact(slots: &[f64]) -> Decoded {
    Decoded { hit: slots[2], dt: slots[1], state: [slots[0], slots[WINDOW], slots[2 * WINDOW], slots[3 * WINDOW]] }
}
