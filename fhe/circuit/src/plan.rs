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

/// `value[slot] = Σ_c coef[c][slot] · X_c[slot] + offset[slot]`, with `X` the
/// state at the slot's step (see [`Plan::state`]): unused components are `None`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub coef: [Option<Vec<f64>>; 4],
    pub offset: Vec<f64>,
}

impl Linear {
    fn new(components: &[usize], offset: f64) -> Linear {
        Linear {
            coef: std::array::from_fn(|c| components.contains(&c).then(|| vec![0.0; SLOTS])),
            offset: vec![offset; SLOTS],
        }
    }

    fn set(&mut self, slot: usize, coef: [f64; 4], offset: f64) {
        for (c, v) in self.coef.iter_mut().enumerate() {
            if let Some(v) = v {
                v[slot] = coef[c];
            }
        }
        self.offset[slot] = offset;
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
    /// The state at step `k`, in every slot of the step (window 1, and the
    /// last steps of window 3: at step 0).
    pub state: [Affine; 4],
    /// Normalized gap of every grid slot (and of step 0 in window 1): the
    /// ball's clearance at step `k` to face `f`, from the state at the step.
    pub gap: Linear,
    /// The bounce off face `f` at step `k`, for the state components.
    pub delta: [Linear; 4],
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

        let mut state: [Affine; 4] = std::array::from_fn(|_| Affine::zero());
        let mut gap = Linear::new(&[0, 1], PADDING_GAP);
        let mut delta = [
            Linear::new(&[0, 1], 0.0),
            Linear::new(&[0, 1], 0.0),
            Linear::new(&[2, 3], 0.0),
            Linear::new(&[2, 3], 0.0),
        ];
        let mut band = 0.0f64;
        // Window 1 holds every step at step 0, and so do the last steps of
        // window 3, which the first steps of the grid read when they look back
        // FACE_LAG steps (rotations wrap around the ring).
        for (k, maps) in m.iter().enumerate().take(STEPS) {
            let row = |r: usize| -> [f64; 4] { std::array::from_fn(|j| maps[r][j]) };
            for f in 0..FACES {
                for (r, a) in state.iter_mut().enumerate() {
                    a.set(slot(k, f), row(r), maps[r][4]);
                    if k == 0 {
                        for k2 in 0..STEPS {
                            a.set(WINDOW + slot(k2, f), row(r), maps[r][4]);
                        }
                        for k2 in STEPS - FACE_LAG..STEPS {
                            a.set(3 * WINDOW + slot(k2, f), row(r), maps[r][4]);
                        }
                    }
                }
            }
            for d in &hole.detectors {
                let e = -(1.0 + d.restitution);
                for (i, face) in d.faces(&hole.faces).iter().enumerate() {
                    let f = d.first + i;
                    let s = slot(k, f);
                    let [nx, ny] = face.n;
                    // Clearance n·q − b and normal speed n·v from the state at the
                    // step; mirror the first and reflect the second, scaled by 1 + e.
                    delta[0].set(s, [e * nx * nx, e * nx * ny, 0.0, 0.0], -e * nx * face.b);
                    delta[1].set(s, [e * ny * nx, e * ny * ny, 0.0, 0.0], -e * ny * face.b);
                    delta[2].set(s, [0.0, 0.0, e * nx * nx, e * nx * ny], 0.0);
                    delta[3].set(s, [0.0, 0.0, e * ny * nx, e * ny * ny], 0.0);
                    // Each slot's gap, divided by a bound on its magnitude over
                    // the shots, lies in [−1, 1]: the comparison resolves a fraction of it.
                    let g: [f64; 4] = std::array::from_fn(|j| nx * maps[0][j] + ny * maps[1][j]);
                    let norm = bound(g, nx * maps[0][4] + ny * maps[1][4] - face.b) * NORM_MARGIN;
                    band = band.max(norm * 2f64.powf(-poly::SIGN_LOG_TAU));
                    let mut slots = vec![s];
                    if k == 0 {
                        slots.extend((0..STEPS).map(|k2| WINDOW + slot(k2, f)));
                        slots.extend((STEPS - FACE_LAG..STEPS).map(|k2| 3 * WINDOW + slot(k2, f)));
                    }
                    for x in slots {
                        gap.set(x, [nx / norm, ny / norm, 0.0, 0.0], -face.b / norm);
                    }
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
