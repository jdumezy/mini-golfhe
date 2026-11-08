//! Evaluation of a hole's circuit on any [`Engine`]: the contacts of a shot,
//! one after the other.
//!
//! Each contact runs in two halves, each within one bootstrap's capacity.
//!
//! 1. From the state, the gap of every (step, face) slot, an affine map, and
//!    through a composite sign and a smoothstep whether the ball is behind
//!    the face. The state at every step and the bounce off every face, also
//!    affine, are kept for the second half. The face indicators are bootstrapped.
//! 2. For each detector, whether the ball is inside it and was not at the
//!    start of the segment. Whether that holds for any detector, multiplied
//!    over the steps so far (a prefix product), is the ball still *free*; the
//!    contact weights are its differences from one step to the next, which
//!    always sum to one, even when the comparison is unsure over a few steps
//!    (the state is then a blend of those steps). The weighted state and
//!    bounces are summed out of the grid into the next state, bootstrapped.

use anyhow::Result;

use crate::engine::Engine;
use mini_golfhe_physics::FACE_LAG;

use crate::plan::{Affine, FACES, LOG_FACES, LOG_STEPS, LOG_WINDOW, Linear, Plan, WINDOW};

/// An [`Affine`] map: a transform of the state over its four windows, and the offset.
struct PreparedAffine<E: Engine> {
    transform: E::Transform,
    offset: E::Pt,
}

/// A [`Plan`] with every plaintext encoded for an engine.
pub struct Prepared<E: Engine> {
    pub max_contacts: usize,
    /// The state at every step, component `j` in window `j`.
    state: PreparedAffine<E>,
    gap: PreparedAffine<E>,
    /// The bounce off every face, component `j` in window `j`.
    delta: PreparedAffine<E>,
    inside: Vec<E::Poly>,
    rails: E::Pt,
    /// Rail faces on the grid only.
    grid_rails: E::Pt,
    blocks: Vec<(usize, E::Pt)>,
    /// Each block's gate, from its first face's slot to all of its faces.
    spread: Option<E::Transform>,
    spread_shifts: Vec<i64>,
    first_face: E::Pt,
    last_step: E::Pt,
    first_faces: E::Pt,
    step_time: E::Pt,
    window_starts: E::Pt,
    one: E::Pt,
    three: E::Pt,
    is_positive: E::Poly,
    multi_face: Option<PreparedMultiFace<E>>,
}

/// [`crate::plan::MultiFace`], encoded.
struct PreparedMultiFace<E: Engine> {
    masks: Vec<E::Pt>,
    counts: E::Pt,
    one_minus_scale: E::Pt,
    scale: E::Pt,
    is_one: E::Poly,
}

/// Shifts bringing each state window to window 0.
const STATE_SHIFTS: [i64; 4] = [0, WINDOW as i64, 2 * WINDOW as i64, 3 * WINDOW as i64];

fn prepare_affine<E: Engine>(engine: &E, a: &Affine) -> Result<PreparedAffine<E>> {
    let diagonals: Vec<(i64, Vec<f64>)> = STATE_SHIFTS.iter().zip(&a.coef).map(|(&k, c)| (k, c.clone())).collect();
    Ok(PreparedAffine { transform: engine.encode_transform(&diagonals)?, offset: engine.encode(&a.offset)? })
}

/// A linear map of the state at every step, as an affine map of the shot's state.
fn compose(l: &Linear, state: &[Affine; 4]) -> Affine {
    let mut out = Affine { coef: std::array::from_fn(|_| vec![0.0; l.offset.len()]), offset: l.offset.clone() };
    for (c, a) in l.coef.iter().zip(state) {
        let Some(c) = c else { continue };
        for (o, coef) in out.coef.iter_mut().zip(&a.coef) {
            o.iter_mut().zip(c.iter().zip(coef)).for_each(|(o, (c, x))| *o += c * x);
        }
        out.offset.iter_mut().zip(c.iter().zip(&a.offset)).for_each(|(o, (c, x))| *o += c * x);
    }
    out
}

/// One affine map holding the grid (window 0) of `maps[j]` in window `j`.
fn by_window(maps: &[Affine; 4]) -> Affine {
    let mut out = Affine { coef: std::array::from_fn(|_| vec![0.0; 4 * WINDOW]), offset: vec![0.0; 4 * WINDOW] };
    for (j, a) in maps.iter().enumerate() {
        for q in 0..WINDOW {
            // In window 0, `coef[c]` multiplies component `c`; in window `j`, `coef[(c − j) mod 4]` does.
            for c in 0..4 {
                out.coef[(c + 4 - j) % 4][j * WINDOW + q] = a.coef[c][q];
            }
            out.offset[j * WINDOW + q] = a.offset[q];
        }
    }
    out
}

impl<E: Engine> Prepared<E> {
    pub fn new(engine: &E, plan: &Plan) -> Result<Prepared<E>> {
        Ok(Prepared {
            max_contacts: plan.max_contacts,
            state: prepare_affine(engine, &by_window(&plan.state))?,
            gap: prepare_affine(engine, &compose(&plan.gap, &plan.state))?,
            delta: prepare_affine(engine, &by_window(&plan.delta.each_ref().map(|d| compose(d, &plan.state))))?,
            inside: plan.inside.iter().map(|p| engine.encode_poly(p)).collect::<Result<_>>()?,
            rails: engine.encode(&plan.rails)?,
            grid_rails: engine.encode(
                &plan.rails.iter().enumerate().map(|(i, &v)| if i < WINDOW { v } else { 0.0 }).collect::<Vec<_>>(),
            )?,
            blocks: plan.blocks.iter().map(|(n, m)| Ok((*n, engine.encode(m)?))).collect::<Result<_>>()?,
            spread: if plan.block_faces.is_empty() {
                None
            } else {
                let diagonals: Vec<(i64, Vec<f64>)> =
                    plan.block_faces.iter().enumerate().map(|(i, m)| (-(i as i64), m.clone())).collect();
                Some(engine.encode_transform(&diagonals)?)
            },
            spread_shifts: (1..plan.block_faces.len() as i64).map(|i| -i).collect(),
            first_face: engine.encode(&plan.first_face)?,
            last_step: engine.encode(&plan.last_step)?,
            first_faces: engine.encode(&plan.first_faces)?,
            step_time: engine.encode(&plan.step_time)?,
            window_starts: engine.encode(&plan.window_starts)?,
            one: engine.encode(&vec![1.0; plan.window_starts.len()])?,
            three: engine.encode(&vec![3.0; plan.window_starts.len()])?,
            is_positive: engine.encode_poly(&plan.is_positive)?,
            multi_face: match &plan.multi_face {
                Some(m) => Some(PreparedMultiFace {
                    masks: m.masks.iter().map(|v| engine.encode(v)).collect::<Result<_>>()?,
                    counts: engine.encode(&m.counts)?,
                    one_minus_scale: engine.encode(&m.one_minus_scale)?,
                    scale: engine.encode(&m.scale)?,
                    is_one: engine.encode_poly(&m.is_one)?,
                }),
                None => None,
            },
        })
    }
}

/// What the evaluator is doing, for progress reports and tests.
pub enum Event<'a, C> {
    /// Contact `index` starts.
    Contact { index: usize },
    /// Bootstrapping, in contact `index`: the indicators (`state == false`) or the next state.
    Refresh { index: usize, state: bool },
    /// Internals of contact `index`, for tests: whether the ball is still free after each step.
    Free { index: usize, free: &'a C },
}

/// Runs `max_contacts` contacts from the shot's state ciphertext (see
/// [`crate::plan::encode_state`]); returns one output ciphertext per contact
/// (see [`crate::plan::decode_contact`]).
pub fn evaluate<E: Engine>(
    engine: &E,
    plan: &Prepared<E>,
    shot: &E::Ct,
    mut inspect: impl FnMut(Event<'_, E::Ct>),
) -> Result<Vec<E::Ct>> {
    let mut state = engine.copy(shot)?;
    let mut outputs = Vec::with_capacity(plan.max_contacts);
    for index in 0..plan.max_contacts {
        inspect(Event::Contact { index });
        let (output, next) = contact(engine, plan, &state, index, &mut inspect)?;
        outputs.push(output);
        if index + 1 < plan.max_contacts {
            inspect(Event::Refresh { index, state: true });
            state = engine.refresh(&next)?;
        }
    }
    Ok(outputs)
}

fn contact<E: Engine>(
    en: &E,
    plan: &Prepared<E>,
    state: &E::Ct,
    index: usize,
    inspect: &mut impl FnMut(Event<'_, E::Ct>),
) -> Result<(E::Ct, E::Ct)> {
    let w = WINDOW as i64;
    let f = FACES as i64;

    // --- First half: gaps, indicators, and what the second half sums. ---
    let windows = en.rotations_of(state, &STATE_SHIFTS[1..])?;
    let affine = |a: &PreparedAffine<E>| -> Result<E::Ct> {
        let mut acc = en.transform(state, &windows, &a.transform)?;
        en.add_pt_assign(&mut acc, &a.offset)?;
        Ok(acc)
    };
    let at_step = affine(&plan.state)?;
    let mut h = affine(&plan.gap)?;
    for p in &plan.inside {
        h = en.eval_poly(&h, p)?;
    }
    // Smoothstep 3h² − 2h³ keeps 0 and 1 and squares the error near both.
    let mut three_minus = en.copy(&h)?;
    en.add_assign(&mut three_minus, &h)?;
    en.neg_assign(&mut three_minus)?;
    en.add_pt_assign(&mut three_minus, &plan.three)?;
    let c = en.mul(&en.mul(&h, &h)?, &three_minus)?;
    let bounce = affine(&plan.delta)?;
    inspect(Event::Refresh { index, state: false });
    let c = en.refresh(&c)?;

    // --- Second half: detectors, the first one entered, the contact. ---
    // Products of runs of 1, 2, 4 and 8 consecutive faces.
    let mut runs = vec![en.copy(&c)?];
    let widest = plan.blocks.iter().map(|(n, _)| *n).max().unwrap_or(0);
    while (1 << (runs.len() - 1)) * 2 <= widest {
        let last = runs.last().unwrap();
        let shifted = en.rotate(last, 1 << (runs.len() - 1))?;
        runs.push(en.mul(last, &shifted)?);
    }
    // Inside each detector, at its first face's slot (window 1: at step 0).
    let mut detector = en.mul_pt(&c, &plan.rails)?;
    for (n, mask) in &plan.blocks {
        let mut product: Option<E::Ct> = None;
        let mut offset = 0i64;
        for b in (0..runs.len()).rev() {
            if n & (1 << b) != 0 {
                let term = if offset == 0 { en.copy(&runs[b])? } else { en.rotate(&runs[b], offset)? };
                product = Some(match product {
                    Some(p) => en.mul(&p, &term)?,
                    None => term,
                });
                offset += 1 << b;
            }
        }
        en.add_assign(&mut detector, &en.mul_pt(&product.expect("blocks have faces"), mask)?)?;
    }
    // Inside now, and not at the start of the segment.
    let mut entered = en.copy(&detector)?;
    en.sub_assign(&mut entered, &en.mul(&detector, &en.rotate(&detector, w)?)?)?;

    // Each detector entered, as 0 or 1 (the same map that turns the count of
    // entered detectors into "any", so that one detector's bounce weights
    // sum to one with the state's).
    let mut gate = en.eval_poly(&entered, &plan.is_positive)?;
    // Whether the ball was behind each face a few steps before.
    let face_before = if plan.spread.is_some() { Some(en.rotate(&c, -f * FACE_LAG as i64)?) } else { None };
    if let (Some(m), Some(behind)) = (&plan.multi_face, &face_before) {
        // The faces of each scaled block it was in front of, counted at its
        // first face: the count of faces minus a segmented sum of `behind`.
        let mut sum = en.copy(behind)?;
        for (i, mask) in m.masks.iter().enumerate() {
            let next = en.mul_pt(&en.rotate(&sum, 1 << i)?, mask)?;
            en.add_assign(&mut sum, &next)?;
        }
        en.neg_assign(&mut sum)?;
        en.add_pt_assign(&mut sum, &m.counts)?;
        // The block's scale unless it was in front of a single face.
        let mut scale = en.mul_pt(&en.eval_poly(&sum, &m.is_one)?, &m.one_minus_scale)?;
        en.add_pt_assign(&mut scale, &m.scale)?;
        gate = en.mul(&gate, &scale)?;
    }
    // The faces bounced off: a rail's own; for a block, those it was in front
    // of a few steps before (steps of an unsure comparison included).
    let mut faces = en.mul_pt(&gate, &plan.grid_rails)?;
    if let (Some(transform), Some(face_before)) = (&plan.spread, face_before) {
        // Each block's value, from its first face's slot to all of its faces.
        let spread = en.transform(&gate, &en.rotations_of(&gate, &plan.spread_shifts)?, transform)?;
        en.add_assign(&mut faces, &spread)?;
        en.sub_assign(&mut faces, &en.mul(&spread, &face_before)?)?;
    }
    // The faces, on the grid only, copied to every window: each component's bounce.
    let bounces = en.mul(&en.replicate(&faces, LOG_WINDOW, 2)?, &bounce)?;

    // Free after step k: no detector newly entered at any step up to k.
    // At a coarser scale: the noise of the hits, differences of consecutive
    // steps, cancels out of the contact's state (a sum by parts over the steps).
    let mut free = en.eval_poly(&en.inner_sum(&entered, 0, LOG_FACES)?, &plan.is_positive)?;
    en.neg_assign(&mut free)?;
    en.add_pt_assign(&mut free, &plan.one)?;
    free = en.coarsen(&free)?;
    for i in 0..LOG_STEPS {
        let shifted = en.rotate(&free, -(f << i))?;
        free = en.mul(&free, &shifted)?;
    }
    let free = en.refine(&free)?;
    inspect(Event::Free { index, free: &free });
    // Free before step k (step 0: the slots wrap to the ring's end, where it is 1).
    let before = en.rotate(&free, -f)?;
    let mut hit = en.copy(&before)?;
    en.sub_assign(&mut hit, &free)?;
    let hit = en.mul_pt(&hit, &plan.first_face)?;
    // Still free at the last step: the segment ends there, without a bounce.
    let mut at_contact = en.mul_pt(&free, &plan.last_step)?;
    en.add_assign(&mut at_contact, &hit)?;
    let before = en.replicate(&en.mul_pt(&before, &plan.first_face)?, 0, LOG_FACES)?;

    // Each component in its window, summed over the grid: faces, then steps.
    let mut packed = en.mul(&en.replicate(&at_contact, LOG_WINDOW, 2)?, &at_step)?;
    en.add_assign(&mut packed, &en.mul(&en.replicate(&before, LOG_WINDOW, 2)?, &bounces)?)?;
    let mut by_step = en.mul_pt(&en.inner_sum(&packed, 0, LOG_FACES)?, &plan.first_faces)?;
    // The step's time at face 1, whether there was a contact at face 2.
    en.add_assign(&mut by_step, &en.rotate(&en.mul_pt(&at_contact, &plan.step_time)?, -1)?)?;
    en.add_assign(&mut by_step, &en.rotate(&hit, -2)?)?;
    let output = en.inner_sum(&by_step, LOG_FACES, LOG_STEPS)?;
    let next = en.replicate(&en.mul_pt(&output, &plan.window_starts)?, 0, LOG_WINDOW)?;
    Ok((output, next))
}
