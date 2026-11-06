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

use crate::plan::{Affine, FACES, LOG_FACES, LOG_STEPS, LOG_WINDOW, Plan, WINDOW};

struct PreparedAffine<E: Engine> {
    coef: [E::Pt; 4],
    offset: E::Pt,
}

/// A [`Plan`] with every plaintext encoded for an engine.
pub struct Prepared<E: Engine> {
    pub max_contacts: usize,
    gap: PreparedAffine<E>,
    state: [PreparedAffine<E>; 4],
    delta: [PreparedAffine<E>; 4],
    inside: Vec<E::Poly>,
    rails: E::Pt,
    blocks: Vec<(usize, E::Pt)>,
    block_faces: Vec<E::Pt>,
    first_face: E::Pt,
    last_step: E::Pt,
    first_faces: E::Pt,
    step_time: E::Pt,
    window_starts: E::Pt,
    one: E::Pt,
    three: E::Pt,
    is_positive: E::Poly,
}

fn prepare_affine<E: Engine>(engine: &E, a: &Affine) -> Result<PreparedAffine<E>> {
    Ok(PreparedAffine {
        coef: [
            engine.encode(&a.coef[0])?,
            engine.encode(&a.coef[1])?,
            engine.encode(&a.coef[2])?,
            engine.encode(&a.coef[3])?,
        ],
        offset: engine.encode(&a.offset)?,
    })
}

impl<E: Engine> Prepared<E> {
    pub fn new(engine: &E, plan: &Plan) -> Result<Prepared<E>> {
        let affines = |a: &[Affine; 4]| -> Result<[PreparedAffine<E>; 4]> {
            Ok([
                prepare_affine(engine, &a[0])?,
                prepare_affine(engine, &a[1])?,
                prepare_affine(engine, &a[2])?,
                prepare_affine(engine, &a[3])?,
            ])
        };
        Ok(Prepared {
            max_contacts: plan.max_contacts,
            gap: prepare_affine(engine, &plan.gap)?,
            state: affines(&plan.state)?,
            delta: affines(&plan.delta)?,
            inside: plan.inside.iter().map(|p| engine.encode_poly(p)).collect::<Result<_>>()?,
            rails: engine.encode(&plan.rails)?,
            blocks: plan.blocks.iter().map(|(n, m)| Ok((*n, engine.encode(m)?))).collect::<Result<_>>()?,
            block_faces: plan.block_faces.iter().map(|m| engine.encode(m)).collect::<Result<_>>()?,
            first_face: engine.encode(&plan.first_face)?,
            last_step: engine.encode(&plan.last_step)?,
            first_faces: engine.encode(&plan.first_faces)?,
            step_time: engine.encode(&plan.step_time)?,
            window_starts: engine.encode(&plan.window_starts)?,
            one: engine.encode(&vec![1.0; plan.window_starts.len()])?,
            three: engine.encode(&vec![3.0; plan.window_starts.len()])?,
            is_positive: engine.encode_poly(&plan.is_positive)?,
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
    let components = [en.copy(state)?, en.rotate(state, w)?, en.rotate(state, 2 * w)?, en.rotate(state, 3 * w)?];
    let affine = |a: &PreparedAffine<E>| -> Result<E::Ct> {
        let mut acc = en.mul_pt(&components[0], &a.coef[0])?;
        for (component, coef) in components.iter().zip(&a.coef).skip(1) {
            en.add_assign(&mut acc, &en.mul_pt(component, coef)?)?;
        }
        en.add_pt_assign(&mut acc, &a.offset)?;
        Ok(acc)
    };
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
    let at_step: Vec<E::Ct> = plan.state.iter().map(&affine).collect::<Result<_>>()?;
    let bounce: Vec<E::Ct> = plan.delta.iter().map(&affine).collect::<Result<_>>()?;
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
    let gate = en.eval_poly(&entered, &plan.is_positive)?;
    // The faces bounced off: a rail's own; for a block, those it was in front
    // of a few steps before (steps of an unsure comparison included).
    let mut faces = en.mul_pt(&gate, &plan.rails)?;
    if let Some((first_faces, others)) = plan.block_faces.split_first() {
        // Each block's value, from its first face's slot to all of its faces.
        let mut spread = en.mul_pt(&gate, first_faces)?;
        for (i, mask) in others.iter().enumerate() {
            en.add_assign(&mut spread, &en.mul_pt(&en.rotate(&gate, -(i as i64 + 1))?, mask)?)?;
        }
        let face_before = en.rotate(&c, -f * FACE_LAG as i64)?;
        en.add_assign(&mut faces, &spread)?;
        en.sub_assign(&mut faces, &en.mul(&spread, &face_before)?)?;
    }
    let bounces: Vec<E::Ct> = bounce.iter().map(|d| en.mul(&faces, d)).collect::<Result<_>>()?;

    // Free after step k: no detector newly entered at any step up to k.
    let mut free = en.eval_poly(&en.inner_sum(&entered, 0, LOG_FACES)?, &plan.is_positive)?;
    en.neg_assign(&mut free)?;
    en.add_pt_assign(&mut free, &plan.one)?;
    for i in 0..LOG_STEPS {
        let shifted = en.rotate(&free, -(f << i))?;
        free = en.mul(&free, &shifted)?;
    }
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

    // Sum each component over the grid into its window: faces, then steps.
    let mut packed: Option<E::Ct> = None;
    for j in 0..4 {
        let mut cj = en.mul(&at_contact, &at_step[j])?;
        en.add_assign(&mut cj, &en.mul(&before, &bounces[j])?)?;
        let cj = if j == 0 { cj } else { en.rotate(&cj, -(j as i64) * w)? };
        match &mut packed {
            Some(p) => en.add_assign(p, &cj)?,
            None => packed = Some(cj),
        }
    }
    let mut by_step = en.mul_pt(&en.inner_sum(&packed.unwrap(), 0, LOG_FACES)?, &plan.first_faces)?;
    // The step's time at face 1, whether there was a contact at face 2.
    en.add_assign(&mut by_step, &en.rotate(&en.mul_pt(&at_contact, &plan.step_time)?, -1)?)?;
    en.add_assign(&mut by_step, &en.rotate(&hit, -2)?)?;
    let output = en.inner_sum(&by_step, LOG_FACES, LOG_STEPS)?;
    let next = en.replicate(&en.mul_pt(&output, &plan.window_starts)?, 0, LOG_WINDOW)?;
    Ok((output, next))
}
