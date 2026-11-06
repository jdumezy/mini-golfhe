//! The operations the circuit is made of, and an exact slot simulator.
//!
//! The evaluator only talks to an [`Engine`], so the CKKS evaluation and the
//! plaintext simulation run the very same sequence of operations.

use anyhow::{Context, Result};

use crate::poly::Poly;

pub trait Engine: Sync {
    type Ct: Send + Sync;
    type Pt: Send + Sync;
    type Poly: Send + Sync;
    /// A linear transform with plaintext diagonals (see [`Engine::encode_transform`]).
    type Transform: Send + Sync;
    /// The rotations of a ciphertext that transforms read (see [`Engine::rotations_of`]).
    type Rotated;

    fn slots(&self) -> usize;

    /// Encodes public slot values (masks, coefficients).
    fn encode(&self, values: &[f64]) -> Result<Self::Pt>;
    fn encode_poly(&self, poly: &Poly) -> Result<Self::Poly>;

    /// A copy of `a` (device buffers are not `Clone`).
    fn copy(&self, a: &Self::Ct) -> Result<Self::Ct>;
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> Result<Self::Ct>;
    fn mul_pt(&self, a: &Self::Ct, b: &Self::Pt) -> Result<Self::Ct>;
    fn add_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> Result<()>;
    fn sub_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> Result<()>;
    fn add_pt_assign(&self, a: &mut Self::Ct, b: &Self::Pt) -> Result<()>;
    fn neg_assign(&self, a: &mut Self::Ct) -> Result<()>;
    /// `out[j] = a[j + k]`, indices modulo the slot count.
    fn rotate(&self, a: &Self::Ct, k: i64) -> Result<Self::Ct>;
    /// `Σ_{j < 2^log_count} rot(a, j · 2^log_batch)`.
    fn inner_sum(&self, a: &Self::Ct, log_batch: usize, log_count: usize) -> Result<Self::Ct>;
    /// `Σ_{j < 2^log_count} rot(a, -j · 2^log_batch)`.
    fn replicate(&self, a: &Self::Ct, log_batch: usize, log_count: usize) -> Result<Self::Ct>;
    fn eval_poly(&self, a: &Self::Ct, poly: &Self::Poly) -> Result<Self::Ct>;

    /// `a ↦ Σ_(k, d) d ⊙ rot(a, k)` over `diagonals`, which must not repeat a shift.
    fn encode_transform(&self, diagonals: &[(i64, Vec<f64>)]) -> Result<Self::Transform>;
    /// The rotations of `a` by `shifts` (and by 0), for the transforms of `a`
    /// whose shifts they cover: rotated once, read by each.
    fn rotations_of(&self, a: &Self::Ct, shifts: &[i64]) -> Result<Self::Rotated>;
    /// The transform `t` of `a`, from its rotations `r`.
    fn transform(&self, a: &Self::Ct, r: &Self::Rotated, t: &Self::Transform) -> Result<Self::Ct>;

    /// Bootstraps `a`: same values, capacity restored.
    fn refresh(&self, a: &Self::Ct) -> Result<Self::Ct>;
}

/// Exact evaluation on plaintext slot vectors, with unbounded capacity.
pub struct SlotSim {
    pub slots: usize,
}

pub(crate) fn rotate(a: &[f64], k: i64) -> Vec<f64> {
    let n = a.len() as i64;
    (0..n).map(|j| a[(j + k).rem_euclid(n) as usize]).collect()
}

impl Engine for SlotSim {
    type Ct = Vec<f64>;
    type Pt = Vec<f64>;
    type Poly = Poly;
    type Transform = Vec<(i64, Vec<f64>)>;
    type Rotated = std::collections::BTreeMap<i64, Vec<f64>>;

    fn slots(&self) -> usize {
        self.slots
    }

    fn encode(&self, values: &[f64]) -> Result<Vec<f64>> {
        Ok(values.to_vec())
    }

    fn encode_poly(&self, poly: &Poly) -> Result<Poly> {
        Ok(poly.clone())
    }

    fn copy(&self, a: &Vec<f64>) -> Result<Vec<f64>> {
        Ok(a.clone())
    }

    fn mul(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<Vec<f64>> {
        Ok(a.iter().zip(b).map(|(x, y)| x * y).collect())
    }

    fn mul_pt(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<Vec<f64>> {
        self.mul(a, b)
    }

    fn add_assign(&self, a: &mut Vec<f64>, b: &Vec<f64>) -> Result<()> {
        a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        Ok(())
    }

    fn sub_assign(&self, a: &mut Vec<f64>, b: &Vec<f64>) -> Result<()> {
        a.iter_mut().zip(b).for_each(|(x, y)| *x -= y);
        Ok(())
    }

    fn add_pt_assign(&self, a: &mut Vec<f64>, b: &Vec<f64>) -> Result<()> {
        self.add_assign(a, b)
    }

    fn neg_assign(&self, a: &mut Vec<f64>) -> Result<()> {
        a.iter_mut().for_each(|x| *x = -*x);
        Ok(())
    }

    fn rotate(&self, a: &Vec<f64>, k: i64) -> Result<Vec<f64>> {
        Ok(rotate(a, k))
    }

    fn inner_sum(&self, a: &Vec<f64>, log_batch: usize, log_count: usize) -> Result<Vec<f64>> {
        // Doubling, as the CKKS inner sum does.
        let mut out = a.clone();
        for i in 0..log_count {
            let r = rotate(&out, (1i64 << i) << log_batch);
            self.add_assign(&mut out, &r)?;
        }
        Ok(out)
    }

    fn replicate(&self, a: &Vec<f64>, log_batch: usize, log_count: usize) -> Result<Vec<f64>> {
        let mut out = a.clone();
        for i in 0..log_count {
            let r = rotate(&out, -((1i64 << i) << log_batch));
            self.add_assign(&mut out, &r)?;
        }
        Ok(out)
    }

    fn eval_poly(&self, a: &Vec<f64>, poly: &Poly) -> Result<Vec<f64>> {
        Ok(a.iter().map(|&x| poly.eval(x)).collect())
    }

    fn encode_transform(&self, diagonals: &[(i64, Vec<f64>)]) -> Result<Self::Transform> {
        Ok(diagonals.to_vec())
    }

    fn rotations_of(&self, a: &Vec<f64>, shifts: &[i64]) -> Result<Self::Rotated> {
        let n = self.slots as i64;
        Ok(shifts.iter().chain([&0]).map(|&k| (k.rem_euclid(n), rotate(a, k))).collect())
    }

    fn transform(&self, _: &Vec<f64>, r: &Self::Rotated, t: &Self::Transform) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.slots];
        for (k, d) in t {
            let rotated = r.get(&k.rem_euclid(self.slots as i64)).context("a shift the rotations miss")?;
            out.iter_mut().zip(rotated.iter().zip(d)).for_each(|(o, (x, y))| *o += x * y);
        }
        Ok(out)
    }

    fn refresh(&self, a: &Vec<f64>) -> Result<Vec<f64>> {
        Ok(a.clone())
    }
}

/// The rotations and sums a circuit performs, which need evaluation keys.
#[derive(Default, Debug)]
pub struct Rotations {
    pub shifts: std::collections::BTreeSet<i64>,
    /// `(log_batch, log_count)` of the inner sums and of the replications.
    pub inner_sums: std::collections::BTreeSet<(usize, usize)>,
    pub replicates: std::collections::BTreeSet<(usize, usize)>,
}

/// Runs a circuit without data, recording its rotations.
pub struct Recorder {
    pub slots: usize,
    pub seen: std::sync::Mutex<Rotations>,
}

impl Engine for Recorder {
    type Ct = ();
    type Pt = ();
    type Poly = ();
    type Transform = ();
    type Rotated = ();

    fn slots(&self) -> usize {
        self.slots
    }
    fn encode(&self, _: &[f64]) -> Result<()> {
        Ok(())
    }
    fn encode_poly(&self, _: &Poly) -> Result<()> {
        Ok(())
    }
    fn copy(&self, _: &()) -> Result<()> {
        Ok(())
    }
    fn mul(&self, _: &(), _: &()) -> Result<()> {
        Ok(())
    }
    fn mul_pt(&self, _: &(), _: &()) -> Result<()> {
        Ok(())
    }
    fn add_assign(&self, _: &mut (), _: &()) -> Result<()> {
        Ok(())
    }
    fn sub_assign(&self, _: &mut (), _: &()) -> Result<()> {
        Ok(())
    }
    fn add_pt_assign(&self, _: &mut (), _: &()) -> Result<()> {
        Ok(())
    }
    fn neg_assign(&self, _: &mut ()) -> Result<()> {
        Ok(())
    }
    fn rotate(&self, _: &(), k: i64) -> Result<()> {
        self.seen.lock().unwrap().shifts.insert(k.rem_euclid(self.slots as i64));
        Ok(())
    }
    fn inner_sum(&self, _: &(), log_batch: usize, log_count: usize) -> Result<()> {
        self.seen.lock().unwrap().inner_sums.insert((log_batch, log_count));
        Ok(())
    }
    fn replicate(&self, _: &(), log_batch: usize, log_count: usize) -> Result<()> {
        self.seen.lock().unwrap().replicates.insert((log_batch, log_count));
        Ok(())
    }
    fn eval_poly(&self, _: &(), _: &()) -> Result<()> {
        Ok(())
    }
    fn encode_transform(&self, _: &[(i64, Vec<f64>)]) -> Result<()> {
        Ok(())
    }
    fn rotations_of(&self, _: &(), shifts: &[i64]) -> Result<()> {
        let n = self.slots as i64;
        let mut seen = self.seen.lock().unwrap();
        seen.shifts.extend(shifts.iter().map(|k| k.rem_euclid(n)).filter(|&k| k != 0));
        Ok(())
    }
    fn transform(&self, _: &(), _: &(), _: &()) -> Result<()> {
        Ok(())
    }
    fn refresh(&self, _: &()) -> Result<()> {
        Ok(())
    }
}

/// The rotations of `plan`'s circuit.
pub fn rotations(plan: &crate::Plan) -> Result<Rotations> {
    let recorder = Recorder { slots: crate::plan::SLOTS, seen: Default::default() };
    let prepared = crate::Prepared::new(&recorder, plan)?;
    crate::evaluate(&recorder, &prepared, &(), |_| {})?;
    Ok(recorder.seen.into_inner().unwrap())
}
