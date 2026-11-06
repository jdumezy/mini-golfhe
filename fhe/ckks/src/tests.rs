use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mini_golfhe_circuit::plan::SLOTS;
use mini_golfhe_circuit::poly::Poly;
use mini_golfhe_circuit::{Engine, Plan, Prepared, SlotSim, decode_contact, encode_state, evaluate, rotations};
use mini_golfhe_course::Course;
use mini_golfhe_physics::{Flow, State};

use crate::engine::DeviceCiphertext;
use poulpy_core::layouts::GGLWEInfos;

use crate::{CkksEngine, Params, Ring};

/// A keyed engine and the client's secret for `params`; the keys go through
/// their byte format, as between the processes.
fn setup(params: Params, plan: &Plan) -> (CkksEngine, crate::SecretKey) {
    let ring = Ring::new(params.clone()).unwrap();
    let sk = ring.generate_secret([1; 32]);
    let galois = ring.circuit_galois_elements(&rotations(plan).unwrap());
    let keys = ring.generate_eval_keys(&sk, &galois, [2; 32]).unwrap();
    let mut bytes = Vec::new();
    crate::wire::write_eval_keys(&mut bytes, &keys).unwrap();
    drop(keys);
    eprintln!("evaluation keys: {} rotations, {:.2} GB", galois.len(), bytes.len() as f64 / 1e9);
    let mut engine = CkksEngine::new(Ring::new(params).unwrap()).unwrap();
    let keys = engine.prepare_keys_from(&mut bytes.as_slice()).unwrap();
    engine.set_keys(Some(keys));
    (engine, sk)
}

fn plan(hole: usize, contacts: usize) -> Plan {
    let course = Course::get();
    let h = &course.holes[hole];
    let flow = Flow::new(&h.terrain, &course.physics);
    Plan { max_contacts: contacts, ..Plan::new(h, &flow, &course.physics).unwrap() }
}

/// Forwards to an engine and records the time spent in each kind of
/// operation, waiting for the device after each so that queued work is
/// charged to its call, and the budget of every ciphertext refreshed.
struct Timed<'a> {
    inner: &'a CkksEngine,
    times: Mutex<BTreeMap<&'static str, (u32, Duration)>>,
    budgets: Mutex<Vec<usize>>,
}

impl<'a> Timed<'a> {
    fn new(inner: &'a CkksEngine) -> Self {
        Timed { inner, times: Default::default(), budgets: Default::default() }
    }

    fn time<T>(&self, op: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.inner.synchronize();
        let mut times = self.times.lock().unwrap();
        let entry = times.entry(op).or_default();
        entry.0 += 1;
        entry.1 += t.elapsed();
        out
    }

    fn report(&self, total: Duration) {
        for (op, (n, d)) in self.times.lock().unwrap().iter() {
            eprintln!(
                "  {op:<12} {n:>5} calls {:>8.0} ms {:>5.1}%",
                d.as_secs_f64() * 1e3,
                100.0 * d.as_secs_f64() / total.as_secs_f64()
            );
        }
    }
}

type R<T> = anyhow::Result<T>;

impl Engine for Timed<'_> {
    type Ct = DeviceCiphertext;
    type Pt = crate::engine::Plaintext;
    type Poly = crate::engine::DevicePoly;
    type Transform = crate::engine::DeviceTransform;
    type Rotated = crate::engine::DeviceRotations;
    fn slots(&self) -> usize {
        self.inner.slots()
    }
    fn encode(&self, v: &[f64]) -> R<Self::Pt> {
        self.inner.encode(v)
    }
    fn encode_poly(&self, p: &Poly) -> R<Self::Poly> {
        self.inner.encode_poly(p)
    }
    fn copy(&self, a: &Self::Ct) -> R<Self::Ct> {
        self.time("add/copy", || self.inner.copy(a))
    }
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> R<Self::Ct> {
        self.time("mul", || self.inner.mul(a, b))
    }
    fn mul_pt(&self, a: &Self::Ct, b: &Self::Pt) -> R<Self::Ct> {
        self.time("mul_pt", || self.inner.mul_pt(a, b))
    }
    fn add_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> R<()> {
        self.time("add/copy", || self.inner.add_assign(a, b))
    }
    fn sub_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> R<()> {
        self.time("add/copy", || self.inner.sub_assign(a, b))
    }
    fn add_pt_assign(&self, a: &mut Self::Ct, b: &Self::Pt) -> R<()> {
        self.time("add/copy", || self.inner.add_pt_assign(a, b))
    }
    fn neg_assign(&self, a: &mut Self::Ct) -> R<()> {
        self.time("add/copy", || self.inner.neg_assign(a))
    }
    fn rotate(&self, a: &Self::Ct, k: i64) -> R<Self::Ct> {
        self.time("rotate", || self.inner.rotate(a, k))
    }
    fn inner_sum(&self, a: &Self::Ct, lb: usize, lc: usize) -> R<Self::Ct> {
        self.time("inner_sum", || self.inner.inner_sum(a, lb, lc))
    }
    fn replicate(&self, a: &Self::Ct, lb: usize, lc: usize) -> R<Self::Ct> {
        self.time("replicate", || self.inner.replicate(a, lb, lc))
    }
    fn eval_poly(&self, a: &Self::Ct, p: &Self::Poly) -> R<Self::Ct> {
        self.time("polynomial", || self.inner.eval_poly(a, p))
    }
    fn encode_transform(&self, d: &[(i64, Vec<f64>)]) -> R<Self::Transform> {
        self.inner.encode_transform(d)
    }
    fn rotations_of(&self, a: &Self::Ct, shifts: &[i64]) -> R<Self::Rotated> {
        self.time("rotations", || self.inner.rotations_of(a, shifts))
    }
    fn transform(&self, a: &Self::Ct, r: &Self::Rotated, t: &Self::Transform) -> R<Self::Ct> {
        self.time("transform", || self.inner.transform(a, r, t))
    }
    fn refresh(&self, a: &Self::Ct) -> R<Self::Ct> {
        self.budgets.lock().unwrap().push(self.inner.log_budget(a));
        self.time("bootstrap", || self.inner.refresh(a))
    }
}

fn simulate(plan: &Plan, shot: &State) -> Vec<Vec<f64>> {
    let sim = SlotSim { slots: SLOTS };
    evaluate(&sim, &Prepared::new(&sim, plan).unwrap(), &encode_state(shot), |_| {}).unwrap()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Runs `plan` under CKKS and compares each contact with the simulator.
fn compare(engine: &CkksEngine, sk: &crate::SecretKey, plan: &Plan, shot: &State) {
    let start = engine.upload(&engine.ring.encrypt(sk, &encode_state(shot), [3; 32]).unwrap());
    // Without waiting for the device after every operation, as the server runs.
    let plain = Prepared::new(engine, plan).unwrap();
    let t = Instant::now();
    evaluate(engine, &plain, &start, |_| {}).unwrap();
    engine.synchronize();
    eprintln!("evaluation in {:.2?}", t.elapsed());
    drop(plain);
    let timed = Timed::new(engine);
    let prepared = Prepared::new(&timed, plan).unwrap();
    let fresh = engine.log_budget(&start);
    let t = Instant::now();
    let outputs = evaluate(&timed, &prepared, &start, |_| {}).unwrap();
    engine.synchronize();
    let total = t.elapsed();
    eprintln!(
        "{} contacts in {total:.2?} (fresh budget {fresh} bits, at refreshes {:?}, after the last {})",
        outputs.len(),
        timed.budgets.lock().unwrap(),
        outputs.last().map_or(0, |o| engine.log_budget(o))
    );
    timed.report(total);
    let want = simulate(plan, shot);
    for (i, (ct, w)) in outputs.iter().zip(&want).enumerate() {
        let got = engine.ring.decrypt(sk, &engine.download(ct).unwrap()).unwrap();
        let (g, w) = (decode_contact(&got), decode_contact(w));
        eprintln!("  contact {i}: ckks {g:?}\n             sim  {w:?}");
        assert!((g.hit - w.hit).abs() < 1e-2 && (g.dt - w.dt).abs() < 1e-2, "contact {i}");
        assert!(max_err(&g.state, &w.state) < 1e-2, "contact {i}: {:?} vs {:?}", g.state, w.state);
    }
}

#[test]
#[ignore = "one contact without bootstrapping at logN 16: about a minute on the CPU"]
fn one_contact_matches_the_simulator() {
    // Hexagon: rails and a six-faced block.
    let plan = plan(6, 1);
    let (engine, sk) = setup(Params::insecure_test(), &plan);
    compare(&engine, &sk, &plan, &[-1.9, 0.0, 2.5, 1.2]);
}

#[test]
#[ignore = "secure parameters with bootstrapping: several GB of keys, run with --ignored --nocapture"]
fn shot_under_secure_parameters() {
    let params = Params::secure().unwrap();
    let plan = plan(6, Course::get().physics.max_contacts);
    let t = Instant::now();
    let (engine, sk) = setup(params, &plan);
    eprintln!("keys generated and prepared in {:.2?}", t.elapsed());
    // A first run compiles the device's kernels.
    engine.warm_up(&Prepared::new(&engine, &plan).unwrap()).unwrap();
    compare(&engine, &sk, &plan, &[-1.9, 0.0, 2.5, 1.2]);
}

/// Every key and ciphertext within the modulus bound of its secret for 128-bit
/// security at `N = 2^16`: 1707 bits for the dense secret (Hamming weight
/// 1024), 345 for the sparse one (weight 32). A key's modulus spans its row
/// offset, its gadget digits and its guard.
#[test]
fn secure_parameters_fit_the_modulus_bounds() {
    let params = Params::secure().unwrap();
    let preset = params.bootstrap.as_deref().unwrap();
    assert_eq!(params.hamming_weight, 1024);
    eprintln!("{}: fresh {} bits, bootstrap {} bits", params.name, params.fresh_k, preset.bootstrap_k);
    assert!(preset.bootstrap_k <= 1707 && params.fresh_k <= 1707);
    assert!(preset.usable() >= crate::params::USABLE_BITS);
    let circuit = [
        ("circuit tensor".to_string(), params.tensor_key_layout().gglwe_layout(), 1707),
        ("circuit rotation".to_string(), params.rotation_key_layout().gglwe_layout(), 1707),
    ];
    let mut worst = (0, 0);
    for (name, l, limit) in preset.key_moduli().unwrap().into_iter().chain(circuit) {
        let total = l.k_offset.as_usize() + l.dnum.as_usize() * l.k_digit.as_usize() + l.k_aux.as_usize();
        eprintln!(
            "  {name:<22} {} × {:>4} + guard {:>4} + offset {:>4} = {total:>4} / {limit}",
            l.dnum.as_usize(),
            l.k_digit.as_usize(),
            l.k_aux.as_usize(),
            l.k_offset.as_usize()
        );
        assert!(total <= limit, "{name}: {total} bits");
        if limit == 1707 {
            worst = worst.max((total, 0));
        }
    }
    eprintln!("largest dense-secret key: {} bits", worst.0);
}

/// Runs every operation under CKKS and in the simulator, and reports the
/// first whose decrypted result strays from the simulated one.
struct Shadow<'a> {
    engine: &'a CkksEngine,
    sk: &'a crate::SecretKey,
    sim: SlotSim,
    step: Mutex<usize>,
}

impl Shadow<'_> {
    fn check(&self, op: &str, out: (DeviceCiphertext, Vec<f64>)) -> (DeviceCiphertext, Vec<f64>) {
        let got = self.engine.ring.decrypt(self.sk, &self.engine.download(&out.0).unwrap()).unwrap();
        let (i, err) = got
            .iter()
            .zip(&out.1)
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .fold((0, 0.0), |m, (i, e)| if e > m.1 { (i, e) } else { m });
        let scale = out.1.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let mut step = self.step.lock().unwrap();
        *step += 1;
        if err > 1e-3 * scale {
            eprintln!(
                "step {step} {op}: max error {err:.3e} at slot {i} (ckks {}, sim {}), budget {}",
                got[i],
                out.1[i],
                self.engine.log_budget(&out.0)
            );
        }
        out
    }
}

impl Engine for Shadow<'_> {
    type Ct = (DeviceCiphertext, Vec<f64>);
    type Pt = (crate::engine::Plaintext, Vec<f64>);
    type Poly = (crate::engine::DevicePoly, Poly);
    type Transform = (crate::engine::DeviceTransform, Vec<(i64, Vec<f64>)>);
    type Rotated = (crate::engine::DeviceRotations, BTreeMap<i64, Vec<f64>>);
    fn slots(&self) -> usize {
        SLOTS
    }
    fn encode(&self, v: &[f64]) -> R<Self::Pt> {
        Ok((self.engine.encode(v)?, v.to_vec()))
    }
    fn encode_poly(&self, p: &Poly) -> R<Self::Poly> {
        Ok((self.engine.encode_poly(p)?, p.clone()))
    }
    fn copy(&self, a: &Self::Ct) -> R<Self::Ct> {
        Ok(self.check("copy", (self.engine.copy(&a.0)?, a.1.clone())))
    }
    fn mul(&self, a: &Self::Ct, b: &Self::Ct) -> R<Self::Ct> {
        Ok(self.check("mul", (self.engine.mul(&a.0, &b.0)?, self.sim.mul(&a.1, &b.1)?)))
    }
    fn mul_pt(&self, a: &Self::Ct, b: &Self::Pt) -> R<Self::Ct> {
        Ok(self.check("mul_pt", (self.engine.mul_pt(&a.0, &b.0)?, self.sim.mul_pt(&a.1, &b.1)?)))
    }
    fn add_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> R<()> {
        self.engine.add_assign(&mut a.0, &b.0)?;
        self.sim.add_assign(&mut a.1, &b.1)
    }
    fn sub_assign(&self, a: &mut Self::Ct, b: &Self::Ct) -> R<()> {
        self.engine.sub_assign(&mut a.0, &b.0)?;
        self.sim.sub_assign(&mut a.1, &b.1)
    }
    fn add_pt_assign(&self, a: &mut Self::Ct, b: &Self::Pt) -> R<()> {
        self.engine.add_pt_assign(&mut a.0, &b.0)?;
        self.sim.add_pt_assign(&mut a.1, &b.1)
    }
    fn neg_assign(&self, a: &mut Self::Ct) -> R<()> {
        self.engine.neg_assign(&mut a.0)?;
        self.sim.neg_assign(&mut a.1)
    }
    fn rotate(&self, a: &Self::Ct, k: i64) -> R<Self::Ct> {
        Ok(self.check("rotate", (self.engine.rotate(&a.0, k)?, self.sim.rotate(&a.1, k)?)))
    }
    fn inner_sum(&self, a: &Self::Ct, lb: usize, lc: usize) -> R<Self::Ct> {
        Ok(self.check("inner_sum", (self.engine.inner_sum(&a.0, lb, lc)?, self.sim.inner_sum(&a.1, lb, lc)?)))
    }
    fn replicate(&self, a: &Self::Ct, lb: usize, lc: usize) -> R<Self::Ct> {
        Ok(self.check("replicate", (self.engine.replicate(&a.0, lb, lc)?, self.sim.replicate(&a.1, lb, lc)?)))
    }
    fn eval_poly(&self, a: &Self::Ct, p: &Self::Poly) -> R<Self::Ct> {
        Ok(self.check("polynomial", (self.engine.eval_poly(&a.0, &p.0)?, self.sim.eval_poly(&a.1, &p.1)?)))
    }
    fn encode_transform(&self, d: &[(i64, Vec<f64>)]) -> R<Self::Transform> {
        Ok((self.engine.encode_transform(d)?, d.to_vec()))
    }
    fn rotations_of(&self, a: &Self::Ct, shifts: &[i64]) -> R<Self::Rotated> {
        Ok((self.engine.rotations_of(&a.0, shifts)?, self.sim.rotations_of(&a.1, shifts)?))
    }
    fn transform(&self, a: &Self::Ct, r: &Self::Rotated, t: &Self::Transform) -> R<Self::Ct> {
        Ok(self.check("transform", (self.engine.transform(&a.0, &r.0, &t.0)?, self.sim.transform(&a.1, &r.1, &t.1)?)))
    }
    fn refresh(&self, a: &Self::Ct) -> R<Self::Ct> {
        Ok(self.check("refresh", (self.engine.refresh(&a.0)?, a.1.clone())))
    }
}

#[test]
#[ignore = "diagnostic: the first operation where CKKS and the simulator part"]
fn shadow_one_contact() {
    let plan = plan(6, 1);
    let (engine, sk) = setup(Params::insecure_test(), &plan);
    let shadow = Shadow { engine: &engine, sk: &sk, sim: SlotSim { slots: SLOTS }, step: Mutex::new(0) };
    let prepared = Prepared::new(&shadow, &plan).unwrap();
    let shot = encode_state(&[-1.9, 0.0, 2.5, 1.2]);
    let start = (engine.upload(&engine.ring.encrypt(&sk, &shot, [3; 32]).unwrap()), shot);
    evaluate(&shadow, &prepared, &start, |_| {}).unwrap();
}
