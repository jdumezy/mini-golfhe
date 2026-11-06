//! CKKS implementation of the circuit [`Engine`], on poulpy.
//!
//! The client's side ([`Ring`]: secret, keys, encryption) runs on the CPU
//! backend [`Be`], which also holds the keys and ciphertexts the processes
//! exchange. The server's [`CkksEngine`] runs the circuit and its bootstrap on
//! the device backend [`Dev`], a GPU when the build enables one: keys move to
//! the device as they are prepared, ciphertexts on the way in and out.

use std::collections::BTreeMap;
use std::sync::Mutex;

use anyhow::{Context, Result};
use mini_golfhe_circuit::poly::Poly;
use mini_golfhe_circuit::{Engine, Prepared, Rotations as CircuitRotations, evaluate};
use poulpy_ckks::api::{
    CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSCopyOps, CKKSEncodingHostOps, CKKSInnerSumOps,
    CKKSPolynomialEvaluationOps, CKKSPow2Ops,
};
use poulpy_ckks::api::{
    CKKSLinearTransformationOps, LinearTransformationBabySteps, LinearTransformationPlan, LinearTransformationPrepared,
    LinearTransformationStrategy, ckks_plaintext_diagonal_layout,
};
use poulpy_ckks::encoding::ckks_linear_transformation_encode_from_diagonals;
use poulpy_ckks::layouts::ComplexDiagonals;
use poulpy_ckks::layouts::{
    BootstrappingContextPrepared, BootstrappingKeySet, BootstrappingKeysLayout, BootstrappingKeysPrepared,
    CKKSCiphertextOwned, CKKSPlaintextOwned, EncapsulationKeys,
};
use poulpy_ckks::polynomial::{BSGSPolynomial, Basis, CKKSPolynomialEncode, Polynomial, SplitStrategy};
use poulpy_ckks::prelude::*;
use poulpy_core::layouts::Diagonals;
use poulpy_core::layouts::prepared::{GLWEAutomorphismKeyPrepared, GLWESwitchingKeyPrepared, GLWETensorKeyPrepared};
use poulpy_core::layouts::prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef};
use poulpy_core::layouts::{
    BackendGLWEAutomorphismKey, BackendGLWEAutomorphismKeyPrepared, BackendGLWESecret, BackendGLWESecretPrepared,
    BackendGLWETensorKeyPrepared, Degree, GGLWEBackendRef, GGLWEInfos, GGLWEToBackendRef, GGLWEViewMut, GLWEInfos,
    GLWESwitchingKeyDegrees, GLWETensorKey, GetGaloisElement, LWEInfos,
};
use poulpy_core::layouts::{GLWEAutomorphismKey, GLWECIEmbedKey, GLWECITraceKey, GLWESwitchingKey};
use poulpy_core::layouts::{
    GLWEAutomorphismKeyAlloc, GLWESecretAlloc, GLWETensorKeyAlloc, GetAutomorphismKey, GetTensorKey, TorusPrecision,
};
use poulpy_core::{
    GGLWEBytesOf, GLWEAutomorphismKeyEncryptSk, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory,
    GLWESecretSampling, GLWESwitchingKeyEncryptSk, GLWESwitchingKeyPreparedFactory, GLWETensorKeyEncryptSk,
    GLWETensorKeyPreparedFactory, ScratchArenaTakeGGLWE, TransferInto,
};
use poulpy_hal::api::{ModuleSynchronize, ScratchOwnedAlloc, ScratchOwnedBorrow};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena, ScratchOwned};
use poulpy_hal::source::Source;

use crate::backend::{Be, Dev};
use crate::params::Params;
use crate::wire::{KeySink, read_eval_keys_into};

/// A ciphertext as the client encrypts it and the processes exchange it.
pub type Ciphertext = CKKSCiphertextOwned<Be>;
/// A ciphertext on the server's device.
pub type DeviceCiphertext = CKKSCiphertextOwned<Dev>;
pub type Plaintext = CKKSPlaintextOwned<Dev>;
/// A polynomial's encoded coefficients on the device, and for an interval
/// `[0, 2^k]` the map onto `[-1, 1]`: a division by `2^(k-1)` and `−1`.
pub struct DevicePoly {
    bsgs: BSGSPolynomial<CKKSPlaintextOwned<Dev>>,
    map: Option<(usize, Plaintext)>,
}
/// A linear transform's diagonals, prepared on the device, and its schedule.
pub struct DeviceTransform {
    prepared: LinearTransformationPrepared<Dev>,
    plan: LinearTransformationPlan,
}
/// The rotations of a ciphertext, prepared for the products of transforms.
pub struct DeviceRotations(LinearTransformationBabySteps<Dev>);
type Word = <Be as Backend>::ZnxWord;
type Buf = <Be as Backend>::OwnedBuf;
type DevBuf = <Dev as Backend>::OwnedBuf;

/// Extra bits above the plaintext precision, for the integer part of codes (≤ 18).
const PT_INTEGER_BITS: usize = 8;

fn pt_meta(log_delta: usize) -> GLWEPlaintextMeta {
    GLWEPlaintextMeta { scale: Scale::Log(log_delta), slots: SlotsKind::Complex, log_sparsity: 0 }
}

/// Bootstrapping context, compiled from the preset on the device.
struct Bootstrap {
    context: BootstrappingContextPrepared<Dev, f64>,
}

impl Bootstrap {
    fn new(module: &Module<Dev>, params: &Params) -> Result<Option<Bootstrap>> {
        let Some(preset) = &params.bootstrap else { return Ok(None) };
        let bytes = BootstrappingContextPrepared::<Dev, f64>::compile_unprepared_tmp_bytes(module, &preset.plan)?;
        let unprepared = BootstrappingContextPrepared::<Dev, f64>::compile_unprepared(
            module,
            preset.base2k.into(),
            &preset.plan,
            &mut ScratchOwned::<Dev>::alloc(bytes).borrow(),
        )?;
        let bytes = unprepared.prepare_tmp_bytes(module);
        let context = unprepared.prepare(module, &mut ScratchOwned::<Dev>::alloc(bytes).borrow())?;
        Ok(Some(Bootstrap { context }))
    }
}

/// Scratch to generate the bootstrapping keys of `layout`.
fn bootstrap_keygen_tmp_bytes(module: &Module<Be>, layout: &BootstrappingKeysLayout) -> usize {
    let mut bytes = module.glwe_tensor_key_encrypt_sk_tmp_bytes(&layout.tensor_key);
    for key in layout.rotation_key_layouts() {
        bytes = bytes.max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&key));
    }
    // A first-factor key is encrypted beside the sparse secret and its rotation, both carved from the scratch.
    if let Some(key) = &layout.first_factor_key {
        let (n, rank) = (key.n().as_usize(), key.rank_out().as_usize());
        let secret = Be::scratch_aligned(Be::scalar_znx_bytes_of(n, rank));
        bytes = bytes.max(2 * secret + module.glwe_switching_key_encrypt_sk_tmp_bytes(key));
    }
    if let Some(e) = &layout.encapsulation {
        for key in [Some(e.dense_to_sparse), e.sparse_to_dense].into_iter().flatten() {
            bytes = bytes.max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&key));
        }
    }
    bytes
}

/// Public evaluation keys, as the client generates them.
pub struct EvalKeys {
    pub tensor_key: GLWETensorKey<Buf, Word>,
    pub rotation_keys: BTreeMap<i64, BackendGLWEAutomorphismKey<Be>>,
    pub bootstrap: Option<BootstrappingKeySet<Buf, Word>>,
}

/// The circuit's ring on the CPU, keyless: what both sides derive from the
/// parameters, and where the client works.
pub struct Ring {
    pub params: Params,
    pub module: Module<Be>,
}

impl Ring {
    pub fn new(params: Params) -> Result<Ring> {
        let module = Module::<Be>::new(params.n() as u64);
        Ok(Ring { params, module })
    }

    fn scratch(&self) -> ScratchOwned<Be> {
        ScratchOwned::<Be>::alloc(self.scratch_bytes())
    }

    fn scratch_bytes(&self) -> usize {
        let p = &self.params;
        let ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&p.glwe_layout());
        let atk = p.rotation_key_layout();
        self.module
            .ckks_all_ops_tmp_bytes(&ct, &ct, &p.tensor_key_layout())
            .max(self.module.ckks_inner_sum_tmp_bytes(&ct, &ct, &atk))
    }

    /// Galois elements of every rotation the circuit performs.
    pub fn circuit_galois_elements(&self, rotations: &CircuitRotations) -> Vec<i64> {
        let n = Degree(self.params.n() as u32);
        let mut out = std::collections::BTreeSet::new();
        for &k in &rotations.shifts {
            out.insert(self.module.ckks_galois_element(k));
        }
        for &(log_batch, log_count) in &rotations.inner_sums {
            out.extend(self.module.ckks_inner_sum_galois_elements(n, log_batch, log_count));
        }
        for &(log_batch, log_count) in &rotations.replicates {
            out.extend(self.module.ckks_replicate_galois_elements(n, log_batch, log_count));
        }
        out.into_iter().collect()
    }
}

/// The client's secret key, on the circuit ring.
pub struct SecretKey {
    pub raw: BackendGLWESecret<Be>,
    pub prepared: BackendGLWESecretPrepared<Be>,
}

impl Ring {
    pub fn generate_secret(&self, seed: [u8; 32]) -> SecretKey {
        let layout = self.params.glwe_layout();
        let mut raw = self.module.glwe_secret_alloc_from_infos(&layout);
        self.module.glwe_secret_fill_ternary_hw(&mut raw, self.params.hamming_weight, &mut Source::new(seed));
        let mut prepared = self.module.glwe_secret_prepared_alloc_from_infos(&layout);
        self.module.glwe_secret_prepare(&mut prepared, &raw);
        SecretKey { raw, prepared }
    }

    /// Evaluation keys: relinearization, the circuit's rotations, and the
    /// bootstrapping keys when the parameters bootstrap.
    pub fn generate_eval_keys(&self, sk: &SecretKey, galois_elements: &[i64], seed: [u8; 32]) -> Result<EvalKeys> {
        let p = &self.params;
        let mut scratch = self.scratch();
        let mut xa = Source::new(seed);
        let mut xe = Source::new([seed, [0x5a; 32]].concat()[16..48].try_into().unwrap());

        let mut tensor_key = self.module.glwe_tensor_key_alloc_from_infos(&p.tensor_key_layout());
        self.module.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk.raw, &mut xa, &mut xe, &mut scratch.borrow());

        let bootstrap = match &p.bootstrap {
            Some(preset) => {
                let layout = &preset.keys_layout;
                let bytes = bootstrap_keygen_tmp_bytes(&self.module, layout);
                let mut xs = Source::new([seed, [0xc3; 32]].concat()[8..40].try_into().unwrap());
                Some(preset.plan.generate_keys(
                    &self.module,
                    &sk.raw,
                    layout,
                    &mut xs,
                    &mut xe,
                    &mut xa,
                    &mut ScratchOwned::<Be>::alloc(bytes).borrow(),
                )?)
            }
            None => None,
        };
        let mut rotation_keys = BTreeMap::new();
        // The bootstrapping keys also serve the circuit's rotations, where their
        // gadget covers the circuit's widest ciphertexts (some are narrower).
        let shared = |g: i64| {
            let Some((b, preset)) = bootstrap.as_ref().zip(p.bootstrap.as_ref()) else { return false };
            let Some(layout) = preset.keys_layout.rotation_keys.get(&g) else { return false };
            b.rotation_keys.contains_key(&g) && (layout.dnum.0 * layout.k_digit.0) as usize >= p.fresh_k
        };
        for &g in galois_elements.iter().filter(|&&g| !shared(g)) {
            let mut key = self.module.glwe_automorphism_key_alloc_from_infos(&p.rotation_key_layout());
            self.module.glwe_automorphism_key_encrypt_sk(&mut key, g, &sk.raw, &mut xe, &mut xa, &mut scratch.borrow());
            rotation_keys.insert(g, key);
        }

        Ok(EvalKeys { tensor_key, rotation_keys, bootstrap })
    }

    /// Encrypts slot values at the fresh width and scale.
    pub fn encrypt(&self, sk: &SecretKey, values: &[f64], seed: [u8; 32]) -> Result<Ciphertext> {
        let p = &self.params;
        let mut scratch = self.scratch();
        let mut pt = self.module.ckks_plaintext_alloc_from_slots(p.slots(), p.base2k.into(), p.fresh_k.into());
        pt.set_meta(pt_meta(p.log_delta));
        self.module.ckks_encode_slots_host_into(&mut pt, values, &vec![0.0; values.len()], &mut scratch.borrow())?;
        let mut ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&p.glwe_layout());
        let mut xe = Source::new([seed, [0x17; 32]].concat()[24..56].try_into().unwrap());
        self.module.ckks_encrypt_sk(
            &mut ct,
            &pt,
            &sk.prepared,
            &mut xe,
            &mut Source::new(seed),
            &mut scratch.borrow(),
        )?;
        Ok(ct)
    }

    /// Decrypts slot values (real parts).
    pub fn decrypt(&self, sk: &SecretKey, ct: &Ciphertext) -> Result<Vec<f64>> {
        let p = &self.params;
        let mut scratch = self.scratch();
        let budget = ct.log_budget().min(PT_INTEGER_BITS + 8);
        let mut pt =
            self.module.ckks_plaintext_alloc_from_slots(p.slots(), p.base2k.into(), (ct.log_delta() + budget).into());
        pt.set_meta(ct.meta());
        self.module.ckks_decrypt(&mut pt, ct, &sk.prepared, &mut scratch.borrow())?;
        let (mut re, mut im) = (vec![0.0; p.slots()], vec![0.0; p.slots()]);
        self.module.ckks_decode_slots_host_into(&pt, &mut re, &mut im, &mut scratch.borrow())?;
        Ok(re)
    }
}

/// Evaluation keys prepared on the device.
pub struct PreparedKeys {
    tensor_key: BackendGLWETensorKeyPrepared<Dev>,
    rotation_keys: std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Dev>>,
    bootstrap: Option<BootstrappingKeysPrepared<DevBuf, Dev>>,
}

/// The CKKS [`Engine`]: the circuit ring on the device, with prepared
/// evaluation keys.
pub struct CkksEngine {
    pub ring: Ring,
    module: Module<Dev>,
    bootstrap: Option<Bootstrap>,
    keys: Option<PreparedKeys>,
    /// One scratch arena, and its size, for everything that runs on the
    /// device: circuit operations, bootstrapping and key preparation never
    /// overlap, so they share the largest arena any of them needs.
    scratch: Mutex<(usize, ScratchOwned<Dev>)>,
}

impl CkksEngine {
    pub fn new(ring: Ring) -> Result<CkksEngine> {
        let module = Module::<Dev>::new(ring.params.n() as u64);
        let bootstrap = Bootstrap::new(&module, &ring.params)?;
        let mut engine =
            CkksEngine { ring, module, bootstrap, keys: None, scratch: Mutex::new((0, ScratchOwned::alloc(0))) };
        let p = &engine.ring.params;
        // The circuit's keys are prepared beside their copy on the device.
        let circuit_keys = engine.moved_bytes(&p.tensor_key_layout())
            + engine.module.glwe_tensor_key_prepare_tmp_bytes(&p.tensor_key_layout());
        let rotation_keys = engine.moved_bytes(&p.rotation_key_layout())
            + engine.module.glwe_automorphism_key_prepare_tmp_bytes(&p.rotation_key_layout());
        let mut bytes = engine.ops_scratch_bytes().max(circuit_keys).max(rotation_keys);
        if let (Some(b), Some(preset)) = (&engine.bootstrap, p.bootstrap.as_deref()) {
            let boot = engine.module.ckks_bootstrap_tmp_bytes(
                &engine.alloc_like(preset.bootstrap_k),
                &engine.alloc_like(preset.input_k),
                &b.context,
                &preset.keys_layout,
            );
            bytes = bytes.max(boot).max(preset.keys_layout.prepare_tmp_bytes(&engine.module));
        }
        engine.scratch = Mutex::new((bytes, ScratchOwned::<Dev>::alloc(bytes)));
        Ok(engine)
    }

    pub fn has_keys(&self) -> bool {
        self.keys.is_some()
    }

    /// Runs `f` on the scratch arena, grown to `bytes` first if it is smaller.
    fn with_scratch<T>(&self, bytes: usize, f: impl FnOnce(&mut ScratchArena<'_, Dev>) -> Result<T>) -> Result<T> {
        let mut scratch = self.scratch.lock().unwrap();
        if scratch.0 < bytes {
            *scratch = (bytes, ScratchOwned::<Dev>::alloc(bytes));
        }
        f(&mut scratch.1.borrow())
    }

    /// Bytes of the scratch arena.
    pub fn scratch_size(&self) -> usize {
        self.scratch.lock().unwrap().0
    }

    /// Scratch of the circuit's operations.
    fn ops_scratch_bytes(&self) -> usize {
        let p = &self.ring.params;
        let ct = self.alloc_like(p.fresh_k);
        let atk = p.rotation_key_layout();
        self.module
            .ckks_all_ops_tmp_bytes(&ct, &ct, &p.tensor_key_layout())
            .max(self.module.ckks_inner_sum_tmp_bytes(&ct, &ct, &atk))
    }

    /// Scratch a key of `layout` takes once moved to the device.
    fn moved_bytes(&self, layout: &impl GGLWEInfos) -> usize {
        Dev::scratch_aligned(self.module.gglwe_bytes_of_from_infos(layout))
    }

    /// The circuit's rotation keys, falling back on the bootstrapping keys.
    fn rotations(&self) -> Result<Rotations<'_>> {
        let keys = self.keys()?;
        Ok(Rotations {
            own: &keys.rotation_keys,
            shared: keys.bootstrap.as_ref().map(|b| &b.rotation_keys),
            tensor_key: &keys.tensor_key,
        })
    }

    fn keys(&self) -> Result<&PreparedKeys> {
        self.keys.as_ref().context("no evaluation keys installed")
    }

    /// A ciphertext of width `k`. Its storage spans the fresh width at least,
    /// whatever `k`: the device reuses a freed buffer only for a request of
    /// the same size, and a ciphertext narrowing stage by stage would
    /// otherwise make every allocation a new, zero-filled device buffer.
    fn alloc_like(&self, k: usize) -> DeviceCiphertext {
        let mut layout = self.ring.params.glwe_layout();
        layout.k = k.max(self.ring.params.fresh_k).into();
        let mut ct = self.module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        ct.set_k(k.into());
        ct
    }

    /// Runs the circuit once on an encryption of zero. A GPU compiles each
    /// kernel the first time it runs, which made the first evaluation much
    /// slower than the next ones; this moves that cost to key setup.
    pub fn warm_up(&self, prepared: &Prepared<CkksEngine>) -> Result<()> {
        let p = &self.ring.params;
        let mut zero = self.alloc_like(p.fresh_k);
        zero.set_meta(pt_meta(p.log_delta));
        evaluate(self, prepared, &zero, |_| {})?;
        self.synchronize();
        Ok(())
    }

    /// Waits for the work queued on the device.
    pub fn synchronize(&self) {
        self.module.synchronize();
    }

    /// Moves a received ciphertext to the device.
    pub fn upload(&self, ct: &Ciphertext) -> DeviceCiphertext {
        let mut out = self.alloc_like(ct.k().as_usize());
        (**ct).transfer_into(&mut *out);
        out.set_meta(ct.meta());
        out
    }

    /// Brings a ciphertext back from the device, to send or decrypt, with the
    /// storage of its width: device ciphertexts hold more (see [`Self::alloc_like`]).
    pub fn download(&self, ct: &DeviceCiphertext) -> Result<Ciphertext> {
        let module = &self.ring.module;
        let mut layout = ct.glwe_layout();
        layout.k = (ct.data().size() * self.ring.params.base2k).into();
        let mut held = module.ckks_ciphertext_alloc_from_glwe_infos(&layout);
        held.set_k(ct.k());
        (**ct).transfer_into(&mut *held);
        held.set_meta(ct.meta());
        let mut out = module.ckks_ciphertext_alloc_from_glwe_infos(&ct.glwe_layout());
        module.ckks_copy(&mut out, &held, &mut self.ring.scratch().borrow())?;
        Ok(out)
    }
}

impl Engine for CkksEngine {
    type Ct = DeviceCiphertext;
    type Pt = Plaintext;
    type Poly = DevicePoly;
    type Transform = DeviceTransform;
    type Rotated = DeviceRotations;

    fn slots(&self) -> usize {
        self.ring.params.slots()
    }

    fn encode(&self, values: &[f64]) -> Result<Plaintext> {
        let p = &self.ring.params;
        let mut pt = self.module.ckks_plaintext_alloc_from_slots(
            p.slots(),
            p.base2k.into(),
            (p.pt_log_delta + PT_INTEGER_BITS).into(),
        );
        pt.set_meta(pt_meta(p.pt_log_delta));
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_encode_slots_host_into(&mut pt, values, &vec![0.0; values.len()], &mut scratch.borrow())?;
        Ok(pt)
    }

    /// Chebyshev coefficients, split for a minimum-depth evaluation. The
    /// interval must be `[-1, 1]` or `[0, 2^k]`, whose map is exact and free.
    fn encode_poly(&self, poly: &Poly) -> Result<DevicePoly> {
        let p = &self.ring.params;
        let map = match poly.interval {
            (-1.0, 1.0) => None,
            (0.0, hi) if hi >= 2.0 && hi.log2().fract() == 0.0 => {
                Some((hi.log2() as usize - 1, self.encode(&vec![-1.0; p.slots()])?))
            }
            other => anyhow::bail!("unsupported polynomial interval {other:?}"),
        };
        let host = Module::<HostBytesBackend>::new(p.n() as u64);
        let bsgs = Polynomial::new(Basis::Chebyshev, poly.cheb.clone()).encode_bsgs_with(
            &host,
            p.base2k.into(),
            CoeffsMeta::from_delta_budget(p.poly_log_delta, PT_INTEGER_BITS),
            SplitStrategy::MinDepth,
        )?;
        let bsgs = bsgs.map_baby_steps_ref(|pt| {
            let mut dev = self.module.ckks_plaintext_alloc(pt.n(), pt.base2k(), pt.k());
            (**pt).transfer_into(&mut *dev);
            dev.set_meta(pt.meta());
            dev
        });
        Ok(DevicePoly { bsgs, map })
    }

    fn copy(&self, a: &DeviceCiphertext) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_copy(&mut out, a, &mut scratch.borrow())?;
        Ok(out)
    }

    fn mul(&self, a: &DeviceCiphertext, b: &DeviceCiphertext) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize().min(b.k().as_usize()));
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_mul_into(&mut out, a, b, &self.keys()?.tensor_key, &mut scratch.borrow())?;
        Ok(out)
    }

    fn mul_pt(&self, a: &DeviceCiphertext, b: &Plaintext) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_mul_pt_vec_into(&mut out, a, b, &mut scratch.borrow())?;
        Ok(out)
    }

    fn add_assign(&self, a: &mut DeviceCiphertext, b: &DeviceCiphertext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_add_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn sub_assign(&self, a: &mut DeviceCiphertext, b: &DeviceCiphertext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_sub_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn add_pt_assign(&self, a: &mut DeviceCiphertext, b: &Plaintext) -> Result<()> {
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_add_pt_vec_assign(a, b, &mut scratch.borrow())?;
        Ok(())
    }

    fn neg_assign(&self, a: &mut DeviceCiphertext) -> Result<()> {
        self.module.ckks_neg_assign(a)?;
        Ok(())
    }

    fn rotate(&self, a: &DeviceCiphertext, k: i64) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        let k = k.rem_euclid(self.ring.params.slots() as i64);
        self.module.ckks_rotate_into(k, &mut out, a, &self.rotations()?, &mut scratch.borrow())?;
        Ok(out)
    }

    fn inner_sum(&self, a: &DeviceCiphertext, log_batch: usize, log_count: usize) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_inner_sum(
            log_batch,
            log_count,
            false,
            &mut out,
            a,
            &self.rotations()?,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    fn replicate(&self, a: &DeviceCiphertext, log_batch: usize, log_count: usize) -> Result<DeviceCiphertext> {
        let mut out = self.alloc_like(a.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_replicate(log_batch, log_count, &mut out, a, &self.rotations()?, &mut scratch.borrow())?;
        Ok(out)
    }

    fn eval_poly(&self, a: &DeviceCiphertext, poly: &DevicePoly) -> Result<DeviceCiphertext> {
        let mapped;
        let input = match &poly.map {
            None => a,
            Some((bits, minus_one)) => {
                let mut x = self.copy(a)?;
                self.module.ckks_div_pow2_assign(*bits, &mut x)?;
                self.add_pt_assign(&mut x, minus_one)?;
                mapped = x;
                &mapped
            }
        };
        let mut out = self.alloc_like(input.k().as_usize());
        let scratch = &mut self.scratch.lock().unwrap().1;
        self.module.ckks_eval_poly_real_const_coeffs(
            &mut out,
            input,
            &poly.bsgs,
            &self.keys()?.tensor_key,
            &mut scratch.borrow(),
        )?;
        Ok(out)
    }

    /// All diagonals in one giant step: the products read the rotations of the
    /// source, and their sum needs no rotation of its own.
    fn encode_transform(&self, diagonals: &[(i64, Vec<f64>)]) -> Result<DeviceTransform> {
        let p = &self.ring.params;
        let slots = p.slots();
        let (mut re, mut im) = (Diagonals::<f64>::new(slots), Diagonals::<f64>::new(slots));
        for (k, d) in diagonals {
            re.set(k.rem_euclid(slots as i64), d.clone());
            im.set(k.rem_euclid(slots as i64), vec![0.0; slots]);
        }
        let lt = {
            let scratch = &mut self.scratch.lock().unwrap().1;
            ckks_linear_transformation_encode_from_diagonals(
                &self.module,
                p.base2k.into(),
                CoeffsMeta::from_delta_budget(p.pt_log_delta, PT_INTEGER_BITS),
                &ComplexDiagonals::new(re, im),
                LinearTransformationStrategy::Bsgs { giant_step: slots },
                false,
                &mut scratch.borrow(),
            )?
        };
        let first = lt.first_diagonal_plaintext().context("a transform without diagonals")?;
        let plan = lt.index();
        let mut prepared = LinearTransformationPrepared::<Dev>::alloc_prepared_from_index(&self.module, &plan, first);
        let bytes = self.module.ckks_linear_transformation_rhs_prepare_tmp_bytes(first);
        self.with_scratch(bytes, |s| Ok(self.module.ckks_linear_transformation_rhs_prepare(&mut prepared, &lt, s)?))?;
        Ok(DeviceTransform { prepared, plan })
    }

    fn rotations_of(&self, a: &DeviceCiphertext, shifts: &[i64]) -> Result<DeviceRotations> {
        let slots = self.ring.params.slots() as i64;
        let mut steps: Vec<i64> = shifts.iter().map(|k| k.rem_euclid(slots)).chain([0]).collect();
        steps.sort_unstable();
        steps.dedup();
        let plan = LinearTransformationPlan {
            baby_steps: steps.clone(),
            giant_steps: vec![0],
            index: vec![steps.clone()],
            rotations: Default::default(),
        };
        let mut babies = LinearTransformationBabySteps::alloc(&self.module, &steps, a);
        let bytes = self.module.ckks_linear_transformation_baby_steps_prepare_tmp_bytes(
            a,
            &plan,
            &self.ring.params.rotation_key_layout(),
        );
        self.with_scratch(bytes, |s| {
            Ok(self.module.ckks_linear_transformation_baby_steps_prepare(&mut babies, a, &self.rotations()?, s)?)
        })?;
        Ok(DeviceRotations(babies))
    }

    fn transform(&self, a: &DeviceCiphertext, r: &DeviceRotations, t: &DeviceTransform) -> Result<DeviceCiphertext> {
        let p = &self.ring.params;
        let mut out = self.alloc_like(a.k().as_usize());
        let bytes = self.module.ckks_linear_transformation_eval_tmp_bytes(
            a,
            &ckks_plaintext_diagonal_layout(a),
            &t.plan,
            &p.tensor_key_layout(),
            &p.rotation_key_layout(),
        );
        self.with_scratch(bytes, |s| {
            Ok(self.module.ckks_linear_transformation_eval_into(
                &mut out,
                a,
                &r.0,
                &t.prepared,
                &self.rotations()?,
                s,
            )?)
        })?;
        Ok(out)
    }

    /// Bootstraps `a`: its budget is what the bootstrap reads, at the
    /// circuit's scale; the values are real, so the bootstrap skips the
    /// imaginary half.
    fn refresh(&self, a: &DeviceCiphertext) -> Result<DeviceCiphertext> {
        let (Some(b), Some(preset), Some(keys)) =
            (&self.bootstrap, &self.ring.params.bootstrap, &self.keys()?.bootstrap)
        else {
            // Parameters without bootstrapping must be wide enough for the whole circuit.
            return self.copy(a);
        };
        let scratch = &mut self.scratch.lock().unwrap().1;
        let mut input = self.alloc_like(preset.input_k);
        self.module.ckks_copy(&mut input, a, &mut scratch.borrow())?;
        input.set_slots(SlotsKind::Real);
        // The bootstrap runs at the width of its output, and leaves it at `output_k`.
        let mut out = self.alloc_like(preset.bootstrap_k);
        self.module.ckks_bootstrap(&mut out, &input, &b.context, keys, &mut scratch.borrow())?;
        out.set_slots(SlotsKind::Complex);
        Ok(out)
    }
}

impl CkksEngine {
    /// Remaining capacity of `a`, in bits.
    pub fn log_budget(&self, a: &DeviceCiphertext) -> usize {
        a.log_budget()
    }
}

/// A received key's matrix moved into a carve of the device scratch, with what
/// its preparation reads besides: its Galois element and its degrees.
struct Moved<'a> {
    matrix: GGLWEViewMut<'a, Dev>,
    p: i64,
    degrees: (Degree, Degree),
}

impl<'a> Moved<'a> {
    /// Moves `key` into the front of `scratch`, returning the rest.
    fn of<K>(
        key: &K,
        p: i64,
        degrees: (Degree, Degree),
        scratch: ScratchArena<'a, Dev>,
    ) -> (Self, ScratchArena<'a, Dev>)
    where
        K: GGLWEInfos + TransferInto<GGLWEViewMut<'a, Dev>>,
    {
        let (mut matrix, rest) = scratch.take_gglwe_scratch(key);
        key.transfer_into(&mut matrix);
        (Moved { matrix, p, degrees }, rest)
    }
}

impl GGLWEToBackendRef<Dev> for Moved<'_> {
    fn to_backend_ref(&self) -> GGLWEBackendRef<'_, Dev> {
        self.matrix.to_backend_ref()
    }
}

impl GetGaloisElement for Moved<'_> {
    fn p(&self) -> i64 {
        self.p
    }
}

impl GLWESwitchingKeyDegrees for Moved<'_> {
    fn input_degree(&self) -> &Degree {
        &self.degrees.0
    }

    fn output_degree(&self) -> &Degree {
        &self.degrees.1
    }
}

/// Prepares the keys of a stream one by one, as they arrive, so that raw keys
/// never pile up next to their prepared form.
struct Preparing<'a> {
    engine: &'a CkksEngine,
    scratch: &'a mut ScratchOwned<Dev>,
    tensor_key: Option<BackendGLWETensorKeyPrepared<Dev>>,
    rotation_keys: std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Dev>>,
    boot_rotation_keys: std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<DevBuf, Dev>>,
    boot_tensor_key: Option<GLWETensorKeyPrepared<DevBuf, Dev>>,
    encapsulation: Option<EncapsulationKeys<GLWESwitchingKeyPrepared<DevBuf, Dev>>>,
    first_factor_keys: std::collections::HashMap<i64, GLWESwitchingKeyPrepared<DevBuf, Dev>>,
}

impl<'a> Preparing<'a> {
    fn new(engine: &'a CkksEngine, scratch: &'a mut ScratchOwned<Dev>) -> Self {
        Preparing {
            engine,
            scratch,
            tensor_key: None,
            rotation_keys: Default::default(),
            boot_rotation_keys: Default::default(),
            boot_tensor_key: None,
            encapsulation: None,
            first_factor_keys: Default::default(),
        }
    }

    /// The module, for a bootstrapping key.
    fn boot(&self) -> Result<&'a Module<Dev>> {
        self.engine.bootstrap.as_ref().context("bootstrapping keys for parameters without bootstrapping")?;
        Ok(&self.engine.module)
    }

    fn tensor(&mut self, key: &GLWETensorKey<Buf, Word>) -> GLWETensorKeyPrepared<DevBuf, Dev> {
        let module = &self.engine.module;
        let (mut moved, mut rest) = self.scratch.borrow().take_gglwe_scratch(key);
        key.transfer_into(&mut moved);
        let mut prepared = module.glwe_tensor_key_prepared_alloc_from_infos(key);
        module.glwe_tensor_key_prepare(&mut prepared, &moved, &mut rest);
        prepared
    }

    fn automorphism(&mut self, key: &GLWEAutomorphismKey<Buf, Word>) -> GLWEAutomorphismKeyPrepared<DevBuf, Dev> {
        let module = &self.engine.module;
        let (moved, mut rest) = Moved::of(key, key.p(), (key.n(), key.n()), self.scratch.borrow());
        let mut prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(key);
        module.glwe_automorphism_key_prepare(&mut prepared, &moved, &mut rest);
        prepared
    }

    fn switching_key(&mut self, key: &GLWESwitchingKey<Buf, Word>) -> Result<GLWESwitchingKeyPrepared<DevBuf, Dev>> {
        let module = self.boot()?;
        let degrees = (*key.input_degree(), *key.output_degree());
        let (moved, mut rest) = Moved::of(key, 0, degrees, self.scratch.borrow());
        let mut prepared = module.glwe_switching_key_prepared_alloc_from_infos(key);
        module.glwe_switching_key_prepare(&mut prepared, &moved, &mut rest);
        Ok(prepared)
    }

    fn finish(self, bootstrap: bool) -> Result<PreparedKeys> {
        let bootstrap = if bootstrap {
            Some(BootstrappingKeysPrepared {
                rotation_keys: self.boot_rotation_keys,
                tensor_key: self.boot_tensor_key.context("missing bootstrapping tensor key")?,
                encapsulation_keys: self.encapsulation,
                first_factor_keys: self.first_factor_keys,
                conversion_keys: None,
            })
        } else {
            None
        };
        if bootstrap.is_some() != self.engine.bootstrap.is_some() {
            anyhow::bail!("bootstrapping keys do not match the parameters");
        }
        Ok(PreparedKeys {
            tensor_key: self.tensor_key.context("missing tensor key")?,
            rotation_keys: self.rotation_keys,
            bootstrap,
        })
    }
}

impl KeySink for Preparing<'_> {
    fn tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> Result<()> {
        self.tensor_key = Some(self.tensor(&key));
        Ok(())
    }

    fn rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> Result<()> {
        let prepared = self.automorphism(&key);
        self.rotation_keys.insert(galois, prepared);
        Ok(())
    }

    fn bootstrap_rotation_key(&mut self, galois: i64, key: GLWEAutomorphismKey<Buf, Word>) -> Result<()> {
        self.boot()?;
        let prepared = self.automorphism(&key);
        self.boot_rotation_keys.insert(galois, prepared);
        Ok(())
    }

    fn bootstrap_tensor_key(&mut self, key: GLWETensorKey<Buf, Word>) -> Result<()> {
        self.boot()?;
        self.boot_tensor_key = Some(self.tensor(&key));
        Ok(())
    }

    fn encapsulation_keys(
        &mut self,
        dense_to_sparse: GLWESwitchingKey<Buf, Word>,
        sparse_to_dense: Option<GLWESwitchingKey<Buf, Word>>,
    ) -> Result<()> {
        let dense_to_sparse = self.switching_key(&dense_to_sparse)?;
        let sparse_to_dense = sparse_to_dense.map(|k| self.switching_key(&k)).transpose()?;
        self.encapsulation = Some((dense_to_sparse, sparse_to_dense));
        Ok(())
    }

    fn first_factor_key(&mut self, rotation: i64, key: GLWESwitchingKey<Buf, Word>) -> Result<()> {
        let prepared = self.switching_key(&key)?;
        self.first_factor_keys.insert(rotation, prepared);
        Ok(())
    }

    fn conversion_keys(&mut self, _: GLWECITraceKey<Buf, Word>, _: GLWECIEmbedKey<Buf, Word>) -> Result<()> {
        anyhow::bail!("conversion keys for a bootstrap on the standard ring")
    }
}

impl CkksEngine {
    /// Reads a key stream and prepares every key on the device as it arrives.
    pub fn prepare_keys_from<R: std::io::Read>(&self, r: &mut R) -> Result<PreparedKeys> {
        let mut guard = self.scratch.lock().unwrap();
        let mut sink = Preparing::new(self, &mut guard.1);
        let bootstrap = read_eval_keys_into(r, &self.ring, &mut sink)?;
        sink.finish(bootstrap)
    }

    /// Installs prepared keys, or drops the current ones with `None`.
    pub fn set_keys(&mut self, keys: Option<PreparedKeys>) {
        self.keys = keys;
    }
}

/// Rotation keys of the circuit, then those of the bootstrap, and the
/// circuit's tensor key, which linear transformations also take.
struct Rotations<'a> {
    own: &'a std::collections::HashMap<i64, BackendGLWEAutomorphismKeyPrepared<Dev>>,
    shared: Option<&'a std::collections::HashMap<i64, GLWEAutomorphismKeyPrepared<DevBuf, Dev>>>,
    tensor_key: &'a BackendGLWETensorKeyPrepared<Dev>,
}

impl GetTensorKey<Dev> for Rotations<'_> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, Dev>> {
        self.tensor_key.lookup_tensor_key(k)
    }
}

impl GetAutomorphismKey<Dev> for Rotations<'_> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, Dev>> {
        match self.shared {
            Some(shared) if !self.own.contains_key(&p) => shared.lookup_automorphism_key(p, k),
            _ => self.own.lookup_automorphism_key(p, k),
        }
    }
}
