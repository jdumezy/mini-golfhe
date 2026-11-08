//! CKKS parameters: the secure set with bootstrapping, and small insecure sets
//! for tests.

use std::sync::Arc;

use anyhow::{Context, Result};
use poulpy_ckks::presets::bootstrapping::{BootstrappingPreset, n16_d35_k733_p20_s2c};
use poulpy_core::layouts::{GLWEAutomorphismKeyLayout, GLWELayout, GLWETensorKeyLayout, Rank};

use crate::backend::{Be, Dev};
use crate::preset::Preset;

/// Largest modulus of a key under the dense secret, for 128-bit security at `N = 2^16`.
const MAX_KEY_MODULUS: usize = crate::preset::DENSE_MAX_MODULUS;

/// Failure probability `2^-128` per bootstrap when adapting the radix to the backend.
const BOOTSTRAP_FAILURE_BITS: usize = 128;

/// Bits each half of a contact may consume between two bootstraps, rounded up
/// to whole limbs: 27 limbs for the bootstrap, 811-bit ciphertexts. The first
/// half leaves 6 bits to spare, the second, whose prefix product runs at a
/// coarser scale, 44.
pub const USABLE_BITS: usize = 647;

/// CoeffsToSlots as `(layers, giant step)` factors, the first applied by the
/// modulus raise: four small factors instead of the preset's three of five
/// layers, about a third faster on a GPU (measured on Metal), with smaller keys.
const FIRST_FACTOR_ON_RAISE: [(usize, usize); 4] = [(3, 1), (4, 1024), (4, 64), (4, 4)];

/// The largest moduli of a parameter set against the 128-bit bounds of their
/// secrets, in bits.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Moduli {
    /// The widest key under the dense secret.
    pub max_key: usize,
    /// The widest ciphertext: the bootstrap's working width, or a fresh one.
    pub max_ciphertext: usize,
    /// Bound for the dense secret.
    pub bound: usize,
    pub hamming_weight: usize,
    /// The bootstrap's key under its sparse secret, and that secret's bound and weight.
    pub sparse_key: Option<usize>,
    pub sparse_bound: usize,
    pub sparse_hamming_weight: usize,
    pub log_n: usize,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub name: String,
    pub log_n: usize,
    pub base2k: usize,
    /// Scale of the ciphertexts, `2^log_delta`.
    pub log_delta: usize,
    /// Precision of the public plaintexts (masks, affine coefficients).
    pub pt_log_delta: usize,
    /// Scale of the polynomial coefficients.
    pub poly_log_delta: usize,
    /// Width of fresh ciphertexts, the widest the circuit sees.
    pub fresh_k: usize,
    pub hamming_weight: usize,
    pub bootstrap: Option<Arc<Preset>>,
}

/// The preset at the largest radix both backends afford for a `2^-128`
/// bootstrap failure probability, as poulpy's `preset_for_backend` derives it:
/// above the nominal radix the keys are re-derived, below it the high-modulus
/// keys take the widest digits that fit the modulus bounds. The client's CPU
/// backend and the server's device must agree on the parameters. (EvalMod runs
/// on the standard ring, so no conjugate-invariant backend is involved.)
fn for_backend(preset: BootstrappingPreset) -> Result<BootstrappingPreset> {
    let max = preset
        .max_backend_base2k::<Be, Be>(BOOTSTRAP_FAILURE_BITS)
        .min(preset.max_backend_base2k::<Dev, Dev>(BOOTSTRAP_FAILURE_BITS));
    if max > preset.base2k() {
        return (1..=max)
            .rev()
            .find_map(|base2k| preset.with_base2k(base2k).ok())
            .with_context(|| format!("no radix up to {max} fits the modulus bounds of {}", preset.name()));
    }
    if max == preset.base2k() {
        return Ok(preset);
    }
    let preset = preset.with_base2k(max)?;
    for dsize in (2..=7).rev() {
        if let Ok(adapted) = preset.with_dsizes(dsize, 1) {
            return Ok(adapted);
        }
    }
    preset.with_dsizes(1, 1)
}

impl Params {
    /// 128-bit secure parameters: logN 16, refreshed by the bootstrap of the
    /// `n16_d35_k733_p20_s2c` preset with the first CoeffsToSlots factor on the
    /// raise ([`FIRST_FACTOR_ON_RAISE`]), widened to [`USABLE_BITS`].
    pub fn secure() -> Result<Params> {
        // Power-of-two rotation keys: fewer bootstrapping keys, faster on a GPU.
        let base =
            n16_d35_k733_p20_s2c()?.with_power_of_two_keys()?.with_first_factor_on_raise(&FIRST_FACTOR_ON_RAISE)?;
        let base = for_backend(base)?;
        let preset = Preset::new(&base, USABLE_BITS).context("widening the bootstrap")?;
        Ok(Params {
            name: format!("{} (power-of-two keys, first factor on the raise)", preset.name),
            log_n: preset.log_n,
            base2k: preset.base2k,
            log_delta: preset.log_delta,
            pt_log_delta: 24,
            // Polynomial coefficients exact to 2^-28: the error stays far
            // below the ciphertexts' and each constant product spends 7 bits less.
            poly_log_delta: 28,
            fresh_k: preset.output_k,
            hamming_weight: preset.dense_hamming_weight,
            bootstrap: Some(Arc::new(preset)),
        })
    }

    /// Small, **insecure** parameters without bootstrapping, wide enough for
    /// one contact. For tests only.
    pub fn insecure_test() -> Params {
        Params {
            name: "insecure-test".into(),
            log_n: 16,
            base2k: 52,
            log_delta: 35,
            pt_log_delta: 24,
            poly_log_delta: 35,
            fresh_k: 30 * 52,
            hamming_weight: 192,
            bootstrap: None,
        }
    }

    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    /// The largest moduli against the 128-bit bounds; `None` for the insecure
    /// test parameters, which have no bound.
    pub fn moduli(&self) -> Option<Moduli> {
        use crate::preset::{DENSE_MAX_MODULUS, SPARSE_HAMMING_WEIGHT, SPARSE_MAX_MODULUS};
        let preset = self.bootstrap.as_deref()?;
        let total = |l: &poulpy_core::layouts::GGLWELayout| {
            l.k_offset.as_usize() + l.dnum.as_usize() * l.k_digit.as_usize() + l.k_aux.as_usize()
        };
        let keys = preset.key_moduli().ok()?;
        let dense = keys.iter().filter(|(_, _, limit)| *limit == DENSE_MAX_MODULUS).map(|(_, l, _)| total(l));
        let sparse = keys.iter().filter(|(_, _, limit)| *limit == SPARSE_MAX_MODULUS).map(|(_, l, _)| total(l));
        Some(Moduli {
            max_key: dense.chain([self.key_modulus()]).max()?,
            max_ciphertext: preset.bootstrap_k.max(self.fresh_k),
            bound: DENSE_MAX_MODULUS,
            hamming_weight: self.hamming_weight,
            sparse_key: sparse.max(),
            sparse_bound: SPARSE_MAX_MODULUS,
            sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
            log_n: self.log_n,
        })
    }

    /// Complex slots of a ciphertext.
    pub fn slots(&self) -> usize {
        self.n() / 2
    }

    pub fn glwe_layout(&self) -> GLWELayout {
        GLWELayout { n: self.n().into(), base2k: self.base2k.into(), k: self.fresh_k.into(), rank: Rank(1) }
    }

    /// Gadget digits covering a fresh ciphertext, as few as the 128-bit
    /// modulus bound admits: `(dnum, k_digit, k_aux)`.
    fn key_shape(&self) -> (usize, usize, usize) {
        let shape = |dnum: usize| {
            let k_digit = self.fresh_k.div_ceil(dnum);
            (dnum, k_digit, k_digit + self.log_n)
        };
        if self.bootstrap.is_none() {
            // Test parameters, insecure anyway.
            return shape(1);
        }
        (1..).map(shape).find(|&(d, k, aux)| d * k + aux <= MAX_KEY_MODULUS).unwrap()
    }

    /// Total modulus of the circuit's keys.
    pub fn key_modulus(&self) -> usize {
        let (dnum, k_digit, k_aux) = self.key_shape();
        dnum * k_digit + k_aux
    }

    pub fn tensor_key_layout(&self) -> GLWETensorKeyLayout {
        let (dnum, k_digit, k_aux) = self.key_shape();
        GLWETensorKeyLayout {
            n: self.n().into(),
            base2k: self.base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            rank: Rank(1),
            k_digit: k_digit.into(),
        }
    }

    pub fn rotation_key_layout(&self) -> GLWEAutomorphismKeyLayout {
        let (dnum, k_digit, k_aux) = self.key_shape();
        GLWEAutomorphismKeyLayout {
            n: self.n().into(),
            base2k: self.base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            rank: Rank(1),
            k_digit: k_digit.into(),
        }
    }
}
