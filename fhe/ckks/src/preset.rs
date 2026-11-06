//! A bootstrapping parameter set: the plan of one of poulpy's S2C-first
//! presets (its DFTs and EvalMod), with a wider output than the preset's, for
//! circuits deeper than 17 levels between two refreshes. The widths and key
//! layouts are derived the way poulpy's preset builder does, checking every
//! key against the 128-bit modulus bounds of its secret.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use poulpy_ckks::layouts::{BootstrappingKeysLayout, BootstrappingPlan, EncapsulationKeysLayout};
use poulpy_ckks::presets::bootstrapping::BootstrappingPreset;
use poulpy_ckks::presets::first_factor::{first_factor_diagonal_growth, first_factor_small_bits, first_factor_work};
use poulpy_core::layouts::{Base2K, Degree, GGLWEInfos, GGLWELayout, Rank, TorusPrecision};

/// Largest modulus, in bits, for 128-bit security at `N = 2^16`, by secret
/// Hamming weight (sparse-key-estimate table).
const DENSE_HAMMING_WEIGHT: usize = 1024;
const DENSE_MAX_MODULUS: usize = 1707;
const SPARSE_HAMMING_WEIGHT: usize = 32;
const SPARSE_MAX_MODULUS: usize = 345;

/// Bits a key keeps between its modulus and the input plus one digit (as in
/// poulpy's presets).
const KEY_NOISE_MARGIN: usize = 2;

pub struct Preset {
    pub name: String,
    pub log_n: usize,
    pub base2k: usize,
    /// Scale of the ciphertexts entering and leaving the bootstrap.
    pub log_delta: usize,
    pub plan: BootstrappingPlan,
    pub keys_layout: BootstrappingKeysLayout,
    /// Width of a ciphertext entering the bootstrap.
    pub input_k: usize,
    /// Width of a ciphertext leaving it.
    pub output_k: usize,
    /// Working width of the bootstrap, in whole limbs.
    pub bootstrap_k: usize,
    pub dense_hamming_weight: usize,
}

impl std::fmt::Debug for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preset")
            .field("name", &self.name)
            .field("input_k", &self.input_k)
            .field("output_k", &self.output_k)
            .field("bootstrap_k", &self.bootstrap_k)
            .finish_non_exhaustive()
    }
}

impl Preset {
    /// Bootstraps with the plan and radix of `base` to outputs with at least
    /// `usable` bits above the input width.
    pub fn new(base: &BootstrappingPreset, usable: usize) -> Result<Preset> {
        let (log_n, base2k) = (base.n().trailing_zeros() as usize, base.base2k());
        ensure!(log_n == 16, "the modulus bounds are those of N = 2^16");
        ensure!(
            base.dense_secret_hamming_weight() == DENSE_HAMMING_WEIGHT
                && base.plan().sparse_secret_hamming_weight() == Some(SPARSE_HAMMING_WEIGHT),
            "the modulus bounds are those of Hamming weights {DENSE_HAMMING_WEIGHT} and {SPARSE_HAMMING_WEIGHT}"
        );
        let plan = base.plan().clone();
        let (log_delta, log_modulus) = (base.log_delta(), base.log_modulus());
        let input_k = plan.input_k(log_modulus);
        let overhead = plan.bootstrap_k(0, log_delta);
        let bootstrap_k = (input_k + usable + overhead).next_multiple_of(base2k);
        let output_k = bootstrap_k - overhead;
        ensure!(
            bootstrap_k <= DENSE_MAX_MODULUS,
            "bootstrap width {bootstrap_k} exceeds the dense-secret bound {DENSE_MAX_MODULUS}"
        );

        let keys = Shape { log_n, base2k };
        // A key only used on narrower ciphertexts takes the digit of fewest rows those admit.
        let widths = plan.key_widths(log_n, bootstrap_k, input_k, log_delta);
        let shape = |name, k| keys.key_layout(name, k, keys.widest_k_digit(k, DENSE_MAX_MODULUS), DENSE_MAX_MODULUS);
        let rotation_keys = widths
            .rotation_keys
            .iter()
            .map(|(&p, &k)| Ok((p, shape("rotation", k)?.into())))
            .collect::<Result<BTreeMap<_, _>>>()?;
        // With the first CoeffsToSlots factor on the raise, its keys replace the sparse-to-dense one.
        let first_factor_on_raise = base.keys_layout().first_factor_key.is_some();
        let raised = plan.raised_window(log_modulus);
        let first_factor_key = if first_factor_on_raise {
            let (work, _) = first_factor_work(base2k, bootstrap_k);
            let growth = first_factor_diagonal_growth(base2k, &plan.coeffs_to_slots().coeffs_meta());
            let small = first_factor_small_bits(base2k, log_n, bootstrap_k, raised, growth).div_ceil(base2k);
            ensure!(
                small < work,
                "first-factor key: a term of {small} limbs does not fit the {work} the raise works at"
            );
            let k_offset = (work - small) * base2k;
            let below = DENSE_MAX_MODULUS.saturating_sub(k_offset);
            let input = small * base2k;
            let layout = keys.key_layout("first-factor", input, keys.widest_k_digit(input, below), below)?;
            Some(GGLWELayout { k_offset: TorusPrecision(k_offset as u32), ..layout }.into())
        } else {
            None
        };
        ensure!(base.keys_layout().conversion_keys.is_none(), "EvalMod must run on the standard ring");
        let keys_layout = BootstrappingKeysLayout {
            rotation_keys,
            tensor_key: shape("tensor", widths.tensor_key)?.into(),
            encapsulation: Some(EncapsulationKeysLayout {
                dense_to_sparse: keys.key_layout("dense-to-sparse", log_modulus, base2k, SPARSE_MAX_MODULUS)?.into(),
                sparse_to_dense: if first_factor_on_raise {
                    None
                } else {
                    Some(keys.sparse_to_dense_layout(raised, bootstrap_k, DENSE_MAX_MODULUS)?.into())
                },
            }),
            first_factor_key,
            conversion_keys: None,
        };
        let preset = Preset {
            name: format!("{} widened to k{output_k}", base.name()),
            log_n,
            base2k,
            log_delta,
            plan,
            keys_layout,
            input_k,
            output_k,
            bootstrap_k,
            dense_hamming_weight: DENSE_HAMMING_WEIGHT,
        };
        preset.validate()?;
        Ok(preset)
    }

    /// Bits a ciphertext can consume between two bootstraps.
    pub fn usable(&self) -> usize {
        self.output_k - self.input_k
    }

    /// Every key within the modulus bound of the secret it is encrypted under.
    fn validate(&self) -> Result<()> {
        ensure!(
            self.bootstrap_k <= DENSE_MAX_MODULUS,
            "bootstrap modulus {} exceeds {DENSE_MAX_MODULUS}",
            self.bootstrap_k
        );
        for (name, layout, limit) in self.key_moduli()? {
            let total = layout.k_offset.as_usize()
                + layout.dnum.as_usize() * layout.k_digit.as_usize()
                + layout.k_aux.as_usize();
            ensure!(total <= limit, "{name} key modulus {total} exceeds {limit}");
        }
        Ok(())
    }

    /// Every key of the bootstrap with the modulus bound of its secret.
    pub fn key_moduli(&self) -> Result<Vec<(String, GGLWELayout, usize)>> {
        let keys = &self.keys_layout;
        let mut out: Vec<(String, GGLWELayout, usize)> = keys
            .rotation_keys
            .iter()
            .map(|(p, k)| (format!("rotation {p}"), k.gglwe_layout(), DENSE_MAX_MODULUS))
            .collect();
        out.push(("tensor".into(), keys.tensor_key.gglwe_layout(), DENSE_MAX_MODULUS));
        let enc = keys.encapsulation.as_ref().context("missing encapsulation keys")?;
        out.push(("dense-to-sparse".into(), enc.dense_to_sparse.gglwe_layout(), SPARSE_MAX_MODULUS));
        if let Some(key) = &enc.sparse_to_dense {
            out.push(("sparse-to-dense".into(), key.gglwe_layout(), DENSE_MAX_MODULUS));
        }
        if let Some(key) = &keys.first_factor_key {
            out.push(("first-factor".into(), key.gglwe_layout(), DENSE_MAX_MODULUS));
        }
        Ok(out)
    }
}

/// The key-shape rules of poulpy's preset builder.
struct Shape {
    log_n: usize,
    base2k: usize,
}

impl Shape {
    /// Guard behind the digits: a digit and `log_n` bits, or what the bound leaves.
    fn guard(&self, k_digit: usize, gadget: usize, limit: usize) -> usize {
        (k_digit + self.log_n).min(limit.saturating_sub(gadget))
    }

    fn fits(&self, input_k: usize, k_digit: usize, limit: usize) -> bool {
        let gadget = input_k.div_ceil(k_digit) * k_digit;
        let guard = self.guard(k_digit, gadget, limit);
        guard >= k_digit && gadget + guard >= input_k + k_digit + KEY_NOISE_MARGIN
    }

    /// Whole-limb digit of the key of fewest rows that fits, narrowest at that count.
    fn widest_k_digit(&self, input_k: usize, limit: usize) -> usize {
        (1..=input_k.max(1).div_ceil(self.base2k))
            .map(|limbs| limbs * self.base2k)
            .filter(|&k_digit| self.fits(input_k, k_digit, limit))
            .min_by_key(|&k_digit| (input_k.div_ceil(k_digit), k_digit))
            .unwrap_or(self.base2k)
    }

    fn key_layout(&self, name: &str, input_k: usize, k_digit: usize, limit: usize) -> Result<GGLWELayout> {
        let dnum = GGLWELayout::dnum_for_input(TorusPrecision(input_k as u32), TorusPrecision(k_digit as u32));
        let gadget = dnum.as_usize() * k_digit;
        ensure!(
            self.fits(input_k, k_digit, limit),
            "{name} key: {k_digit}-bit digits on a {input_k}-bit input do not fit {limit} bits"
        );
        Ok(GGLWELayout {
            n: Degree(1 << self.log_n),
            base2k: Base2K(self.base2k as u32),
            dnum,
            k_aux: TorusPrecision(self.guard(k_digit, gadget, limit) as u32),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: TorusPrecision(k_digit as u32),
            k_offset: TorusPrecision(0),
            stride: 1,
        })
    }

    /// The sparse-to-dense key switches the raise: its rows sit under the raised
    /// ciphertext, `window` bits at the bottom of the working width, and end at it.
    fn sparse_to_dense_layout(&self, window: usize, bootstrap_k: usize, limit: usize) -> Result<GGLWELayout> {
        let shape = |k_digit: usize| {
            let gadget = window.div_ceil(k_digit) * k_digit;
            bootstrap_k.checked_sub(gadget).map(|k_offset| (gadget, k_offset))
        };
        let fits = |k_digit: usize| {
            shape(k_digit).is_some_and(|(gadget, k_offset)| self.fits(gadget, k_digit, limit.saturating_sub(k_offset)))
        };
        let k_digit = (1..=window.div_ceil(self.base2k))
            .map(|limbs| limbs * self.base2k)
            .filter(|&k_digit| fits(k_digit))
            .min_by_key(|&k_digit| (window.div_ceil(k_digit), k_digit))
            .unwrap_or(self.base2k);
        let (gadget, k_offset) = shape(k_digit).context("sparse-to-dense key: digits overshoot the working width")?;
        Ok(GGLWELayout {
            k_offset: TorusPrecision(k_offset as u32),
            ..self.key_layout("sparse-to-dense", gadget, k_digit, limit.saturating_sub(k_offset))?
        })
    }
}
