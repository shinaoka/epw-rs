//! Phonon linewidths, mode-resolved coupling strengths and the Eliashberg function.
//!
//! Formulas:
//! - Imaginary part of the phonon self-energy in the Migdal approximation,
//!   S. Poncé et al., Comput. Phys. Commun. 209, 116 (2016), Eq. 9:
//!   `gamma_{q nu} = -pi sum_{k m n} w_k |g_{mn nu}(k,q)|^2 / (2 omega_{q nu})
//!                   [f(e_{m k+q}) - f(e_{n k})] delta(e_{m k+q} - e_{n k} - omega_{q nu})`,
//!   with Fermi–Dirac occupations at temperature T and a Gaussian delta.
//! - `lambda_{q nu} = gamma_{q nu} / (pi N_F omega_{q nu}^2)` with `N_F` the DOS per spin.
//! - `alpha^2F(omega) = 1/2 sum_{q nu} w_q omega_{q nu} lambda_{q nu} delta(omega - omega_{q nu})`,
//!   `lambda = 2 int alpha^2F / omega`, `omega_log = exp[(2/lambda) int alpha^2F ln(omega) / omega]`.
//! - Allen–Dynes T_c: P. B. Allen and R. C. Dynes, Phys. Rev. B 12, 905 (1975).
//!
//! Conventions chosen to agree numerically with EPW 6.1 (QE 7.6): only
//! (k, k+q) pairs with a band within `fsthick` of E_F at both points
//! contribute, summed over the bands of the Fermi window; modes with
//! `omega <= eps_acoustic` are dropped; gamma and lambda are averaged over
//! modes whose frequencies differ by less than 1e-6 Ry; negative lambda is
//! ignored in alpha^2F. The code was written independently from the formulas.
//!
//! # Examples
//!
//! ```
//! let tc = eph_selfen::allen_dynes_tc(100.0, 1.0, 0.1).expect("lambda above threshold");
//! assert!(tc > 0.0 && tc < 100.0);
//! ```

use eph_comm::ReduceKind;
use num_complex::Complex64;
use std::ops::Range;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_tensor::TypedTensor;

/// Errors from the self-energy and alpha^2F evaluation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tensor operation failed.
    #[error("tensor operation failed")]
    Tensor(#[from] tenferro_tensor::Error),
    /// Einsum failed.
    #[error("einsum failed")]
    Einsum(#[from] tenferro_einsum::Error),
    /// Inputs have inconsistent sizes.
    #[error("{what}: expected {expected} elements, got {got}")]
    Length {
        /// Which input.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
}

/// Settings of the phonon self-energy (energies in Ry).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinewidthParams {
    /// Half-width of the Fermi-surface window.
    pub fsthick: f64,
    /// Gaussian broadening of the energy-conserving delta.
    pub sigma: f64,
    /// Temperature of the occupations (as an energy).
    pub temperature: f64,
    /// Frequencies at or below this are treated as acoustic and excluded.
    pub eps_acoustic: f64,
}

/// Electronic quantities fixed for the whole q loop.
#[derive(Debug, Clone, PartialEq)]
pub struct FermiSurface {
    /// Fermi energy (Ry).
    pub fermi_energy: f64,
    /// DOS per spin at E_F (states / Ry / cell).
    pub dos_per_spin: f64,
    /// Bands with an energy within `fsthick` of E_F somewhere on the k mesh.
    pub band_window: Range<usize>,
}

/// Data of one q-point needed for its linewidth.
#[derive(Debug, Clone, Copy)]
pub struct QPointData<'a> {
    /// Number of bands.
    pub nbnd: usize,
    /// Frequencies `(nmodes)`.
    pub omega: &'a [f64],
    /// Vertex `g[m, n, k, nu]` (band m at k+q, band n at k), not divided by sqrt(2 omega).
    pub g: &'a [Complex64],
    /// Energies at k, `(nbnd, nk)`.
    pub energies_k: &'a [f64],
    /// Energies at k+q, `(nbnd, nk)`.
    pub energies_kq: &'a [f64],
    /// k-point weights including spin, `(nk)`.
    pub weights: &'a [f64],
}

/// Linewidth and coupling of every mode at one q-point (Ry).
#[derive(Debug, Clone, PartialEq)]
pub struct Linewidth {
    /// Frequencies.
    pub omega: Vec<f64>,
    /// Linewidths gamma (half width, Ry).
    pub gamma: Vec<f64>,
    /// Mode coupling strengths lambda.
    pub lambda: Vec<f64>,
}

/// A per-q quantity together with how it is combined across ranks.
pub trait QQuantity {
    /// Reduction applied by the driver.
    const REDUCE: ReduceKind;
}

impl QQuantity for Linewidth {
    const REDUCE: ReduceKind = ReduceKind::GatherByIndex;
}

impl QQuantity for EliashbergFunction {
    const REDUCE: ReduceKind = ReduceKind::Sum;
}

/// Phonon linewidths and coupling strengths at one q-point.
///
/// # Errors
///
/// [`Error::Length`] for inconsistent inputs, tensor errors otherwise.
///
/// # Examples
///
/// ```
/// use eph_selfen::*;
/// use num_complex::Complex64;
/// // One band crossing E_F, one mode, two k-points; zero coupling gives zero width.
/// let params = LinewidthParams { fsthick: 1.0, sigma: 0.01, temperature: 0.001, eps_acoustic: 1e-6 };
/// let fs = FermiSurface { fermi_energy: 0.0, dos_per_spin: 1.0, band_window: 0..1 };
/// let data = QPointData { nbnd: 1, omega: &[0.01], g: &[Complex64::default(); 2],
///     energies_k: &[0.0, 0.0], energies_kq: &[0.0, 0.0], weights: &[1.0, 1.0] };
/// let mut engine = tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().build().unwrap());
/// let lw = phonon_linewidth(&mut engine, &data, &fs, &params)?;
/// assert_eq!(lw.gamma, vec![0.0]);
/// # Ok::<(), eph_selfen::Error>(())
/// ```
pub fn phonon_linewidth(
    backend: &mut CpuEngine,
    data: &QPointData<'_>,
    fs: &FermiSurface,
    params: &LinewidthParams,
) -> Result<Linewidth, Error> {
    let nb = data.nbnd;
    let nk = data.weights.len();
    let nm = data.omega.len();
    check("g", nb * nb * nk * nm, data.g.len())?;
    check("energies_k", nb * nk, data.energies_k.len())?;
    check("energies_kq", nb * nk, data.energies_kq.len())?;

    let window = fs.band_window.clone();
    let nw = window.len();
    let ef = fs.fermi_energy;
    let near_fermi = |bands: &[f64]| bands.iter().any(|e| (e - ef).abs() < params.fsthick);
    let k_active: Vec<bool> = (0..nk)
        .map(|k| {
            near_fermi(&data.energies_k[k * nb..(k + 1) * nb])
                && near_fermi(&data.energies_kq[k * nb..(k + 1) * nb])
        })
        .collect();
    let inv_two_omega: Vec<f64> = data
        .omega
        .iter()
        .map(|&w| {
            if w > params.eps_acoustic {
                1.0 / (2.0 * w)
            } else {
                0.0
            }
        })
        .collect();

    // Band pairs x = (m, n, k) in the window: k weight times occupation change
    // (zero outside the Fermi-surface filter) and the energy gap e_{m,k+q} - e_{n,k}.
    let sigma = params.sigma;
    let mut pair_factor = Vec::with_capacity(nw * nw * nk);
    let mut pair_gap = Vec::with_capacity(nw * nw * nk);
    let mut g_index = Vec::with_capacity(nw * nw * nk);
    for (k, (&active, &wk)) in k_active.iter().zip(data.weights).enumerate() {
        for n in window.clone() {
            for m in window.clone() {
                let e_n = data.energies_k[n + nb * k] - ef;
                let e_m = data.energies_kq[m + nb * k] - ef;
                let occupation_change = smearing::fermi_dirac(e_m, params.temperature)
                    - smearing::fermi_dirac(e_n, params.temperature);
                pair_factor.push(if active {
                    -phys_units::PI * wk * occupation_change
                } else {
                    0.0
                });
                pair_gap.push(e_m - e_n);
                g_index.push(m + nb * (n + nb * k));
            }
        }
    }
    // Energy-conserving delta and coupling |g|^2 / (2 omega) over (x, nu).
    let per_mode = nb * nb * nk;
    let mut delta = Vec::with_capacity(pair_gap.len() * nm);
    let mut coupling = Vec::with_capacity(pair_gap.len() * nm);
    for (nu, (&omega, &inv_2w)) in data.omega.iter().zip(&inv_two_omega).enumerate() {
        let g_mode = &data.g[nu * per_mode..(nu + 1) * per_mode];
        delta.extend(
            pair_gap
                .iter()
                .map(|gap| smearing::Smearing::Gaussian.delta((gap - omega) / sigma) / sigma),
        );
        coupling.extend(g_index.iter().map(|&i| g_mode[i].norm_sqr() * inv_2w));
    }
    let npair = pair_factor.len();
    let pair_factor = TypedTensor::from_vec_col_major(vec![npair], pair_factor)?;
    let delta = TypedTensor::from_vec_col_major(vec![npair, nm], delta)?;
    let coupling = TypedTensor::from_vec_col_major(vec![npair, nm], coupling)?;
    // Pairwise in a fixed order: multi-operand einsum paths are chosen by the
    // planner and may change the last bits between calls.
    let gamma_raw = {
        let s = &mut *backend;
        let weighted = [&delta, &coupling].einsum("xv,xv->xv", s)?;
        [&pair_factor, &weighted].einsum("x,xv->v", s)?
    };
    let gamma_raw = gamma_raw.as_slice()?;

    // Average over degenerate modes.
    let mut gamma = vec![0.0; nm];
    let mut lambda = vec![0.0; nm];
    for i in 0..nm {
        let group: Vec<usize> = (0..nm)
            .filter(|&j| (data.omega[i] - data.omega[j]).abs() < 1.0e-6)
            .collect();
        let count = group.len() as f64;
        let w2 = data.omega[i] * data.omega[i];
        gamma[i] = group.iter().map(|&j| gamma_raw[j]).sum::<f64>() / count;
        lambda[i] = group
            .iter()
            .filter(|&&j| data.omega[j] > params.eps_acoustic)
            .map(|&j| gamma_raw[j] / phys_units::PI / w2 / fs.dos_per_spin)
            .sum::<f64>()
            / count;
    }
    Ok(Linewidth {
        omega: data.omega.to_vec(),
        gamma,
        lambda,
    })
}

/// Settings of the Eliashberg function (energies in Ry).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EliashbergParams {
    /// Number of frequency points.
    pub nstep: usize,
    /// Smallest Gaussian broadening.
    pub sigma: f64,
    /// Number of broadenings.
    pub nsmear: usize,
    /// Increment between broadenings.
    pub sigma_step: f64,
    /// Acoustic cut-off as in [`LinewidthParams::eps_acoustic`].
    pub eps_acoustic: f64,
}

/// alpha^2F(omega) and derived quantities.
#[derive(Debug, Clone, PartialEq)]
pub struct EliashbergFunction {
    /// Frequency grid `omega_i = i * d omega`, i = 1..nstep (Ry).
    pub omega: Vec<f64>,
    /// alpha^2F, shape `(nstep, nsmear)`.
    pub a2f: Vec<f64>,
    /// Phonon DOS, shape `(nstep, nsmear)` (states / Ry).
    pub phonon_dos: Vec<f64>,
    /// Broadenings (Ry).
    pub sigmas: Vec<f64>,
    /// lambda = sum_{q nu} w_q lambda_{q nu} (positive, non-acoustic modes).
    pub lambda: f64,
    /// lambda = 2 int alpha^2F / omega for each broadening.
    pub lambda_integrated: Vec<f64>,
    /// omega_log for each broadening (Ry).
    pub omega_log: Vec<f64>,
}

/// Eliashberg function from the linewidths of all q-points (weights `q_weight` each).
///
/// # Errors
///
/// Tensor errors.
///
/// # Examples
///
/// ```
/// use eph_selfen::*;
/// let lw = Linewidth { omega: vec![0.01], gamma: vec![0.0], lambda: vec![1.0] };
/// let p = EliashbergParams { nstep: 2000, sigma: 2e-4, nsmear: 1, sigma_step: 0.0, eps_acoustic: 1e-8 };
/// let mut engine = tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().build().unwrap());
/// let a2f = eliashberg_function(&mut engine, &[lw], 1.0, &p)?;
/// assert!((a2f.lambda_integrated[0] - 1.0).abs() < 1e-2);
/// assert!((a2f.omega_log[0] - 0.01).abs() < 1e-4);
/// # Ok::<(), eph_selfen::Error>(())
/// ```
pub fn eliashberg_function(
    backend: &mut CpuEngine,
    linewidths: &[Linewidth],
    q_weight: f64,
    params: &EliashbergParams,
) -> Result<EliashbergFunction, Error> {
    let modes: Vec<(f64, f64)> = linewidths
        .iter()
        .flat_map(|lw| lw.omega.iter().copied().zip(lw.lambda.iter().copied()))
        .collect();
    let omega_max = 1.1 * modes.iter().map(|m| m.0).fold(f64::MIN, f64::max);
    let dw = omega_max / params.nstep as f64;
    let omega: Vec<f64> = (1..=params.nstep).map(|i| i as f64 * dw).collect();
    let sigmas: Vec<f64> = (0..params.nsmear)
        .map(|s| params.sigma + s as f64 * params.sigma_step)
        .collect();

    let active: Vec<(f64, f64)> = modes
        .into_iter()
        .filter(|&(w, _)| w > params.eps_acoustic)
        .collect();
    let nx = active.len();
    // Gaussian kernel (omega_i, sigma_s, mode x).
    let mut kernel = Vec::with_capacity(params.nstep * params.nsmear * nx);
    for &(w0, _) in &active {
        for &sigma in &sigmas {
            for &w in &omega {
                kernel.push(smearing::Smearing::Gaussian.delta((w - w0) / sigma) / sigma);
            }
        }
    }
    let strength: Vec<f64> = active
        .iter()
        .map(|&(w, l)| q_weight * w * l.max(0.0) / 2.0)
        .collect();
    let dos_weight = vec![q_weight; nx];
    let kernel = TypedTensor::from_vec_col_major(vec![params.nstep, params.nsmear, nx], kernel)?;
    let strength = TypedTensor::from_vec_col_major(vec![nx], strength)?;
    let dos_weight = TypedTensor::from_vec_col_major(vec![nx], dos_weight)?;
    let inv_omega = TypedTensor::from_vec_col_major(
        vec![params.nstep],
        omega.iter().map(|w| 2.0 * dw / w).collect(),
    )?;
    let log_inv_omega = TypedTensor::from_vec_col_major(
        vec![params.nstep],
        omega.iter().map(|w| 2.0 * dw * w.ln() / w).collect(),
    )?;
    let (a2f, dos, lambda_int, log_moment) = {
        let s = &mut *backend;
        let a2f = [&kernel, &strength].einsum("wsx,x->ws", s)?;
        let dos = [&kernel, &dos_weight].einsum("wsx,x->ws", s)?;
        let lambda_int = [&a2f, &inv_omega].einsum("ws,w->s", s)?;
        let log_moment = [&a2f, &log_inv_omega].einsum("ws,w->s", s)?;
        (a2f, dos, lambda_int, log_moment)
    };
    let lambda_integrated = lambda_int.as_slice()?.to_vec();
    let omega_log = log_moment
        .as_slice()?
        .iter()
        .zip(&lambda_integrated)
        .map(|(m, l)| (m / l).exp())
        .collect();
    let lambda = linewidths
        .iter()
        .flat_map(|lw| lw.omega.iter().zip(&lw.lambda))
        .filter(|&(&w, &l)| l > 0.0 && w > params.eps_acoustic)
        .map(|(_, &l)| q_weight * l)
        .sum();
    Ok(EliashbergFunction {
        omega,
        a2f: a2f.as_slice()?.to_vec(),
        phonon_dos: dos.as_slice()?.to_vec(),
        sigmas,
        lambda,
        lambda_integrated,
        omega_log,
    })
}

/// Allen–Dynes estimate of T_c (same unit as `omega_log`).
///
/// Returns `None` when `lambda <= mu_star (1 + 0.62 lambda)`, where the formula
/// has no superconducting solution.
///
/// # Examples
///
/// ```
/// assert!(eph_selfen::allen_dynes_tc(1.0, 0.1, 0.1).is_none());
/// let tc = eph_selfen::allen_dynes_tc(100.0, 1.0, 0.1).unwrap();
/// assert!(tc > 0.0 && tc < 100.0);
/// ```
pub fn allen_dynes_tc(omega_log: f64, lambda: f64, mu_star: f64) -> Option<f64> {
    let denominator = lambda - mu_star * (1.0 + 0.62 * lambda);
    (denominator > 0.0).then(|| omega_log / 1.2 * (-1.04 * (1.0 + lambda) / denominator).exp())
}

fn check(what: &'static str, expected: usize, got: usize) -> Result<(), Error> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::Length {
            what,
            expected,
            got,
        })
    }
}
