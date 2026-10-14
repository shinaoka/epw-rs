//! Anisotropic full-bandwidth Migdal–Eliashberg equations on the imaginary axis.
//!
//! States `s = (n, k)` are electronic states inside the Fermi window. With the
//! state-pair kernel `L_{ss'}(i nu) = sum_{q: k+q = k'} w_q 2 sum_nu |g|^2/(2 omega) * omega / (omega^2 + nu^2)`
//! (E. R. Margine and F. Giustino, PRB 87, 024505 (2013); Pickett, PRB 26, 1186 (1982)):
//!
//! - `Z_s(n) = 1 + (T / omega_n) sum_{s', m} L_{ss'}(n - m) omega_m Z_{s'}(m) / Theta_{s'}(m)`
//! - `Z_s(n) Delta_s(n) = T sum_{s', m} [L_{ss'}(n - m) - mu* M_{ss'}] Delta_{s'}(m) Z_{s'}(m) / Theta_{s'}(m)`
//! - `chi_s(n) = -T sum_{s', m} L_{ss'}(n - m) (e_{s'} - mu + chi_{s'}(m)) / Theta_{s'}(m)`
//!
//! with `Theta = (omega Z)^2 + (e - mu + chi)^2 + (Delta Z)^2` and
//! `M_{ss'} = sum_{q: k+q = k'} w_q / N_F`. Sums over `m` run over all fermionic
//! frequencies; a [`StateFrequencySum`] decides how (uniform cutoff or IR).

use crate::matsubara::{fermionic_frequency, Parity};
use crate::Error;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_tensor::{BackendSessionHost, TypedTensor};

/// Matsubara convolutions coupling states through a kernel matrix `L_{ss'}(i nu)`.
pub trait StateFrequencySum {
    /// Temperature (eV).
    fn temperature(&self) -> f64;

    /// Indices `n >= 0` of the positive fermionic frequencies used.
    fn indices(&self) -> &[i64];

    /// `C_s(n) = sum_{s', m in Z} L_{ss'}(n - m) f_{s'}(m)`; `f` and the result have shape `(nfreq, nstates)`.
    ///
    /// # Errors
    ///
    /// Tensor or basis errors, [`Error::InvalidInput`] for a wrong shape.
    fn convolve(&mut self, f: &TypedTensor<f64>, parity: Parity)
        -> Result<TypedTensor<f64>, Error>;

    /// `sum_{m in Z} f_s(m)` for each state of an even `f` of shape `(nfreq, nstates)`.
    ///
    /// # Errors
    ///
    /// As [`StateFrequencySum::convolve`].
    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error>;
}

/// Electronic states, weights and the e-ph kernel of the anisotropic problem.
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicModel {
    /// Energies of the states relative to the chemical potential (eV).
    pub energies: Vec<f64>,
    /// Fermi-surface weight of each state for averages (sums to ~1).
    pub fs_weights: Vec<f64>,
    /// Kernel spectrum: entries `(s, s', omega, c)` with
    /// `L_{ss'}(i nu) = sum c * 2 omega / (omega^2 + nu^2)`.
    pub kernel_entries: Vec<KernelEntry>,
    /// Coulomb weights `M_{ss'}` (row-major `s * nstates + s'`).
    pub coulomb: Vec<f64>,
}

/// One phonon contribution to the kernel between two states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelEntry {
    /// Row state.
    pub s: usize,
    /// Column state.
    pub s_prime: usize,
    /// Phonon frequency (eV).
    pub omega: f64,
    /// Weight (eV^2 times the q weight).
    pub weight: f64,
}

impl AnisotropicModel {
    /// Number of states.
    ///
    /// # Examples
    ///
    /// ```
    /// let m = eliashberg::AnisotropicModel { energies: vec![0.0; 2], fs_weights: vec![0.5; 2],
    ///     kernel_entries: vec![], coulomb: vec![0.0; 4] };
    /// assert_eq!(m.nstates(), 2);
    /// ```
    pub fn nstates(&self) -> usize {
        self.energies.len()
    }

    /// `L_{ss'}(i nu)` for a batch of bosonic frequencies, shape `(nstates, nstates, nnu)`.
    ///
    /// # Errors
    ///
    /// Tensor errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::{AnisotropicModel, KernelEntry};
    /// let m = AnisotropicModel { energies: vec![0.0], fs_weights: vec![1.0], coulomb: vec![0.0],
    ///     kernel_entries: vec![KernelEntry { s: 0, s_prime: 0, omega: 0.1, weight: 0.05 }] };
    /// let l = m.kernel_matsubara(&[0.0])?;
    /// assert!((l.as_slice()?[0] - 1.0).abs() < 1e-15);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn kernel_matsubara(&self, nu: &[f64]) -> Result<TypedTensor<f64>, Error> {
        let ns = self.nstates();
        let mut out = vec![0.0; ns * ns * nu.len()];
        for e in &self.kernel_entries {
            let pair = e.s + ns * e.s_prime;
            for (l, v) in nu.iter().enumerate() {
                out[pair + ns * ns * l] += e.weight * 2.0 * e.omega / (e.omega * e.omega + v * v);
            }
        }
        Ok(TypedTensor::from_vec_col_major(
            vec![ns, ns, nu.len()],
            out,
        )?)
    }
}

/// Uniform-grid [`StateFrequencySum`]: frequencies `n = 0..N-1` (EPW `gridsamp = 0`).
#[derive(Debug)]
pub struct UniformStateSum {
    temperature: f64,
    indices: Vec<i64>,
    /// `L[s, s', l]` at `nu_l = 2 l pi T`, `l = 0..2N-1`.
    kernel: TypedTensor<f64>,
    backend: CpuBackend,
}

impl UniformStateSum {
    /// Build the sum with `nfreq` positive frequencies.
    ///
    /// # Errors
    ///
    /// Tensor errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::*;
    /// let m = AnisotropicModel { energies: vec![0.0], fs_weights: vec![1.0], coulomb: vec![0.0],
    ///     kernel_entries: vec![KernelEntry { s: 0, s_prime: 0, omega: 0.1, weight: 0.05 }] };
    /// let sum = UniformStateSum::new(tenferro_cpu::CpuBackend::new(), &m, 0.01, 8)?;
    /// assert_eq!(sum.indices().len(), 8);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(
        backend: CpuBackend,
        model: &AnisotropicModel,
        temperature: f64,
        nfreq: usize,
    ) -> Result<Self, Error> {
        let two_pi_t = 2.0 * std::f64::consts::PI * temperature;
        let nu: Vec<f64> = (0..2 * nfreq).map(|l| l as f64 * two_pi_t).collect();
        Ok(Self {
            temperature,
            indices: (0..nfreq as i64).collect(),
            kernel: model.kernel_matsubara(&nu)?,
            backend,
        })
    }
}

impl StateFrequencySum for UniformStateSum {
    fn temperature(&self) -> f64 {
        self.temperature
    }

    fn indices(&self) -> &[i64] {
        &self.indices
    }

    fn convolve(
        &mut self,
        f: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        let nf = self.indices.len();
        let ns = self.kernel.shape()[0];
        check_shape(f, nf, ns)?;
        // G[s, l, m] = sum_{s'} L[s, s', l] f[m, s']
        let g = self
            .backend
            .with_backend_session(|s| [&self.kernel, f].einsum("sql,mq->slm", s))?;
        let g = g.as_slice()?;
        let nl = 2 * nf;
        let sign = match parity {
            Parity::Odd => -1.0,
            Parity::Even => 1.0,
        };
        // C[n, s] = sum_m G[s, |n-m|, m] ± G[s, n+m+1, m]
        let mut out = vec![0.0; nf * ns];
        for s in 0..ns {
            for n in 0..nf {
                let mut acc = 0.0;
                for m in 0..nf {
                    acc += g[s + ns * (n.abs_diff(m) + nl * m)]
                        + sign * g[s + ns * (n + m + 1 + nl * m)];
                }
                out[n + nf * s] = acc;
            }
        }
        Ok(TypedTensor::from_vec_col_major(vec![nf, ns], out)?)
    }

    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        let nf = self.indices.len();
        let ns = self.kernel.shape()[0];
        check_shape(f, nf, ns)?;
        let twos = TypedTensor::from_vec_col_major(vec![nf], vec![2.0; nf])?;
        let out = self
            .backend
            .with_backend_session(|s| [f, &twos].einsum("ms,m->s", s))?;
        Ok(out.as_slice()?.to_vec())
    }
}

pub(crate) fn check_shape(f: &TypedTensor<f64>, nf: usize, ns: usize) -> Result<(), Error> {
    if f.shape() == [nf, ns] {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "expected shape ({nf}, {ns}), got {:?}",
            f.shape()
        )))
    }
}

/// Settings of the anisotropic solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnisotropicParams {
    /// Coulomb pseudopotential mu*.
    pub mu_star: f64,
    /// Convergence threshold on `sum |Delta - Delta_old| / sum |Delta|` over all states and frequencies.
    pub conv_thr: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Linear mixing weight.
    pub mixing: f64,
}

/// Converged anisotropic solution, arrays of shape `(nfreq, nstates)`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicSolution {
    /// Matsubara frequencies (eV).
    pub omega: Vec<f64>,
    /// Z.
    pub z: Vec<f64>,
    /// Delta (eV).
    pub delta: Vec<f64>,
    /// chi (eV).
    pub shift: Vec<f64>,
    /// Normal-state Z evaluated with the converged Z and chi (as EPW's `naznormi`).
    pub z_normal: Vec<f64>,
    /// Iterations.
    pub iterations: usize,
    /// Final relative change.
    pub residual: f64,
}

impl AnisotropicSolution {
    /// `-2 pi T sum_s w_s sum_n (Omega - omega)(Z - Z_N omega / Omega)` with `w_s` the
    /// Fermi-surface weights times `N_F` (EPW's `0.5 wkfs w0g`).
    ///
    /// # Examples
    ///
    /// ```
    /// let sol = eliashberg::AnisotropicSolution { omega: vec![1.0], z: vec![1.0], delta: vec![0.0],
    ///     shift: vec![0.0], z_normal: vec![1.0], iterations: 1, residual: 0.0 };
    /// assert_eq!(sol.free_energy(&[1.0], 0.01), 0.0);
    /// ```
    pub fn free_energy(&self, state_weights: &[f64], temperature: f64) -> f64 {
        let nf = self.omega.len();
        let mut sum = 0.0;
        for (s, w) in state_weights.iter().enumerate() {
            for n in 0..nf {
                let i = n + nf * s;
                let om = self.omega[n];
                let big = (om * om + self.delta[i] * self.delta[i]).sqrt();
                sum += w * (big - om) * (self.z[i] - self.z_normal[i] * om / big);
            }
        }
        -2.0 * std::f64::consts::PI * temperature * sum
    }
}

/// Solve the anisotropic FBW equations.
///
/// `initial_gap(omega)` gives the starting gap (the same for every state).
///
/// # Errors
///
/// [`Error::InvalidInput`], [`Error::NotConverged`], and errors of the frequency sum.
///
/// # Examples
///
/// ```
/// use eliashberg::*;
/// // One state at E_F coupled to itself: reduces to an Einstein model.
/// let m = AnisotropicModel { energies: vec![0.0], fs_weights: vec![1.0], coulomb: vec![0.0],
///     kernel_entries: vec![KernelEntry { s: 0, s_prime: 0, omega: 0.05, weight: 0.025 }] };
/// let mut sum = UniformStateSum::new(tenferro_cpu::CpuBackend::new(), &m, 0.001, 64)?;
/// let p = AnisotropicParams { mu_star: 0.0, conv_thr: 1e-8, max_iter: 5000, mixing: 0.5 };
/// let sol = solve_anisotropic_fbw(&mut sum, &m, &p, |_| 0.005)?;
/// assert!(sol.z[0] > 1.0);
/// # Ok::<(), eliashberg::Error>(())
/// ```
pub fn solve_anisotropic_fbw(
    sum: &mut dyn StateFrequencySum,
    model: &AnisotropicModel,
    params: &AnisotropicParams,
    initial_gap: impl Fn(f64) -> f64,
) -> Result<AnisotropicSolution, Error> {
    if !(params.mixing > 0.0 && params.mixing <= 1.0) {
        return Err(Error::InvalidInput(format!(
            "mixing must be in (0, 1], got {}",
            params.mixing
        )));
    }
    let t = sum.temperature();
    let omega: Vec<f64> = sum
        .indices()
        .iter()
        .map(|&n| fermionic_frequency(n, t))
        .collect();
    let (nf, ns) = (omega.len(), model.nstates());
    let len = nf * ns;
    let mut z_in = vec![1.0; len];
    let mut delta_in: Vec<f64> = (0..len).map(|i| initial_gap(omega[i % nf])).collect();
    let mut shift_in = vec![0.0; len];
    let mut previous = delta_in.clone();
    let mut backend = CpuBackend::new();
    let coulomb = TypedTensor::from_vec_col_major(vec![ns, ns], {
        // stored row-major (s, s'); column-major tensor index (s, s') = s + ns * s'
        let mut m = vec![0.0; ns * ns];
        for s in 0..ns {
            for sp in 0..ns {
                m[s + ns * sp] = model.coulomb[s * ns + sp];
            }
        }
        m
    })?;
    let tensor = |data: Vec<f64>| TypedTensor::from_vec_col_major(vec![nf, ns], data);

    for iteration in 1..=params.max_iter {
        let mut z_src = vec![0.0; len];
        let mut z0_src = vec![0.0; len];
        let mut d_src = vec![0.0; len];
        let mut s_src = vec![0.0; len];
        for (s, &energy) in model.energies.iter().enumerate() {
            for (m, &w) in omega.iter().enumerate() {
                let i = m + nf * s;
                let wz = w * z_in[i];
                let xi = energy + shift_in[i];
                let dz = delta_in[i] * z_in[i];
                let inv = 1.0 / (wz * wz + xi * xi + dz * dz);
                z_src[i] = wz * inv;
                d_src[i] = dz * inv;
                s_src[i] = xi * inv;
                z0_src[i] = wz / (wz * wz + xi * xi);
            }
        }
        let d_src = tensor(d_src)?;
        let z_conv = sum.convolve(&tensor(z_src)?, Parity::Odd)?;
        let z0_conv = sum.convolve(&tensor(z0_src)?, Parity::Odd)?;
        let d_conv = sum.convolve(&d_src, Parity::Even)?;
        let s_conv = sum.convolve(&tensor(s_src)?, Parity::Even)?;
        let d_sums = TypedTensor::from_vec_col_major(vec![ns], sum.sum_even(&d_src)?)?;
        let mu_term = backend.with_backend_session(|s| [&coulomb, &d_sums].einsum("sq,q->s", s))?;
        let mu_term = mu_term.as_slice()?;

        let (zc, z0c, dc, sc) = (
            z_conv.as_slice()?,
            z0_conv.as_slice()?,
            d_conv.as_slice()?,
            s_conv.as_slice()?,
        );
        let mut z = vec![0.0; len];
        let mut z_normal = vec![0.0; len];
        let mut delta = vec![0.0; len];
        let mut shift = vec![0.0; len];
        for (s, &coulomb_s) in mu_term.iter().enumerate() {
            for (n, &w) in omega.iter().enumerate() {
                let i = n + nf * s;
                z[i] = 1.0 + t * zc[i] / w;
                z_normal[i] = 1.0 + t * z0c[i] / w;
                delta[i] = t * (dc[i] - params.mu_star * coulomb_s) / z[i];
                shift[i] = -t * sc[i];
            }
        }
        let change: f64 = delta
            .iter()
            .zip(&previous)
            .map(|(a, b)| (a - b).abs())
            .sum();
        let size: f64 = delta.iter().map(|a| a.abs()).sum();
        let residual = change / size;
        if residual < params.conv_thr {
            return Ok(AnisotropicSolution {
                omega,
                z,
                delta,
                shift,
                z_normal,
                iterations: iteration,
                residual,
            });
        }
        for (input, output) in [
            (&mut z_in, &z),
            (&mut delta_in, &delta),
            (&mut shift_in, &shift),
        ] {
            for (i, o) in input.iter_mut().zip(output.iter()) {
                *i += params.mixing * (o - *i);
            }
        }
        previous = delta;
    }
    Err(Error::NotConverged {
        iterations: params.max_iter,
    })
}
