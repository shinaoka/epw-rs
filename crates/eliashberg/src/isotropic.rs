//! Isotropic Migdal–Eliashberg equations on the imaginary axis.
//!
//! Equations (energies in eV, `T` = k_B T):
//!
//! - Fermi-surface-restricted (FSR), F. Marsiglio, M. Schossmann and
//!   J. P. Carbotte, PRB 37, 4965 (1988); E. R. Margine and F. Giustino,
//!   PRB 87, 024505 (2013), Eqs. (34)–(35):
//!   `Z_n = 1 + (pi T / omega_n) sum_m lambda(n-m) omega_m / sqrt(omega_m^2 + Delta_m^2)`,
//!   `Z_n Delta_n = pi T sum_m [lambda(n-m) - mu*] Delta_m / sqrt(omega_m^2 + Delta_m^2)`.
//! - Full bandwidth (FBW) with the electronic DOS N(e) (per spin),
//!   W. E. Pickett, PRB 26, 1186 (1982):
//!   `Z_n = 1 + (T / omega_n) sum_m lambda(n-m) <omega_m Z_m / Theta_m>`,
//!   `Z_n Delta_n = T sum_m [lambda(n-m) - mu*] <Delta_m Z_m / Theta_m>`,
//!   `chi_n = -T sum_m lambda(n-m) <(e - mu + chi_m) / Theta_m>`,
//!   with `Theta_m(e) = Z_m^2 (omega_m^2 + Delta_m^2) + (e - mu + chi_m)^2` and
//!   `<f> = int de N(e) f(e) / N(E_F)`.
//!
//! The Matsubara sums are delegated to a [`FrequencySum`]; with
//! [`crate::UniformSum`] they run over `|omega_m| < omega_c` exactly as in EPW
//! (`gridsamp = 0`). Self-consistency uses linear mixing of the input fields;
//! convergence is `sum_n |Delta_n - Delta_n^old| / sum_n |Delta_n| < threshold`.

use crate::matsubara::{fermionic_frequency, FrequencySum, Parity};
use crate::Error;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_tensor::{BackendSessionHost, TypedTensor};

/// Solver settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsotropicParams {
    /// Coulomb pseudopotential mu*.
    pub mu_star: f64,
    /// Convergence threshold on the relative change of Delta.
    pub conv_thr: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Linear mixing weight of the new output (0 < mixing <= 1).
    pub mixing: f64,
    /// Initial gap (eV), constant in frequency.
    pub initial_gap: f64,
}

/// Electronic DOS for the full-bandwidth equations (per spin).
#[derive(Debug, Clone, PartialEq)]
pub struct ElectronDos {
    /// Bin energies (eV).
    pub energies: Vec<f64>,
    /// DOS per spin (1/eV).
    pub dos: Vec<f64>,
    /// Bin width (eV).
    pub step: f64,
    /// Chemical potential (eV).
    pub chemical_potential: f64,
    /// DOS per spin at the chemical potential (1/eV), used to normalise lambda.
    pub dos_at_fermi: f64,
}

/// Converged solution on the positive Matsubara frequencies of the [`FrequencySum`].
#[derive(Debug, Clone, PartialEq)]
pub struct IsotropicSolution {
    /// Matsubara frequencies `omega_n` (eV).
    pub omega: Vec<f64>,
    /// Renormalisation Z(i omega_n).
    pub z: Vec<f64>,
    /// Gap Delta(i omega_n) (eV).
    pub delta: Vec<f64>,
    /// Energy shift chi(i omega_n) (eV); zero for FSR.
    pub shift: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final relative change of Delta.
    pub residual: f64,
}

fn tensor(rows: usize, data: Vec<f64>) -> Result<TypedTensor<f64>, Error> {
    let nfreq = data.len() / rows;
    Ok(TypedTensor::from_vec_col_major(vec![nfreq, rows], data)?)
}

fn check_params(params: &IsotropicParams) -> Result<(), Error> {
    if !(params.mixing > 0.0 && params.mixing <= 1.0) {
        return Err(Error::InvalidInput(format!(
            "mixing must be in (0, 1], got {}",
            params.mixing
        )));
    }
    Ok(())
}

fn relative_change(new: &[f64], old: &[f64]) -> f64 {
    let change: f64 = new.iter().zip(old).map(|(a, b)| (a - b).abs()).sum();
    let size: f64 = new.iter().map(|a| a.abs()).sum();
    change / size
}

/// Solve the FSR equations.
///
/// # Errors
///
/// [`Error::InvalidInput`] for bad parameters, [`Error::NotConverged`] if the
/// threshold is not reached, and errors of the frequency sum.
///
/// # Examples
///
/// ```
/// use eliashberg::*;
/// // Einstein spectrum, lambda = 1, at T well below T_c.
/// let spectrum = EliashbergSpectrum::new(vec![0.02], vec![0.5])?;
/// let t = 0.0005;
/// let nfreq = frequencies_below(0.4, t);
/// let mut sum = UniformSum::new(tenferro_cpu::CpuBackend::new(), t, nfreq, |nu| {
///     spectrum.lambda_matsubara(&mut tenferro_cpu::CpuBackend::new(), nu)
/// })?;
/// let params = IsotropicParams { mu_star: 0.1, conv_thr: 1e-10, max_iter: 2000, mixing: 0.5, initial_gap: 0.002 };
/// let sol = solve_isotropic_fsr(&mut sum, &params)?;
/// assert!(sol.delta[0] > 0.0 && sol.z[0] > 1.0);
/// # Ok::<(), eliashberg::Error>(())
/// ```
pub fn solve_isotropic_fsr(
    sum: &mut dyn FrequencySum,
    params: &IsotropicParams,
) -> Result<IsotropicSolution, Error> {
    check_params(params)?;
    let t = sum.temperature();
    let omega: Vec<f64> = sum
        .indices()
        .iter()
        .map(|&n| fermionic_frequency(n, t))
        .collect();
    let nf = omega.len();
    let mut delta_in = vec![params.initial_gap; nf];
    let mut previous = delta_in.clone();
    let pi_t = std::f64::consts::PI * t;
    for iteration in 1..=params.max_iter {
        let root: Vec<f64> = omega
            .iter()
            .zip(&delta_in)
            .map(|(w, d)| (w * w + d * d).sqrt())
            .collect();
        let z_source: Vec<f64> = omega.iter().zip(&root).map(|(w, r)| w / r).collect();
        let d_source = tensor(1, delta_in.iter().zip(&root).map(|(d, r)| d / r).collect())?;
        let z_conv = sum.convolve(&tensor(1, z_source)?, Parity::Odd)?;
        let d_conv = sum.convolve(&d_source, Parity::Even)?;
        let coulomb = params.mu_star * sum.sum_even(&d_source)?[0];
        let z: Vec<f64> = omega
            .iter()
            .zip(z_conv.as_slice()?)
            .map(|(w, c)| 1.0 + pi_t * c / w)
            .collect();
        let delta: Vec<f64> = d_conv
            .as_slice()?
            .iter()
            .zip(&z)
            .map(|(c, z)| pi_t * (c - coulomb) / z)
            .collect();
        let residual = relative_change(&delta, &previous);
        if residual < params.conv_thr {
            return Ok(IsotropicSolution {
                omega,
                z,
                shift: vec![0.0; nf],
                delta,
                iterations: iteration,
                residual,
            });
        }
        for (d_in, d_out) in delta_in.iter_mut().zip(&delta) {
            *d_in += params.mixing * (d_out - *d_in);
        }
        previous = delta;
    }
    Err(Error::NotConverged {
        iterations: params.max_iter,
    })
}

/// Solve the FBW equations with the electronic DOS; `backend` runs the energy integrals.
///
/// # Errors
///
/// As [`solve_isotropic_fsr`], plus tensor errors.
///
/// # Examples
///
/// ```
/// use eliashberg::*;
/// // Flat band of half-width 1 eV, Einstein spectrum with lambda = 1.
/// let spectrum = EliashbergSpectrum::new(vec![0.02], vec![0.5])?;
/// let n = 2001;
/// let energies: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect();
/// let dos = ElectronDos { dos: vec![1.0; n], step: 0.001, energies, chemical_potential: 0.0, dos_at_fermi: 1.0 };
/// let t = 0.0005;
/// let mut sum = UniformSum::new(tenferro_cpu::CpuBackend::new(), t, frequencies_below(0.4, t), |nu| {
///     spectrum.lambda_matsubara(&mut tenferro_cpu::CpuBackend::new(), nu)
/// })?;
/// let params = IsotropicParams { mu_star: 0.1, conv_thr: 1e-8, max_iter: 2000, mixing: 0.5, initial_gap: 0.002 };
/// let sol = solve_isotropic_fbw(&mut tenferro_cpu::CpuBackend::new(), &mut sum, &dos, &params)?;
/// assert!(sol.delta[0] > 0.0);
/// assert!(sol.shift[0].abs() < 1e-6); // particle-hole symmetric band
/// # Ok::<(), eliashberg::Error>(())
/// ```
pub fn solve_isotropic_fbw(
    backend: &mut CpuBackend,
    sum: &mut dyn FrequencySum,
    dos: &ElectronDos,
    params: &IsotropicParams,
) -> Result<IsotropicSolution, Error> {
    check_params(params)?;
    let t = sum.temperature();
    let omega: Vec<f64> = sum
        .indices()
        .iter()
        .map(|&n| fermionic_frequency(n, t))
        .collect();
    let nf = omega.len();
    let ne = dos.energies.len();
    // Energy-integration weights N(e) de / N(E_F) and energies relative to mu.
    let weight = TypedTensor::from_vec_col_major(
        vec![ne],
        dos.dos
            .iter()
            .map(|n| n * dos.step / dos.dos_at_fermi)
            .collect(),
    )?;
    let xi: Vec<f64> = dos
        .energies
        .iter()
        .map(|e| e - dos.chemical_potential)
        .collect();
    let xi_tensor = TypedTensor::from_vec_col_major(vec![ne], xi.clone())?;

    let mut z_in = vec![1.0; nf];
    let mut delta_in = vec![params.initial_gap; nf];
    let mut shift_in = vec![0.0; nf];
    let mut previous = delta_in.clone();
    for iteration in 1..=params.max_iter {
        // inv_theta[e, m] = 1 / Theta_m(e)
        let mut inv_theta = Vec::with_capacity(ne * nf);
        for m in 0..nf {
            let gap_part = z_in[m] * z_in[m] * (omega[m] * omega[m] + delta_in[m] * delta_in[m]);
            inv_theta.extend(xi.iter().map(|x| {
                let shifted = x + shift_in[m];
                1.0 / (gap_part + shifted * shifted)
            }));
        }
        let inv_theta = TypedTensor::from_vec_col_major(vec![ne, nf], inv_theta)?;
        let (moment0, moment1) = backend.with_backend_session(|s| -> Result<_, Error> {
            let m0 = [&weight, &inv_theta].einsum("e,em->m", s)?;
            let weighted_xi = [&weight, &xi_tensor].einsum("e,e->e", s)?;
            let m1 = [&weighted_xi, &inv_theta].einsum("e,em->m", s)?;
            Ok((m0, m1))
        })?;
        let (m0, m1) = (moment0.as_slice()?, moment1.as_slice()?);
        let z_source: Vec<f64> = (0..nf).map(|m| omega[m] * z_in[m] * m0[m]).collect();
        let d_source = tensor(1, (0..nf).map(|m| delta_in[m] * z_in[m] * m0[m]).collect())?;
        let s_source: Vec<f64> = (0..nf).map(|m| m1[m] + shift_in[m] * m0[m]).collect();

        let z_conv = sum.convolve(&tensor(1, z_source)?, Parity::Odd)?;
        let d_conv = sum.convolve(&d_source, Parity::Even)?;
        let s_conv = sum.convolve(&tensor(1, s_source)?, Parity::Even)?;
        let coulomb = params.mu_star * sum.sum_even(&d_source)?[0];

        let z: Vec<f64> = omega
            .iter()
            .zip(z_conv.as_slice()?)
            .map(|(w, c)| 1.0 + t * c / w)
            .collect();
        let delta: Vec<f64> = d_conv
            .as_slice()?
            .iter()
            .zip(&z)
            .map(|(c, z)| t * (c - coulomb) / z)
            .collect();
        let shift: Vec<f64> = s_conv.as_slice()?.iter().map(|c| -t * c).collect();

        let residual = relative_change(&delta, &previous);
        if residual < params.conv_thr {
            return Ok(IsotropicSolution {
                omega,
                z,
                delta,
                shift,
                iterations: iteration,
                residual,
            });
        }
        let mix = |input: &mut Vec<f64>, output: &[f64]| {
            for (i, o) in input.iter_mut().zip(output) {
                *i += params.mixing * (o - *i);
            }
        };
        mix(&mut z_in, &z);
        mix(&mut delta_in, &delta);
        mix(&mut shift_in, &shift);
        previous = delta;
    }
    Err(Error::NotConverged {
        iterations: params.max_iter,
    })
}

/// Superconducting minus normal free energy in the FSR approximation (eV),
/// `-2 pi T sum_{n>=0} (Omega_n - omega_n)(Z_n - Z^N_n omega_n / Omega_n)`, `Omega_n = sqrt(omega_n^2 + Delta_n^2)`,
/// summed over the frequencies of the solution (Bardeen–Stephen form as in EPW).
///
/// `z_normal` is the normal-state renormalisation on the same frequencies.
///
/// # Examples
///
/// ```
/// let sol = eliashberg::IsotropicSolution { omega: vec![1.0], z: vec![1.0], delta: vec![0.0],
///     shift: vec![0.0], iterations: 1, residual: 0.0 };
/// assert_eq!(eliashberg::free_energy_difference(&sol, &[1.0], 0.01), 0.0);
/// ```
pub fn free_energy_difference(
    solution: &IsotropicSolution,
    z_normal: &[f64],
    temperature: f64,
) -> f64 {
    let sum: f64 = (0..solution.omega.len())
        .map(|n| {
            let w = solution.omega[n];
            let big = (w * w + solution.delta[n] * solution.delta[n]).sqrt();
            (big - w) * (solution.z[n] - z_normal[n] * w / big)
        })
        .sum();
    -2.0 * std::f64::consts::PI * temperature * sum
}

/// Normal-state renormalisation `Z^N_n = 1 + (pi T / omega_n) sum_m lambda(n-m) sign(omega_m)`.
///
/// # Errors
///
/// Errors of the frequency sum.
///
/// # Examples
///
/// ```
/// use eliashberg::*;
/// let mut sum = UniformSum::new(tenferro_cpu::CpuBackend::new(), 0.001, 8, |nu| Ok(vec![0.0; nu.len()]))?;
/// assert_eq!(normal_state_z(&mut sum)?, vec![1.0; 8]);
/// # Ok::<(), eliashberg::Error>(())
/// ```
pub fn normal_state_z(sum: &mut dyn FrequencySum) -> Result<Vec<f64>, Error> {
    let t = sum.temperature();
    let omega: Vec<f64> = sum
        .indices()
        .iter()
        .map(|&n| fermionic_frequency(n, t))
        .collect();
    let conv = sum.convolve(&tensor(1, vec![1.0; omega.len()])?, Parity::Odd)?;
    Ok(omega
        .iter()
        .zip(conv.as_slice()?)
        .map(|(w, c)| 1.0 + std::f64::consts::PI * t * c / w)
        .collect())
}
