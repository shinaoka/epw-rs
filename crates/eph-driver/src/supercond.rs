//! Build the anisotropic Eliashberg problem from EPW's Fermi-surface e-ph data.

use eliashberg::{AnisotropicModel, KernelEntry};
use eph_io_epw::ephmat::EphmatData;

/// Anisotropic model plus the bookkeeping needed to compare with EPW output.
#[derive(Debug, Clone)]
pub struct EpwAnisotropicProblem {
    /// The model passed to the solver.
    pub model: AnisotropicModel,
    /// `(k, band)` of each state, in EPW's output order (k outer, band inner).
    pub states: Vec<(usize, usize)>,
    /// EPW's `0.5 wkfs w0g` per state (free-energy weights).
    pub free_energy_weights: Vec<f64>,
}

/// Turn EPW data into an [`AnisotropicModel`] with the chemical potential at `ef0`.
///
/// - Kernel entries: `w_q |g|^2/(2 omega)` for every (k, q, mode, i, j) with `omega > 0`
///   (acoustic modes are already zeroed by the reader).
/// - Coulomb weights: `M_{ss'} = sum_{q: k+q = k'} w_q / N_F`.
/// - Fermi-surface weights: `0.5 wkfs w0g / N_F` with a Gaussian `w0g` of width `degaussw`.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/mgb2_epw_super/aniso");
/// let eps = phys_units::cmm1_to_ry(5.0) * phys_units::RY_TO_EV;
/// let data = eph_io_epw::ephmat::read_ephmat(&dir, "MgB2", eps).unwrap();
/// let problem = eph_driver::supercond::anisotropic_problem(&data);
/// assert_eq!(problem.model.nstates(), 140);
/// ```
pub fn anisotropic_problem(data: &EphmatData) -> EpwAnisotropicProblem {
    let mut states = Vec::new();
    let mut state_of = vec![Vec::new(); data.k_points.len()];
    for (k, bands) in data.window_bands.iter().enumerate() {
        for &b in bands {
            state_of[k].push(states.len());
            states.push((k, b));
        }
    }
    let ns = states.len();
    let nq = data.q_points.len();
    let wq = 1.0 / nq as f64;
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let energies: Vec<f64> = states
        .iter()
        .map(|&(k, b)| data.energies[b + data.nbnd * k] - data.ef0)
        .collect();
    let free_energy_weights: Vec<f64> = states
        .iter()
        .zip(&energies)
        .map(|(&(k, _), e)| {
            let x = e / data.degaussw;
            0.5 * data.k_weights[k] * inv_sqrt_pi * (-(x * x).min(200.0)).exp() / data.degaussw
        })
        .collect();
    let fs_weights = free_energy_weights
        .iter()
        .map(|w| w / data.dos_at_fermi)
        .collect();

    let mut kernel_entries = Vec::new();
    let mut coulomb = vec![0.0; ns * ns];
    for block in &data.blocks {
        let (rows, cols) = (&state_of[block.k], &state_of[block.kq]);
        for &s in rows {
            for &sp in cols {
                coulomb[s * ns + sp] += wq / data.dos_at_fermi;
            }
        }
        let (nb_k, nb_kq) = (rows.len(), cols.len());
        for mode in 0..data.nmodes {
            let omega = data.frequencies[mode + data.nmodes * block.q];
            for (i, &s) in rows.iter().enumerate() {
                for (j, &sp) in cols.iter().enumerate() {
                    let g2 = block.g2[(mode * nb_k + i) * nb_kq + j];
                    if g2 != 0.0 {
                        kernel_entries.push(KernelEntry {
                            s,
                            s_prime: sp,
                            omega,
                            weight: wq * g2,
                        });
                    }
                }
            }
        }
    }
    EpwAnisotropicProblem {
        model: AnisotropicModel {
            energies,
            fs_weights,
            kernel_entries,
            coulomb,
        },
        states,
        free_energy_weights,
    }
}
