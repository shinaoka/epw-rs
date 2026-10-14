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
                            q: block.q,
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

/// Diagnostics of [`coulomb_states`], printed by EPW as well.
#[derive(Debug, Clone, PartialEq)]
pub struct CoulombSetup {
    /// The Coulomb states.
    pub states: eliashberg::CoulombStates,
    /// Number of nscf bands in the energy window.
    pub nbnd_window: usize,
    /// Band offset between the nscf bands (in the window) and the Wannier bands.
    pub band_offset: i64,
    /// Largest |e_Wannier - e_nscf| over matched Fermi-window states (eV).
    pub max_mismatch: f64,
}

fn grid_index(x: [f64; 3], mesh: [usize; 3]) -> [usize; 3] {
    std::array::from_fn(|a| {
        let n = mesh[a] as i64;
        ((x[a] * n as f64).round() as i64).rem_euclid(n) as usize
    })
}

/// States for EPW's Coulomb term (`icoulomb = 1`) from the bands of `nscf2supercond.x`.
///
/// Follows EPW 6.1 (`supercond_coul.f90`): nscf bands with an energy in
/// `(E_F + emin, E_F + emax)` are kept; nscf k points are matched to Fermi-window k
/// points by grid index and unfolded to the full grid with the symmetry operations;
/// the band offset minimises `sum |e_Wannier - e_nscf|` over Fermi-window states
/// (offsets -5..5 around the bands closest to E_F); every (band, full-grid k) is either
/// a Fermi-window state or, if inside the energy window, an outer state.
/// The weight is `mu* / (N_F N_k)`.
///
/// # Errors
///
/// [`crate::Error::Unsupported`] if no Wannier or nscf band lies near E_F.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/mgb2_epw_super/aniso");
/// let eps = phys_units::cmm1_to_ry(5.0) * phys_units::RY_TO_EV;
/// let data = eph_io_epw::ephmat::read_ephmat(&dir, "MgB2", eps).unwrap();
/// let problem = eph_driver::supercond::anisotropic_problem(&data);
/// let bands = eph_io_epw::supercond::read_coulomb_bands(&dir.join("MgB2.bands.6x6x6.dat")).unwrap();
/// let setup = eph_driver::supercond::coulomb_states(&data, &problem, &bands, -10.0, 10.0, 0.2)?;
/// assert_eq!(setup.nbnd_window, 8);
/// # Ok::<(), eph_driver::Error>(())
/// ```
pub fn coulomb_states(
    data: &EphmatData,
    problem: &EpwAnisotropicProblem,
    bands: &eph_io_epw::supercond::CoulombBands,
    emin: f64,
    emax: f64,
    mu_star: f64,
) -> Result<CoulombSetup, crate::Error> {
    let ef0 = data.ef0;
    let fsthick = data.fsthick;
    let nk_cl = bands.k_points.len();
    let in_window = |e: f64| e - ef0 > emin && e - ef0 < emax;

    // nscf bands kept: the range of bands with any energy in the window.
    let (mut bmin, mut bmax) = (usize::MAX, 0);
    for k in 0..nk_cl {
        for b in 0..bands.nbnd {
            if in_window(bands.energies[b + bands.nbnd * k]) {
                bmin = bmin.min(b);
                bmax = bmax.max(b);
            }
        }
    }
    if bmin > bmax {
        return Err(crate::Error::Unsupported(
            "no nscf band in the Coulomb energy window".into(),
        ));
    }
    let nbnd_cl = bmax - bmin + 1;
    let ek_cl = |b: usize, k: usize| bands.energies[bmin + b + bands.nbnd * k];

    // nscf k -> Fermi-window k by grid point.
    let fs_of_cl: Vec<Option<usize>> = bands
        .k_points
        .iter()
        .map(|x| {
            let g = grid_index(*x, data.k_mesh);
            data.k_points
                .iter()
                .position(|y| grid_index(*y, data.k_mesh) == g)
        })
        .collect();

    // Full nscf grid -> irreducible nscf k through the symmetry operations (last match wins).
    let mesh = bands.k_mesh;
    let nfull: usize = mesh.iter().product();
    let mut ibz_of_bz = vec![usize::MAX; nfull];
    for (ik, x) in bands.k_points.iter().enumerate() {
        for (rot, &trev) in bands.rotations.iter().zip(&bands.time_reversal) {
            let mut y: [f64; 3] =
                std::array::from_fn(|i| (0..3).map(|j| rot[i][j] as f64 * x[j]).sum());
            for v in y.iter_mut() {
                *v -= (*v + 1e-8).round();
            }
            if trev {
                y = y.map(|v| -v);
            }
            let c: [usize; 3] = std::array::from_fn(|a| {
                let n = mesh[a] as i64;
                ((y[a] * n as f64 + 2.0 * n as f64).round() as i64).rem_euclid(n) as usize
            });
            ibz_of_bz[c[2] + mesh[2] * (c[1] + mesh[1] * c[0])] = ik;
        }
    }
    if ibz_of_bz.contains(&usize::MAX) {
        return Err(crate::Error::Unsupported(
            "symmetry operations do not cover the nscf grid".into(),
        ));
    }

    // Wannier window bands per Fermi-window k (all-band indices) and their energies.
    let window = &data.window_bands;
    let ekfs = |k: usize, all: usize| data.energies[all + data.nbnd * k];

    // Bands closest to E_F in both sets (strict improvements, scan order as EPW).
    let (mut min_cl, mut best_cl) = (1e5, None);
    let (mut min_fs, mut best_fs) = (1e5, None);
    for (k_cl, fs) in fs_of_cl.iter().enumerate() {
        let Some(k) = *fs else { continue };
        for b in 0..nbnd_cl {
            let d = (ek_cl(b, k_cl) - ef0).abs();
            if d < fsthick && d < min_cl {
                min_cl = d;
                best_cl = Some(b);
            }
        }
        for &all in &window[k] {
            let d = (ekfs(k, all) - ef0).abs();
            if d < fsthick && d < min_fs {
                min_fs = d;
                best_fs = Some(all);
            }
        }
    }
    let (Some(best_cl), Some(best_fs)) = (best_cl, best_fs) else {
        return Err(crate::Error::Unsupported(
            "no band within fsthick of E_F".into(),
        ));
    };
    let mismatch = |offset: i64| -> Option<(f64, f64)> {
        let (mut sum, mut max) = (0.0_f64, 0.0_f64);
        for (k_cl, fs) in fs_of_cl.iter().enumerate() {
            let Some(k) = *fs else { continue };
            for &all in &window[k] {
                let b_cl = all as i64 + offset;
                if b_cl < 0 || b_cl >= nbnd_cl as i64 {
                    return None;
                }
                if (ekfs(k, all) - ef0).abs() < fsthick {
                    let d = (ekfs(k, all) - ek_cl(b_cl as usize, k_cl)).abs();
                    sum += d;
                    max = max.max(d);
                }
            }
        }
        Some((sum, max))
    };
    let mut best: Option<(f64, i64, f64)> = None;
    for ipol in -5i64..=5 {
        let offset = ipol + best_cl as i64 - best_fs as i64;
        if let Some((sum, max)) = mismatch(offset) {
            if best.is_none_or(|(s, _, _)| sum < s) {
                best = Some((sum, offset, max));
            }
        }
    }
    let Some((_, band_offset, max_mismatch)) = best else {
        return Err(crate::Error::Unsupported(
            "no consistent band offset".into(),
        ));
    };

    // Classify every (band, full-grid k).
    let state_index: std::collections::HashMap<(usize, usize), usize> = problem
        .states
        .iter()
        .enumerate()
        .map(|(s, &kb)| (kb, s))
        .collect();
    let mut inside_counts = vec![0.0; problem.states.len()];
    let mut outer_counts: std::collections::BTreeMap<(usize, usize), f64> = Default::default();
    for &k_cl in &ibz_of_bz {
        for b_cl in 0..nbnd_cl {
            let all = b_cl as i64 - band_offset;
            let inside = fs_of_cl[k_cl].and_then(|k| {
                if all < 0 || all >= data.nbnd as i64 {
                    return None;
                }
                let all = all as usize;
                window[k].contains(&all).then_some(())?;
                ((ekfs(k, all) - ef0).abs() < fsthick).then(|| state_index[&(k, all)])
            });
            match inside {
                Some(s) => inside_counts[s] += 1.0,
                None => {
                    if in_window(ek_cl(b_cl, k_cl)) {
                        *outer_counts.entry((k_cl, b_cl)).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
    }
    let outer = outer_counts
        .into_iter()
        .map(|((k, b), n)| (ek_cl(b, k) - ef0, n))
        .collect();
    let weight = if (mu_star / data.dos_at_fermi).abs() > 1e-8 / data.dos_at_fermi {
        mu_star / data.dos_at_fermi / nfull as f64
    } else {
        0.0
    };
    Ok(CoulombSetup {
        states: eliashberg::CoulombStates {
            weight,
            inside_counts,
            outer,
        },
        nbnd_window: nbnd_cl,
        band_offset,
        max_mismatch,
    })
}
