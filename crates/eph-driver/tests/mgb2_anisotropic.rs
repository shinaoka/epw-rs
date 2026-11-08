//! L2 test: anisotropic FBW Eliashberg equations for MgB2 against EPW (QE 7.6).
//!
//! EPW solved `epw6.in` (uniform grid) with `conv_thr_iaxis = 1e-10` at 20 and
//! 30 K (`tests/fixtures/mgb2_epw_super/aniso`). The Rust solver reads the same
//! `MgB2.ephmat` data.

use eliashberg::*;
use eph_driver::supercond::{anisotropic_problem, EpwAnisotropicProblem};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tenferro_cpu::{CpuEngine, Pool};

const MU_STAR: f64 = 0.16;
const WSCUT: f64 = 0.5;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mgb2_epw_super/aniso")
}

fn problem() -> &'static EpwAnisotropicProblem {
    static PROBLEM: OnceLock<EpwAnisotropicProblem> = OnceLock::new();
    PROBLEM.get_or_init(|| {
        let eps = phys_units::cmm1_to_ry(5.0) * phys_units::RY_TO_EV;
        let data = eph_io_epw::ephmat::read_ephmat(&dir(), "MgB2", eps).unwrap();
        anisotropic_problem(&data)
    })
}

/// EPW's temperature in eV: K -> Ry -> eV, as in readin.f90 and supercond.f90.
fn kelvin(t: f64) -> f64 {
    phys_units::kelvin_to_ry(t) * phys_units::RY_TO_EV
}

/// Rows of `MgB2.imag_aniso_<T>`: omega, e - ef, Z, Delta, chi (omega outer, then states).
fn reference(kind: &str, label: &str) -> Vec<[f64; 5]> {
    std::fs::read_to_string(dir().join(kind).join(format!("MgB2.imag_aniso_{label}")))
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4]]
        })
        .collect()
}

fn printed_free_energy(kind: &str, itemp: usize) -> f64 {
    let text = std::fs::read_to_string(dir().join(kind).join(format!("aniso_{kind}.out"))).unwrap();
    let line = text
        .lines()
        .filter(|l| l.contains("Free energy"))
        .nth(itemp)
        .unwrap();
    line.split_whitespace()
        .rev()
        .nth(1)
        .unwrap()
        .parse()
        .unwrap()
}

/// Max relative difference, measured against the largest |value| of each quantity.
fn rel(actual: impl Iterator<Item = f64>, expected: impl Iterator<Item = f64>) -> f64 {
    let pairs: Vec<(f64, f64)> = actual.zip(expected).collect();
    let scale = pairs.iter().fold(0.0_f64, |m, p| m.max(p.1.abs()));
    pairs.iter().fold(0.0_f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

const PARAMS: AnisotropicParams = AnisotropicParams {
    mu_star: MU_STAR,
    conv_thr: 1e-12,
    max_iter: 20_000,
    mixing: 0.3,
};

#[test]
fn uniform_grid_matches_epw() {
    let p = problem();
    for (itemp, (t, label)) in [(20.0, "020.00"), (30.0, "030.00")].into_iter().enumerate() {
        let tk = kelvin(t);
        let mut sum = UniformStateSum::new(
            CpuEngine::new(Pool::builder().build().unwrap()),
            &p.model,
            tk,
            frequencies_below(WSCUT, tk),
        )
        .unwrap();
        let sol = solve_anisotropic_fbw(&mut sum, &p.model, &PARAMS, |_| 0.004).unwrap();
        let rows = reference("uni", label);
        let (nf, ns) = (sol.omega.len(), p.model.nstates());
        assert_eq!(rows.len(), nf * ns);
        // EPW rows: omega outer, states inner; ours: (nfreq, nstates) column-major.
        // EPW prints 11 significant digits (ES20.10).
        let ours = |a: &[f64]| {
            (0..nf * ns)
                .map(|r| a[r / ns + nf * (r % ns)])
                .collect::<Vec<_>>()
        };
        let ez = rel(ours(&sol.z).into_iter(), rows.iter().map(|r| r[2]));
        let ed = rel(ours(&sol.delta).into_iter(), rows.iter().map(|r| r[3]));
        let es = rel(ours(&sol.shift).into_iter(), rows.iter().map(|r| r[4]));
        let ee = rel(
            (0..nf * ns).map(|r| p.model.energies[r % ns]),
            rows.iter().map(|r| r[1]),
        );
        println!(
            "{t} K: {nf} freq x {ns} states, {} iterations; max diff / max |value|: Z {ez:.1e}, Delta {ed:.1e}, chi {es:.1e}, e {ee:.1e}",
            sol.iterations
        );
        assert!(ee < 1e-10 && ez < 1e-8 && ed < 1e-8 && es < 1e-8);
        let fe = sol.free_energy(&p.free_energy_weights, tk) * 1000.0;
        let expected = printed_free_energy("uni", itemp);
        println!("{t} K: free energy {fe:.8} meV (EPW {expected})");
        assert!((fe - expected).abs() < 6e-7);
    }
}

/// IR basis pinned to the sampling points of EPW's IR object file, against EPW `epw7` (tight).
#[cfg(feature = "sparse-ir")]
#[test]
fn ir_with_epw_points_matches_epw() {
    ir_against("ir", Some(1e-5));
}

/// The same without EPW's coefficient cut (`eps_cut_ir = 0` on both sides).
#[cfg(feature = "sparse-ir")]
#[test]
fn ir_without_cut_matches_epw() {
    ir_against("ir_nocut", None);
}

#[cfg(feature = "sparse-ir")]
fn ir_against(kind: &str, cut: Option<f64>) {
    let p = problem();
    let ir = eph_io_epw::supercond::read_ir_object(&dir().join("ir_nlambda5_ndigit6.dat")).unwrap();
    let points = IrSettings {
        tau: Some(ir.x.iter().map(|x| (x + 1.0) / 2.0).collect()),
        fermionic: Some(
            ir.fermionic
                .iter()
                .filter(|&&f| f > 0)
                .map(|f| (f - 1) / 2)
                .collect(),
        ),
        bosonic: Some(
            ir.bosonic
                .iter()
                .filter(|&&b| b >= 0)
                .map(|b| b / 2)
                .collect(),
        ),
        epw_drop_last_coefficient: true,
        epw_coefficient_cut: cut,
    };
    let set = IrBasisSet::new(ir.lambda, ir.epsilon).unwrap();
    for (t, label) in [(20.0, "020.00"), (30.0, "030.00")] {
        let tk = kelvin(t);
        let mut sum = IrStateSum::new(
            CpuEngine::new(Pool::builder().build().unwrap()),
            &set,
            &p.model,
            tk,
            &points,
        )
        .unwrap();
        let sol = solve_anisotropic_fbw(&mut sum, &p.model, &PARAMS, |_| 0.004).unwrap();
        let rows = reference(kind, label);
        let (nf, ns) = (sol.omega.len(), p.model.nstates());
        assert_eq!(rows.len(), nf * ns);
        let ours = |a: &[f64]| {
            (0..nf * ns)
                .map(|r| a[r / ns + nf * (r % ns)])
                .collect::<Vec<_>>()
        };
        let ew = rel(
            ours(&sol.omega.repeat(ns)).into_iter(),
            rows.iter().map(|r| r[0]),
        );
        let ez = rel(ours(&sol.z).into_iter(), rows.iter().map(|r| r[2]));
        let ed = rel(ours(&sol.delta).into_iter(), rows.iter().map(|r| r[3]));
        let es = rel(ours(&sol.shift).into_iter(), rows.iter().map(|r| r[4]));
        println!(
            "IR ({kind}) {t} K: {nf} freq x {ns} states, {} iterations; max diff / max |value|: omega {ew:.1e}, Z {ez:.1e}, Delta {ed:.1e}, chi {es:.1e}",
            sol.iterations
        );
        assert!(ew < 1e-10); // printed with 11 digits
                             // With EPW's points, its odd-size handling and its coefficient cut reproduced,
                             // the remaining difference is the basis itself (Python sparse-ir file versus
                             // sparse-ir-rs), observed at ~1e-8 or below.
        assert!(ez < 5e-8 && ed < 5e-8 && es < 5e-8);
    }
}

/// IR with EPW's Coulomb term (`icoulomb = 1`, `muc = 0.20`, states within ±10 eV), against
/// EPW `epw8` rerun with a tight threshold.
#[cfg(feature = "sparse-ir")]
#[test]
fn ir_with_coulomb_matches_epw() {
    let p = problem();
    let eps = phys_units::cmm1_to_ry(5.0) * phys_units::RY_TO_EV;
    let data = eph_io_epw::ephmat::read_ephmat(&dir(), "MgB2", eps).unwrap();
    let bands =
        eph_io_epw::supercond::read_coulomb_bands(&dir().join("MgB2.bands.6x6x6.dat")).unwrap();
    let setup = eph_driver::supercond::coulomb_states(&data, p, &bands, -10.0, 10.0, 0.20).unwrap();
    assert_eq!((setup.nbnd_window, setup.band_offset), (8, 0));
    let ir = eph_io_epw::supercond::read_ir_object(&dir().join("ir_nlambda5_ndigit6.dat")).unwrap();
    let settings = IrSettings {
        tau: Some(ir.x.iter().map(|x| (x + 1.0) / 2.0).collect()),
        fermionic: Some(
            ir.fermionic
                .iter()
                .filter(|&&f| f > 0)
                .map(|f| (f - 1) / 2)
                .collect(),
        ),
        bosonic: Some(
            ir.bosonic
                .iter()
                .filter(|&&b| b >= 0)
                .map(|b| b / 2)
                .collect(),
        ),
        epw_drop_last_coefficient: true,
        epw_coefficient_cut: Some(1e-5),
    };
    let set = IrBasisSet::new(ir.lambda, ir.epsilon).unwrap();
    for (itemp, (t, label)) in [(20.0, "020.00"), (30.0, "030.00")].into_iter().enumerate() {
        let tk = kelvin(t);
        let mut sum = IrStateSum::new(
            CpuEngine::new(Pool::builder().build().unwrap()),
            &set,
            &p.model,
            tk,
            &settings,
        )
        .unwrap();
        let sol = solve_anisotropic_fbw_coulomb(
            &mut sum,
            &p.model,
            &PARAMS,
            |_| 0.004,
            Some(&setup.states),
        )
        .unwrap();
        let rows = reference("coul", label);
        let (nf, ns) = (sol.omega.len(), p.model.nstates());
        assert_eq!(rows.len(), nf * ns);
        let ours = |a: &[f64]| {
            (0..nf * ns)
                .map(|r| a[r / ns + nf * (r % ns)])
                .collect::<Vec<_>>()
        };
        let ez = rel(ours(&sol.z).into_iter(), rows.iter().map(|r| r[2]));
        let ed = rel(ours(&sol.delta).into_iter(), rows.iter().map(|r| r[3]));
        let es = rel(ours(&sol.shift).into_iter(), rows.iter().map(|r| r[4]));
        let fe = sol.free_energy(&p.free_energy_weights, tk) * 1000.0;
        let expected = printed_free_energy("coul", itemp);
        println!(
            "Coulomb {t} K: {} iterations, outer gap {:.6} meV; max diff / max |value|: Z {ez:.1e}, Delta {ed:.1e}, chi {es:.1e}; free energy {fe:.8} meV (EPW {expected})",
            sol.iterations,
            sol.outer_gap[0] * 1e3
        );
        assert!(ez < 5e-8 && ed < 5e-8 && es < 5e-8);
        assert!((fe - expected).abs() < 6e-7);
    }
}

/// Real-axis grid of EPW's Padé continuation: `omega_i = i * d omega`, `i = 1..nqstep * round(wscut / omega_max)`,
/// with `d omega = omega_max / nqstep` and `omega_max` the last frequency of `MgB2.a2f`.
fn pade_grid() -> Vec<f64> {
    let a2f = eph_io_epw::supercond::read_a2f(&dir().join("MgB2.a2f")).unwrap();
    let omega_max = *a2f.omega.last().unwrap();
    let nqstep = 500;
    let n = nqstep * (WSCUT / omega_max).round() as usize;
    (1..=n)
        .map(|i| i as f64 * (omega_max / nqstep as f64))
        .collect()
}

/// Rows `(omega, e, Re Z, Im Z, Re Delta, Im Delta, Re chi, Im chi)` of the every-100th extract.
fn pade_reference(kind: &str, label: &str) -> Vec<[f64; 8]> {
    let path = dir()
        .join(kind)
        .join(format!("MgB2.pade_aniso_{label}.every100"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            std::array::from_fn(|i| v[i])
        })
        .collect()
}

/// Min/max Padé gap edges printed after the continuation (second "Min. / Max." line per temperature).
fn printed_pade_gaps(kind: &str, itemp: usize) -> (f64, f64) {
    let text = std::fs::read_to_string(dir().join(kind).join(format!("aniso_{kind}.out"))).unwrap();
    let line = text
        .lines()
        .filter(|l| l.contains("Min. / Max. values of superconducting gap"))
        .nth(2 * itemp + 1)
        .unwrap();
    let v: Vec<f64> = line
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    (v[0], v[1])
}

fn check_pade(kind: &str, itemp: usize, label: &str, sol: &AnisotropicSolution, ns: usize) {
    let grid = pade_grid();
    let npoints = pade_point_count(sol.omega.len(), 90);
    let real = continue_to_real_axis(sol, npoints, &grid).unwrap();
    let nw = grid.len();
    let rows = pade_reference(kind, label);
    assert_eq!(rows.len(), ns * nw / 100);
    let picked = |field: &[num_complex::Complex64]| -> Vec<num_complex::Complex64> {
        (0..ns)
            .flat_map(|s| (1..=nw / 100).map(move |j| (s, j * 100 - 1)))
            .map(|(s, i)| field[i + nw * s])
            .collect()
    };
    let max_rel = |ours: Vec<num_complex::Complex64>, re: usize| {
        let scale = rows.iter().fold(0.0_f64, |m, r| m.max(r[re].abs()));
        ours.iter().zip(&rows).fold(0.0_f64, |m, (o, r)| {
            m.max((o - num_complex::Complex64::new(r[re], r[re + 1])).norm())
        }) / scale
    };
    let (ez, ed, es) = (
        max_rel(picked(&real.z), 2),
        max_rel(picked(&real.delta), 4),
        max_rel(picked(&real.shift), 6),
    );
    let edges: Vec<f64> = (0..ns)
        .map(|s| gap_edge(&grid, &real.delta[nw * s..nw * (s + 1)], 500) * 1000.0)
        .collect();
    let (min, max) = edges
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &e| (a.min(e), b.max(e)));
    let (emin, emax) = printed_pade_gaps(kind, itemp);
    println!(
        "Padé ({kind}) {label}: N = {npoints}; max diff / max |value|: Z {ez:.1e}, Delta {ed:.1e}, chi {es:.1e}; gap edges {min:.6}/{max:.6} meV (EPW {emin}/{emax})"
    );
    // The continuation is ill conditioned, so EPW cannot be matched to the precision of
    // the Matsubara data. The bound is measured here: perturb the input by 1e-10
    // (relative) and require the difference to EPW to stay within 10 times the change.
    let first = |field: &[num_complex::Complex64],
                 reference: Option<&[[f64; 8]]>,
                 other: Option<&[num_complex::Complex64]>|
     -> f64 {
        (0..ns).fold(0.0_f64, |m, s| {
            let ours = field[99 + nw * s];
            let theirs = match (reference, other) {
                (Some(rows), _) => {
                    num_complex::Complex64::new(rows[s * (nw / 100)][4], rows[s * (nw / 100)][5])
                }
                (None, Some(o)) => o[99 + nw * s],
                _ => unreachable!(),
            };
            m.max((ours - theirs).norm())
        })
    };
    let mut perturbed = sol.clone();
    for (i, d) in perturbed.delta.iter_mut().enumerate() {
        *d *= 1.0 + 1e-10 * (((i * 7919) % 13) as f64 - 6.0) / 6.0;
    }
    let real_p = continue_to_real_axis(&perturbed, npoints, &grid).unwrap();
    let edges_p: Vec<f64> = (0..ns)
        .map(|s| gap_edge(&grid, &real_p.delta[nw * s..nw * (s + 1)], 500) * 1000.0)
        .collect();
    let edge_sensitivity = edges
        .iter()
        .zip(&edges_p)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    let delta_sensitivity = first(&real.delta, None, Some(&real_p.delta));
    let d_first = first(&real.delta, Some(&rows), None);
    println!(
        "  at {:.4} eV: max |Delta - Delta_EPW| {d_first:.2e} eV (sensitivity {delta_sensitivity:.2e}); gap-edge sensitivity {edge_sensitivity:.2e} meV",
        grid[99]
    );
    assert!(d_first <= 10.0 * delta_sensitivity + 1e-9);
    let edge_bound = 10.0 * edge_sensitivity + 1e-5;
    assert!((min - emin).abs() <= edge_bound && (max - emax).abs() <= edge_bound);
}

/// Padé continuation of the uniform-grid solution against EPW (`lpade`, `iverbosity = 2`).
#[test]
fn pade_uniform_matches_epw() {
    let p = problem();
    for (itemp, (t, label)) in [(20.0, "020.00"), (30.0, "030.00")].into_iter().enumerate() {
        let tk = kelvin(t);
        let mut sum = UniformStateSum::new(
            CpuEngine::new(Pool::builder().build().unwrap()),
            &p.model,
            tk,
            frequencies_below(WSCUT, tk),
        )
        .unwrap();
        let sol = solve_anisotropic_fbw(&mut sum, &p.model, &PARAMS, |_| 0.004).unwrap();
        check_pade("pade_uni", itemp, label, &sol, p.model.nstates());
    }
}

/// Padé continuation of the IR solution (EPW points, odd-size handling and cut) against EPW.
#[cfg(feature = "sparse-ir")]
#[test]
fn pade_ir_matches_epw() {
    let p = problem();
    let ir = eph_io_epw::supercond::read_ir_object(&dir().join("ir_nlambda5_ndigit6.dat")).unwrap();
    let settings = IrSettings {
        tau: Some(ir.x.iter().map(|x| (x + 1.0) / 2.0).collect()),
        fermionic: Some(
            ir.fermionic
                .iter()
                .filter(|&&f| f > 0)
                .map(|f| (f - 1) / 2)
                .collect(),
        ),
        bosonic: Some(
            ir.bosonic
                .iter()
                .filter(|&&b| b >= 0)
                .map(|b| b / 2)
                .collect(),
        ),
        epw_drop_last_coefficient: true,
        epw_coefficient_cut: Some(1e-5),
    };
    let set = IrBasisSet::new(ir.lambda, ir.epsilon).unwrap();
    for (itemp, (t, label)) in [(20.0, "020.00"), (30.0, "030.00")].into_iter().enumerate() {
        let tk = kelvin(t);
        let mut sum = IrStateSum::new(
            CpuEngine::new(Pool::builder().build().unwrap()),
            &set,
            &p.model,
            tk,
            &settings,
        )
        .unwrap();
        let sol = solve_anisotropic_fbw(&mut sum, &p.model, &PARAMS, |_| 0.004).unwrap();
        check_pade("pade_ir", itemp, label, &sol, p.model.nstates());
    }
}
