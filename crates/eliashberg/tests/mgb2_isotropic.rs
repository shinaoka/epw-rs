//! L2 test: isotropic Eliashberg equations for MgB2 against EPW (QE 7.6).
//!
//! EPW read the same `MgB2.a2f` (and `MgB2.dos` for FBW) with `fila2f` and
//! solved with `conv_thr_iaxis = 1e-12` (`tests/fixtures/mgb2_epw_super/iso`).

use eliashberg::*;
use std::path::{Path, PathBuf};

const MU_STAR: f64 = 0.16;
const WSCUT: f64 = 0.5;

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mgb2_epw_super/iso")
}

fn spectrum() -> EliashbergSpectrum {
    let table = eph_io_epw::supercond::read_a2f(&dir().join("MgB2.a2f")).unwrap();
    EliashbergSpectrum::new(table.omega, table.a2f).unwrap()
}

fn dos() -> ElectronDos {
    let table = eph_io_epw::supercond::read_dos(&dir().join("MgB2.dos")).unwrap();
    // The file holds both spins; EPW divides by two.
    ElectronDos {
        energies: table.energies,
        dos: table.dos.iter().map(|d| d / 2.0).collect(),
        step: table.step,
        chemical_potential: table.fermi_energy,
        dos_at_fermi: table.dos_at_fermi / 2.0,
    }
}

fn kelvin(t: f64) -> f64 {
    t * phys_units::KELVIN_TO_EV
}

/// Columns of `MgB2.imag_iso_<T>`: omega, Z, Delta (and chi for FBW).
fn reference(kind: &str, temp: &str) -> Vec<Vec<f64>> {
    let path = dir().join(kind).join(format!("MgB2.imag_iso_{temp}"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect()
}

fn printed_free_energy(kind: &str, itemp: usize) -> f64 {
    let text = std::fs::read_to_string(dir().join(kind).join(format!("iso_{kind}.out"))).unwrap();
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

fn max_rel(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-300))
        .fold(0.0, f64::max)
}

fn uniform_sum(spectrum: &EliashbergSpectrum, t: f64) -> UniformSum {
    let n = frequencies_below(WSCUT, t);
    UniformSum::new(tenferro_cpu::CpuBackend::new(), t, n, |nu| {
        spectrum.lambda_matsubara(&mut tenferro_cpu::CpuBackend::new(), nu)
    })
    .unwrap()
}

fn params() -> IsotropicParams {
    IsotropicParams {
        mu_star: MU_STAR,
        conv_thr: 1e-13,
        max_iter: 20_000,
        mixing: 0.3,
        initial_gap: 0.004,
    }
}

#[test]
fn mcmillan_tc_matches() {
    let tc = spectrum().mcmillan_tc(MU_STAR).unwrap() / phys_units::KELVIN_TO_EV;
    println!("McMillan Tc = {tc:.6} K (EPW 26.4143 K)");
    assert!((tc - 26.4143).abs() < 5e-5);
}

#[test]
fn fsr_matches_epw() {
    let spectrum = spectrum();
    for (itemp, (t, label, nfreq)) in [(15.0, "015.00", 62), (20.0, "020.00", 46)]
        .into_iter()
        .enumerate()
    {
        let mut sum = uniform_sum(&spectrum, kelvin(t));
        assert_eq!(sum.indices().len(), nfreq);
        let sol = solve_isotropic_fsr(&mut sum, &params()).unwrap();
        let rows = reference("fsr", label);
        let col = |c: usize| rows.iter().map(|r| r[c]).collect::<Vec<_>>();
        let (ew, ez, ed) = (
            max_rel(&sol.omega, &col(0)),
            max_rel(&sol.z, &col(1)),
            max_rel(&sol.delta, &col(2)),
        );
        println!(
            "FSR {t} K: {} iterations; max rel diff omega {ew:.1e}, Z {ez:.1e}, Delta {ed:.1e}",
            sol.iterations
        );
        assert!(ew < 1e-10 && ez < 1e-9 && ed < 1e-9);

        let z_normal = normal_state_z(&mut sum).unwrap();
        let fe = free_energy_difference(&sol, &z_normal, kelvin(t)) * 1000.0;
        let expected = printed_free_energy("fsr", itemp);
        println!("FSR {t} K: free energy {fe:.8} meV (EPW {expected})");
        assert!((fe - expected).abs() < 6e-7);
    }
}

#[test]
fn fbw_matches_epw() {
    let spectrum = spectrum();
    let dos = dos();
    for (t, label) in [(15.0, "015.00"), (20.0, "020.00")] {
        let mut sum = uniform_sum(&spectrum, kelvin(t));
        let sol = solve_isotropic_fbw(&mut sum, &dos, &params()).unwrap();
        let rows = reference("fbw", label);
        let col = |c: usize| rows.iter().map(|r| r[c]).collect::<Vec<_>>();
        let (ez, ed, es) = (
            max_rel(&sol.z, &col(1)),
            max_rel(&sol.delta, &col(2)),
            max_rel(&sol.shift, &col(3)),
        );
        println!(
            "FBW {t} K: {} iterations; max rel diff Z {ez:.1e}, Delta {ed:.1e}, chi {es:.1e}",
            sol.iterations
        );
        assert!(ez < 1e-9 && ed < 1e-9 && es < 1e-9);
    }
}

/// EPW prints for FBW the FSR Bardeen–Stephen expression with
/// `Z^N = 1 + (T / omega) sum lambda`, i.e. without the factor pi of the FSR `Z^N`.
/// This test records that convention; the FSR form is in any case not the exact
/// FBW free energy (it ignores chi and the energy integral).
#[test]
fn fbw_free_energy_convention() {
    let spectrum = spectrum();
    let dos = dos();
    for (itemp, t) in [15.0, 20.0].into_iter().enumerate() {
        let tk = kelvin(t);
        let mut sum = uniform_sum(&spectrum, tk);
        let sol = solve_isotropic_fbw(&mut sum, &dos, &params()).unwrap();
        let z_with_pi = normal_state_z(&mut sum).unwrap();
        let z_without_pi: Vec<f64> = z_with_pi
            .iter()
            .map(|z| 1.0 + (z - 1.0) / std::f64::consts::PI)
            .collect();
        let with_pi = free_energy_difference(&sol, &z_with_pi, tk) * 1000.0;
        let without_pi = free_energy_difference(&sol, &z_without_pi, tk) * 1000.0;
        let expected = printed_free_energy("fbw", itemp);
        println!("FBW {t} K: free energy with pi {with_pi:.8}, without pi {without_pi:.8} meV (EPW {expected})");
        assert!((without_pi - expected).abs() < 6e-7);
        assert!((with_pi - expected).abs() > 1e-3);
    }
}
