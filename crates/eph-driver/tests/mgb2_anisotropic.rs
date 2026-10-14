//! L2 test: anisotropic FBW Eliashberg equations for MgB2 against EPW (QE 7.6).
//!
//! EPW solved `epw6.in` (uniform grid) with `conv_thr_iaxis = 1e-10` at 20 and
//! 30 K (`tests/fixtures/mgb2_epw_super/aniso`). The Rust solver reads the same
//! `MgB2.ephmat` data.

use eliashberg::*;
use eph_driver::supercond::{anisotropic_problem, EpwAnisotropicProblem};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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
            tenferro_cpu::CpuBackend::new(),
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
    let p = problem();
    let ir = eph_io_epw::supercond::read_ir_object(&dir().join("ir_nlambda5_ndigit6.dat")).unwrap();
    let points = SamplingPoints {
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
    };
    let set = IrBasisSet::new(ir.lambda, ir.epsilon).unwrap();
    for (t, label) in [(20.0, "020.00"), (30.0, "030.00")] {
        let tk = kelvin(t);
        let mut sum =
            IrStateSum::new(tenferro_cpu::CpuBackend::new(), &set, &p.model, tk, &points).unwrap();
        let sol = solve_anisotropic_fbw(&mut sum, &p.model, &PARAMS, |_| 0.004).unwrap();
        let rows = reference("ir", label);
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
            "IR {t} K: {nf} freq x {ns} states, {} iterations; max diff / max |value|: omega {ew:.1e}, Z {ez:.1e}, Delta {ed:.1e}, chi {es:.1e}",
            sol.iterations
        );
        assert!(ew < 1e-10); // printed with 11 digits
                             // Remaining differences (~1e-5) come mainly from EPW's coefficient cut
                             // (`eps_cut_ir = 1e-5`), which is not reproduced, and from the basis itself
                             // (Python sparse-ir file versus sparse-ir-rs), both at the epsilon = 1e-6 level.
        assert!(ez < 5e-5 && ed < 5e-5 && es < 5e-5);
    }
}
