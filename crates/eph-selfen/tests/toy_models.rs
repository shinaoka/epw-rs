//! L0 checks of the linewidth and alpha^2F formulas on toy inputs.

use eph_selfen::*;
use num_complex::Complex64;
use tenferro_cpu::{CpuEngine, Pool};

/// lambda = 2 int alpha^2F / omega for a set of Einstein-like modes.
#[test]
fn integrated_lambda_matches_sum() {
    let modes = [(0.004, 0.3), (0.007, 0.5), (0.011, 0.2)];
    let lws: Vec<Linewidth> = modes
        .iter()
        .map(|&(w, l)| Linewidth {
            omega: vec![w],
            gamma: vec![0.0],
            lambda: vec![l],
        })
        .collect();
    let params = EliashbergParams {
        nstep: 4000,
        sigma: 5e-5,
        nsmear: 2,
        sigma_step: 5e-5,
        eps_acoustic: 1e-8,
    };
    let a2f = eliashberg_function(
        &mut CpuEngine::new(Pool::builder().build().unwrap()),
        &lws,
        1.0,
        &params,
    )
    .unwrap();
    assert!((a2f.lambda - 1.0).abs() < 1e-14);
    for (s, l) in a2f.lambda_integrated.iter().enumerate() {
        assert!((l - 1.0).abs() < 2e-3, "smearing {s}: {l}");
    }
}

/// Single band, constant coupling: gamma follows the closed form of the formula.
#[test]
fn constant_coupling_single_pair() {
    let omega = 0.002;
    let g0 = Complex64::new(0.01, 0.0);
    let (e_k, e_kq) = (-0.001, 0.0012);
    let params = LinewidthParams {
        fsthick: 1.0,
        sigma: 0.003,
        temperature: 0.0005,
        eps_acoustic: 1e-8,
    };
    let fs = FermiSurface {
        fermi_energy: 0.0,
        dos_per_spin: 2.5,
        band_window: 0..1,
    };
    let data = QPointData {
        nbnd: 1,
        omega: &[omega],
        g: &[g0],
        energies_k: &[e_k],
        energies_kq: &[e_kq],
        weights: &[2.0],
    };
    let lw = phonon_linewidth(
        &mut CpuEngine::new(Pool::builder().build().unwrap()),
        &data,
        &fs,
        &params,
    )
    .unwrap();
    let f = |e: f64| smearing::fermi_dirac(e, params.temperature);
    let gauss = (-((e_kq - e_k - omega) / params.sigma).powi(2)).exp()
        / (std::f64::consts::PI.sqrt() * params.sigma);
    let expected =
        -std::f64::consts::PI * 2.0 * (f(e_kq) - f(e_k)) * gauss * g0.norm_sqr() / (2.0 * omega);
    assert!((lw.gamma[0] - expected).abs() < 1e-15 * expected.abs().max(1.0));
    let lambda = expected / (std::f64::consts::PI * omega * omega * fs.dos_per_spin);
    assert!((lw.lambda[0] - lambda).abs() < 1e-12 * lambda);
}

/// Degenerate modes share the average gamma; acoustic modes give no lambda.
#[test]
fn degenerate_average_and_acoustic_cut() {
    let params = LinewidthParams {
        fsthick: 1.0,
        sigma: 0.003,
        temperature: 0.0005,
        eps_acoustic: 1e-6,
    };
    let fs = FermiSurface {
        fermi_energy: 0.0,
        dos_per_spin: 1.0,
        band_window: 0..1,
    };
    let g = [
        Complex64::new(0.01, 0.0),
        Complex64::new(0.03, 0.0),
        Complex64::new(0.05, 0.0),
    ];
    let data = QPointData {
        nbnd: 1,
        omega: &[1e-9, 0.002, 0.002],
        g: &g,
        energies_k: &[-0.001],
        energies_kq: &[0.001],
        weights: &[2.0],
    };
    let lw = phonon_linewidth(
        &mut CpuEngine::new(Pool::builder().build().unwrap()),
        &data,
        &fs,
        &params,
    )
    .unwrap();
    assert_eq!(lw.gamma[0], 0.0);
    assert_eq!(lw.lambda[0], 0.0);
    assert_eq!(lw.gamma[1], lw.gamma[2]);
    assert_eq!(lw.lambda[1], lw.lambda[2]);
}
