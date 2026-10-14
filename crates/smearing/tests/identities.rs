//! L0 checks of the smearing functions and the Fermi-level search.

use smearing::{fermi_level, BandSet, FermiStrategy, Smearing};

const ALL: [Smearing; 5] = [
    Smearing::Gaussian,
    Smearing::MethfesselPaxton(1),
    Smearing::MethfesselPaxton(2),
    Smearing::MarzariVanderbilt,
    Smearing::FermiDirac,
];

/// The occupation is the integral of the delta: d occ / dt = delta(t).
#[test]
fn occupation_derivative_is_delta() {
    for s in ALL {
        for &t in &[-1.7, -0.4, 0.0, 0.3, 1.1] {
            let h = 1e-5;
            let fd = (s.occupation(t + h) - s.occupation(t - h)) / (2.0 * h);
            let err = (fd - s.delta(t)).abs();
            assert!(err < 1e-8, "{s:?} at t={t}: |d occ/dt - delta| = {err:e}");
        }
    }
}

/// Every delta approximation integrates to one.
#[test]
fn delta_is_normalised() {
    for s in ALL {
        let (h, n) = (1e-3, 40_000);
        let integral: f64 = (-n..=n).map(|i| s.delta(i as f64 * h) * h).sum();
        assert!(
            (integral - 1.0).abs() < 1e-9,
            "{s:?}: integral = {integral}"
        );
    }
}

/// Flat band set: E_F sits between two levels for every smearing type.
#[test]
fn fermi_level_between_two_levels() {
    let energies = [-0.5, 0.5];
    let weights = [2.0];
    let bands = BandSet::new(&energies, &weights, 2).unwrap();
    for s in ALL {
        let ef = fermi_level(&bands, 2.0, 0.02, s).unwrap();
        let count = bands.electron_count(ef.energy, 0.02, s);
        assert!((count - 2.0).abs() < 1e-8, "{s:?}: N(E_F) = {count}");
        assert!(ef.energy.abs() < 0.5, "{s:?}: E_F = {}", ef.energy);
    }
}

/// Plain Gaussian smearing never needs the Newton step.
#[test]
fn gaussian_uses_bisection() {
    let bands = BandSet::new(&[0.0, 1.0, 0.2, 1.3], &[1.0, 1.0], 2).unwrap();
    let ef = fermi_level(&bands, 2.0, 0.05, Smearing::Gaussian).unwrap();
    assert_eq!(ef.strategy, FermiStrategy::Bisection);
}
