//! IR-basis Matsubara sums (feature `sparse-ir`) against exact results and dense sums.
#![cfg(feature = "sparse-ir")]

use eliashberg::*;
use std::f64::consts::PI;
use tenferro_tensor::TypedTensor;

const T: f64 = 0.002;

/// Einstein kernel lambda(i nu) = lambda0 w0^2 / (w0^2 + nu^2).
fn einstein(nu: &[f64]) -> Result<Vec<f64>, Error> {
    let (lambda0, w0) = (0.8, 0.05);
    Ok(nu
        .iter()
        .map(|v| lambda0 * w0 * w0 / (w0 * w0 + v * v))
        .collect())
}

fn column(values: Vec<f64>) -> TypedTensor<f64> {
    let n = values.len();
    TypedTensor::from_vec_col_major(vec![n, 1], values).unwrap()
}

fn basis_set() -> &'static IrBasisSet {
    static SET: std::sync::OnceLock<IrBasisSet> = std::sync::OnceLock::new();
    // omega_max = Lambda T = 2 eV covers the Einstein mode and the test functions.
    SET.get_or_init(|| IrBasisSet::new(1e3, 1e-10).unwrap())
}

fn ir_sum(points: &IrSettings) -> IrSum {
    IrSum::new(basis_set(), T, points, einstein).unwrap()
}

/// sum_{m in Z} 1 / (omega_m^2 + a^2) = tanh(a / 2T) / (2 a T).
#[test]
fn even_sum_is_exact() {
    let mut sum = ir_sum(&IrSettings::default());
    let a = 0.03;
    let omega: Vec<f64> = sum
        .indices()
        .iter()
        .map(|&n| fermionic_frequency(n, T))
        .collect();
    let f = column(omega.iter().map(|w| 1.0 / (w * w + a * a)).collect());
    let got = sum.sum_even(&f).unwrap()[0];
    let exact = (a / (2.0 * T)).tanh() / (2.0 * a * T);
    println!(
        "sum = {got:.15e}, exact = {exact:.15e}, rel {:.1e}",
        (got - exact).abs() / exact
    );
    assert!((got - exact).abs() < 1e-9 * exact);
}

/// Convolutions agree with a dense uniform sum (truncated where f is negligible).
#[test]
fn convolution_matches_dense_sum() {
    let mut ir = ir_sum(&IrSettings::default());
    let n_dense = 200_000;
    let a = 0.03;
    let even = |w: f64| a / (w * w + a * a);
    let odd = |w: f64| w / (w * w + a * a);
    for (parity, f) in [
        (Parity::Even, &even as &dyn Fn(f64) -> f64),
        (Parity::Odd, &odd),
    ] {
        let omega_ir: Vec<f64> = ir
            .indices()
            .iter()
            .map(|&n| fermionic_frequency(n, T))
            .collect();
        let c_ir = ir
            .convolve(&column(omega_ir.iter().map(|&w| f(w)).collect()), parity)
            .unwrap();
        // Dense reference: C(n) = sum_m lambda(n - m) f(m), m over |m| < n_dense.
        let f_dense: Vec<f64> = (-n_dense..n_dense)
            .map(|m| f(fermionic_frequency(m, T)))
            .collect();
        let (mut worst, mut worst_abs, mut scale) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (i, &n) in ir.indices().iter().enumerate().filter(|(_, n)| **n < 2000) {
            let nu: Vec<f64> = (-n_dense..n_dense)
                .map(|m| 2.0 * PI * T * (n - m) as f64)
                .collect();
            let lambda = einstein(&nu).unwrap();
            let c: f64 = lambda.iter().zip(&f_dense).map(|(l, f)| l * f).sum();
            let got = c_ir.as_slice().unwrap()[i];
            worst = worst.max((got - c).abs() / c.abs().max(1e-12));
            worst_abs = worst_abs.max((got - c).abs());
            scale = scale.max(c.abs());
        }
        println!(
            "{parity:?}: IR vs dense: max |diff| / max |C| = {:.1e}, max entrywise rel diff = {worst:.1e}",
            worst_abs / scale
        );
        // Accuracy of the IR fit is relative to the largest values (epsilon = 1e-10 basis).
        assert!(worst_abs < 1e-8 * scale);
    }
}

/// tau points in [-beta/2, beta/2] (sparse-ir-rs) and in (0, beta) give the same convolution.
#[test]
fn tau_range_does_not_matter() {
    let mut shifted = ir_sum(&IrSettings::default());
    let unfolded: Vec<f64> = shifted
        .tau_points()
        .iter()
        .map(|&t| {
            if t < 0.0 {
                t / (1.0 / T) + 1.0
            } else {
                t / (1.0 / T)
            }
        })
        .collect();
    let mut positive = ir_sum(&IrSettings {
        tau: Some(unfolded),
        ..Default::default()
    });
    assert!(positive.tau_points().iter().all(|&t| t > 0.0));
    let omega: Vec<f64> = shifted
        .indices()
        .iter()
        .map(|&n| fermionic_frequency(n, T))
        .collect();
    for parity in [Parity::Even, Parity::Odd] {
        let f = column(
            omega
                .iter()
                .map(|w| if parity == Parity::Odd { *w } else { 0.03 } / (w * w + 9e-4))
                .collect(),
        );
        let a = shifted.convolve(&f, parity).unwrap();
        let b = positive.convolve(&f, parity).unwrap();
        let diff = a
            .as_slice()
            .unwrap()
            .iter()
            .zip(b.as_slice().unwrap())
            .map(|(x, y)| (x - y).abs() / y.abs())
            .fold(0.0, f64::max);
        println!("{parity:?}: [-beta/2, beta/2] vs (0, beta): max rel diff {diff:.1e}");
        assert!(diff < 1e-9);
    }
}
