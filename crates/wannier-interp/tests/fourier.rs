//! L0/L1 checks of the interpolation on the Pb fixture against direct (non-FFT) sums.

use bz_mesh::UniformMesh;
use num_complex::Complex64;
use std::path::Path;
use wannier_interp::Interpolator;

fn model() -> eph_wannier_data::WannierModel {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal/wannier");
    eph_io_epw::read_wannier_model(&dir).unwrap()
}

/// H(k) = sum_R exp(2 pi i k.R) H(R) / N_R evaluated directly, Hermitised.
fn direct_hk(model: &eph_wannier_data::WannierModel, k: [f64; 3]) -> Vec<Complex64> {
    let nb = model.nbnd;
    let mut h = vec![Complex64::default(); nb * nb];
    for (ir, (r, &deg)) in model
        .ws_k
        .vectors
        .iter()
        .zip(&model.ws_k.degeneracy)
        .enumerate()
    {
        let arg = 2.0 * std::f64::consts::PI * (0..3).map(|a| k[a] * r[a] as f64).sum::<f64>();
        let phase = Complex64::from_polar(1.0 / f64::from(deg), arg);
        let block = &model.hamiltonian[nb * nb * ir..nb * nb * (ir + 1)];
        for (hi, &hr) in h.iter_mut().zip(block) {
            *hi += phase * hr;
        }
    }
    let mut herm = h.clone();
    for i in 0..nb {
        for j in 0..nb {
            herm[i + nb * j] = 0.5 * (h[i + nb * j] + h[j + nb * i].conj());
        }
    }
    herm
}

/// The FFT path reproduces the direct Fourier sum (sign and folding conventions).
#[test]
fn hamiltonian_matches_direct_sum() {
    let model = model();
    let nb = model.nbnd;
    let mesh = UniformMesh::new([6, 6, 6]).unwrap();
    let bands = Interpolator::new().electrons(&model, &mesh).unwrap();
    let u = bands.rotations.as_slice().unwrap();
    let mut worst = 0.0_f64;
    for ik in [0, 1, 7, 43, 100, 215] {
        let h = direct_hk(&model, mesh.point(ik));
        // Residual of H U - U E with the interpolated eigenpairs.
        for n in 0..nb {
            for a in 0..nb {
                let hu: Complex64 = (0..nb)
                    .map(|b| h[a + nb * b] * u[b + nb * (n + nb * ik)])
                    .sum();
                let ue = u[a + nb * (n + nb * ik)] * bands.energies[n + nb * ik];
                worst = worst.max((hu - ue).norm());
            }
        }
    }
    println!("max |H U - U E| = {worst:e} Ry");
    assert!(worst < 1e-12);
}

/// Eigenvectors are orthonormal and energies ascend.
#[test]
fn rotations_are_unitary() {
    let model = model();
    let nb = model.nbnd;
    let mesh = UniformMesh::new([3, 3, 3]).unwrap();
    let bands = Interpolator::new().electrons(&model, &mesh).unwrap();
    let u = bands.rotations.as_slice().unwrap();
    for ik in 0..mesh.len() {
        for m in 0..nb {
            for n in 0..nb {
                let dot: Complex64 = (0..nb)
                    .map(|a| u[a + nb * (m + nb * ik)].conj() * u[a + nb * (n + nb * ik)])
                    .sum();
                let expected = if m == n { 1.0 } else { 0.0 };
                assert!((dot - expected).norm() < 1e-12);
            }
        }
        let e = &bands.energies[ik * nb..(ik + 1) * nb];
        assert!(e.windows(2).all(|w| w[0] <= w[1]));
    }
}

/// Acoustic sum rule at Gamma and real, non-negative frequencies on the mesh.
#[test]
fn phonons_obey_acoustic_sum_rule() {
    let model = model();
    let mesh = UniformMesh::new([4, 4, 4]).unwrap();
    let ph = Interpolator::new().phonons(&model, &mesh).unwrap();
    let gamma_max = ph.frequencies[..3]
        .iter()
        .fold(0.0_f64, |m, w| m.max(w.abs()));
    assert!(gamma_max < 1e-7, "omega(Gamma) = {gamma_max:e} Ry");
    assert!(ph.frequencies.iter().all(|&w| w > -1e-7));
}

/// The vertex is the same whether R_p is read in one block or in small blocks.
#[test]
fn vertex_independent_of_rp_blocking() {
    let model = model();
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal/wannier");
    let g =
        eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
    let q = [[0.5, 0.0, 1.0 / 3.0], [0.0, 0.25, 0.0]];
    let mut interp = Interpolator::new();
    let whole = interp.vertex_rp_to_q(&g, &model.ws_g, &q, 43).unwrap();
    let pieces = interp.vertex_rp_to_q(&g, &model.ws_g, &q, 5).unwrap();
    let diff = whole
        .as_slice()
        .unwrap()
        .iter()
        .zip(pieces.as_slice().unwrap())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    assert!(diff < 1e-14, "max diff {diff:e}");
}
