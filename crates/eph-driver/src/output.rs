//! Plain-text output of [`crate::Results`] (meV, eV and K where convenient).

use crate::Results;
use phys_units::{RY_TO_EV, RY_TO_MEV};
use std::fmt::Write as _;

/// One line per (q, mode): crystal coordinates, omega and gamma in meV, lambda.
///
/// # Examples
///
/// ```
/// # let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal/wannier");
/// # let model = eph_io_epw::read_wannier_model(&dir).unwrap();
/// # let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
/// # let config = eph_driver::Config { k_mesh: [1, 1, 1], q_mesh: [1, 1, 1], ..eph_driver::Config::epw_metal_test() };
/// # let results = eph_driver::run(&model, &g, &config, &eph_comm::SelfComm).unwrap();
/// let table = eph_driver::output::linewidth_table(&results);
/// assert_eq!(table.lines().filter(|l| !l.starts_with('#')).count(), 3);
/// ```
pub fn linewidth_table(results: &Results) -> String {
    let mut s = String::from("#  iq  mode        q1        q2        q3       omega(meV)       gamma(meV)           lambda\n");
    for (iq, (q, lw)) in results.q_points.iter().zip(&results.linewidths).enumerate() {
        for nu in 0..lw.omega.len() {
            let _ = writeln!(
                s,
                "{:5} {:5} {:9.5} {:9.5} {:9.5} {:16.9e} {:16.9e} {:16.9e}",
                iq + 1,
                nu + 1,
                q[0],
                q[1],
                q[2],
                lw.omega[nu] * RY_TO_MEV,
                lw.gamma[nu] * RY_TO_MEV,
                lw.lambda[nu]
            );
        }
    }
    s
}

/// alpha^2F table: omega (meV) followed by one column per phonon smearing.
///
/// # Examples
///
/// ```
/// # let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal/wannier");
/// # let model = eph_io_epw::read_wannier_model(&dir).unwrap();
/// # let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
/// # let config = eph_driver::Config { k_mesh: [2, 2, 2], q_mesh: [2, 2, 2], ..eph_driver::Config::epw_metal_test() };
/// # let results = eph_driver::run(&model, &g, &config, &eph_comm::SelfComm).unwrap();
/// let table = eph_driver::output::a2f_table(&results);
/// assert_eq!(table.lines().filter(|l| !l.starts_with('#')).count(), 500);
/// ```
pub fn a2f_table(results: &Results) -> String {
    let a = &results.eliashberg;
    let nstep = a.omega.len();
    let mut s = String::from("# omega(meV)  alpha^2F for phonon smearings (meV):");
    for sigma in &a.sigmas {
        let _ = write!(s, " {:.4}", sigma * RY_TO_MEV);
    }
    s.push('\n');
    for (i, w) in a.omega.iter().enumerate() {
        let _ = write!(s, "{:14.8}", w * RY_TO_MEV);
        for js in 0..a.sigmas.len() {
            let _ = write!(s, " {:14.8e}", a.a2f[i + nstep * js]);
        }
        s.push('\n');
    }
    s
}

/// Human-readable summary: E_F, N_F, lambda, omega_log and Allen–Dynes T_c.
///
/// # Examples
///
/// ```
/// # let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal/wannier");
/// # let model = eph_io_epw::read_wannier_model(&dir).unwrap();
/// # let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
/// # let config = eph_driver::Config { k_mesh: [2, 2, 2], q_mesh: [2, 2, 2], ..eph_driver::Config::epw_metal_test() };
/// # let results = eph_driver::run(&model, &g, &config, &eph_comm::SelfComm).unwrap();
/// assert!(eph_driver::output::summary(&results).contains("lambda"));
/// ```
pub fn summary(results: &Results) -> String {
    let a = &results.eliashberg;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Fermi energy (coarse DFT)   : {:12.6} eV",
        results.fermi_energy_coarse * RY_TO_EV
    );
    let _ = writeln!(
        s,
        "Fermi energy (fine k mesh)  : {:12.6} eV  [{:?}, {:?}]",
        results.fermi.energy * RY_TO_EV,
        results.smearing,
        results.fermi.strategy
    );
    let _ = writeln!(
        s,
        "DOS at E_F per spin         : {:12.6} states/eV/cell",
        results.dos_per_spin / RY_TO_EV
    );
    let _ = writeln!(
        s,
        "Bands in Fermi window       : {}..={}",
        results.band_window.start + 1,
        results.band_window.end
    );
    let _ = writeln!(s, "lambda (sum over q, nu)     : {:12.7}", a.lambda);
    let _ = writeln!(
        s,
        "lambda (2 int a2F/w)        : {:12.7}",
        a.lambda_integrated[0]
    );
    let _ = writeln!(
        s,
        "omega_log                   : {:12.7} Ry",
        a.omega_log[0]
    );
    for i in 0..6 {
        let mu = 0.1 + 0.02 * i as f64;
        match eph_selfen::allen_dynes_tc(a.omega_log[0], a.lambda_integrated[0], mu) {
            Some(tc) => {
                let kelvin = tc * RY_TO_EV / phys_units::KELVIN_TO_EV;
                let _ = writeln!(s, "Allen-Dynes Tc (mu* = {mu:.2})  : {kelvin:18.12} K");
            }
            None => {
                let _ = writeln!(
                    s,
                    "Allen-Dynes Tc (mu* = {mu:.2})  : no solution (lambda too small)"
                );
            }
        }
    }
    s
}
