//! L1 test: reproduce EPW (QE 7.6) on the Pb `epw_metal` fixture.
//!
//! The reference files were produced by `epw.x` from exactly the Wannier files
//! read here (`tests/fixtures/pb_epw_metal/reference/epw_read`). Tolerances
//! follow the number of digits EPW prints; each check reports its residual.

use eph_comm::{SelfComm, ThreadComm};
use eph_driver::{run, Config, Results};
use phys_units::{RY_TO_EV, RY_TO_MEV};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal")
}

fn numbers(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

fn data_rows(path: &Path) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .map(numbers)
        .filter(|v| !v.is_empty())
        .collect()
}

/// Value printed after `key` in the EPW standard output.
fn stdout_value(key: &str) -> f64 {
    let text = std::fs::read_to_string(fixture().join("reference/epw_read/epw_read.out")).unwrap();
    let line = text
        .lines()
        .find(|l| l.contains(key))
        .unwrap_or_else(|| panic!("{key} not found"));
    let rest = &line[line.find(key).unwrap() + key.len()..];
    numbers(&rest.replace('=', " "))[0]
}

fn results() -> &'static Results {
    static RESULTS: OnceLock<Results> = OnceLock::new();
    RESULTS.get_or_init(|| {
        let dir = fixture().join("wannier");
        let model = eph_io_epw::read_wannier_model(&dir).unwrap();
        let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model))
            .unwrap();
        run(&model, &g, &Config::epw_metal_test(), &SelfComm).unwrap()
    })
}

/// Max of |a - b| / (abs + rel |b|); <= 1 passes.
fn worst(actual: &[f64], expected: &[f64], abs: f64, rel: f64) -> (f64, usize) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs() / (abs + rel * b.abs()))
        .enumerate()
        .fold((0.0, 0), |acc, (i, r)| if r > acc.0 { (r, i) } else { acc })
}

fn assert_close(what: &str, actual: &[f64], expected: &[f64], abs: f64, rel: f64) {
    let (ratio, at) = worst(actual, expected, abs, rel);
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("{what}: max |diff| = {max_abs:.3e}, worst ratio to tolerance = {ratio:.3} at {at}");
    assert!(
        ratio <= 1.0,
        "{what}: entry {at}: got {} expected {} (abs {abs:e}, rel {rel:e})",
        actual[at],
        expected[at]
    );
}

#[test]
fn fermi_level_and_dos() {
    let r = results();
    let ef = r.fermi.energy * RY_TO_EV;
    assert_close("E_F (eV)", &[ef], &[stdout_value("Ef =")], 1.0e-6, 0.0);
    assert_close(
        "N_F (1/eV)",
        &[r.dos_per_spin / RY_TO_EV],
        &[stdout_value("DOS =")],
        1.0e-6,
        0.0,
    );
    assert_eq!(r.band_window, 1..4);
}

#[test]
fn phonon_frequencies_and_linewidths() {
    let rows = data_rows(&fixture().join("reference/epw_read/linewidth.phself.300.000K"));
    let r = results();
    let omega: Vec<f64> = r
        .linewidths
        .iter()
        .flat_map(|l| l.omega.iter().map(|w| w * RY_TO_MEV))
        .collect();
    // EPW writes the full width 2 gamma to linewidth.phself (stdout prints gamma).
    let gamma: Vec<f64> = r
        .linewidths
        .iter()
        .flat_map(|l| l.gamma.iter().map(|g| 2.0 * g * RY_TO_MEV))
        .collect();
    let ref_omega: Vec<f64> = rows.iter().map(|v| v[2]).collect();
    let ref_gamma: Vec<f64> = rows.iter().map(|v| v[3]).collect();
    // 8 significant digits printed; acoustic modes at Gamma are numerical noise (~1e-7 meV).
    assert_close("omega (meV)", &omega, &ref_omega, 1.0e-5, 1.0e-7);
    // 10 significant digits printed. Entries far below the largest width (down to
    // 1e-9 of it) are sums of Gaussian/Fermi-Dirac tails; they are compared with an
    // absolute tolerance of 1e-7 of the largest width plus 1e-6 relative.
    let scale = ref_gamma.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    assert_close("2 gamma (meV)", &gamma, &ref_gamma, 1.0e-7 * scale, 1.0e-6);
}

#[test]
fn mode_coupling_strengths() {
    let rows = data_rows(&fixture().join("reference/epw_read/lambda.phself.300.000K"));
    let r = results();
    let lambda: Vec<f64> = r
        .linewidths
        .iter()
        .flat_map(|l| l.lambda.iter().copied())
        .collect();
    let expected: Vec<f64> = rows.iter().flat_map(|v| v[1..].to_vec()).collect();
    // 5 significant digits printed.
    assert_close("lambda_q nu", &lambda, &expected, 1.0e-12, 1.0e-4);
}

#[test]
fn eliashberg_function_and_lambda() {
    let r = results();
    let a = &r.eliashberg;
    let rows = data_rows(&fixture().join("reference/epw_read/pb.a2f.01.300.000"));
    let table: Vec<&Vec<f64>> = rows
        .iter()
        .filter(|v| v.len() == 1 + a.sigmas.len())
        .collect();
    assert_eq!(table.len(), a.omega.len());
    let omega: Vec<f64> = a.omega.iter().map(|w| w * RY_TO_MEV).collect();
    assert_close(
        "a2F omega grid (meV)",
        &omega,
        &table.iter().map(|v| v[0]).collect::<Vec<_>>(),
        1.0e-7,
        0.0,
    );
    let n = a.omega.len();
    for s in 0..a.sigmas.len() {
        let expected: Vec<f64> = table.iter().map(|v| v[1 + s]).collect();
        // Printed with 7 decimals (f12.7): allow half a unit plus rounding.
        assert_close(
            &format!("a2F smearing {s}"),
            &a.a2f[s * n..(s + 1) * n],
            &expected,
            6.0e-8,
            0.0,
        );
    }
    assert_close(
        "lambda (sum)",
        &[a.lambda],
        &[stdout_value("lambda :")],
        1.0e-7,
        0.0,
    );
    assert_close(
        "lambda (a2F)",
        &[a.lambda_integrated[0]],
        &[stdout_value("l_a2f =")],
        1.0e-7,
        0.0,
    );
    assert_close(
        "omega_log (Ry)",
        &[a.omega_log[0]],
        &[stdout_value("logavg =")],
        1.0e-7,
        0.0,
    );
}

#[test]
fn distributed_run_is_bit_identical() {
    let dir = fixture().join("wannier");
    let model = eph_io_epw::read_wannier_model(&dir).unwrap();
    // Stream the vertex from disk this time (EPW etf_mem = 1 analogue).
    let g = eph_io_epw::EpmatwpFile::open(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model))
        .unwrap();
    let config = Config {
        q_block: 5,
        rp_block: 7,
        ..Config::epw_metal_test()
    };
    let four = run(&model, &g, &config, &ThreadComm::new(4).unwrap()).unwrap();
    let one = run(&model, &g, &config, &SelfComm).unwrap();
    let again = run(&model, &g, &config, &SelfComm).unwrap();
    assert_eq!(one, again, "repeated runs must be bit-identical");
    assert_eq!(
        one, four,
        "4 emulated ranks must reproduce 1 rank bit for bit"
    );
}
