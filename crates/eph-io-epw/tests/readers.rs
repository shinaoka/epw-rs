//! Reader checks on the Pb fixture.

use eph_wannier_data::EphMatrixSource;
use num_complex::Complex64;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pb_epw_metal/wannier")
}

#[test]
fn model_dimensions_and_values() {
    let model = eph_io_epw::read_wannier_model(&dir()).unwrap();
    assert_eq!(model.nbnd, 4);
    assert_eq!(model.crystal.nmodes(), 3);
    assert_eq!(model.crystal.wannier_electrons(), 4.0);
    assert!((model.crystal.alat - 9.27).abs() < 1e-12);
    // Values exactly as printed in epwdata.fmt (parsed from the same text).
    let parse = |s: &str| s.parse::<f64>().unwrap();
    assert_eq!(model.fermi_energy_coarse, parse("0.72013407900204163"));
    let h0 = Complex64::new(
        parse("-7.41120091206255716E-005"),
        parse("-2.65841272606426315E-011"),
    );
    assert_eq!(model.hamiltonian[0], h0);
    assert_eq!(model.ws_k.vectors[0], [-2, 0, 1]);
    assert_eq!(model.ws_k.degeneracy[0], 3);
}

/// Streaming blocks from disk gives the same bytes as loading the whole file.
#[test]
fn file_blocks_match_in_memory() {
    let model = eph_io_epw::read_wannier_model(&dir()).unwrap();
    let dims = eph_io_epw::eph_dims(&model);
    let path = dir().join("pb.epmatwp");
    let memory = eph_io_epw::read_epmatwp(&path, dims).unwrap();
    let file = eph_io_epw::EpmatwpFile::open(&path, dims).unwrap();
    let block = dims.block_len();
    for range in [0..1, 5..12, 40..43] {
        let mut a = vec![Complex64::default(); block * range.len()];
        let mut b = a.clone();
        memory.read_rp_block(range.clone(), &mut a).unwrap();
        file.read_rp_block(range, &mut b).unwrap();
        assert_eq!(a, b);
    }
    let mut out = vec![Complex64::default(); block];
    assert!(file.read_rp_block(42..44, &mut out).is_err());
}

#[test]
fn tokenizer_handles_fortran_forms() {
    use eph_io_epw::Token;
    let t = eph_io_epw::tokenize("  (1.0D0,-2.5E-1)\n T  .false. 3,4");
    assert_eq!(t[0], Token::Complex("1.0D0".into(), "-2.5E-1".into()));
    assert_eq!(t.len(), 5);
}
