//! MgB2 isotropic FBW with the IR basis (feature `sparse-ir`).
//!
//! There is no EPW counterpart (EPW's IR solver is anisotropic only), so the
//! IR solution is checked for stability in (Lambda, epsilon) and against
//! uniform grids with growing cutoff, which approach it monotonically.
#![cfg(feature = "sparse-ir")]

use eliashberg::*;
use std::path::Path;

fn inputs() -> (EliashbergSpectrum, ElectronDos) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mgb2_epw_super/iso");
    let a2f = eph_io_epw::supercond::read_a2f(&dir.join("MgB2.a2f")).unwrap();
    let table = eph_io_epw::supercond::read_dos(&dir.join("MgB2.dos")).unwrap();
    let dos = ElectronDos {
        energies: table.energies,
        dos: table.dos.iter().map(|d| d / 2.0).collect(),
        step: table.step,
        chemical_potential: table.fermi_energy,
        dos_at_fermi: table.dos_at_fermi / 2.0,
    };
    (EliashbergSpectrum::new(a2f.omega, a2f.a2f).unwrap(), dos)
}

const PARAMS: IsotropicParams = IsotropicParams {
    mu_star: 0.16,
    conv_thr: 1e-11,
    max_iter: 50_000,
    mixing: 0.3,
    initial_gap: 0.004,
};

#[test]
fn ir_solution_is_basis_independent_and_approached_by_uniform_grids() {
    let (spectrum, dos) = inputs();
    let t = 15.0 * phys_units::KELVIN_TO_EV;
    let kernel = |nu: &[f64]| spectrum.lambda_matsubara(&mut tenferro_cpu::CpuBackend::new(), nu);

    let mut gaps = Vec::new();
    for lambda in [1e5, 1e6] {
        let set = IrBasisSet::new(lambda, 1e-6).unwrap();
        let mut sum = IrSum::new(&set, t, TauPoints::Default, kernel).unwrap();
        let sol = solve_isotropic_fbw(&mut sum, &dos, &PARAMS).unwrap();
        assert_eq!(sum.indices()[0], 0);
        println!(
            "IR Lambda {lambda:.0e}: {} points, Delta(0) = {:.8} meV, Z(0) = {:.8}",
            sum.indices().len(),
            sol.delta[0] * 1e3,
            sol.z[0]
        );
        gaps.push(sol.delta[0]);
    }
    let ir_gap = gaps[1];
    // EPW's basis (Lambda 1e5, eps 1e-6) is accurate to ~3e-5 here; a 1e-10 basis
    // reproduces the Lambda = 1e6 value to ~1e-7 (see examples/mgb2_ir_vs_uniform.rs).
    assert!(
        (gaps[0] - gaps[1]).abs() < 1e-4 * ir_gap,
        "IR depends on Lambda"
    );

    let mut previous = f64::INFINITY;
    for wscut in [0.5, 2.0, 8.0] {
        let mut sum = UniformSum::new(
            tenferro_cpu::CpuBackend::new(),
            t,
            frequencies_below(wscut, t),
            kernel,
        )
        .unwrap();
        let sol = solve_isotropic_fbw(&mut sum, &dos, &PARAMS).unwrap();
        let gap_error = (sol.delta[0] - ir_gap).abs() / ir_gap;
        println!(
            "uniform wscut {wscut} eV: Delta(0) = {:.8} meV, rel diff to IR {gap_error:.2e}",
            sol.delta[0] * 1e3
        );
        assert!(gap_error < previous);
        previous = gap_error;
    }
}
