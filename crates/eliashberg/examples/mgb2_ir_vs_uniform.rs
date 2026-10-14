//! MgB2 isotropic FBW: IR-basis solution versus uniform grids with growing cutoff.
//!
//! Run with `cargo run --release -p eliashberg --features sparse-ir --example mgb2_ir_vs_uniform`.

use eliashberg::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mgb2_epw_super/iso");
    let a2f = eph_io_epw::supercond::read_a2f(&dir.join("MgB2.a2f"))?;
    let spectrum = EliashbergSpectrum::new(a2f.omega, a2f.a2f)?;
    let table = eph_io_epw::supercond::read_dos(&dir.join("MgB2.dos"))?;
    let dos = ElectronDos {
        energies: table.energies,
        dos: table.dos.iter().map(|d| d / 2.0).collect(),
        step: table.step,
        chemical_potential: table.fermi_energy,
        dos_at_fermi: table.dos_at_fermi / 2.0,
    };
    let t = 15.0 * phys_units::KELVIN_TO_EV;
    let params = IsotropicParams {
        mu_star: 0.16,
        conv_thr: 1e-12,
        max_iter: 50_000,
        mixing: 0.3,
        initial_gap: 0.004,
    };
    let kernel = |nu: &[f64]| spectrum.lambda_matsubara(&mut tenferro_cpu::CpuBackend::new(), nu);

    let start = std::time::Instant::now();
    let set = IrBasisSet::new(1e5, 1e-10)?;
    let mut ir = IrSum::new(&set, t, &SamplingPoints::default(), kernel)?;
    let sol_ir = solve_isotropic_fbw(&mut tenferro_cpu::CpuBackend::new(), &mut ir, &dos, &params)?;
    println!(
        "IR (Lambda 1e5, eps 1e-10): {} frequencies, {} iterations, {:.2} s",
        ir.indices().len(),
        sol_ir.iterations,
        start.elapsed().as_secs_f64()
    );
    println!(
        "  n=0: Z = {:.10}, Delta = {:.10} meV, chi = {:.10} meV",
        sol_ir.z[0],
        sol_ir.delta[0] * 1e3,
        sol_ir.shift[0] * 1e3
    );

    println!("wscut(eV)  nfreq   iters  time(s)   Z(0)          Delta(0) meV    chi(0) meV     |dDelta|/Delta vs IR");
    for wscut in [0.5, 2.0, 8.0, 32.0] {
        let start = std::time::Instant::now();
        let n = frequencies_below(wscut, t);
        let mut uniform = UniformSum::new(tenferro_cpu::CpuBackend::new(), t, n, kernel)?;
        let sol = solve_isotropic_fbw(
            &mut tenferro_cpu::CpuBackend::new(),
            &mut uniform,
            &dos,
            &params,
        )?;
        println!(
            "{wscut:8.1} {n:7} {:7} {:8.2}   {:.10}  {:.10}  {:.10}   {:.2e}",
            sol.iterations,
            start.elapsed().as_secs_f64(),
            sol.z[0],
            sol.delta[0] * 1e3,
            sol.shift[0] * 1e3,
            (sol.delta[0] - sol_ir.delta[0]).abs() / sol_ir.delta[0]
        );
    }
    Ok(())
}
