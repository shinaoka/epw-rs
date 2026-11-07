# epw-rs

> A technical demonstration of verifying **tenferro-rs**' capabilities, and of porting a first-principles
> calculation through AI-agent development. EPW and Quantum ESPRESSO remain the reference implementations;
> every number here is validated against them.

epw-rs is an experimental Rust re-implementation of the EPW *downstream*: everything after the Wannier-representation data (`epwwrite`/`ephwrite`) has been written. It is meant to show the EPW developers a working, testable minimal example of what a modular downstream could look like.

The current scope is:

- **Wannier interpolation and phonon self-energy.** Fermi level, DOS, phonon linewidths, λ, α²F, ω_log and the Allen–Dynes T_c (Pb, `test-suite/epw_metal`).
- **Migdal–Eliashberg equations** (MgB₂, `test-suite/epw_super`):
  - isotropic FSR and FBW;
  - anisotropic FBW on the uniform Matsubara grid and on the IR basis (sparse-ir), with EPW's `eps_cut_ir` coefficient cut;
  - the Coulomb term of `icoulomb = 1`;
  - Padé continuation to the real axis.

## Agreement with EPW at a glance

Every row below is an automated test (`cargo test`) that reads EPW's own output files in `tests/fixtures/` and compares them with epw-rs. All rows were re-run and pass at the commit that added this table (2026-10-15).

- **Reference data.** QE 7.6 (EPW v6.1): gfortran 13.3, OpenBLAS, one MPI rank. For the Eliashberg runs, EPW was re-run with `conv_thr_iaxis = 1e-10` or tighter, so that the comparison is limited by the solution and not by the stopping criterion.
- **"Max rel. diff"** is max |epw-rs − EPW| / max |EPW| over all entries.
- **"Printed digits"** means that epw-rs agrees with every digit EPW writes.

### Pb: Wannier interpolation, linewidths, λ, α²F (`epw_metal`)

Test: `crates/eph-driver/tests/pb_epw_metal.rs`. The input is only EPW's Wannier-representation files (`crystal.fmt`, `epwdata.fmt`, `wigner.fmt`, `pb.epmatwp`).

| Quantity | EPW | epw-rs | Agreement |
|---|---|---|---|
| E_F (fine grid) | 11.755998 eV | 11.7559985 eV | printed digits |
| N_F (per spin) | 0.518460 /eV | 0.518460 /eV | printed digits |
| ω_qν (648 values) | `linewidth.phself` | | max abs. diff 1.7e-7 meV |
| γ_qν (648 values) | `linewidth.phself` | | max abs. diff 2.5e-9 meV (max γ = 0.085 meV) |
| λ_qν (648 values) | `lambda.phself` | | printed digits |
| α²F(ω) (500 ω × 10 smearings) | `pb.a2f.01` | | ≤ 5.0e-8 (EPW prints 7 decimals) |
| λ (sum over q) | 0.1608789 | 0.1608789 | printed digits |
| λ (2∫α²F/ω) | 0.1608847 | 0.1608847 | printed digits |
| ω_log | 0.0004572 Ry | 0.0004572 Ry | printed digits |

### MgB₂: isotropic Eliashberg (`epw_super`, from `MgB2.a2f` and `MgB2.dos`)

Test: `crates/eliashberg/tests/mgb2_isotropic.rs`. Z, Δ and χ are compared at every Matsubara frequency.

| Quantity | EPW | epw-rs | Agreement |
|---|---|---|---|
| McMillan T_c | 26.4143 K | 26.414261 K | printed digits |
| FSR Z(iω), Δ(iω) at 15 K and 20 K | `MgB2.imag_iso_*` | | max rel. diff 4.4e-11 |
| FBW Z(iω), Δ(iω), χ(iω) at 15 K and 20 K | `MgB2.imag_iso_*` | | max rel. diff 4.8e-11 |
| FSR free energy, 15 K | −0.018341 meV | −0.01834058 meV | printed digits |
| FBW free energy, 20 K | −0.011441 meV | −0.01144090 meV | printed digits |

### MgB₂: anisotropic FBW (`epw6`, `epw7`, `epw8` settings, 140 Fermi-surface states)

Test: `crates/eph-driver/tests/mgb2_anisotropic.rs`. The input is EPW's `MgB2.ephmat/` and `selecq.fmt`. Every Matsubara frequency of every state at 20 K and 30 K is compared.

| Run | Z | Δ | χ | Free energy (EPW / epw-rs) |
|---|---|---|---|---|
| Uniform grid (`epw6`) | 2.2e-10 | 4.9e-10 | 1.3e-10 | 20 K: −0.130522 / −0.13052154 meV<br>30 K: −0.113593 / −0.11359347 meV |
| IR basis, EPW's sampling points and `eps_cut_ir = 1e-5` (`epw7`) | 1.0e-8 | 3.5e-9 | 1.0e-9 | |
| IR basis, `eps_cut_ir = 0` | 3.4e-10 | 4.7e-10 | 4.9e-10 | |
| IR basis + Coulomb term, μ* = 0.20 (`epw8`) | 1.1e-10 | 2.8e-10 | 6.7e-11 | 20 K: −0.125777 / −0.12577704 meV<br>30 K: −0.109032 / −0.10903169 meV |

The Z, Δ and χ columns are max rel. diffs. In the Coulomb run, epw-rs builds the 1080 inside-window and 594 outside-window states from `nscf2supercond.x` output with EPW's rules. The outside-window gap comes out as −5.05 meV at 20 K.

### MgB₂: Padé continuation to the real axis

Gap edges (min / max over states) where Re Δ(ω) = ω, in meV:

| Matsubara points | T | EPW | epw-rs |
|---|---|---|---|
| IR basis (`epw7` + `lpade`) | 20 K | 2.70138 / 31.535633 | 2.701380 / 31.535633 |
| | 30 K | 2.711355 / 31.715349 | 2.711355 / 31.715349 |
| Uniform grid (`epw6` + `lpade`) | 20 K | 2.264043 / 30.762298 | 2.264044 / 30.760812 |
| | 30 K | 2.256066 / 30.866918 | 2.255980 / 30.826385 |

Padé continuation is severely ill-conditioned. A relative perturbation of 1e-10 on the Matsubara input moves the uniform-grid gap edges by up to 0.045 meV (20 K) and 0.59 meV (30 K). The same perturbation moves the IR-point gap edges by only 0.001–0.002 meV.

- **Uniform grid.** The real-axis values cannot be reproduced beyond this sensitivity by any implementation. The test therefore requires the difference from EPW to stay within 10× the measured sensitivity, and it does.
- **IR points.** With the same number of points, Padé is 30–250× more stable, and epw-rs matches all printed digits.

## Running

Rust 1.96 or later is required.

```
# Pb: writes linewidths.dat and a2f.dat, prints E_F, N_F, λ, ω_log and T_c
cargo run --release --bin epw-rs -- tests/fixtures/pb_epw_metal/wannier out/

# All comparisons with EPW (uniform-grid solvers only)
cargo test --release --workspace

# Including the IR-basis solvers (optional Cargo feature, off by default)
cargo test --release --workspace --features eph-driver/sparse-ir
cargo run --release -p eliashberg --features sparse-ir --example mgb2_ir_vs_uniform
```

`--ranks N` runs the Pb calculation on N simulated ranks (threads). The result is bit-identical for any N.

## Design

- **Clean-room implementation.** No EPW Fortran source is included or translated. The equations are implemented from the papers (cited at the top of each crate). File formats and numerical conventions were checked by reading the EPW 6.1 source and are matched exactly. Examples of such conventions are the Fermi-level search, the Fermi window, degenerate-mode averaging, the IR coefficient cut and the Padé recursion order.
- **Downstream-only tests.** The fixtures are EPW's Wannier-representation and `ephwrite` files, so the downstream can be tested without running DFT or DFPT.
- **Tensors.** All tensor operations use [tenferro-rs](https://github.com/tensor4all/tenferro-rs) (tensor4all, column-major): `tenferro-fft` for the Wannier Fourier sums, `tenferro-einsum` for contractions and `tenferro-linalg` for diagonalization. Results are bit-reproducible for a fixed thread count.
- **Crates.** `crates/` is split so that each piece can be reused and tested on its own:
  - `eph-io-epw`: EPW file formats.
  - `wannier-interp`: Wannier interpolation.
  - `eph-selfen`: linewidths, λ and α²F.
  - `eliashberg`: Eliashberg solvers, IR sums and Padé. This crate does not know EPW's file formats.
  - `eph-driver`: maps EPW data onto the solvers.
  - `eph-comm`: distribution.
- **IR basis.** IR sums use [sparse-ir](https://crates.io/crates/sparse-ir) 0.13. The sampling points can be pinned to EPW's IR object file, so that results do not depend on the sparse-ir version. τ = 0⁺ is evaluated exactly from u_l(0⁺).

## Documentation

The design notes and work logs are in Japanese.

| Document | Contents |
|---|---|
| [docs/epw_analysis_ja.md](docs/epw_analysis_ja.md), [docs/epw_structure.svg](docs/epw_structure.svg) | Structure of EPW and how deeply it is coupled to QE. |
| [docs/crate_design_ja.md](docs/crate_design_ja.md) | Crate split of the downstream and the design for adding MPI later. |
| [docs/minimal_example_plan_ja.md](docs/minimal_example_plan_ja.md) | Plan for the downstream-only minimal example. |
| [docs/pb_reproduction_ja.md](docs/pb_reproduction_ja.md) | Pb reproduction: design decisions and the comparison with EPW. |
| [docs/sparse_ir_analysis_ja.md](docs/sparse_ir_analysis_ja.md) | Analysis of EPW's sparse-ir implementation. |
| [docs/eliashberg_ir_ja.md](docs/eliashberg_ir_ja.md) | Eliashberg solvers, IR basis, `eps_cut_ir`, Coulomb term and Padé: work log and result tables. |
| [docs/upstream_findings_ja.md](docs/upstream_findings_ja.md) | Possible issues found in EPW, tenferro-rs and sparse-ir while doing this work (not reported yet). |
| [tests/fixtures/pb_epw_metal/README.md](tests/fixtures/pb_epw_metal/README.md), [tests/fixtures/mgb2_epw_super/README.md](tests/fixtures/mgb2_epw_super/README.md) | Provenance of the fixtures, which inputs we modified, and how to regenerate them. |

## Attribution

- **Programs and test cases.** The fixtures were generated with Quantum ESPRESSO 7.6 and EPW v6.1 (https://gitlab.com/QEF/q-e, https://epw-code.org), from QE's `test-suite/epw_metal` and `test-suite/epw_super` test cases.
- **Copyright.** Copyright of those programs, inputs and outputs stays with their original authors: the Quantum ESPRESSO group and the EPW developers.
- **Licence.** QE and EPW are distributed under the GNU GPL (v2 or later). A copy of the licence is in `tests/fixtures/pb_epw_metal/LICENSE-QE`.
- **Details.** The fixture READMEs list the authors of each test case and say which inputs we modified.

## License

GNU General Public License, version 2 — the same licence as EPW, carried over because the
design, the algorithms and the reference data are EPW's. See [LICENSE](LICENSE) and
[AUTHORS.md](AUTHORS.md) for the copyright and the papers to cite.
