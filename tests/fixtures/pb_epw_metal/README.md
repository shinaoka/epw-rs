# Fixture: Pb (`epw_metal`)

Wannier-representation data written by EPW for fcc Pb, plus EPW's own results computed from exactly that data. The Rust downstream reads `wannier/` and must reproduce `reference/epw_read/`.

## Origin and authors

All QE/EPW-derived files here come from **Quantum ESPRESSO 7.6** (tag `qe-7.6`, commit `9f93ddec427d2b9a45bb72d828c6d324f62fcabd`, https://gitlab.com/QEF/q-e), which ships **EPW v6.1**. Quantum ESPRESSO and EPW are distributed under the GNU General Public License (v2 or later). A copy of QE's `License` file is included here as `LICENSE-QE`. Copyright for the inputs and for the programs that generated the outputs stays with the original authors listed below.

- **Test case `test-suite/epw_metal`** (inputs and the phonon data in `save/` used to generate the fixture). Commit authors in the QE history: Samuel Poncé (`scf_epw.in`, `nscf_epw.in`, `save/`, `epw1.in`), Felix Goudreault, Hyungjun Lee, and `zx199323 <xiaozha@umich.edu>` (`epw1.in`). The EPW collaboration is described at https://epw-code.org.
- **Pseudopotential `pseudo/pb_s.UPF`.** Generated with the `atomic` code by A. Dal Corso (Quantum ESPRESSO distribution); added to the QE tree by Samuel Poncé. It is not copied here; `regenerate.sh` takes it from the QE tree.
- **Programs.** `pw.x` (Quantum ESPRESSO group) and `epw.x` (EPW developers; see `EPW.bib` written by EPW, and cite it as the EPW output requests).

| File | Status |
|---|---|
| `qe_inputs/scf_epw.in`, `nscf_epw.in`, `epw1.in` | Verbatim copies of `test-suite/epw_metal/*` (byte-identical). |
| `qe_inputs/epw_read.in` | **Modified** from `epw1.in` by us: `epbwrite`→`.false.`, `epwwrite`→`.false.`, `epwread`→`.true.`, `wannierize`→`.false.`, the `wdata(*)` band-plot lines removed. Nothing else changed. |
| `wannier/*` | Written by `epw.x` while running `epw1.in`. |
| `reference/epw1/*` | Outputs of `pw.x` and `epw.x` for the three upstream inputs, plus the Wannier90 files `pb.win` and `pb.wout` (for gauge provenance). |
| `reference/epw_read/*` | Outputs of `epw.x` for `epw_read.in`, run in a directory that contained **only** `wannier/*`. |

The `scf_epw.in` that was actually run had `pseudo_dir` changed to an absolute path, so `reference/epw1/scf_epw.out` prints that path. Otherwise the run is identical to the verbatim copy.

## How it was generated (2026-10-15)

- Build: QE 7.6 CMake build with gfortran 13.3.0, OpenMPI (Ubuntu 24.04), system OpenBLAS, `-DQE_FFTW_VENDOR=Internal`, no OpenMP. Targets: `pw ph epw pp`.
- All runs used `mpirun -np 1 ... -npool 1` and `OMP_NUM_THREADS=1`.
- Steps: `scf_epw.in` → `nscf_epw.in` → `epw1.in` (in a copy of `test-suite/epw_metal`), then `epw_read.in` in a clean directory. `regenerate.sh` repeats all of this.

## Checks done

- `epw1.out` matches the upstream benchmark `benchmark.out.git.inp=epw1.in.args=3`: E_F(fine) = 11.755998 eV, λ = 0.1608789, λ_tr = 0.1349579, and identical per-q `lambda_tr` lines.
- The `epw_read` pass reproduces `epw1` exactly. Every result file (`lambda.phself`, `linewidth.phself`/`elself`, `pb.a2f*`, `pb.phdos*`, `pb.res*`, `selecq.fmt`) is byte-identical, and `logavg` = 0.0004572, `l_a2f` = 0.1608847. So `wannier/` is a complete input set for this downstream calculation.
- `pb.epmatwp` is 1,420,032 bytes = 16 × nbndsub² × nrr_k × nmodes × nrr_g = 16 × 4² × 43 × 3 × 43. That means raw complex128 with no Fortran record markers. The second line of `epwdata.fmt` is `4 43 3 43 43`. The field order is `nbndsub nrr_k nmodes nrr_q nrr_g` (from the EPW 6.1 reader `epw_read`). nrr_k, nrr_q and nrr_g are all 43 here, so this fixture alone would not catch a swap. The order is `(nbnd, nbnd, nrr_k, nmodes, nrr_g)` column-major, with R_p outermost. The Rust reader relies on this, and the agreement with EPW confirms it.

## Caveats

These settings are a regression test, not physics. The 3³ coarse and 6³ fine grids give λ ≈ 0.16 (Pb is ≈ 1.2 experimentally). The Fermi-level minimisation falls back to bisection, and E_F jumps from 9.80 eV (coarse) to 11.76 eV (fine). Use this fixture to check bit-level agreement with EPW, not to check physical values.
