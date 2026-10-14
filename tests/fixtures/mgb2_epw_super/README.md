# Fixture: MgB2 (`epw_super`), isotropic Eliashberg

Inputs and EPW reference solutions for the isotropic Migdal–Eliashberg equations of MgB2. The Rust solvers (`crates/eliashberg`) read `iso/MgB2.a2f` and `iso/MgB2.dos` and must reproduce `iso/fsr/*` and `iso/fbw/*`.

## Origin and authors

Generated with **Quantum ESPRESSO 7.6** (tag `qe-7.6`, commit `9f93ddec427d2b9a45bb72d828c6d324f62fcabd`), which ships **EPW v6.1**. Quantum ESPRESSO and EPW are GPL (v2 or later); a copy of the license is in `../pb_epw_metal/LICENSE-QE`. Copyright of the inputs and of the programs stays with their authors.

- **Test case `test-suite/epw_super`.** Commit authors in the QE history: Samuel Poncé (`scf.in`, `ph.in`, `epw1.in`), Hyungjun Lee, Hari (GitHub user name as recorded) and Felix Goudreault (`epw1.in`).
- **Pseudopotentials** `Mg.pz-n-vbc.UPF` and `B.pz-vbc.UPF`, downloaded from https://pseudopotentials.quantum-espresso.org/upf_files/. They are not copied here.
- **Programs.** `pw.x`, `ph.x` (Quantum ESPRESSO group) and `epw.x` (EPW developers).

| File | Status |
|---|---|
| `iso/MgB2.a2f`, `iso/MgB2.dos`, `iso/crystal.fmt` | Written by `epw.x` running the unmodified `epw_super/epw1.in` after `scf`, `ph`, `scf_epw` and `nscf_epw` (with `pp.py`). |
| `iso/fsr/iso_fsr.in`, `iso/fbw/iso_fbw.in` | **Written by us.** They take the Eliashberg settings of `epw1.in` (`muc = 0.16`, `wscut = 0.5` eV) and read `MgB2.a2f` (and `MgB2.dos`) through `fila2f`. They solve the FSR or FBW equations at 15 and 20 K with `conv_thr_iaxis = 1e-12`. |
| `iso/fsr/*`, `iso/fbw/*` (other files) | Output of `epw.x` for those inputs: standard output and `MgB2.imag_iso_<T>` (omega, Z, Delta and, for FBW, chi). |

## How it was generated (2026-10-15)

- Same build as `../pb_epw_metal`: gfortran 13.3, OpenMPI, OpenBLAS, internal FFTW.
- `pw.x` and `ph.x` ran on 8 MPI ranks (`-npool 8`). `epw.x` ran on 1 rank.
- For the `fila2f` runs, EPW still needs `crystal.fmt` and the k and q meshes in the input. Without the meshes, `supercond.f90:104` evaluates `MOD(nkf1, nqf1)` before testing `fila2f` (Fortran does not short-circuit) and stops with an integer division by zero.
- The McMillan T_c printed by this build is 26.4143 K; the upstream benchmark prints 26.4082 K. The difference comes from the DFT/DFPT stage on a different platform. The Rust code is compared with the outputs in this directory.

## Regeneration

There is no script yet; the steps are manual:

1. Copy `test-suite/epw_super` of QE 7.6. Set `pseudo_dir` to a directory holding the two pseudopotentials.
2. Run `scf.in`, `ph.in`, `python3 EPW/bin/pp.py < pp.in`, `scf_epw.in`, `nscf_epw.in` and `epw1.in`, in that order.
3. Copy `MgB2.a2f`, `MgB2.dos` and `crystal.fmt` to a new directory, together with `iso/fsr/iso_fsr.in` or `iso/fbw/iso_fbw.in`, and run `epw.x` there.

## Checks done

- FSR and FBW: Z, Delta (and chi) agree with EPW to a relative 5e-11 at every Matsubara frequency, for both temperatures (`crates/eliashberg/tests/mgb2_isotropic.rs`). The FSR free energy and the McMillan T_c agree to the printed digits.
- EPW's FBW free energy matches the FSR Bardeen–Stephen expression with the normal-state Z_N **without** the factor pi of the FSR Z_N. The test records this convention.
