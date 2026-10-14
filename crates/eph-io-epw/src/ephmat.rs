//! Readers for the Fermi-surface e-ph data written by EPW with `ephwrite = .true.`
//! (`<prefix>.ephmat/` and `selecq.fmt`), used by the anisotropic Eliashberg solvers.
//!
//! All binary files are Fortran stream files of native-endian 4-byte integers
//! and 8-byte reals (EPW 6.1, `io_supercond.f90`):
//!
//! - `freq`: `nqtot, nq1, nq2, nq3, nmodes`, then per q its crystal
//!   coordinates and `nmodes` frequencies (Ry).
//! - `egnv`: `nktot, nk1, nk2, nk3, nkfs`, `nbnd`, `ef, ef0, dosef, degaussw,
//!   fsthick` (Ry, dosef per spin per Ry), then per irreducible k: weight,
//!   crystal coordinates, its index, the Fermi-surface index (0 if outside),
//!   and `nbnd` energies (Ry).
//! - `ikmap`: full-grid -> Fermi-surface index, then full-grid -> irreducible index.
//! - `npool`, `ephmat1..N`: per file `pool id, nks`, then for every selected q,
//!   every k of the file and every (k, k+q) pair with bands in the window:
//!   `|g|^2 / (2 omega)` (Ry) for each mode, band at k and band at k+q.
//!
//! Energies are converted to eV here, as EPW does after reading.

use crate::Error;
use phys_units::RY_TO_EV;
use std::path::Path;

/// One (k, q) block of `|g|^2 / (2 omega)` between the window bands at k and k+q.
#[derive(Debug, Clone, PartialEq)]
pub struct EphBlock {
    /// Fermi-surface index of k (0-based).
    pub k: usize,
    /// Full-grid q index (0-based).
    pub q: usize,
    /// Fermi-surface index of k+q (0-based).
    pub kq: usize,
    /// `g2[(mode * nb_k + i) * nb_kq + j]` in eV^2 (zero for modes at or below `eps_acoustic`),
    /// with `i`, `j` running over the window bands of k and k+q.
    pub g2: Vec<f64>,
}

/// Contents of `<prefix>.ephmat/` (energies in eV).
#[derive(Debug, Clone, PartialEq)]
pub struct EphmatData {
    /// Phonon frequencies `(nmodes, nq)` in eV.
    pub frequencies: Vec<f64>,
    /// Number of modes.
    pub nmodes: usize,
    /// q points (crystal coordinates).
    pub q_points: Vec<[f64; 3]>,
    /// k mesh.
    pub k_mesh: [usize; 3],
    /// Fermi energy (eV).
    pub ef0: f64,
    /// DOS at E_F per spin (1/eV).
    pub dos_at_fermi: f64,
    /// Electron smearing (eV).
    pub degaussw: f64,
    /// Fermi window half-width (eV).
    pub fsthick: f64,
    /// Bands per k in the files.
    pub nbnd: usize,
    /// Fermi-surface k points: weights (summing to 2, spin included).
    pub k_weights: Vec<f64>,
    /// Fermi-surface k points: crystal coordinates.
    pub k_points: Vec<[f64; 3]>,
    /// Energies `(nbnd, nkfs)` in eV.
    pub energies: Vec<f64>,
    /// Window bands (0-based) of each Fermi-surface k.
    pub window_bands: Vec<Vec<usize>>,
    /// e-ph blocks in file order.
    pub blocks: Vec<EphBlock>,
}

struct Stream<'a> {
    path: &'a Path,
    bytes: Vec<u8>,
    pos: usize,
}

impl<'a> Stream<'a> {
    fn open(path: &'a Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            path,
            bytes,
            pos: 0,
        })
    }

    fn take<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], Error> {
        let end = self.pos + N;
        let chunk = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| Error::UnexpectedEnd {
                path: self.path.to_path_buf(),
                what,
            })?;
        self.pos = end;
        Ok(chunk.try_into().expect("slice of length N"))
    }

    fn i32(&mut self, what: &'static str) -> Result<i32, Error> {
        Ok(i32::from_ne_bytes(self.take::<4>(what)?))
    }

    fn usize(&mut self, what: &'static str) -> Result<usize, Error> {
        let v = self.i32(what)?;
        usize::try_from(v).map_err(|_| Error::Parse {
            path: self.path.to_path_buf(),
            token: v.to_string(),
            what,
        })
    }

    fn f64(&mut self, what: &'static str) -> Result<f64, Error> {
        Ok(f64::from_ne_bytes(self.take::<8>(what)?))
    }

    fn vec3(&mut self, what: &'static str) -> Result<[f64; 3], Error> {
        Ok([self.f64(what)?, self.f64(what)?, self.f64(what)?])
    }

    fn finished(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

/// Index of `k + q` on the full k mesh (EPW `kpmq_map` with `backtoBZ`).
fn shifted_index(k: [f64; 3], q: [f64; 3], mesh: [usize; 3]) -> usize {
    let c: [usize; 3] = std::array::from_fn(|a| {
        let n = mesh[a] as i64;
        (((k[a] + q[a]) * n as f64).round() as i64).rem_euclid(n) as usize
    });
    c[2] + mesh[2] * (c[1] + mesh[1] * c[0])
}

/// Read `dir/<prefix>.ephmat/` and `dir/selecq.fmt`.
///
/// Couplings of modes with frequency `<= eps_acoustic` (eV) are set to zero, as in EPW.
///
/// # Errors
///
/// [`Error::Io`], [`Error::UnexpectedEnd`] or [`Error::Parse`] for missing or
/// malformed files, [`Error::Unsupported`] if a file has trailing data.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/mgb2_epw_super/aniso");
/// let eps_acoustic = phys_units::cmm1_to_ry(5.0) * phys_units::RY_TO_EV;
/// let data = eph_io_epw::ephmat::read_ephmat(&dir, "MgB2", eps_acoustic)?;
/// assert_eq!(data.k_points.len(), 28);
/// assert_eq!(data.blocks.len(), 216 * 28);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_ephmat(dir: &Path, prefix: &str, eps_acoustic: f64) -> Result<EphmatData, Error> {
    let ephdir = dir.join(format!("{prefix}.ephmat"));

    let freq_path = ephdir.join("freq");
    let mut s = Stream::open(&freq_path)?;
    let nq = s.usize("nqtot")?;
    let _q_mesh = [s.usize("nq1")?, s.usize("nq2")?, s.usize("nq3")?];
    let nmodes = s.usize("nmodes")?;
    let mut q_points = Vec::with_capacity(nq);
    let mut frequencies = Vec::with_capacity(nq * nmodes);
    for _ in 0..nq {
        q_points.push(s.vec3("q point")?);
        for _ in 0..nmodes {
            frequencies.push(s.f64("frequency")? * RY_TO_EV);
        }
    }

    let egnv_path = ephdir.join("egnv");
    let mut s = Stream::open(&egnv_path)?;
    let nktot = s.usize("nktot")?;
    let k_mesh = [s.usize("nk1")?, s.usize("nk2")?, s.usize("nk3")?];
    let nkfs = s.usize("nkfs")?;
    let nbnd = s.usize("nbnd")?;
    let _ef = s.f64("ef")?;
    let ef0 = s.f64("ef0")? * RY_TO_EV;
    let dos_at_fermi = s.f64("dosef")? / RY_TO_EV;
    let degaussw = s.f64("degaussw")? * RY_TO_EV;
    let fsthick = s.f64("fsthick")? * RY_TO_EV;
    let mut fs_of_irr = vec![0usize; nktot];
    let mut k_weights = vec![0.0; nkfs];
    let mut k_points = vec![[0.0; 3]; nkfs];
    let mut energies = vec![0.0; nbnd * nkfs];
    for slot in fs_of_irr.iter_mut() {
        let w = s.f64("k weight")?;
        let x = s.vec3("k point")?;
        let _ik = s.i32("k index")?;
        let fs = s.usize("fermi-surface index")?;
        *slot = fs;
        let e: Vec<f64> = (0..nbnd)
            .map(|_| s.f64("energy"))
            .collect::<Result<_, _>>()?;
        if fs > 0 {
            k_weights[fs - 1] = w;
            k_points[fs - 1] = x;
            for (b, &en) in e.iter().enumerate() {
                energies[b + nbnd * (fs - 1)] = en * RY_TO_EV;
            }
        }
    }
    let window_bands: Vec<Vec<usize>> = (0..nkfs)
        .map(|k| {
            (0..nbnd)
                .filter(|&b| (energies[b + nbnd * k] - ef0).abs() < fsthick)
                .collect()
        })
        .collect();

    let ikmap_path = ephdir.join("ikmap");
    let mut s = Stream::open(&ikmap_path)?;
    let nfull: usize = k_mesh.iter().product();
    let _ixkff: Vec<i32> = (0..nfull)
        .map(|_| s.i32("ixkff"))
        .collect::<Result<_, _>>()?;
    let bztoibz: Vec<usize> = (0..nfull)
        .map(|_| s.usize("bztoibz"))
        .collect::<Result<_, _>>()?;

    let selecq = std::fs::read_to_string(dir.join("selecq.fmt")).map_err(|source| Error::Io {
        path: dir.join("selecq.fmt"),
        source,
    })?;
    let numbers: Vec<usize> = selecq
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let totq = numbers.first().copied().unwrap_or(0);
    let selected: Vec<usize> = numbers.iter().skip(2).take(totq).map(|q| q - 1).collect();

    let npool_path = ephdir.join("npool");
    let npool = Stream::open(&npool_path)?.usize("npool")?;
    let mut blocks = Vec::new();
    let mut k_offset = 0;
    for pool in 1..=npool {
        let path = ephdir.join(format!("ephmat{pool}"));
        let mut s = Stream::open(&path)?;
        let _pool_id = s.i32("pool id")?;
        let nks = s.usize("nks")?;
        for &q in &selected {
            for k in k_offset..k_offset + nks {
                let full = shifted_index(k_points[k], q_points[q], k_mesh);
                let fs = fs_of_irr[bztoibz[full] - 1];
                if fs == 0 || window_bands[k].is_empty() || window_bands[fs - 1].is_empty() {
                    continue;
                }
                let kq = fs - 1;
                let (nb_k, nb_kq) = (window_bands[k].len(), window_bands[kq].len());
                let mut g2 = Vec::with_capacity(nmodes * nb_k * nb_kq);
                for mode in 0..nmodes {
                    let keep = frequencies[mode + nmodes * q] > eps_acoustic;
                    for _ in 0..nb_k * nb_kq {
                        let v = s.f64("g2")? * RY_TO_EV * RY_TO_EV;
                        g2.push(if keep { v } else { 0.0 });
                    }
                }
                blocks.push(EphBlock { k, q, kq, g2 });
            }
        }
        if !s.finished() {
            return Err(Error::Unsupported {
                path,
                detail: "trailing data after the expected e-ph blocks".into(),
            });
        }
        k_offset += nks;
    }

    Ok(EphmatData {
        frequencies,
        nmodes,
        q_points,
        k_mesh,
        ef0,
        dos_at_fermi,
        degaussw,
        fsthick,
        nbnd,
        k_weights,
        k_points,
        energies,
        window_bands,
        blocks,
    })
}
