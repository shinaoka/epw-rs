//! Wannier-representation boundary data for downstream e-ph calculations.
//!
//! Everything here is plain contiguous numeric arrays (column-major) and
//! scalars, independent of any file format, so the data can be broadcast to
//! other ranks unchanged. Readers live in separate crates (`eph-io-epw`).
//!
//! The e-ph vertex g(R_e, R_p) is not stored as an array but accessed through
//! [`EphMatrixSource`], block by block over R_p, which is the outermost index.
//!
//! Units: energies in Ry, lengths in units of `alat` (Bohr for `alat`), masses
//! in Ry mass units (m_e / 2).
//!
//! # Examples
//!
//! ```
//! use eph_wannier_data::{EphMatrixSource, InMemoryEph, EphDims};
//! use num_complex::Complex64;
//! let dims = EphDims { nbnd: 1, nrr_k: 1, nmodes: 1, nrr_g: 2 };
//! let g = InMemoryEph::new(dims, vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)])?;
//! let mut block = vec![Complex64::default(); 1];
//! g.read_rp_block(1..2, &mut block)?;
//! assert_eq!(block[0].re, 2.0);
//! # Ok::<(), eph_wannier_data::Error>(())
//! ```

use num_complex::Complex64;
use std::ops::Range;

/// Validation errors for boundary data.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An array does not have the length implied by its dimensions.
    #[error("{what}: expected {expected} elements, got {got}")]
    Length {
        /// Name of the array.
        what: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// A requested R_p block is out of range.
    #[error("R_p block {start}..{end} is outside 0..{nrr_g}")]
    BlockRange {
        /// Block start.
        start: usize,
        /// Block end.
        end: usize,
        /// Number of R_p vectors.
        nrr_g: usize,
    },
    /// Reading a block from a backing store failed.
    #[error("reading e-ph block failed")]
    Io(#[source] std::io::Error),
}

/// Crystal structure and electron count.
#[derive(Debug, Clone, PartialEq)]
pub struct Crystal {
    /// Lattice vectors `a_i` as columns, in units of `alat`.
    pub at: [[f64; 3]; 3],
    /// Reciprocal vectors `b_i` as columns, in units of `2 pi / alat`.
    pub bg: [[f64; 3]; 3],
    /// Unit-cell volume (Bohr^3).
    pub omega: f64,
    /// Lattice parameter (Bohr).
    pub alat: f64,
    /// Cartesian atomic positions in units of `alat`.
    pub tau: Vec<[f64; 3]>,
    /// Atomic type of each atom (0-based).
    pub ityp: Vec<usize>,
    /// Mass of each atomic type (Ry mass units).
    pub type_mass: Vec<f64>,
    /// Number of valence electrons in the DFT calculation.
    pub nelec: f64,
    /// Number of low-lying bands excluded from the Wannier manifold.
    pub nbndskip: usize,
    /// Non-collinear (spinor) calculation.
    pub noncolin: bool,
}

impl Crystal {
    /// Number of atoms.
    ///
    /// # Examples
    ///
    /// ```
    /// # let c = eph_wannier_data::Crystal { at: [[0.0; 3]; 3], bg: [[0.0; 3]; 3], omega: 1.0, alat: 1.0,
    /// #   tau: vec![[0.0; 3]], ityp: vec![0], type_mass: vec![1.0], nelec: 1.0, nbndskip: 0, noncolin: false };
    /// assert_eq!(c.nat(), 1);
    /// assert_eq!(c.nmodes(), 3);
    /// ```
    pub fn nat(&self) -> usize {
        self.tau.len()
    }

    /// Number of phonon modes (3 x atoms).
    ///
    /// # Examples
    ///
    /// ```
    /// # let c = eph_wannier_data::Crystal { at: [[0.0; 3]; 3], bg: [[0.0; 3]; 3], omega: 1.0, alat: 1.0,
    /// #   tau: vec![[0.0; 3]; 2], ityp: vec![0, 0], type_mass: vec![1.0], nelec: 1.0, nbndskip: 0, noncolin: false };
    /// assert_eq!(c.nmodes(), 6);
    /// ```
    pub fn nmodes(&self) -> usize {
        3 * self.nat()
    }

    /// Mass of atom `ia` (Ry mass units).
    ///
    /// # Examples
    ///
    /// ```
    /// # let c = eph_wannier_data::Crystal { at: [[0.0; 3]; 3], bg: [[0.0; 3]; 3], omega: 1.0, alat: 1.0,
    /// #   tau: vec![[0.0; 3]], ityp: vec![0], type_mass: vec![7.0], nelec: 1.0, nbndskip: 0, noncolin: false };
    /// assert_eq!(c.atom_mass(0), 7.0);
    /// ```
    pub fn atom_mass(&self, ia: usize) -> f64 {
        self.type_mass[self.ityp[ia]]
    }

    /// Electrons in the Wannier manifold (spin degeneracy 2 unless non-collinear).
    ///
    /// # Examples
    ///
    /// ```
    /// # let c = eph_wannier_data::Crystal { at: [[0.0; 3]; 3], bg: [[0.0; 3]; 3], omega: 1.0, alat: 1.0,
    /// #   tau: vec![[0.0; 3]], ityp: vec![0], type_mass: vec![1.0], nelec: 14.0, nbndskip: 5, noncolin: false };
    /// assert_eq!(c.wannier_electrons(), 4.0);
    /// ```
    pub fn wannier_electrons(&self) -> f64 {
        let per_band = if self.noncolin { 1.0 } else { 2.0 };
        self.nelec - per_band * self.nbndskip as f64
    }
}

/// Wigner–Seitz lattice vectors (crystal units) and their degeneracies.
#[derive(Debug, Clone, PartialEq)]
pub struct WsVectors {
    /// Lattice vectors.
    pub vectors: Vec<[i64; 3]>,
    /// Number of equivalent images of each vector.
    pub degeneracy: Vec<u32>,
}

impl WsVectors {
    /// Number of vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// let ws = eph_wannier_data::WsVectors { vectors: vec![[0, 0, 0]], degeneracy: vec![1] };
    /// assert_eq!(ws.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether there are no vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// let ws = eph_wannier_data::WsVectors { vectors: vec![], degeneracy: vec![] };
    /// assert!(ws.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// Electron Hamiltonian and interatomic force constants in the Wannier/real-space basis.
#[derive(Debug, Clone, PartialEq)]
pub struct WannierModel {
    /// Crystal structure.
    pub crystal: Crystal,
    /// Fermi energy of the coarse DFT calculation (Ry).
    pub fermi_energy_coarse: f64,
    /// Number of Wannier functions.
    pub nbnd: usize,
    /// WS vectors for the electrons.
    pub ws_k: WsVectors,
    /// WS vectors for the phonons.
    pub ws_q: WsVectors,
    /// WS vectors for the e-ph vertex (R_p).
    pub ws_g: WsVectors,
    /// H(R_e), shape `(nbnd, nbnd, nrr_k)`, column-major, not divided by degeneracies.
    pub hamiltonian: Vec<Complex64>,
    /// Force constants C(R), shape `(nmodes, nmodes, nrr_q)`, column-major (Ry/Bohr^2).
    pub force_constants: Vec<Complex64>,
}

impl WannierModel {
    /// Check that array lengths match the dimensions.
    ///
    /// # Errors
    ///
    /// [`Error::Length`] naming the inconsistent array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use eph_wannier_data::*;
    /// # let crystal = Crystal { at: [[0.0; 3]; 3], bg: [[0.0; 3]; 3], omega: 1.0, alat: 1.0,
    /// #   tau: vec![[0.0; 3]], ityp: vec![0], type_mass: vec![1.0], nelec: 1.0, nbndskip: 0, noncolin: false };
    /// let ws = WsVectors { vectors: vec![[0, 0, 0]], degeneracy: vec![1] };
    /// let model = WannierModel { crystal, fermi_energy_coarse: 0.0, nbnd: 1,
    ///     ws_k: ws.clone(), ws_q: ws.clone(), ws_g: ws,
    ///     hamiltonian: vec![Default::default(); 1], force_constants: vec![Default::default(); 9] };
    /// model.validate()?;
    /// # Ok::<(), eph_wannier_data::Error>(())
    /// ```
    pub fn validate(&self) -> Result<(), Error> {
        let nm = self.crystal.nmodes();
        check(
            "hamiltonian",
            self.nbnd * self.nbnd * self.ws_k.len(),
            self.hamiltonian.len(),
        )?;
        check(
            "force_constants",
            nm * nm * self.ws_q.len(),
            self.force_constants.len(),
        )?;
        for ws in [&self.ws_k, &self.ws_q, &self.ws_g] {
            check("ws degeneracy", ws.vectors.len(), ws.degeneracy.len())?;
        }
        Ok(())
    }
}

/// Dimensions of g(R_e, R_p): array `(nbnd, nbnd, nrr_k, nmodes, nrr_g)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EphDims {
    /// Wannier functions.
    pub nbnd: usize,
    /// Electron WS vectors R_e.
    pub nrr_k: usize,
    /// Phonon modes (Cartesian displacement basis).
    pub nmodes: usize,
    /// Vertex WS vectors R_p.
    pub nrr_g: usize,
}

impl EphDims {
    /// Elements per R_p block, `nbnd^2 * nrr_k * nmodes`.
    ///
    /// # Examples
    ///
    /// ```
    /// let d = eph_wannier_data::EphDims { nbnd: 4, nrr_k: 43, nmodes: 3, nrr_g: 43 };
    /// assert_eq!(d.block_len() * d.nrr_g * 16, 1_420_032);
    /// ```
    pub fn block_len(&self) -> usize {
        self.nbnd * self.nbnd * self.nrr_k * self.nmodes
    }
}

/// Access to g(R_e, R_p) by contiguous blocks of R_p.
pub trait EphMatrixSource: Sync {
    /// Dimensions of the vertex.
    fn dims(&self) -> EphDims;

    /// Copy g for `rp` into `out`, shape `(nbnd, nbnd, nrr_k, nmodes, rp.len())`.
    ///
    /// # Errors
    ///
    /// [`Error::BlockRange`] for an invalid range, [`Error::Length`] if `out`
    /// has the wrong size, [`Error::Io`] if a backing file cannot be read.
    fn read_rp_block(&self, rp: Range<usize>, out: &mut [Complex64]) -> Result<(), Error>;
}

/// g(R_e, R_p) held fully in memory.
#[derive(Debug, Clone)]
pub struct InMemoryEph {
    dims: EphDims,
    data: Vec<Complex64>,
}

impl InMemoryEph {
    /// Wrap a column-major array of shape `(nbnd, nbnd, nrr_k, nmodes, nrr_g)`.
    ///
    /// # Errors
    ///
    /// [`Error::Length`] if `data` does not match `dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use eph_wannier_data::{EphDims, InMemoryEph};
    /// let dims = EphDims { nbnd: 1, nrr_k: 1, nmodes: 1, nrr_g: 1 };
    /// assert!(InMemoryEph::new(dims, vec![]).is_err());
    /// ```
    pub fn new(dims: EphDims, data: Vec<Complex64>) -> Result<Self, Error> {
        check("e-ph vertex", dims.block_len() * dims.nrr_g, data.len())?;
        Ok(Self { dims, data })
    }
}

impl EphMatrixSource for InMemoryEph {
    fn dims(&self) -> EphDims {
        self.dims
    }

    fn read_rp_block(&self, rp: Range<usize>, out: &mut [Complex64]) -> Result<(), Error> {
        if rp.start > rp.end || rp.end > self.dims.nrr_g {
            return Err(Error::BlockRange {
                start: rp.start,
                end: rp.end,
                nrr_g: self.dims.nrr_g,
            });
        }
        let block = self.dims.block_len();
        check("e-ph block output", block * rp.len(), out.len())?;
        out.copy_from_slice(&self.data[block * rp.start..block * rp.end]);
        Ok(())
    }
}

fn check(what: &'static str, expected: usize, got: usize) -> Result<(), Error> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::Length {
            what,
            expected,
            got,
        })
    }
}
