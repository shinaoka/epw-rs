//! Readers for the Wannier-representation files written by EPW (`epwwrite = .true.`).
//!
//! - `crystal.fmt`: structure, masses, electron count.
//! - `epwdata.fmt`: coarse E_F, dimensions, H(R_e), force constants.
//! - `wigner.fmt`: WS vectors and degeneracies for electrons, phonons and the vertex.
//! - `<prefix>.epmatwp`: g(R_e, R_p) as raw native-endian complex128 values,
//!   column-major `(nbnd, nbnd, nrr_k, nmodes, nrr_g)`, no record markers.
//!
//! The text files are Fortran list-directed output; complex numbers appear as
//! `(re,im)`. The field order follows the EPW 6.1 writer (QE 7.6). Only the
//! `use_ws = .false.` layout of `wigner.fmt` (one degeneracy per vector) is
//! supported.
//!
//! # Examples
//!
//! ```
//! let tokens = eph_io_epw::tokenize(" 3 (1.5,-2.0E-1) T\n 4.0D0");
//! assert_eq!(tokens.len(), 4);
//! ```

use eph_wannier_data::{Crystal, EphDims, EphMatrixSource, InMemoryEph, WannierModel, WsVectors};
use num_complex::Complex64;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

/// Number of type slots QE writes for `amass` (`ntypx`).
const QE_NTYPX: usize = 10;

/// Errors while reading EPW files.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file could not be opened or read.
    #[error("cannot read {path}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The file ended before all expected values were read.
    #[error("{path}: unexpected end of data while reading {what}")]
    UnexpectedEnd {
        /// File path.
        path: PathBuf,
        /// Field being read.
        what: &'static str,
    },
    /// A token could not be parsed.
    #[error("{path}: cannot parse {token:?} as {what}")]
    Parse {
        /// File path.
        path: PathBuf,
        /// Offending token.
        token: String,
        /// Expected kind.
        what: &'static str,
    },
    /// The file uses a layout this reader does not implement.
    #[error("{path}: {detail}")]
    Unsupported {
        /// File path.
        path: PathBuf,
        /// What is unsupported.
        detail: String,
    },
    /// Sizes in different files disagree or the data is inconsistent.
    #[error("inconsistent data")]
    Data(#[from] eph_wannier_data::Error),
}

/// One list-directed token.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A plain value (number or logical).
    Scalar(String),
    /// A parenthesised complex `(re,im)`.
    Complex(String, String),
}

/// Split Fortran list-directed text into tokens.
///
/// # Examples
///
/// ```
/// use eph_io_epw::{tokenize, Token};
/// assert_eq!(tokenize("(1,2)")[0], Token::Complex("1".into(), "2".into()));
/// ```
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(|c: char| !c.is_whitespace() && c != ',') {
        rest = &rest[start..];
        if let Some(body) = rest.strip_prefix('(') {
            let end = body.find(')').unwrap_or(body.len());
            let mut parts = body[..end].splitn(2, ',');
            let re = parts.next().unwrap_or("").trim().to_string();
            let im = parts.next().unwrap_or("").trim().to_string();
            tokens.push(Token::Complex(re, im));
            rest = body.get(end + 1..).unwrap_or("");
        } else {
            let end = rest
                .find(|c: char| c.is_whitespace() || c == ',')
                .unwrap_or(rest.len());
            tokens.push(Token::Scalar(rest[..end].to_string()));
            rest = &rest[end..];
        }
    }
    tokens
}

struct Reader {
    path: PathBuf,
    tokens: std::vec::IntoIter<Token>,
}

impl Reader {
    fn open(path: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            tokens: tokenize(&text).into_iter(),
        })
    }

    fn next(&mut self, what: &'static str) -> Result<Token, Error> {
        self.tokens.next().ok_or_else(|| Error::UnexpectedEnd {
            path: self.path.clone(),
            what,
        })
    }

    fn parse_error(&self, token: impl Into<String>, what: &'static str) -> Error {
        Error::Parse {
            path: self.path.clone(),
            token: token.into(),
            what,
        }
    }

    fn scalar(&mut self, what: &'static str) -> Result<String, Error> {
        match self.next(what)? {
            Token::Scalar(s) => Ok(s),
            Token::Complex(re, im) => Err(self.parse_error(format!("({re},{im})"), what)),
        }
    }

    fn f64(&mut self, what: &'static str) -> Result<f64, Error> {
        let s = self.scalar(what)?;
        parse_real(&s).ok_or_else(|| self.parse_error(s, what))
    }

    fn int(&mut self, what: &'static str) -> Result<i64, Error> {
        let s = self.scalar(what)?;
        s.parse().map_err(|_| self.parse_error(s, what))
    }

    fn usize(&mut self, what: &'static str) -> Result<usize, Error> {
        let v = self.int(what)?;
        usize::try_from(v).map_err(|_| self.parse_error(v.to_string(), what))
    }

    fn logical(&mut self, what: &'static str) -> Result<bool, Error> {
        let s = self.scalar(what)?;
        match s.trim_start_matches('.').chars().next() {
            Some('T' | 't') => Ok(true),
            Some('F' | 'f') => Ok(false),
            _ => Err(self.parse_error(s, what)),
        }
    }

    fn complex(&mut self, what: &'static str) -> Result<Complex64, Error> {
        match self.next(what)? {
            Token::Complex(re, im) => match (parse_real(&re), parse_real(&im)) {
                (Some(re), Some(im)) => Ok(Complex64::new(re, im)),
                _ => Err(self.parse_error(format!("({re},{im})"), what)),
            },
            Token::Scalar(s) => Err(self.parse_error(s, what)),
        }
    }

    fn f64_vec(&mut self, n: usize, what: &'static str) -> Result<Vec<f64>, Error> {
        (0..n).map(|_| self.f64(what)).collect()
    }

    fn vec3s(&mut self, n: usize, what: &'static str) -> Result<Vec<[f64; 3]>, Error> {
        let flat = self.f64_vec(3 * n, what)?;
        Ok(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
    }
}

fn parse_real(s: &str) -> Option<f64> {
    s.replace(['D', 'd'], "E").parse().ok()
}

fn mat3(columns: Vec<[f64; 3]>) -> [[f64; 3]; 3] {
    [columns[0], columns[1], columns[2]]
}

/// Read `crystal.fmt`.
///
/// # Errors
///
/// [`Error::Io`], [`Error::UnexpectedEnd`] or [`Error::Parse`] for unreadable
/// or malformed files.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/pb_epw_metal/wannier");
/// let crystal = eph_io_epw::read_crystal(&dir.join("crystal.fmt"))?;
/// assert_eq!(crystal.nat(), 1);
/// assert_eq!(crystal.wannier_electrons(), 4.0);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_crystal(path: &Path) -> Result<Crystal, Error> {
    let mut r = Reader::open(path)?;
    let nat = r.usize("nat")?;
    let _nmodes = r.usize("nmodes")?;
    let nelec = r.f64("nelec")?;
    let nbndskip = r.usize("nbndskip")?;
    let at = mat3(r.vec3s(3, "at")?);
    let bg = mat3(r.vec3s(3, "bg")?);
    let omega = r.f64("omega")?;
    let alat = r.f64("alat")?;
    let tau = r.vec3s(nat, "tau")?;
    let type_mass = r.f64_vec(QE_NTYPX, "amass")?;
    let ityp = (0..nat)
        .map(|_| r.usize("ityp").map(|t| t.saturating_sub(1)))
        .collect::<Result<Vec<_>, _>>()?;
    let noncolin = r.logical("noncolin")?;
    Ok(Crystal {
        at,
        bg,
        omega,
        alat,
        tau,
        ityp,
        type_mass,
        nelec,
        nbndskip,
        noncolin,
    })
}

/// Read `wigner.fmt` (electron, phonon and vertex WS sets).
///
/// # Errors
///
/// As [`read_crystal`]; [`Error::Unsupported`] for the per-orbital (`use_ws`) layout.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/pb_epw_metal/wannier");
/// let (k, q, g) = eph_io_epw::read_wigner(&dir.join("wigner.fmt"))?;
/// assert_eq!((k.len(), q.len(), g.len()), (43, 43, 43));
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_wigner(path: &Path) -> Result<(WsVectors, WsVectors, WsVectors), Error> {
    let mut r = Reader::open(path)?;
    let nrr = [r.usize("nrr_k")?, r.usize("nrr_q")?, r.usize("nrr_g")?];
    let (dims, dims2) = (r.usize("dims")?, r.usize("dims2")?);
    if dims != 1 || dims2 != 1 {
        return Err(Error::Unsupported {
            path: path.to_path_buf(),
            detail: format!("per-orbital WS degeneracies (dims = {dims}, dims2 = {dims2}); rerun EPW with use_ws = .false."),
        });
    }
    let mut read_set = |n: usize| -> Result<WsVectors, Error> {
        let mut vectors = Vec::with_capacity(n);
        let mut degeneracy = Vec::with_capacity(n);
        for _ in 0..n {
            vectors.push([r.int("irvec")?, r.int("irvec")?, r.int("irvec")?]);
            let _length = r.f64("wslen")?;
            let d = r.usize("ndegen")?;
            degeneracy.push(u32::try_from(d).map_err(|_| r.parse_error(d.to_string(), "ndegen"))?);
        }
        Ok(WsVectors {
            vectors,
            degeneracy,
        })
    };
    Ok((read_set(nrr[0])?, read_set(nrr[1])?, read_set(nrr[2])?))
}

/// Dimensions stored on line 2 of `epwdata.fmt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EpwdataDims {
    nbnd: usize,
    nrr_k: usize,
    nmodes: usize,
    nrr_q: usize,
    nrr_g: usize,
}

/// Read `crystal.fmt`, `epwdata.fmt` and `wigner.fmt` from `dir`.
///
/// # Errors
///
/// As the individual readers; [`Error::Data`] if the files disagree on sizes.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/pb_epw_metal/wannier");
/// let model = eph_io_epw::read_wannier_model(&dir)?;
/// assert_eq!(model.nbnd, 4);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_wannier_model(dir: &Path) -> Result<WannierModel, Error> {
    let crystal = read_crystal(&dir.join("crystal.fmt"))?;
    let (ws_k, ws_q, ws_g) = read_wigner(&dir.join("wigner.fmt"))?;

    let path = dir.join("epwdata.fmt");
    let mut r = Reader::open(&path)?;
    let fermi_energy_coarse = r.f64("ef")?;
    let d = EpwdataDims {
        nbnd: r.usize("nbndsub")?,
        nrr_k: r.usize("nrr_k")?,
        nmodes: r.usize("nmodes")?,
        nrr_q: r.usize("nrr_q")?,
        nrr_g: r.usize("nrr_g")?,
    };
    // Born effective charges and dielectric tensor (unused for metals).
    let _ = r.f64_vec(9 * crystal.nat() + 9, "zstar/epsi")?;
    let read_blocks =
        |r: &mut Reader, n: usize, nr: usize, what| -> Result<Vec<Complex64>, Error> {
            // File order: i outer, j, R inner. Stored column-major (i, j, R).
            let mut out = vec![Complex64::default(); n * n * nr];
            for i in 0..n {
                for j in 0..n {
                    for ir in 0..nr {
                        out[i + n * (j + n * ir)] = r.complex(what)?;
                    }
                }
            }
            Ok(out)
        };
    let hamiltonian = read_blocks(&mut r, d.nbnd, d.nrr_k, "H(R)")?;
    let force_constants = read_blocks(&mut r, d.nmodes, d.nrr_q, "force constants")?;

    let model = WannierModel {
        crystal,
        fermi_energy_coarse,
        nbnd: d.nbnd,
        ws_k,
        ws_q,
        ws_g,
        hamiltonian,
        force_constants,
    };
    model.validate()?;
    if model.ws_k.len() != d.nrr_k || model.ws_q.len() != d.nrr_q || model.ws_g.len() != d.nrr_g {
        return Err(Error::Unsupported {
            path,
            detail: "epwdata.fmt and wigner.fmt disagree on the number of WS vectors".into(),
        });
    }
    Ok(model)
}

/// Vertex dimensions implied by a model.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/pb_epw_metal/wannier");
/// let model = eph_io_epw::read_wannier_model(&dir)?;
/// assert_eq!(eph_io_epw::eph_dims(&model).nrr_g, 43);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn eph_dims(model: &WannierModel) -> EphDims {
    EphDims {
        nbnd: model.nbnd,
        nrr_k: model.ws_k.len(),
        nmodes: model.crystal.nmodes(),
        nrr_g: model.ws_g.len(),
    }
}

const COMPLEX_BYTES: usize = 16;

fn decode(bytes: &[u8]) -> Vec<Complex64> {
    bytes
        .chunks_exact(COMPLEX_BYTES)
        .map(|c| {
            let re = f64::from_ne_bytes(c[..8].try_into().expect("chunk of 16 bytes"));
            let im = f64::from_ne_bytes(c[8..].try_into().expect("chunk of 16 bytes"));
            Complex64::new(re, im)
        })
        .collect()
}

/// Load `<prefix>.epmatwp` fully into memory.
///
/// # Errors
///
/// [`Error::Io`] if unreadable; [`Error::Data`] if the size does not match `dims`.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/pb_epw_metal/wannier");
/// let model = eph_io_epw::read_wannier_model(&dir)?;
/// let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model))?;
/// # use eph_wannier_data::EphMatrixSource;
/// assert_eq!(g.dims().nmodes, 3);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_epmatwp(path: &Path, dims: EphDims) -> Result<InMemoryEph, Error> {
    let bytes = fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(InMemoryEph::new(dims, decode(&bytes))?)
}

/// `<prefix>.epmatwp` read from disk one R_p block at a time (EPW `etf_mem = 1`).
#[derive(Debug)]
pub struct EpmatwpFile {
    file: fs::File,
    dims: EphDims,
}

impl EpmatwpFile {
    /// Open the file and check its size.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if it cannot be opened; [`Error::Data`] on a size mismatch.
    ///
    /// # Examples
    ///
    /// ```
    /// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    ///     .join("../../tests/fixtures/pb_epw_metal/wannier");
    /// let model = eph_io_epw::read_wannier_model(&dir)?;
    /// let g = eph_io_epw::EpmatwpFile::open(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model))?;
    /// # use eph_wannier_data::EphMatrixSource;
    /// assert_eq!(g.dims().nbnd, 4);
    /// # Ok::<(), eph_io_epw::Error>(())
    /// ```
    pub fn open(path: &Path, dims: EphDims) -> Result<Self, Error> {
        let io = |source| Error::Io {
            path: path.to_path_buf(),
            source,
        };
        let file = fs::File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len() as usize;
        let expected = dims.block_len() * dims.nrr_g * COMPLEX_BYTES;
        if len != expected {
            return Err(eph_wannier_data::Error::Length {
                what: "epmatwp bytes",
                expected,
                got: len,
            }
            .into());
        }
        Ok(Self { file, dims })
    }
}

impl EphMatrixSource for EpmatwpFile {
    fn dims(&self) -> EphDims {
        self.dims
    }

    fn read_rp_block(
        &self,
        rp: Range<usize>,
        out: &mut [Complex64],
    ) -> Result<(), eph_wannier_data::Error> {
        use std::os::unix::fs::FileExt;
        let nrr_g = self.dims.nrr_g;
        if rp.start > rp.end || rp.end > nrr_g {
            return Err(eph_wannier_data::Error::BlockRange {
                start: rp.start,
                end: rp.end,
                nrr_g,
            });
        }
        let block = self.dims.block_len();
        if out.len() != block * rp.len() {
            return Err(eph_wannier_data::Error::Length {
                what: "e-ph block output",
                expected: block * rp.len(),
                got: out.len(),
            });
        }
        let mut bytes = vec![0_u8; out.len() * COMPLEX_BYTES];
        let offset = (block * rp.start * COMPLEX_BYTES) as u64;
        self.file
            .read_exact_at(&mut bytes, offset)
            .map_err(eph_wannier_data::Error::Io)?;
        out.copy_from_slice(&decode(&bytes));
        Ok(())
    }
}
