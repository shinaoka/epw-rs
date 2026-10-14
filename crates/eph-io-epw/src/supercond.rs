//! Readers for the inputs of EPW's isotropic Eliashberg solvers.
//!
//! - `<prefix>.a2f`: one header line, then `nqstep` rows of
//!   `omega [meV]` followed by alpha^2F for several smearings (and their
//!   integrated lambda). EPW's isotropic solver uses the first two columns only.
//! - `<prefix>.dos`: two header lines with E_F, N(E_F), the energy step and
//!   the number of bins, a column header, then `E [eV]  dos [1/eV]  int dos`.

use crate::Error;
use std::path::Path;

/// alpha^2F(omega) as used by EPW: first smearing column, frequencies in eV.
#[derive(Debug, Clone, PartialEq)]
pub struct A2fTable {
    /// Frequencies (eV), as printed in the file.
    pub omega: Vec<f64>,
    /// alpha^2F for the first phonon smearing.
    pub a2f: Vec<f64>,
}

/// Electronic density of states written by EPW for full-bandwidth runs.
#[derive(Debug, Clone, PartialEq)]
pub struct DosTable {
    /// Fermi energy (eV).
    pub fermi_energy: f64,
    /// DOS at the Fermi energy (1/eV) from the header.
    pub dos_at_fermi: f64,
    /// Energy step (eV).
    pub step: f64,
    /// Bin energies (eV).
    pub energies: Vec<f64>,
    /// DOS per bin (1/eV).
    pub dos: Vec<f64>,
}

fn read_text(path: &Path) -> Result<String, Error> {
    std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_error(path: &Path, token: &str, what: &'static str) -> Error {
    Error::Parse {
        path: path.to_path_buf(),
        token: token.to_string(),
        what,
    }
}

/// Read `<prefix>.a2f`: data rows up to the first non-numeric line.
///
/// # Errors
///
/// [`Error::Io`] if unreadable, [`Error::Parse`] for a malformed row,
/// [`Error::UnexpectedEnd`] if there is no data row.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/mgb2_epw_super/iso");
/// let a2f = eph_io_epw::supercond::read_a2f(&dir.join("MgB2.a2f"))?;
/// assert_eq!(a2f.omega.len(), 500);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_a2f(path: &Path) -> Result<A2fTable, Error> {
    let text = read_text(path)?;
    let mut omega = Vec::new();
    let mut a2f = Vec::new();
    for line in text.lines().skip(1) {
        let mut fields = line.split_whitespace();
        let (Some(w), Some(a)) = (fields.next(), fields.next()) else {
            break;
        };
        let Ok(w) = w.parse::<f64>() else { break };
        let a = a
            .parse::<f64>()
            .map_err(|_| parse_error(path, a, "alpha^2F"))?;
        omega.push(w / 1000.0);
        a2f.push(a);
    }
    if omega.is_empty() {
        return Err(Error::UnexpectedEnd {
            path: path.to_path_buf(),
            what: "alpha^2F rows",
        });
    }
    Ok(A2fTable { omega, a2f })
}

/// Read `<prefix>.dos`.
///
/// # Errors
///
/// [`Error::Io`] if unreadable, [`Error::Parse`] or [`Error::UnexpectedEnd`]
/// for a malformed header or body.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/mgb2_epw_super/iso");
/// let dos = eph_io_epw::supercond::read_dos(&dir.join("MgB2.dos"))?;
/// assert_eq!(dos.energies.len(), 4000);
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_dos(path: &Path) -> Result<DosTable, Error> {
    let text = read_text(path)?;
    let mut lines = text.lines();
    let mut header_value = |what: &'static str, index: usize| -> Result<Vec<f64>, Error> {
        let line = lines.next().ok_or_else(|| Error::UnexpectedEnd {
            path: path.to_path_buf(),
            what,
        })?;
        let numbers: Vec<f64> = line
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if numbers.len() <= index {
            return Err(parse_error(path, line, what));
        }
        Ok(numbers)
    };
    // "# EFermi[eV] <ef> dos_EFermi[eV^-1] <dos>"
    let first = header_value("E_F header", 1)?;
    let _electrons = header_value("window header", 1)?;
    // "# dos_del[eV] <step> Nr_bins <n>"
    let third = header_value("dos step header", 1)?;
    let _columns = lines.next();
    let nbins = third[1] as usize;
    let mut energies = Vec::with_capacity(nbins);
    let mut dos = Vec::with_capacity(nbins);
    for line in lines.take(nbins) {
        let mut fields = line.split_whitespace();
        let (Some(e), Some(d)) = (fields.next(), fields.next()) else {
            return Err(parse_error(path, line, "dos row"));
        };
        energies.push(e.parse().map_err(|_| parse_error(path, e, "energy"))?);
        dos.push(d.parse().map_err(|_| parse_error(path, d, "dos"))?);
    }
    if energies.len() != nbins {
        return Err(Error::UnexpectedEnd {
            path: path.to_path_buf(),
            what: "dos rows",
        });
    }
    Ok(DosTable {
        fermi_energy: first[0],
        dos_at_fermi: first[1],
        step: third[0],
        energies,
        dos,
    })
}

/// Sampling points of an EPW IR object file (`EPW/irobjs/*.dat`, version 1).
#[derive(Debug, Clone, PartialEq)]
pub struct IrObjectPoints {
    /// Lambda from the header.
    pub lambda: f64,
    /// Accuracy epsilon from the header.
    pub epsilon: f64,
    /// Basis size.
    pub size: usize,
    /// Imaginary-time points `x` in `[-1, 1]` (`tau = beta (x + 1) / 2`).
    pub x: Vec<f64>,
    /// Fermionic Matsubara integers `2n + 1` (both signs).
    pub fermionic: Vec<i64>,
    /// Bosonic Matsubara integers `2m` (both signs).
    pub bosonic: Vec<i64>,
}

/// Read the header and sampling points of an EPW IR object file.
///
/// # Errors
///
/// [`Error::Io`] if unreadable, [`Error::Parse`] or [`Error::UnexpectedEnd`] for a
/// malformed file, [`Error::Unsupported`] for a version other than 1.
///
/// # Examples
///
/// ```
/// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
///     .join("../../tests/fixtures/mgb2_epw_super/aniso");
/// let ir = eph_io_epw::supercond::read_ir_object(&dir.join("ir_nlambda5_ndigit6.dat"))?;
/// assert_eq!((ir.size, ir.x.len(), ir.fermionic.len(), ir.bosonic.len()), (61, 63, 62, 61));
/// # Ok::<(), eph_io_epw::Error>(())
/// ```
pub fn read_ir_object(path: &Path) -> Result<IrObjectPoints, Error> {
    let text = read_text(path)?;
    let mut lines = text.lines();
    let mut header = |what: &'static str| -> Result<String, Error> {
        lines
            .next()
            .map(str::to_string)
            .ok_or_else(|| Error::UnexpectedEnd {
                path: path.to_path_buf(),
                what,
            })
    };
    let last_number = |line: &str, what: &'static str| -> Result<f64, Error> {
        line.split_whitespace()
            .last()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| parse_error(path, line, what))
    };
    let version = last_number(&header("version")?, "version")?;
    if version != 1.0 {
        return Err(Error::Unsupported {
            path: path.to_path_buf(),
            detail: format!("IR object version {version}; only version 1 is supported"),
        });
    }
    let lambda = last_number(&header("lambda")?, "lambda")?;
    let epsilon = last_number(&header("eps")?, "eps")?;
    // The rest is a sequence of "# title" lines, each followed by a count and values.
    let rest: Vec<&str> = text.lines().skip(3).collect();
    let section = |title: &str, what: &'static str| -> Result<Vec<&str>, Error> {
        let start = rest
            .iter()
            .position(|l| l.trim_start().starts_with('#') && l.contains(title))
            .ok_or_else(|| Error::UnexpectedEnd {
                path: path.to_path_buf(),
                what,
            })?;
        let count: usize = rest
            .get(start + 1)
            .and_then(|l| l.trim().parse().ok())
            .ok_or_else(|| parse_error(path, rest.get(start + 1).unwrap_or(&""), what))?;
        rest.get(start + 2..start + 2 + count)
            .map(<[&str]>::to_vec)
            .ok_or_else(|| Error::UnexpectedEnd {
                path: path.to_path_buf(),
                what,
            })
    };
    let size = section("singular_values", "singular values")?.len();
    let x = section("sampling times", "sampling times")?
        .iter()
        .map(|l| {
            l.trim()
                .parse()
                .map_err(|_| parse_error(path, l, "tau point"))
        })
        .collect::<Result<Vec<f64>, _>>()?;
    let ints = |title: &str, what: &'static str| -> Result<Vec<i64>, Error> {
        section(title, what)?
            .iter()
            .map(|l| l.trim().parse().map_err(|_| parse_error(path, l, what)))
            .collect()
    };
    let fermionic = ints("sampling frequencies F", "fermionic frequency")?;
    let bosonic = ints("sampling frequencies B", "bosonic frequency")?;
    Ok(IrObjectPoints {
        lambda,
        epsilon,
        size,
        x,
        fermionic,
        bosonic,
    })
}
