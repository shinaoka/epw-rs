//! Smearing functions, the Fermi-level search and the density of states at E_F.
//!
//! Formulas:
//! - Gaussian and Methfessel–Paxton occupations and delta functions:
//!   M. Methfessel and A. T. Paxton, Phys. Rev. B 40, 3616 (1989).
//! - Marzari–Vanderbilt ("cold") smearing:
//!   N. Marzari et al., Phys. Rev. Lett. 82, 3296 (1999).
//!
//! All functions take a dimensionless argument `t`; for occupations
//! `t = (E_F - e) / sigma`, and `delta(t) = d occupation / dt`. The numerical guards (argument caps) and the Fermi-level search
//! strategy (tolerances, bracket and the Newton step on the squared electron
//! count error, with bisection fallback) are chosen to match QE 7.6 `efermig`
//! so that E_F agrees with EPW to the printed precision. The code was written
//! independently from the published formulas.
//!
//! # Examples
//!
//! ```
//! use smearing::Smearing;
//! let mp1 = Smearing::MethfesselPaxton(1);
//! assert!((mp1.occupation(0.0) - 0.5).abs() < 1e-15);
//! assert!((Smearing::Gaussian.delta(0.0) - 1.0 / std::f64::consts::PI.sqrt()).abs() < 1e-15);
//! ```

use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

/// 1 / sqrt(pi), formed as `1 / sqrt(pi)` like QE's `sqrtpm1`.
const INV_SQRT_PI: f64 = 1.0 / 1.772_453_850_905_516;
const MAX_ARG: f64 = 200.0;

/// Errors from the Fermi-level search.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The electron count at the bracket ends does not enclose the target.
    #[error("cannot bracket the Fermi level: N({lower}) = {n_lower}, N({upper}) = {n_upper}, target {target}; check the number of electrons and bands")]
    CannotBracket {
        /// Lower bracket energy.
        lower: f64,
        /// Upper bracket energy.
        upper: f64,
        /// Electron count at the lower end.
        n_lower: f64,
        /// Electron count at the upper end.
        n_upper: f64,
        /// Requested number of electrons.
        target: f64,
    },
    /// Input arrays have inconsistent sizes.
    #[error("eigenvalue array has {eigenvalues} entries, expected nbnd * nk = {nbnd} * {nk}")]
    Shape {
        /// Length of the eigenvalue slice.
        eigenvalues: usize,
        /// Bands per k-point.
        nbnd: usize,
        /// Number of k-points (length of the weights).
        nk: usize,
    },
}

/// Type of broadening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Smearing {
    /// Plain Gaussian.
    Gaussian,
    /// Methfessel–Paxton of the given order (order 0 is Gaussian).
    MethfesselPaxton(u32),
    /// Marzari–Vanderbilt cold smearing.
    MarzariVanderbilt,
    /// Fermi–Dirac.
    FermiDirac,
}

impl Smearing {
    /// Map the QE/EPW integer code (`ngauss`, `ngaussw`): 0, n > 0, -1, -99.
    ///
    /// # Examples
    ///
    /// ```
    /// use smearing::Smearing;
    /// assert_eq!(Smearing::from_qe_code(1), Some(Smearing::MethfesselPaxton(1)));
    /// assert_eq!(Smearing::from_qe_code(-2), None);
    /// ```
    pub fn from_qe_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Gaussian),
            n if n > 0 => Some(Self::MethfesselPaxton(n as u32)),
            -1 => Some(Self::MarzariVanderbilt),
            -99 => Some(Self::FermiDirac),
            _ => None,
        }
    }

    /// Occupation (approximate step function) at `t = (E_F - e) / sigma`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = smearing::Smearing::Gaussian;
    /// assert!(s.occupation(10.0) > 0.999_999);
    /// assert!(s.occupation(-10.0) < 1e-6);
    /// ```
    pub fn occupation(self, t: f64) -> f64 {
        match self {
            Self::FermiDirac => {
                if t < -MAX_ARG {
                    0.0
                } else if t > MAX_ARG {
                    1.0
                } else {
                    1.0 / (1.0 + (-t).exp())
                }
            }
            Self::MarzariVanderbilt => {
                let u = t - FRAC_1_SQRT_2;
                0.5 * libm::erf(u) + (-(u * u).min(MAX_ARG)).exp() / (2.0 * PI).sqrt() + 0.5
            }
            Self::Gaussian => gaussian_step(t),
            Self::MethfesselPaxton(order) => {
                // S_N(t) = S_0(t) - sum_n A_n H_{2n-1}(t) exp(-t^2),
                // A_n = (-1)^n / (n! 4^n sqrt(pi)), Hermite polynomials by recursion.
                let mut value = gaussian_step(t);
                let gauss = (-(t * t).min(MAX_ARG)).exp();
                let mut hermite = HermiteRecursion::new(t, gauss);
                let mut coefficient = INV_SQRT_PI;
                for n in 1..=order {
                    coefficient = -coefficient / (4.0 * f64::from(n));
                    let odd = hermite.advance();
                    value -= coefficient * odd;
                    hermite.advance();
                }
                value
            }
        }
    }

    /// Delta-function approximation, `d occupation / dt` (divide by sigma for an energy density).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = smearing::Smearing::FermiDirac;
    /// assert!((s.delta(0.0) - 0.25).abs() < 1e-15);
    /// ```
    pub fn delta(self, t: f64) -> f64 {
        match self {
            Self::FermiDirac => {
                if t.abs() <= 36.0 {
                    1.0 / (2.0 + (-t).exp() + t.exp())
                } else {
                    0.0
                }
            }
            Self::MarzariVanderbilt => {
                let u = t - FRAC_1_SQRT_2;
                INV_SQRT_PI * (-(u * u).min(MAX_ARG)).exp() * (2.0 - SQRT_2 * t)
            }
            Self::Gaussian => INV_SQRT_PI * (-(t * t).min(MAX_ARG)).exp(),
            Self::MethfesselPaxton(order) => {
                // D_N(t) = sum_{n=0}^N A_n H_{2n}(t) exp(-t^2)
                let gauss = (-(t * t).min(MAX_ARG)).exp();
                let mut value = gauss * INV_SQRT_PI;
                let mut hermite = HermiteRecursion::new(t, gauss);
                let mut coefficient = INV_SQRT_PI;
                for n in 1..=order {
                    hermite.advance();
                    let even = hermite.advance();
                    coefficient = -coefficient / (4.0 * f64::from(n));
                    value += coefficient * even;
                }
                value
            }
        }
    }

    /// Derivative of [`Smearing::delta`] with respect to `t`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = smearing::Smearing::MethfesselPaxton(1);
    /// let h = 1e-6;
    /// let fd = (s.delta(0.3 + h) - s.delta(0.3 - h)) / (2.0 * h);
    /// assert!((s.delta_derivative(0.3) - fd).abs() < 1e-8);
    /// ```
    pub fn delta_derivative(self, t: f64) -> f64 {
        match self {
            Self::FermiDirac => {
                if t.abs() <= 36.0 {
                    let d = 2.0 + (-t).exp() + t.exp();
                    -(t.exp() - (-t).exp()) / (d * d)
                } else {
                    0.0
                }
            }
            Self::MarzariVanderbilt => {
                let u = t - FRAC_1_SQRT_2;
                INV_SQRT_PI
                    * (-(u * u).min(MAX_ARG)).exp()
                    * (2.0 * SQRT_2 * t * t - 6.0 * t + SQRT_2)
            }
            Self::Gaussian => -2.0 * t * INV_SQRT_PI * (-(t * t).min(MAX_ARG)).exp(),
            Self::MethfesselPaxton(order) => {
                // d/dt [H_{2n} e^{-t^2}] = -H_{2n+1} e^{-t^2}
                let gauss = (-(t * t).min(MAX_ARG)).exp();
                let mut hermite = HermiteRecursion::new(t, gauss);
                let mut coefficient = INV_SQRT_PI;
                let mut value = -coefficient * hermite.advance();
                for n in 1..=order {
                    coefficient = -coefficient / (4.0 * f64::from(n));
                    hermite.advance();
                    value -= coefficient * hermite.advance();
                }
                value
            }
        }
    }
}

/// Fermi–Dirac occupation `1 / (1 + exp(e / kT))` with `kT` in the energy unit of `e`.
///
/// A non-positive temperature gives the step function (1/2 at `e = 0`).
///
/// # Examples
///
/// ```
/// assert_eq!(smearing::fermi_dirac(0.0, 0.01), 0.5);
/// assert_eq!(smearing::fermi_dirac(-1.0, 0.0), 1.0);
/// ```
pub fn fermi_dirac(e: f64, kt: f64) -> f64 {
    if kt > 0.0 {
        let x = e / kt;
        if x > MAX_ARG {
            0.0
        } else if x < -MAX_ARG {
            1.0
        } else {
            1.0 / (1.0 + x.exp())
        }
    } else if e > 0.0 {
        0.0
    } else if e < 0.0 {
        1.0
    } else {
        0.5
    }
}

fn gaussian_step(t: f64) -> f64 {
    if -t < MAX_ARG.sqrt() {
        0.5 * libm::erfc(-t)
    } else {
        0.0
    }
}

/// Generates `H_m(t) exp(-t^2)` for m = 1, 2, 3, ... (Hermite recursion).
struct HermiteRecursion {
    t: f64,
    previous: f64,
    current: f64,
    order: u32,
}

impl HermiteRecursion {
    fn new(t: f64, gauss: f64) -> Self {
        Self {
            t,
            previous: 0.0,
            current: gauss,
            order: 0,
        }
    }

    /// Advance to the next order and return `H_{m+1}(t) exp(-t^2)`.
    fn advance(&mut self) -> f64 {
        let next = 2.0 * self.t * self.current - 2.0 * f64::from(self.order) * self.previous;
        self.previous = self.current;
        self.current = next;
        self.order += 1;
        next
    }
}

/// Band energies on a k-point set with integration weights (weights include spin).
#[derive(Debug, Clone, Copy)]
pub struct BandSet<'a> {
    energies: &'a [f64],
    weights: &'a [f64],
    nbnd: usize,
}

impl<'a> BandSet<'a> {
    /// Wrap column-major energies `(nbnd, nk)` and weights `(nk)`.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] when `energies.len() != nbnd * weights.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let e = [0.0, 1.0];
    /// let w = [2.0];
    /// let bands = smearing::BandSet::new(&e, &w, 2)?;
    /// assert_eq!(bands.nk(), 1);
    /// # Ok::<(), smearing::Error>(())
    /// ```
    pub fn new(energies: &'a [f64], weights: &'a [f64], nbnd: usize) -> Result<Self, Error> {
        if energies.len() != nbnd * weights.len() {
            return Err(Error::Shape {
                eigenvalues: energies.len(),
                nbnd,
                nk: weights.len(),
            });
        }
        Ok(Self {
            energies,
            weights,
            nbnd,
        })
    }

    /// Number of k-points.
    ///
    /// # Examples
    ///
    /// ```
    /// let bands = smearing::BandSet::new(&[0.0; 6], &[1.0; 3], 2)?;
    /// assert_eq!(bands.nk(), 3);
    /// # Ok::<(), smearing::Error>(())
    /// ```
    pub fn nk(&self) -> usize {
        self.weights.len()
    }

    fn weighted_sum(&self, f: impl Fn(f64) -> f64) -> f64 {
        self.energies
            .chunks_exact(self.nbnd)
            .zip(self.weights)
            .map(|(band, w)| w * band.iter().map(|&e| f(e)).sum::<f64>())
            .sum()
    }

    /// Number of electrons below `ef` with the given broadening.
    ///
    /// # Examples
    ///
    /// ```
    /// use smearing::{BandSet, Smearing};
    /// let bands = BandSet::new(&[-1.0, 1.0], &[2.0], 2)?;
    /// assert!((bands.electron_count(0.0, 0.01, Smearing::Gaussian) - 2.0).abs() < 1e-12);
    /// # Ok::<(), smearing::Error>(())
    /// ```
    pub fn electron_count(&self, ef: f64, sigma: f64, smearing: Smearing) -> f64 {
        self.weighted_sum(|e| smearing.occupation((ef - e) / sigma))
    }

    /// Density of states at `energy` (states per energy unit, both spins if the weights include them).
    ///
    /// Evaluates the delta at `(e - energy) / sigma` like EPW's `dos_ef`; for the
    /// asymmetric cold smearing this differs from `d N / d E`.
    ///
    /// # Examples
    ///
    /// ```
    /// use smearing::{BandSet, Smearing};
    /// let bands = BandSet::new(&[0.0], &[1.0], 1)?;
    /// let dos = bands.dos_at(0.0, 0.1, Smearing::Gaussian);
    /// assert!((dos - 10.0 / std::f64::consts::PI.sqrt()).abs() < 1e-12);
    /// # Ok::<(), smearing::Error>(())
    /// ```
    pub fn dos_at(&self, energy: f64, sigma: f64, smearing: Smearing) -> f64 {
        self.weighted_sum(|e| smearing.delta((e - energy) / sigma) / sigma)
    }

    /// dN/dE at `energy`; the argument sign `(E - e)` matters for asymmetric (cold) smearing.
    fn count_derivative(&self, energy: f64, sigma: f64, smearing: Smearing) -> f64 {
        self.weighted_sum(|e| smearing.delta((energy - e) / sigma)) / sigma
    }

    fn count_second_derivative(&self, energy: f64, sigma: f64, smearing: Smearing) -> f64 {
        self.weighted_sum(|e| smearing.delta_derivative((energy - e) / sigma)) / (sigma * sigma)
    }

    fn energy_bounds(&self) -> (f64, f64) {
        let lower = self
            .energies
            .chunks_exact(self.nbnd)
            .map(|b| b[0])
            .fold(1.0e8, f64::min);
        let upper = self
            .energies
            .chunks_exact(self.nbnd)
            .map(|b| b[self.nbnd - 1])
            .fold(-1.0e8, f64::max);
        (lower, upper)
    }
}

/// How the Fermi level was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FermiStrategy {
    /// Bisection with the requested smearing converged directly.
    Bisection,
    /// Newton refinement after a Gaussian bisection.
    Newton,
    /// Newton failed; bisection with the requested smearing was used.
    BisectionFallback,
}

/// Result of [`fermi_level`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FermiLevel {
    /// Fermi energy (same unit as the band energies).
    pub energy: f64,
    /// Strategy that produced it.
    pub strategy: FermiStrategy,
}

const FERMI_TOL: f64 = 1.0e-10;
const FERMI_ACCEPT_NEWTON: f64 = 1.0e-2;
const FERMI_MAX_ITER: usize = 300;

/// Fermi level such that the smeared electron count equals `nelec`.
///
/// A bisection with Gaussian broadening (Fermi–Dirac stays Fermi–Dirac) gives
/// a starting point inside `[min e - 10 sigma, max e + 10 sigma]`. For
/// Methfessel–Paxton and cold smearing, whose occupations are not monotonic,
/// Newton's method minimises `(N(E) - nelec)^2`; if that does not reach
/// `|N - nelec| < 1e-2`, bisection with the requested smearing is used.
///
/// # Errors
///
/// [`Error::CannotBracket`] if the electron count cannot be bracketed.
///
/// # Examples
///
/// ```
/// use smearing::{fermi_level, BandSet, Smearing};
/// let bands = BandSet::new(&[-1.0, 1.0], &[2.0], 2)?;
/// let ef = fermi_level(&bands, 2.0, 0.05, Smearing::Gaussian)?;
/// assert!(ef.energy.abs() < 1e-6);
/// # Ok::<(), smearing::Error>(())
/// ```
pub fn fermi_level(
    bands: &BandSet<'_>,
    nelec: f64,
    sigma: f64,
    smearing: Smearing,
) -> Result<FermiLevel, Error> {
    let (lower, upper) = bands.energy_bounds();
    let lower = lower - 10.0 * sigma;
    let upper = upper + 10.0 * sigma;
    let preliminary = if smearing == Smearing::FermiDirac {
        Smearing::FermiDirac
    } else {
        Smearing::Gaussian
    };
    let count_error = |e: f64, s: Smearing| bands.electron_count(e, sigma, s) - nelec;

    let bracket_error = |s: Smearing| Error::CannotBracket {
        lower,
        upper,
        n_lower: bands.electron_count(lower, sigma, s),
        n_upper: bands.electron_count(upper, sigma, s),
        target: nelec,
    };
    let start = bisect(lower, upper, |e| count_error(e, preliminary))
        .ok_or_else(|| bracket_error(preliminary))?;
    if matches!(
        smearing,
        Smearing::Gaussian | Smearing::FermiDirac | Smearing::MethfesselPaxton(0)
    ) || count_error(start, smearing).abs() < FERMI_TOL
    {
        return Ok(FermiLevel {
            energy: start,
            strategy: FermiStrategy::Bisection,
        });
    }

    // Newton on F(E) = (N(E) - nelec)^2 using |F''| to always step downhill.
    let mut energy = start;
    for _ in 0..FERMI_MAX_ITER {
        let err = count_error(energy, smearing);
        let dn = bands.count_derivative(energy, sigma, smearing);
        let d2n = bands.count_second_derivative(energy, sigma, smearing);
        let gradient = 2.0 * err * dn;
        let curvature = (2.0 * (dn * dn + err * d2n)).abs();
        if curvature <= FERMI_TOL {
            break;
        }
        let next = energy - gradient / curvature;
        let step = (energy - next).abs();
        energy = next;
        if step < FERMI_TOL || count_error(energy, smearing).abs() < FERMI_TOL {
            break;
        }
    }
    if count_error(energy, smearing).abs() < FERMI_ACCEPT_NEWTON {
        return Ok(FermiLevel {
            energy,
            strategy: FermiStrategy::Newton,
        });
    }
    let energy = bisect(lower, upper, |e| count_error(e, smearing))
        .ok_or_else(|| bracket_error(smearing))?;
    Ok(FermiLevel {
        energy,
        strategy: FermiStrategy::BisectionFallback,
    })
}

/// Bisection on the electron-count error; `None` if the bracket does not enclose the root.
fn bisect(lower: f64, upper: f64, count_error: impl Fn(f64) -> f64) -> Option<f64> {
    if count_error(lower) > FERMI_TOL || count_error(upper) < -FERMI_TOL {
        return None;
    }
    let (mut lo, mut hi) = (lower, upper);
    let mut mid = 0.5 * (lo + hi);
    for _ in 0..FERMI_MAX_ITER {
        mid = 0.5 * (hi + lo);
        let err = count_error(mid);
        if err.abs() < FERMI_TOL {
            return Some(mid);
        }
        if err < -FERMI_TOL {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(mid)
}
