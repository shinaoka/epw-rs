//! Physical constants and unit conversions in Rydberg atomic units.
//!
//! Values are derived from the CODATA 2018 SI constants, in the same way as
//! the Quantum ESPRESSO `constants` module (QE 7.6), so that energies converted
//! here agree bit-for-bit with numbers printed by QE and EPW. Only the factors
//! needed by the downstream e-ph calculation are provided.
//!
//! # Examples
//!
//! ```
//! use phys_units::{RY_TO_EV, RY_TO_MEV};
//! assert!((RY_TO_EV - 13.605_693_122_99).abs() < 1e-9);
//! assert_eq!(RY_TO_MEV, RY_TO_EV * 1.0e3);
//! ```

/// pi.
pub const PI: f64 = std::f64::consts::PI;
/// 2 pi.
pub const TWO_PI: f64 = 2.0 * PI;
/// sqrt(pi).
pub const SQRT_PI: f64 = 1.772_453_850_905_516;
/// 1 / sqrt(pi), formed as in QE (`1 / sqrtpi`).
pub const INV_SQRT_PI: f64 = 1.0 / SQRT_PI;

const H_PLANCK_SI: f64 = 6.626_070_15e-34;
const K_BOLTZMANN_SI: f64 = 1.380_649e-23;
const ELECTRONVOLT_SI: f64 = 1.602_176_634e-19;
const ELECTRONMASS_SI: f64 = 9.109_383_701_5e-31;
const HARTREE_SI: f64 = 4.359_744_722_207_1e-18;
const AMU_SI: f64 = 1.660_539_066_60e-27;
const C_SI: f64 = 2.997_924_58e8;

const HARTREE_TO_EV: f64 = HARTREE_SI / ELECTRONVOLT_SI;
const EV_TO_KELVIN: f64 = ELECTRONVOLT_SI / K_BOLTZMANN_SI;
const AU_SEC: f64 = H_PLANCK_SI / TWO_PI / HARTREE_SI;
const RY_TO_THZ: f64 = 1.0 / (AU_SEC * 1.0e12) / (4.0 * PI);

/// Rydberg to eV.
pub const RY_TO_EV: f64 = HARTREE_TO_EV / 2.0;
/// Rydberg to meV.
pub const RY_TO_MEV: f64 = RY_TO_EV * 1.0e3;
/// Rydberg to cm^-1.
pub const RY_TO_CMM1: f64 = 1.0e10 * RY_TO_THZ / C_SI;
/// eV to cm^-1.
pub const EV_TO_CMM1: f64 = RY_TO_CMM1 / RY_TO_EV;
/// Kelvin to eV (k_B in eV/K).
pub const KELVIN_TO_EV: f64 = 1.0 / EV_TO_KELVIN;
/// Atomic mass unit in Rydberg mass units (m_e / 2).
pub const AMU_RY: f64 = AMU_SI / ELECTRONMASS_SI / 2.0;

/// Convert an energy in eV to Ry.
///
/// # Examples
///
/// ```
/// assert!((phys_units::ev_to_ry(phys_units::RY_TO_EV) - 1.0).abs() < 1e-15);
/// ```
pub fn ev_to_ry(ev: f64) -> f64 {
    ev / RY_TO_EV
}

/// Convert an energy in meV to Ry.
///
/// # Examples
///
/// ```
/// assert!((phys_units::mev_to_ry(phys_units::RY_TO_MEV) - 1.0).abs() < 1e-15);
/// ```
pub fn mev_to_ry(mev: f64) -> f64 {
    mev / RY_TO_MEV
}

/// Convert a temperature in K to an energy in Ry.
///
/// # Examples
///
/// ```
/// let t = phys_units::kelvin_to_ry(300.0) * phys_units::RY_TO_EV;
/// assert!((t - 0.025_852).abs() < 1e-6);
/// ```
pub fn kelvin_to_ry(kelvin: f64) -> f64 {
    kelvin * KELVIN_TO_EV / RY_TO_EV
}

/// Convert a wavenumber in cm^-1 to Ry.
///
/// # Examples
///
/// ```
/// assert!((phys_units::cmm1_to_ry(phys_units::RY_TO_CMM1) - 1.0).abs() < 1e-15);
/// ```
pub fn cmm1_to_ry(cmm1: f64) -> f64 {
    cmm1 / EV_TO_CMM1 / RY_TO_EV
}
