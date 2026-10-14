//! Migdal–Eliashberg equations on the imaginary axis, independent of EPW.
//!
//! Inputs are the Eliashberg spectral function alpha^2F(omega) and, for the
//! full-bandwidth equations, the electronic density of states. Energies are in
//! eV and the temperature is `k_B T` in eV, following EPW's superconductivity
//! module.
//!
//! The Matsubara sums are pluggable ([`FrequencySum`]):
//! - [`UniformSum`]: all frequencies below a cutoff (always available);
//! - `IrSum`: the intermediate-representation basis through sparse-ir-rs,
//!   enabled by the Cargo feature `sparse-ir`.
//!
//! # Examples
//!
//! ```
//! use eliashberg::*;
//! let spectrum = EliashbergSpectrum::new(vec![0.02], vec![0.5])?;
//! assert!((spectrum.lambda() - 1.0).abs() < 1e-12);
//! # Ok::<(), eliashberg::Error>(())
//! ```

mod anisotropic;
mod isotropic;
mod matsubara;
mod pade;
mod spectrum;

#[cfg(feature = "sparse-ir")]
mod ir;

pub use anisotropic::{
    solve_anisotropic_fbw, solve_anisotropic_fbw_coulomb, AnisotropicModel, AnisotropicParams,
    AnisotropicSolution, CoulombStates, KernelEntry, StateFrequencySum, UniformStateSum,
};
pub use isotropic::{
    free_energy_difference, normal_state_z, solve_isotropic_fbw, solve_isotropic_fsr, ElectronDos,
    IsotropicParams, IsotropicSolution,
};
pub use matsubara::{fermionic_frequency, frequencies_below, FrequencySum, Parity, UniformSum};
pub use pade::{continue_to_real_axis, gap_edge, pade_point_count, Pade, RealAxisSolution};
pub use spectrum::EliashbergSpectrum;

#[cfg(feature = "sparse-ir")]
pub use ir::{IrBasisSet, IrSettings, IrStateSum, IrSum};

/// Errors of the Eliashberg solvers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid input data or parameters.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The self-consistency did not converge.
    #[error("not converged after {iterations} iterations; increase the iteration limit or reduce the mixing")]
    NotConverged {
        /// Iterations performed.
        iterations: usize,
    },
    /// Tensor operation failed.
    #[error("tensor operation failed")]
    Tensor(#[from] tenferro_tensor::Error),
    /// Einsum failed.
    #[error("einsum failed")]
    Einsum(#[from] tenferro_einsum::Error),
    /// IR basis or sampling failed.
    #[cfg(feature = "sparse-ir")]
    #[error("sparse-ir failed")]
    SparseIr(#[from] sparse_ir::Error),
}
