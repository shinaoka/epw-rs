//! Fermionic Matsubara convolutions with a bosonic kernel.
//!
//! The Eliashberg equations on the imaginary axis need, for functions known at
//! positive fermionic frequencies `omega_n = (2n + 1) pi T` (n >= 0),
//!
//! `C(i omega_n) = sum_{m in Z} lambda(i omega_n - i omega_m) f(i omega_m)`,
//!
//! where `f` is odd or even in `omega` and `lambda(i nu)` is the bosonic
//! kernel. [`FrequencySum`] abstracts how this sum is evaluated:
//! [`UniformSum`] truncates it at `|omega_m| < omega_c` (EPW `gridsamp = 0`),
//! and `IrSum` (feature `sparse-ir`) evaluates it without truncation through
//! the intermediate-representation basis.
//!
//! Arrays have shape `(nfreq, rows)` in column-major order: the frequency axis
//! comes first and `rows` batches independent functions.

use crate::Error;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_tensor::TypedTensor;

/// Parity of a fermionic function under `omega -> -omega`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    /// `f(-omega) = -f(omega)` (e.g. `omega Z / sqrt(omega^2 + Delta^2)`).
    Odd,
    /// `f(-omega) = f(omega)` (e.g. `Delta / sqrt(omega^2 + Delta^2)`).
    Even,
}

/// Evaluates Matsubara convolutions on a fixed set of positive fermionic frequencies.
pub trait FrequencySum {
    /// Temperature `k_B T` (energy unit of the kernel).
    fn temperature(&self) -> f64;

    /// Indices `n >= 0` of the fermionic frequencies `(2n + 1) pi T` used.
    fn indices(&self) -> &[i64];

    /// `C(i omega_n) = sum_{m in Z} lambda(n - m) f(m)` for each row of `f` (shape `(nfreq, rows)`).
    ///
    /// # Errors
    ///
    /// Tensor or basis errors, and [`Error::InvalidInput`] for a wrong shape.
    fn convolve(&mut self, f: &TypedTensor<f64>, parity: Parity)
        -> Result<TypedTensor<f64>, Error>;

    /// `sum_{m in Z} f(m)` for each row of an even function `f` (shape `(nfreq, rows)`).
    ///
    /// # Errors
    ///
    /// As [`FrequencySum::convolve`].
    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error>;
}

/// Fermionic frequency `omega_n = (2n + 1) pi T`.
///
/// # Examples
///
/// ```
/// let w = eliashberg::fermionic_frequency(0, 1.0);
/// assert!((w - std::f64::consts::PI).abs() < 1e-15);
/// ```
pub fn fermionic_frequency(n: i64, temperature: f64) -> f64 {
    (2 * n + 1) as f64 * std::f64::consts::PI * temperature
}

/// Number of positive frequencies below a cutoff, as EPW: `int((omega_c / pi T - 1) / 2) + 1`.
///
/// # Examples
///
/// ```
/// let pi_t = std::f64::consts::PI * 0.01;
/// assert_eq!(eliashberg::frequencies_below(10.0 * pi_t, 0.01), 5);
/// ```
pub fn frequencies_below(cutoff: f64, temperature: f64) -> usize {
    (0.5 * (cutoff / std::f64::consts::PI / temperature - 1.0)) as usize + 1
}

/// Truncated sum over `|omega_m| < omega_c` on the uniform grid `n = 0..N-1`.
///
/// Using `lambda(n - m)` for `m >= 0` and `lambda(n + m + 1)` for the mirrored
/// negative frequencies, the sum becomes `sum_{m=0}^{N-1} K^(±)_{nm} f(m)` with
/// `K^(±) = lambda(n - m) ± lambda(n + m + 1)` (Pickett, PRB 26, 1186 (1982), Eq. 4.4).
#[derive(Debug)]
pub struct UniformSum {
    temperature: f64,
    indices: Vec<i64>,
    k_minus: TypedTensor<f64>,
    k_plus: TypedTensor<f64>,
    backend: CpuEngine,
}

impl UniformSum {
    /// Build the sum for `nfreq` positive frequencies.
    ///
    /// `kernel` returns `lambda(i nu)` for a batch of bosonic frequencies `nu`.
    ///
    /// # Errors
    ///
    /// Errors of `kernel`, tensor errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::{FrequencySum, Parity, UniformSum};
    /// // Constant kernel: sum_{|m| < N} f(m) for an even f = 1 gives 2N.
    /// let mut sum = UniformSum::new(tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().build().unwrap()), 0.01, 4, |nu| Ok(vec![1.0; nu.len()]))?;
    /// let ones = tenferro_tensor::TypedTensor::from_vec_col_major(vec![4, 1], vec![1.0; 4])?;
    /// let c = sum.convolve(&ones, Parity::Even)?;
    /// assert_eq!(c.as_slice()?, &[8.0; 4]);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(
        backend: CpuEngine,
        temperature: f64,
        nfreq: usize,
        kernel: impl FnOnce(&[f64]) -> Result<Vec<f64>, Error>,
    ) -> Result<Self, Error> {
        let two_pi_t = 2.0 * std::f64::consts::PI * temperature;
        let nu: Vec<f64> = (0..2 * nfreq).map(|k| k as f64 * two_pi_t).collect();
        let lambda = kernel(&nu)?;
        let mut k_minus = Vec::with_capacity(nfreq * nfreq);
        let mut k_plus = Vec::with_capacity(nfreq * nfreq);
        for m in 0..nfreq {
            for n in 0..nfreq {
                let same = lambda[n.abs_diff(m)];
                let mirrored = lambda[n + m + 1];
                k_minus.push(same - mirrored);
                k_plus.push(same + mirrored);
            }
        }
        Ok(Self {
            temperature,
            indices: (0..nfreq as i64).collect(),
            k_minus: TypedTensor::from_vec_col_major(vec![nfreq, nfreq], k_minus)?,
            k_plus: TypedTensor::from_vec_col_major(vec![nfreq, nfreq], k_plus)?,
            backend,
        })
    }
}

impl FrequencySum for UniformSum {
    fn temperature(&self) -> f64 {
        self.temperature
    }

    fn indices(&self) -> &[i64] {
        &self.indices
    }

    fn convolve(
        &mut self,
        f: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        check_rows(f, self.indices.len())?;
        let kernel = match parity {
            Parity::Odd => &self.k_minus,
            Parity::Even => &self.k_plus,
        };
        Ok([kernel, f].einsum("nm,mr->nr", &mut self.backend)?)
    }

    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        check_rows(f, self.indices.len())?;
        let twos = TypedTensor::from_vec_col_major(
            vec![self.indices.len()],
            vec![2.0; self.indices.len()],
        )?;
        let out = [f, &twos].einsum("mr,m->r", &mut self.backend)?;
        Ok(out.as_slice()?.to_vec())
    }
}

pub(crate) fn check_rows(f: &TypedTensor<f64>, nfreq: usize) -> Result<(), Error> {
    match f.shape() {
        [n, _] if *n == nfreq => Ok(()),
        shape => Err(Error::InvalidInput(format!(
            "expected shape (nfreq = {nfreq}, rows), got {shape:?}"
        ))),
    }
}
