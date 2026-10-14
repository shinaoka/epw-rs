//! Matsubara convolutions with the intermediate-representation (IR) basis (feature `sparse-ir`).
//!
//! Method: J. Otsuki, M. Ohzeki, H. Shinaoka and K. Yoshimi, PRB 101, 035144
//! (2020); H. Shinaoka et al., SciPost Phys. Lect. Notes 63 (2022). Library:
//! sparse-ir-rs (M. Wallerberger et al., SoftwareX 21, 101266 (2023) for the
//! sparse-ir family). Application to Migdal–Eliashberg theory: H. Mori,
//! T. Nomoto, R. Arita and E. R. Margine, PRB 110, 064505 (2024).
//!
//! With `f(tau) = T sum_n exp(-i omega_n tau) f(i omega_n)` (sparse-ir
//! convention), the convolution `C(i omega_n) = sum_m f(i omega_m) lambda(i omega_n - i omega_m)`
//! is the product `C(tau) = f(tau) lambda(tau) / T`, evaluated on the τ sampling
//! points and fitted back. The product of a fermionic `f` and the bosonic
//! `lambda` is fermionic at every τ, so the result does not depend on whether
//! the τ points lie in `(0, beta)` (sparse-ir for Python/Julia) or in
//! `[-beta/2, beta/2]` (sparse-ir-rs default): sparse-ir evaluates negative τ
//! with the (anti)periodic extension. The sum `sum_m f(i omega_m)` of an even,
//! decaying `f` is `f(tau = 0^+) / T`, evaluated exactly with `u_l(0^+)`.
//!
//! Odd and even real functions of omega are passed as `i f` and `f`; both
//! have real IR coefficients, so the positive-frequency samplings are used.
//! All fits and evaluations are batched over the rows of the input.

use crate::matsubara::{check_rows, FrequencySum, Parity};
use crate::Error;
use num_complex::Complex64;
use sparse_ir::{
    Basis, Bosonic, BosonicBasis, Fermionic, FermionicBasis, LogisticKernel, MatsubaraFreq,
    MatsubaraSamplingPositiveOnly, SVEResult, TauSampling, TworkType,
};
use tenferro_tensor::TypedTensor;

/// Choice of the imaginary-time sampling points.
#[derive(Debug, Clone, PartialEq)]
pub enum TauPoints {
    /// sparse-ir-rs default points.
    Default,
    /// Given points as `tau / beta` in `[-1, 1]`.
    ///
    /// EPW's IR object files store `x` in `[-1, 1]` with `tau = beta (x + 1) / 2`,
    /// so pass `(x + 1) / 2`, not `x`.
    Scaled(Vec<f64>),
}

/// The singular-value expansion of the logistic kernel, computed once and
/// reused for every temperature (`omega_max = Lambda T`).
#[derive(Debug, Clone)]
pub struct IrBasisSet {
    lambda: f64,
    epsilon: f64,
    sve: SVEResult,
}

impl IrBasisSet {
    /// Compute the SVE for `Lambda = beta omega_max` and accuracy `epsilon`.
    ///
    /// # Errors
    ///
    /// [`Error::SparseIr`] if the SVE fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let set = eliashberg::IrBasisSet::new(1e2, 1e-8)?;
    /// assert_eq!(set.lambda(), 1e2);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(lambda: f64, epsilon: f64) -> Result<Self, Error> {
        let sve = sparse_ir::compute_sve(
            LogisticKernel::new(lambda)?,
            Some(epsilon),
            None,
            None,
            TworkType::Auto,
        )?;
        Ok(Self {
            lambda,
            epsilon,
            sve,
        })
    }

    /// Dimensionless cutoff Lambda.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(eliashberg::IrBasisSet::new(10.0, 1e-6)?.lambda(), 10.0);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    fn bases(&self, beta: f64) -> Result<(FermionicBasis, BosonicBasis), Error> {
        let kernel = LogisticKernel::new(self.lambda)?;
        let fermionic = FermionicBasis::from_sve_result(
            kernel,
            beta,
            self.sve.clone(),
            Some(self.epsilon),
            None,
        )?;
        let bosonic = BosonicBasis::from_sve_result(
            kernel,
            beta,
            self.sve.clone(),
            Some(self.epsilon),
            None,
        )?;
        Ok((fermionic, bosonic))
    }
}

/// [`FrequencySum`] on the positive fermionic IR sampling frequencies, without truncation.
pub struct IrSum {
    temperature: f64,
    indices: Vec<i64>,
    matsubara: MatsubaraSamplingPositiveOnly<Fermionic>,
    tau: TauSampling<Fermionic>,
    /// `lambda(tau) / T` on the τ points.
    kernel_tau: Vec<f64>,
    /// `u_l(0^+) / T`.
    u_zero: Vec<f64>,
}

impl std::fmt::Debug for IrSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrSum")
            .field("temperature", &self.temperature)
            .field("n_matsubara", &self.indices.len())
            .field("n_tau", &self.kernel_tau.len())
            .field("basis_size", &self.u_zero.len())
            .finish()
    }
}

impl IrSum {
    /// Build the sum at temperature `T` (energy units of the kernel; `beta = 1/T`).
    ///
    /// `kernel` returns `lambda(i nu)` for a batch of bosonic frequencies; it is
    /// called once at the bosonic sampling frequencies.
    ///
    /// # Errors
    ///
    /// [`Error::SparseIr`] for basis or sampling failures, errors of `kernel`.
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::*;
    /// let set = IrBasisSet::new(1e3, 1e-10)?;
    /// let sum = IrSum::new(&set, 0.01, TauPoints::Default, |nu| Ok(vec![0.0; nu.len()]))?;
    /// assert!(sum.indices().iter().all(|&n| n >= 0));
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(
        set: &IrBasisSet,
        temperature: f64,
        tau_points: TauPoints,
        kernel: impl FnOnce(&[f64]) -> Result<Vec<f64>, Error>,
    ) -> Result<Self, Error> {
        let beta = 1.0 / temperature;
        let (fermionic, bosonic) = set.bases(beta)?;
        let matsubara = MatsubaraSamplingPositiveOnly::<Fermionic>::new(&fermionic)?;
        let tau = match tau_points {
            TauPoints::Default => TauSampling::<Fermionic>::new(&fermionic)?,
            TauPoints::Scaled(points) => TauSampling::<Fermionic>::with_sampling_points(
                &fermionic,
                points.iter().map(|x| x * beta).collect(),
            )?,
        };
        let indices = matsubara
            .sampling_points()
            .iter()
            .map(|f| (f.n() - 1) / 2)
            .collect();

        // lambda(i nu) at the bosonic sampling frequencies -> coefficients -> lambda(tau).
        let boson_points = MatsubaraSamplingPositiveOnly::<Bosonic>::new(&bosonic)?;
        let nu: Vec<f64> = boson_points
            .sampling_points()
            .iter()
            .map(|f: &MatsubaraFreq<Bosonic>| f.n() as f64 * std::f64::consts::PI * temperature)
            .collect();
        let values: Vec<Complex64> = kernel(&nu)?
            .into_iter()
            .map(|v| Complex64::new(v, 0.0))
            .collect();
        let coeffs = boson_points.fit(&values)?;
        let boson_tau =
            TauSampling::<Bosonic>::with_sampling_points(&bosonic, tau.sampling_points().to_vec())?;
        let kernel_tau = boson_tau
            .evaluate(&coeffs)?
            .into_iter()
            .map(|k| k / temperature)
            .collect();

        let u_zero_matrix = fermionic.evaluate_tau(&[0.0])?;
        let u_zero = u_zero_matrix
            .as_slice()?
            .iter()
            .map(|u| u / temperature)
            .collect();
        Ok(Self {
            temperature,
            indices,
            matsubara,
            tau,
            kernel_tau,
            u_zero,
        })
    }

    /// τ sampling points (in `[-beta, beta]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::*;
    /// let set = IrBasisSet::new(1e2, 1e-8)?;
    /// let sum = IrSum::new(&set, 0.1, TauPoints::Default, |nu| Ok(vec![0.0; nu.len()]))?;
    /// assert!(sum.tau_points().iter().any(|&t| t < 0.0)); // sparse-ir-rs: [-beta/2, beta/2]
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn tau_points(&self) -> &[f64] {
        self.tau.sampling_points()
    }

    /// Fit rows of an odd/even real function given at the positive sampling frequencies.
    fn coefficients(
        &self,
        f: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        let values: Vec<Complex64> = f
            .as_slice()?
            .iter()
            .map(|&v| match parity {
                Parity::Odd => Complex64::new(0.0, v),
                Parity::Even => Complex64::new(v, 0.0),
            })
            .collect();
        let values = TypedTensor::from_vec_col_major(f.shape().to_vec(), values)?;
        Ok(self.matsubara.fit_nd(None, &values, 0)?)
    }
}

impl FrequencySum for IrSum {
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
        let coeffs = self.coefficients(f, parity)?;
        let f_tau = self.tau.evaluate_nd(None, &coeffs, 0)?;
        let ntau = self.kernel_tau.len();
        let product: Vec<f64> = f_tau
            .as_slice()?
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.kernel_tau[i % ntau])
            .collect();
        let product = TypedTensor::from_vec_col_major(f_tau.shape().to_vec(), product)?;
        let c_coeffs = self.tau.fit_nd(None, &product, 0)?;
        let c = self.matsubara.evaluate_nd(None, &c_coeffs, 0)?;
        let out: Vec<f64> = c
            .as_slice()?
            .iter()
            .map(|z| match parity {
                Parity::Odd => z.im,
                Parity::Even => z.re,
            })
            .collect();
        Ok(TypedTensor::from_vec_col_major(c.shape().to_vec(), out)?)
    }

    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        check_rows(f, self.indices.len())?;
        let coeffs = self.coefficients(f, Parity::Even)?;
        let size = self.u_zero.len();
        Ok(coeffs
            .as_slice()?
            .chunks_exact(size)
            .map(|row| row.iter().zip(&self.u_zero).map(|(c, u)| c * u).sum())
            .collect())
    }
}
