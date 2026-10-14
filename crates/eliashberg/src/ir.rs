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
//! All fits and evaluations are batched over the trailing axes of the input.
//!
//! Sampling points change between sparse-ir versions and implementations; to
//! reproduce another code, pin them with [`IrSettings`].

use crate::anisotropic::{check_shape, AnisotropicModel, StateFrequencySum};
use crate::matsubara::{check_rows, FrequencySum, Parity};
use crate::Error;
use num_complex::Complex64;
use sparse_ir::{
    Basis, Bosonic, BosonicBasis, Fermionic, FermionicBasis, LogisticKernel, MatsubaraFreq,
    MatsubaraSamplingPositiveOnly, SVEResult, TauSampling, TworkType,
};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_tensor::{BackendSessionHost, TypedTensor};

/// Sampling points; `None` uses the sparse-ir-rs defaults.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IrSettings {
    /// Imaginary times as `tau / beta` in `[-1, 1]`.
    ///
    /// EPW's IR object files store `x` in `[-1, 1]` with `tau = beta (x + 1) / 2`,
    /// so pass `(x + 1) / 2`, not `x`.
    pub tau: Option<Vec<f64>>,
    /// Fermionic indices `n >= 0` (`omega_n = (2n + 1) pi T`).
    pub fermionic: Option<Vec<i64>>,
    /// Bosonic indices `m >= 0` (`nu_m = 2 m pi T`).
    pub bosonic: Option<Vec<i64>>,
    /// Reproduce EPW 6.1: when the basis size is odd, its fermionic
    /// `evaluate_matsubara` omits the last coefficient. Off by default.
    pub epw_drop_last_coefficient: bool,
    /// Reproduce EPW 6.1's `eps_cut_ir` in [`IrStateSum`]: IR coefficients with
    /// `|c_l| / max_l |c_l| < cut` are set to zero in every kernel row `(s, q, s')`
    /// and in the normal Green's-function row of every state. `None` (default) disables it.
    pub epw_coefficient_cut: Option<f64>,
}

/// Zero the coefficients of each column (axis 0 = basis index) below `cut` times the column maximum.
fn cut_coefficients(data: &mut [f64], size: usize, cut: f64) {
    for column in data.chunks_exact_mut(size) {
        let max = column.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for c in column.iter_mut() {
            if c.abs() < cut * max {
                *c = 0.0;
            }
        }
    }
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

/// Samplings shared by [`IrSum`] and [`IrStateSum`].
struct IrSampling {
    temperature: f64,
    indices: Vec<i64>,
    matsubara: MatsubaraSamplingPositiveOnly<Fermionic>,
    tau: TauSampling<Fermionic>,
    boson_matsubara: MatsubaraSamplingPositiveOnly<Bosonic>,
    boson_tau: TauSampling<Bosonic>,
    /// `u_l(0^+) / T`.
    u_zero: Vec<f64>,
    drop_last: bool,
}

impl IrSampling {
    fn new(set: &IrBasisSet, temperature: f64, points: &IrSettings) -> Result<Self, Error> {
        let beta = 1.0 / temperature;
        let (fermionic, bosonic) = set.bases(beta)?;
        let matsubara = match &points.fermionic {
            None => MatsubaraSamplingPositiveOnly::<Fermionic>::new(&fermionic)?,
            Some(ns) => MatsubaraSamplingPositiveOnly::<Fermionic>::with_sampling_points(
                &fermionic,
                ns.iter()
                    .map(|n| MatsubaraFreq::new(2 * n + 1))
                    .collect::<Result<_, _>>()?,
            )?,
        };
        let tau = match &points.tau {
            None => TauSampling::<Fermionic>::new(&fermionic)?,
            Some(xs) => TauSampling::<Fermionic>::with_sampling_points(
                &fermionic,
                xs.iter().map(|x| x * beta).collect(),
            )?,
        };
        let boson_matsubara = match &points.bosonic {
            None => MatsubaraSamplingPositiveOnly::<Bosonic>::new(&bosonic)?,
            Some(ms) => MatsubaraSamplingPositiveOnly::<Bosonic>::with_sampling_points(
                &bosonic,
                ms.iter()
                    .map(|m| MatsubaraFreq::new(2 * m))
                    .collect::<Result<_, _>>()?,
            )?,
        };
        let boson_tau =
            TauSampling::<Bosonic>::with_sampling_points(&bosonic, tau.sampling_points().to_vec())?;
        let indices = matsubara
            .sampling_points()
            .iter()
            .map(|f| (f.n() - 1) / 2)
            .collect();
        let u_zero_matrix = fermionic.evaluate_tau(&[0.0])?;
        let u_zero: Vec<f64> = u_zero_matrix
            .as_slice()?
            .iter()
            .map(|u| u / temperature)
            .collect();
        let drop_last = points.epw_drop_last_coefficient && u_zero.len() % 2 == 1;
        Ok(Self {
            temperature,
            indices,
            matsubara,
            tau,
            boson_matsubara,
            boson_tau,
            u_zero,
            drop_last,
        })
    }

    fn boson_frequencies(&self) -> Vec<f64> {
        let pi_t = std::f64::consts::PI * self.temperature;
        self.boson_matsubara
            .sampling_points()
            .iter()
            .map(|f| f.n() as f64 * pi_t)
            .collect()
    }

    /// Real bosonic kernel values `(nboson, ...)` -> `lambda(tau) / T` on the τ points `(ntau, ...)`.
    fn kernel_tau(&self, values: &TypedTensor<f64>) -> Result<TypedTensor<f64>, Error> {
        self.kernel_tau_from_coefficients(&self.kernel_coefficients(values)?)
    }

    /// Real bosonic kernel values `(nboson, ...)` -> IR coefficients `(size, ...)`.
    fn kernel_coefficients(&self, values: &TypedTensor<f64>) -> Result<TypedTensor<f64>, Error> {
        let complex: Vec<Complex64> = values
            .as_slice()?
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        let complex = TypedTensor::from_vec_col_major(values.shape().to_vec(), complex)?;
        Ok(self.boson_matsubara.fit_nd(None, &complex, 0)?)
    }

    /// Bosonic coefficients `(size, ...)` -> `lambda(tau) / T` on the τ points `(ntau, ...)`.
    fn kernel_tau_from_coefficients(
        &self,
        coeffs: &TypedTensor<f64>,
    ) -> Result<TypedTensor<f64>, Error> {
        let tau = self.boson_tau.evaluate_nd(None, coeffs, 0)?;
        let scaled: Vec<f64> = tau
            .as_slice()?
            .iter()
            .map(|k| k / self.temperature)
            .collect();
        Ok(TypedTensor::from_vec_col_major(
            tau.shape().to_vec(),
            scaled,
        )?)
    }

    fn size(&self) -> usize {
        self.u_zero.len()
    }

    /// Fit odd/even real functions given at the positive fermionic points (axis 0).
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

    fn to_tau(&self, f: &TypedTensor<f64>, parity: Parity) -> Result<TypedTensor<f64>, Error> {
        Ok(self
            .tau
            .evaluate_nd(None, &self.coefficients(f, parity)?, 0)?)
    }

    fn back_from_tau(
        &self,
        c_tau: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        let mut coeffs = self.tau.fit_nd(None, c_tau, 0)?;
        if self.drop_last {
            let size = self.u_zero.len();
            let mut data = coeffs.as_slice()?.to_vec();
            for column in data.chunks_exact_mut(size) {
                column[size - 1] = 0.0;
            }
            coeffs = TypedTensor::from_vec_col_major(coeffs.shape().to_vec(), data)?;
        }
        let c = self.matsubara.evaluate_nd(None, &coeffs, 0)?;
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

    fn sum_even(&self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        let coeffs = self.coefficients(f, Parity::Even)?;
        Ok(coeffs
            .as_slice()?
            .chunks_exact(self.u_zero.len())
            .map(|row| row.iter().zip(&self.u_zero).map(|(c, u)| c * u).sum())
            .collect())
    }
}

/// [`FrequencySum`] on the positive fermionic IR sampling frequencies, without truncation.
pub struct IrSum {
    sampling: IrSampling,
    /// `lambda(tau) / T` on the τ points.
    kernel_tau: Vec<f64>,
}

impl std::fmt::Debug for IrSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrSum")
            .field("temperature", &self.sampling.temperature)
            .field("n_matsubara", &self.sampling.indices.len())
            .field("n_tau", &self.kernel_tau.len())
            .field("basis_size", &self.sampling.u_zero.len())
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
    /// let sum = IrSum::new(&set, 0.01, &IrSettings::default(), |nu| Ok(vec![0.0; nu.len()]))?;
    /// assert!(sum.indices().iter().all(|&n| n >= 0));
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(
        set: &IrBasisSet,
        temperature: f64,
        points: &IrSettings,
        kernel: impl FnOnce(&[f64]) -> Result<Vec<f64>, Error>,
    ) -> Result<Self, Error> {
        let sampling = IrSampling::new(set, temperature, points)?;
        let nu = sampling.boson_frequencies();
        let values = TypedTensor::from_vec_col_major(vec![nu.len()], kernel(&nu)?)?;
        let kernel_tau = sampling.kernel_tau(&values)?.as_slice()?.to_vec();
        Ok(Self {
            sampling,
            kernel_tau,
        })
    }

    /// τ sampling points (in `[-beta, beta]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::*;
    /// let set = IrBasisSet::new(1e2, 1e-8)?;
    /// let sum = IrSum::new(&set, 0.1, &IrSettings::default(), |nu| Ok(vec![0.0; nu.len()]))?;
    /// assert!(sum.tau_points().iter().any(|&t| t < 0.0)); // sparse-ir-rs: [-beta/2, beta/2]
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn tau_points(&self) -> &[f64] {
        self.sampling.tau.sampling_points()
    }
}

impl FrequencySum for IrSum {
    fn temperature(&self) -> f64 {
        self.sampling.temperature
    }

    fn indices(&self) -> &[i64] {
        &self.sampling.indices
    }

    fn convolve(
        &mut self,
        f: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        check_rows(f, self.sampling.indices.len())?;
        let f_tau = self.sampling.to_tau(f, parity)?;
        let ntau = self.kernel_tau.len();
        let product: Vec<f64> = f_tau
            .as_slice()?
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.kernel_tau[i % ntau])
            .collect();
        let product = TypedTensor::from_vec_col_major(f_tau.shape().to_vec(), product)?;
        self.sampling.back_from_tau(&product, parity)
    }

    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        check_rows(f, self.sampling.indices.len())?;
        self.sampling.sum_even(f)
    }
}

/// [`StateFrequencySum`] with the IR basis: the kernel matrix `L_{ss'}` is
/// transformed to τ once; each convolution is a batched matrix-vector product per τ point.
pub struct IrStateSum {
    sampling: IrSampling,
    /// `L_{ss'}(tau) / T`, shape `(ntau, nstates, nstates)`.
    kernel_tau: TypedTensor<f64>,
    backend: CpuBackend,
    cut: Option<f64>,
}

impl std::fmt::Debug for IrStateSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrStateSum")
            .field("temperature", &self.sampling.temperature)
            .field("n_matsubara", &self.sampling.indices.len())
            .field("kernel_tau_shape", &self.kernel_tau.shape())
            .finish()
    }
}

impl IrStateSum {
    /// Build the sum for `model` at temperature `T` (eV).
    ///
    /// # Errors
    ///
    /// [`Error::SparseIr`] for basis or sampling failures, tensor errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use eliashberg::*;
    /// let m = AnisotropicModel { energies: vec![0.0], fs_weights: vec![1.0], coulomb: vec![0.0],
    ///     kernel_entries: vec![KernelEntry { s: 0, s_prime: 0, q: 0, omega: 0.05, weight: 0.025 }] };
    /// let set = IrBasisSet::new(1e3, 1e-10)?;
    /// let sum = IrStateSum::new(tenferro_cpu::CpuBackend::new(), &set, &m, 0.002, &IrSettings::default())?;
    /// assert!(!sum.indices().is_empty());
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(
        backend: CpuBackend,
        set: &IrBasisSet,
        model: &AnisotropicModel,
        temperature: f64,
        points: &IrSettings,
    ) -> Result<Self, Error> {
        let sampling = IrSampling::new(set, temperature, points)?;
        let nu = sampling.boson_frequencies();
        let ns = model.nstates();
        let mut backend = backend;
        let kernel_tau = match points.epw_coefficient_cut {
            None => {
                // L[s, s', l] -> (l, s, s') for the fit along the bosonic axis.
                let kernel = model.kernel_matsubara(&nu)?;
                let kernel = backend.with_backend_session(|s| [&kernel].einsum("sql->lsq", s))?;
                sampling.kernel_tau(&kernel)?
            }
            Some(cut) => cut_kernel_tau(&sampling, model, &nu, cut)?,
        };
        debug_assert_eq!(kernel_tau.shape()[1..], [ns, ns]);
        Ok(Self {
            sampling,
            kernel_tau,
            backend,
            cut: points.epw_coefficient_cut,
        })
    }
}

/// Kernel in τ with EPW's per-row cut: rows `(s, q, s')` are fitted and cut separately,
/// then summed into `L_{ss'}(tau) / T`.
fn cut_kernel_tau(
    sampling: &IrSampling,
    model: &AnisotropicModel,
    nu: &[f64],
    cut: f64,
) -> Result<TypedTensor<f64>, Error> {
    let ns = model.nstates();
    let mut keys: Vec<(usize, usize, usize)> = model
        .kernel_entries
        .iter()
        .map(|e| (e.s, e.s_prime, e.q))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let row_of =
        |key: &(usize, usize, usize)| keys.binary_search(key).expect("key collected above");
    let (nb, nrows) = (nu.len(), keys.len());
    let mut values = vec![0.0; nb * nrows];
    for e in &model.kernel_entries {
        let row = row_of(&(e.s, e.s_prime, e.q));
        for (l, v) in nu.iter().enumerate() {
            values[l + nb * row] += e.weight * 2.0 * e.omega / (e.omega * e.omega + v * v);
        }
    }
    let values = TypedTensor::from_vec_col_major(vec![nb, nrows], values)?;
    let coeffs = sampling.kernel_coefficients(&values)?;
    let mut data = coeffs.as_slice()?.to_vec();
    cut_coefficients(&mut data, sampling.size(), cut);
    let coeffs = TypedTensor::from_vec_col_major(coeffs.shape().to_vec(), data)?;
    let rows_tau = sampling.kernel_tau_from_coefficients(&coeffs)?;
    let rows_tau = rows_tau.as_slice()?;
    let ntau = rows_tau.len() / nrows;
    let mut out = vec![0.0; ntau * ns * ns];
    for (row, &(s, sp, _)) in keys.iter().enumerate() {
        let target = ntau * (s + ns * sp);
        for (o, v) in out[target..target + ntau]
            .iter_mut()
            .zip(&rows_tau[row * ntau..(row + 1) * ntau])
        {
            *o += v;
        }
    }
    Ok(TypedTensor::from_vec_col_major(vec![ntau, ns, ns], out)?)
}

impl IrStateSum {
    fn convolve_tau(
        &mut self,
        f_tau: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        let c_tau = self
            .backend
            .with_backend_session(|s| [&self.kernel_tau, f_tau].einsum("tsq,tq->ts", s))?;
        self.sampling.back_from_tau(&c_tau, parity)
    }
}

impl StateFrequencySum for IrStateSum {
    fn temperature(&self) -> f64 {
        self.sampling.temperature
    }

    fn indices(&self) -> &[i64] {
        &self.sampling.indices
    }

    fn convolve(
        &mut self,
        f: &TypedTensor<f64>,
        parity: Parity,
    ) -> Result<TypedTensor<f64>, Error> {
        let ns = self.kernel_tau.shape()[1];
        check_shape(f, self.sampling.indices.len(), ns)?;
        let f_tau = self.sampling.to_tau(f, parity)?;
        self.convolve_tau(&f_tau, parity)
    }

    fn sum_even(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        let ns = self.kernel_tau.shape()[1];
        check_shape(f, self.sampling.indices.len(), ns)?;
        self.sampling.sum_even(f)
    }

    fn sum_even_rows(&mut self, f: &TypedTensor<f64>) -> Result<Vec<f64>, Error> {
        check_rows(f, self.sampling.indices.len())?;
        let coeffs = self.sampling.coefficients(f, Parity::Even)?;
        let mut data = coeffs.as_slice()?.to_vec();
        if let Some(cut) = self.cut {
            // EPW cuts the anomalous rows of the Coulomb term (but not those of mu*).
            cut_coefficients(&mut data, self.sampling.size(), cut);
        }
        Ok(data
            .chunks_exact(self.sampling.size())
            .map(|row| {
                row.iter()
                    .zip(&self.sampling.u_zero)
                    .map(|(c, u)| c * u)
                    .sum()
            })
            .collect())
    }

    fn convolve_normal(
        &mut self,
        z: &TypedTensor<f64>,
        xi: &TypedTensor<f64>,
    ) -> Result<(TypedTensor<f64>, TypedTensor<f64>), Error> {
        let Some(cut) = self.cut else {
            return Ok((
                self.convolve(z, Parity::Odd)?,
                self.convolve(xi, Parity::Even)?,
            ));
        };
        let ns = self.kernel_tau.shape()[1];
        check_shape(z, self.sampling.indices.len(), ns)?;
        check_shape(xi, self.sampling.indices.len(), ns)?;
        // EPW cuts the coefficients of the complex row -(i omega Z + xi) / Theta as one row;
        // its coefficients are -(c_z + c_xi).
        let cz = self.sampling.coefficients(z, Parity::Odd)?;
        let cx = self.sampling.coefficients(xi, Parity::Even)?;
        let size = self.sampling.size();
        let (mut dz, mut dx) = (cz.as_slice()?.to_vec(), cx.as_slice()?.to_vec());
        for (col_z, col_x) in dz.chunks_exact_mut(size).zip(dx.chunks_exact_mut(size)) {
            let max = col_z
                .iter()
                .zip(col_x.iter())
                .fold(0.0_f64, |m, (a, b)| m.max((a + b).abs()));
            for (a, b) in col_z.iter_mut().zip(col_x.iter_mut()) {
                if (*a + *b).abs() < cut * max {
                    *a = 0.0;
                    *b = 0.0;
                }
            }
        }
        let shape = cz.shape().to_vec();
        let tz = self.sampling.tau.evaluate_nd(
            None,
            &TypedTensor::from_vec_col_major(shape.clone(), dz)?,
            0,
        )?;
        let tx =
            self.sampling
                .tau
                .evaluate_nd(None, &TypedTensor::from_vec_col_major(shape, dx)?, 0)?;
        Ok((
            self.convolve_tau(&tz, Parity::Odd)?,
            self.convolve_tau(&tx, Parity::Even)?,
        ))
    }
}
