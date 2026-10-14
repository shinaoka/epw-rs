//! The isotropic Eliashberg spectral function alpha^2F(omega) and its moments.

use crate::Error;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_tensor::{BackendSessionHost, TypedTensor};

/// alpha^2F on a uniform frequency grid `omega_i = i * d omega` (i = 1..n), energies in eV.
///
/// The integration weight is `d omega = omega_n / n` (rectangle rule), as in EPW.
#[derive(Debug, Clone, PartialEq)]
pub struct EliashbergSpectrum {
    omega: Vec<f64>,
    a2f: Vec<f64>,
    d_omega: f64,
}

impl EliashbergSpectrum {
    /// Wrap frequencies and alpha^2F values.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for empty or mismatched arrays or a non-positive
    /// last frequency.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.01, 0.02], vec![0.0, 1.0])?;
    /// assert_eq!(s.d_omega(), 0.01);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(omega: Vec<f64>, a2f: Vec<f64>) -> Result<Self, Error> {
        if omega.is_empty() || omega.len() != a2f.len() {
            return Err(Error::InvalidInput(format!(
                "alpha^2F needs matching non-empty arrays, got {} frequencies and {} values",
                omega.len(),
                a2f.len()
            )));
        }
        let last = omega[omega.len() - 1];
        if last <= 0.0 {
            return Err(Error::InvalidInput(
                "alpha^2F frequencies must end above zero".into(),
            ));
        }
        let d_omega = last / omega.len() as f64;
        Ok(Self {
            omega,
            a2f,
            d_omega,
        })
    }

    /// Frequency step (eV).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.5, 1.0], vec![1.0, 1.0])?;
    /// assert_eq!(s.d_omega(), 0.5);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn d_omega(&self) -> f64 {
        self.d_omega
    }

    /// Largest frequency (eV).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.5, 1.0], vec![1.0, 1.0])?;
    /// assert_eq!(s.omega_max(), 1.0);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn omega_max(&self) -> f64 {
        self.omega[self.omega.len() - 1]
    }

    /// `lambda(i nu) = 2 int d omega omega alpha^2F(omega) / (omega^2 + nu^2)` for many `nu`.
    ///
    /// # Errors
    ///
    /// Tensor errors.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.01, 0.02], vec![0.0, 1.0])?;
    /// let mut backend = tenferro_cpu::CpuBackend::new();
    /// let l = s.lambda_matsubara(&mut backend, &[0.0])?;
    /// assert!((l[0] - s.lambda()).abs() < 1e-14);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn lambda_matsubara(
        &self,
        backend: &mut CpuBackend,
        nu: &[f64],
    ) -> Result<Vec<f64>, Error> {
        let n = self.omega.len();
        let mut weights = Vec::with_capacity(n * nu.len());
        for &v in nu {
            weights.extend(
                self.omega
                    .iter()
                    .map(|&w| 2.0 * self.d_omega * w / (w * w + v * v)),
            );
        }
        let weights = TypedTensor::from_vec_col_major(vec![n, nu.len()], weights)?;
        let a2f = TypedTensor::from_vec_col_major(vec![n], self.a2f.clone())?;
        let out = backend.with_backend_session(|s| [&weights, &a2f].einsum("ik,i->k", s))?;
        Ok(out.as_slice()?.to_vec())
    }

    /// `lambda = 2 int alpha^2F / omega`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.01, 0.02], vec![0.0, 1.0])?;
    /// assert!((s.lambda() - 2.0 * 0.01 / 0.02).abs() < 1e-15);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn lambda(&self) -> f64 {
        2.0 * self.d_omega
            * self
                .omega
                .iter()
                .zip(&self.a2f)
                .map(|(w, a)| a / w)
                .sum::<f64>()
    }

    /// Logarithmic average frequency `exp[(2/lambda) int alpha^2F ln(omega)/omega]` (eV).
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.01, 0.02], vec![0.0, 1.0])?;
    /// assert!((s.omega_log() - 0.02).abs() < 1e-15);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn omega_log(&self) -> f64 {
        let moment: f64 = self
            .omega
            .iter()
            .zip(&self.a2f)
            .map(|(w, a)| a * w.ln() / w)
            .sum();
        (2.0 * self.d_omega * moment / self.lambda()).exp()
    }

    /// McMillan T_c (eV) for Coulomb pseudopotential `mu_star`:
    /// `omega_log / 1.2 exp[-1.04 (1 + lambda) / (lambda - mu* (1 + 0.62 lambda))]`.
    ///
    /// Returns `None` when the denominator is not positive.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = eliashberg::EliashbergSpectrum::new(vec![0.01, 0.02], vec![0.0, 1.0])?;
    /// assert!(s.mcmillan_tc(0.1).unwrap() > 0.0);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn mcmillan_tc(&self, mu_star: f64) -> Option<f64> {
        let l = self.lambda();
        let denominator = l - mu_star * (1.0 + 0.62 * l);
        (denominator > 0.0)
            .then(|| self.omega_log() / 1.2 * (-1.04 * (1.0 + l) / denominator).exp())
    }
}
