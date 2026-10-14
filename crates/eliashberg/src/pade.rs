//! N-point Padé approximants by the continued-fraction method of
//! H. J. Vidberg and J. W. Serene, J. Low Temp. Phys. 29, 179 (1977).
//!
//! Given values `u_i = f(z_i)` at `N` points, the approximant is
//! `C_N(z) = a_1 / (1 + a_2 (z - z_1) / (1 + a_3 (z - z_2) / (1 + ...)))`
//! with `a_p = g_p(z_p)`, `g_1(z_i) = u_i` and
//! `g_p(z) = (g_{p-1}(z_{p-1}) - g_{p-1}(z)) / ((z - z_{p-1}) g_{p-1}(z))`.
//! It is evaluated with the three-term recursion
//! `A_{n+1} = A_n + (z - z_n) a_{n+1} A_{n-1}` (same for `B`), `C_N = A_N / B_N`.

use crate::Error;
use num_complex::Complex64;

/// A Padé approximant through `N` points.
#[derive(Debug, Clone, PartialEq)]
pub struct Pade {
    points: Vec<Complex64>,
    coefficients: Vec<Complex64>,
}

impl Pade {
    /// Fit the approximant through `(z_i, u_i)`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for mismatched or empty inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64 as C;
    /// // f(z) = 1 / (z - 2) is reproduced exactly by a 2-point approximant.
    /// let z = [C::new(0.0, 1.0), C::new(0.0, 3.0)];
    /// let u: Vec<C> = z.iter().map(|z| 1.0 / (z - 2.0)).collect();
    /// let p = eliashberg::Pade::new(&z, &u)?;
    /// let w = C::new(5.0, 0.5);
    /// assert!((p.evaluate(w) - 1.0 / (w - 2.0)).norm() < 1e-14);
    /// // The approximant interpolates the data.
    /// assert!((p.evaluate(z[1]) - u[1]).norm() < 1e-14);
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn new(points: &[Complex64], values: &[Complex64]) -> Result<Self, Error> {
        let n = points.len();
        if n == 0 || values.len() != n {
            return Err(Error::InvalidInput(format!(
                "Padé needs matching non-empty points and values, got {} and {}",
                n,
                values.len()
            )));
        }
        // g holds g_p(z_i) for i >= p in place.
        let mut g = values.to_vec();
        let mut coefficients = Vec::with_capacity(n);
        coefficients.push(g[0]);
        for p in 1..n {
            let previous = g[p - 1];
            for i in p..n {
                // INVARIANT: `g / g` is kept on purpose: it is EPW 6.1's operation order
                // (`pade_coeff`), and it propagates NaN/inf like EPW instead of returning 1.
                #[allow(clippy::eq_op)]
                let ratio = g[i] / g[i];
                g[i] = (previous / g[i] - ratio) / (points[i] - points[p - 1]);
            }
            coefficients.push(g[p]);
        }
        Ok(Self {
            points: points.to_vec(),
            coefficients,
        })
    }

    /// Evaluate the approximant at `z`.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64 as C;
    /// let z = [C::new(0.0, 1.0)];
    /// let p = eliashberg::Pade::new(&z, &[C::new(3.0, 0.0)])?;
    /// assert_eq!(p.evaluate(C::new(7.0, 0.0)), C::new(3.0, 0.0));
    /// # Ok::<(), eliashberg::Error>(())
    /// ```
    pub fn evaluate(&self, z: Complex64) -> Complex64 {
        let one = Complex64::new(1.0, 0.0);
        let (mut a_prev, mut a) = (Complex64::new(0.0, 0.0), self.coefficients[0]);
        let (mut b_prev, mut b) = (one, one);
        for n in 1..self.coefficients.len() {
            let factor = (z - self.points[n - 1]) * self.coefficients[n];
            let a_next = a + factor * a_prev;
            let b_next = b + factor * b_prev;
            (a_prev, a) = (a, a_next);
            (b_prev, b) = (b, b_next);
        }
        a / b
    }
}

/// Number of Matsubara points used for the continuation, as EPW: `percent * nfreq / 100`,
/// rounded up to an even number and kept at most `nfreq`.
///
/// # Examples
///
/// ```
/// assert_eq!(eliashberg::pade_point_count(31, 90), 28);
/// assert_eq!(eliashberg::pade_point_count(54, 90), 48);
/// ```
pub fn pade_point_count(nfreq: usize, percent: usize) -> usize {
    let mut n = percent * nfreq / 100;
    if !n.is_multiple_of(2) {
        n += 1;
    }
    if n > nfreq {
        n -= 2;
    }
    n
}

/// Z, Delta and chi of every state continued to real frequencies, arrays `(nomega, nstates)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RealAxisSolution {
    /// Real frequencies (eV).
    pub omega: Vec<f64>,
    /// Z(omega).
    pub z: Vec<Complex64>,
    /// Delta(omega) (eV).
    pub delta: Vec<Complex64>,
    /// chi(omega) (eV).
    pub shift: Vec<Complex64>,
}

/// Continue each state's Z, Delta and chi from the first `npoints` positive Matsubara
/// frequencies of `solution` to the real frequencies `omega` (no broadening).
///
/// # Errors
///
/// [`Error::InvalidInput`] if `npoints` is zero or exceeds the number of frequencies.
///
/// # Examples
///
/// ```
/// use eliashberg::*;
/// // A constant gap continues to the same constant.
/// let sol = AnisotropicSolution { omega: vec![0.01, 0.03], z: vec![1.5; 2], delta: vec![0.002; 2],
///     shift: vec![0.0; 2], z_normal: vec![1.5; 2], iterations: 1, residual: 0.0, outer_gap: vec![] };
/// let real = continue_to_real_axis(&sol, 2, &[0.001, 0.005])?;
/// assert!((real.delta[1].re - 0.002).abs() < 1e-15);
/// # Ok::<(), eliashberg::Error>(())
/// ```
pub fn continue_to_real_axis(
    solution: &crate::AnisotropicSolution,
    npoints: usize,
    omega: &[f64],
) -> Result<RealAxisSolution, Error> {
    let nf = solution.omega.len();
    if npoints == 0 || npoints > nf {
        return Err(Error::InvalidInput(format!(
            "Padé needs 1..={nf} points, got {npoints}"
        )));
    }
    let ns = solution.z.len() / nf;
    let points: Vec<Complex64> = solution.omega[..npoints]
        .iter()
        .map(|&w| Complex64::new(0.0, w))
        .collect();
    let nw = omega.len();
    let mut out = [
        vec![Complex64::default(); nw * ns],
        vec![Complex64::default(); nw * ns],
        vec![Complex64::default(); nw * ns],
    ];
    for s in 0..ns {
        for (field, target) in [&solution.z, &solution.delta, &solution.shift]
            .into_iter()
            .zip(out.iter_mut())
        {
            let values: Vec<Complex64> = field[nf * s..nf * s + npoints]
                .iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect();
            let pade = Pade::new(&points, &values)?;
            for (i, &w) in omega.iter().enumerate() {
                target[i + nw * s] = pade.evaluate(Complex64::new(w, 0.0));
            }
        }
    }
    let [z, delta, shift] = out;
    Ok(RealAxisSolution {
        omega: omega.to_vec(),
        z,
        delta,
        shift,
    })
}

/// Gap edge: the first real frequency below `omega[max_index]` where Re Delta(omega) = omega
/// (linear interpolation, Re Delta > 0 on both sides), or Re Delta at the first
/// frequency if there is none (EPW's rule).
///
/// # Examples
///
/// ```
/// let w = [0.001, 0.002, 0.003, 0.004];
/// let d = [0.0025, 0.0025, 0.0025, 0.0025].map(|x| num_complex::Complex64::new(x, 0.0));
/// assert!((eliashberg::gap_edge(&w, &d, 4) - 0.0025).abs() < 1e-15);
/// ```
pub fn gap_edge(omega: &[f64], delta: &[Complex64], max_index: usize) -> f64 {
    for i in 0..omega.len().saturating_sub(1) {
        let (d1, d2) = (delta[i].re, delta[i + 1].re);
        let (v3, v4) = (d1 - omega[i], d2 - omega[i + 1]);
        if i + 1 < max_index && d1 > 0.0 && d2 > 0.0 && v3 * v4 < 0.0 {
            return (v3 * omega[i + 1] - v4 * omega[i]) / (v3 - v4);
        }
    }
    delta[0].re
}
