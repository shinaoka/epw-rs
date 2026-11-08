//! Wannier interpolation of electrons, phonons and the e-ph vertex onto uniform meshes.
//!
//! Formulas (F. Giustino, M. L. Cohen and S. G. Louie, Phys. Rev. B 76,
//! 165108 (2007), Eqs. 22, 31 and 32):
//!
//! - `H(k) = sum_R exp(2 pi i k.R) H(R) / N_R`, diagonalised as `H(k) U_k = U_k e_k`.
//! - `D(q) = sum_R exp(2 pi i q.R) C(R) / (N_R sqrt(M_a M_b))`, eigenvalues `omega^2`,
//!   mode displacement vectors `u_nu = e_nu / sqrt(M_a)`.
//! - `g(k, q) = U_{k+q}^dagger [ sum_{R_e, R_p} exp(2 pi i (k.R_e + q.R_p)) g(R_e, R_p) / (N_{R_e} N_{R_p}) ] U_k`,
//!   followed by the projection on the phonon modes `u_nu(q)`.
//!
//! `N_R` is the Wigner–Seitz degeneracy of `R`. Sums onto a full uniform mesh
//! are evaluated exactly by folding `R` onto the mesh and applying a 3D inverse
//! FFT (`tenferro-fft`); sums onto a block of q-points use a phase matrix and
//! `tenferro-einsum`. Hermitian eigenproblems are solved in batches with
//! `tenferro-linalg`. All arrays are column-major.
//!
//! Matrix element convention (validated against EPW): `g[m, n, k, nu]` couples
//! band `n` at `k` to band `m` at `k+q`. Its square `|g|^2` still has to be divided
//! by `2 omega` to give the e-ph coupling in Ry^2.
//!
//! # Examples
//!
//! ```
//! use wannier_interp::Interpolator;
//! let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
//!     .join("../../tests/fixtures/pb_epw_metal/wannier");
//! let model = eph_io_epw::read_wannier_model(&dir).unwrap();
//! let mesh = bz_mesh::UniformMesh::new([2, 2, 2])?;
//! let mut interp = Interpolator::new();
//! let bands = interp.electrons(&model, &mesh)?;
//! assert_eq!(bands.energies.len(), 4 * 8);
//! # Ok::<(), wannier_interp::Error>(())
//! ```

use bz_mesh::UniformMesh;
use eph_wannier_data::{EphMatrixSource, WannierModel, WsVectors};
use num_complex::Complex64;
use std::ops::Range;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::TypedTensorEinsumExt;
use tenferro_fft::{FftNorm, TensorFftExt};
use tenferro_linalg::TypedTensorLinalgExt;
use tenferro_runtime::TypedTensorSessionOpsExt;
use tenferro_tensor::{Tensor, TypedTensor};

/// Errors from interpolation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tensor operation failed.
    #[error("tensor operation failed")]
    Tensor(#[from] tenferro_tensor::Error),
    /// Einsum failed.
    #[error("einsum failed")]
    Einsum(#[from] tenferro_einsum::Error),
    /// Reading the vertex failed.
    #[error("reading the e-ph vertex failed")]
    Vertex(#[from] eph_wannier_data::Error),
    /// Mesh incompatibility.
    #[error("incompatible meshes")]
    Mesh(#[from] bz_mesh::Error),
}

type C = Complex64;

/// Which Hermitian matrix is diagonalised for H(k).
///
/// The interpolated H(k) is Hermitian only up to the numerical accuracy of
/// H(R) (here ~1e-9 Ry). EPW 6.1 symmetrises it, `(H + H^dagger) / 2`, when it
/// determines E_F and the Fermi window, but its batched routine for the
/// q loop diagonalises the raw H(k) reading only the lower triangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HermitianPart {
    /// `(H + H^dagger) / 2` (default).
    #[default]
    Symmetrized,
    /// The Hermitian matrix defined by the lower triangle of H (EPW q-loop convention).
    LowerTriangle,
}

/// Band energies and Bloch rotations on a k mesh.
#[derive(Debug)]
pub struct ElectronBands {
    /// Number of bands (Wannier functions).
    pub nbnd: usize,
    /// Energies `(nbnd, nk)`, ascending per k (Ry).
    pub energies: Vec<f64>,
    /// Eigenvectors `U[a, n, k]` (Wannier index `a`, band `n`).
    pub rotations: TypedTensor<C>,
}

/// Phonon frequencies and mode displacement vectors on a set of q-points.
#[derive(Debug)]
pub struct Phonons {
    /// Number of modes.
    pub nmodes: usize,
    /// Frequencies `(nmodes, nq)` in Ry; negative for imaginary modes.
    pub frequencies: Vec<f64>,
    /// `u[mu, nu, q] = e_nu[mu](q) / sqrt(M_mu)` (Cartesian displacement `mu`).
    pub displacements: TypedTensor<C>,
}

/// Owns the tensor engine and performs the interpolation steps.
#[derive(Debug)]
pub struct Interpolator {
    backend: CpuEngine,
}

impl Default for Interpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpolator {
    /// Create an interpolator with a default CPU engine.
    ///
    /// # Examples
    ///
    /// ```
    /// let _interp = wannier_interp::Interpolator::new();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the CPU pool cannot be started.
    pub fn new() -> Self {
        Self {
            backend: CpuEngine::new(
                Pool::builder()
                    .build()
                    .expect("cannot start the tenferro CPU pool"),
            ),
        }
    }

    /// Create an interpolator on a configured engine.
    ///
    /// Results are reproducible for a fixed engine thread count but can change
    /// at the last-bit level when the thread count changes.
    ///
    /// # Examples
    ///
    /// ```
    /// let pool = tenferro_cpu::Pool::builder().threads(1).build().unwrap();
    /// let backend = tenferro_cpu::CpuEngine::new(pool);
    /// let _interp = wannier_interp::Interpolator::with_backend(backend);
    /// ```
    pub fn with_backend(backend: CpuEngine) -> Self {
        Self { backend }
    }

    /// Sum `f(R) / N_R` onto every point of `mesh`.
    ///
    /// `values` has shape `(block, nR)`; the result has shape `(block, nk)`.
    fn sum_onto_mesh(
        &mut self,
        values: &[C],
        block: usize,
        ws: &WsVectors,
        mesh: &UniformMesh,
    ) -> Result<TypedTensor<C>, Error> {
        let nk = mesh.len();
        let mut grid = vec![C::default(); block * nk];
        for (ir, (r, &deg)) in ws.vectors.iter().zip(&ws.degeneracy).enumerate() {
            let target = mesh.fold(*r) * block;
            let scale = 1.0 / f64::from(deg);
            for (dst, src) in grid[target..target + block]
                .iter_mut()
                .zip(&values[ir * block..(ir + 1) * block])
            {
                *dst += src * scale;
            }
        }
        let [n3, n2, n1] = mesh.fft_shape();
        let mut tensor = Tensor::from_vec_col_major(vec![block, n3, n2, n1], grid)?;
        // ifft with FftNorm::Forward is the unscaled sum with exp(+2 pi i k.R).
        {
            let s = &mut self.backend;
            for axis in 1..=3 {
                tensor = tensor.ifft(None, axis, FftNorm::Forward, s)?;
            }
        }
        let tensor = tensor.into_typed::<C>().map_err(|f| f.into_parts().1)?;
        let (_, data) = tensor.into_vec_col_major().map_err(|f| f.into_parts().1)?;
        Ok(TypedTensor::from_vec_col_major(vec![block, nk], data)?)
    }

    /// Eigen-decomposition of a batch of matrices `(n, n, batch)` after symmetrisation.
    fn hermitian_eigen(
        &mut self,
        matrices: &TypedTensor<C>,
    ) -> Result<(Vec<f64>, TypedTensor<C>), Error> {
        let s = &mut self.backend;
        let adjoint = [&matrices.conj(s)?].einsum("abk->bak", s)?;
        let doubled = matrices.add(&adjoint, s)?;
        let (values, vectors) = doubled.eigh(s)?;
        // eig(A + A^dagger) / 2 is exact: halving is exact in floating point.
        let values = values.as_slice()?.iter().map(|w| 0.5 * w).collect();
        Ok((values, vectors))
    }

    /// Interpolate the electron Hamiltonian onto `mesh` and diagonalise it.
    ///
    /// # Errors
    ///
    /// [`Error::Tensor`]/[`Error::Einsum`] if a tensor operation fails.
    ///
    /// # Examples
    ///
    /// See the crate-level example.
    pub fn electrons(
        &mut self,
        model: &WannierModel,
        mesh: &UniformMesh,
    ) -> Result<ElectronBands, Error> {
        self.electrons_with(model, mesh, HermitianPart::Symmetrized)
    }

    /// As [`Interpolator::electrons`], choosing how the Hermitian part of H(k) is formed.
    ///
    /// # Errors
    ///
    /// [`Error::Tensor`]/[`Error::Einsum`] if a tensor operation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use wannier_interp::{HermitianPart, Interpolator};
    /// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    ///     .join("../../tests/fixtures/pb_epw_metal/wannier");
    /// let model = eph_io_epw::read_wannier_model(&dir).unwrap();
    /// let mesh = bz_mesh::UniformMesh::new([3, 3, 3])?;
    /// let mut interp = Interpolator::new();
    /// let a = interp.electrons_with(&model, &mesh, HermitianPart::Symmetrized)?;
    /// let b = interp.electrons_with(&model, &mesh, HermitianPart::LowerTriangle)?;
    /// let diff = a.energies.iter().zip(&b.energies).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    /// assert!(diff < 1e-6);
    /// # Ok::<(), wannier_interp::Error>(())
    /// ```
    pub fn electrons_with(
        &mut self,
        model: &WannierModel,
        mesh: &UniformMesh,
        part: HermitianPart,
    ) -> Result<ElectronBands, Error> {
        let nb = model.nbnd;
        let hk = self.sum_onto_mesh(&model.hamiltonian, nb * nb, &model.ws_k, mesh)?;
        let hk = match part {
            HermitianPart::Symmetrized => self.reshape(hk, vec![nb, nb, mesh.len()])?,
            HermitianPart::LowerTriangle => {
                let (_, mut data) = hk.into_vec_col_major().map_err(|f| f.into_parts().1)?;
                for matrix in data.chunks_exact_mut(nb * nb) {
                    for j in 0..nb {
                        matrix[j + nb * j].im = 0.0;
                        for i in 0..j {
                            matrix[i + nb * j] = matrix[j + nb * i].conj();
                        }
                    }
                }
                TypedTensor::from_vec_col_major(vec![nb, nb, mesh.len()], data)?
            }
        };
        // For an exactly Hermitian input the symmetrisation below is the identity.
        let (energies, rotations) = self.hermitian_eigen(&hk)?;
        Ok(ElectronBands {
            nbnd: nb,
            energies,
            rotations,
        })
    }

    /// Interpolate the dynamical matrix onto `mesh` and diagonalise it.
    ///
    /// # Errors
    ///
    /// [`Error::Tensor`]/[`Error::Einsum`] if a tensor operation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    ///     .join("../../tests/fixtures/pb_epw_metal/wannier");
    /// let model = eph_io_epw::read_wannier_model(&dir).unwrap();
    /// let mesh = bz_mesh::UniformMesh::new([1, 1, 1])?;
    /// let ph = wannier_interp::Interpolator::new().phonons(&model, &mesh)?;
    /// // Acoustic sum rule: all three modes vanish at Gamma.
    /// assert!(ph.frequencies.iter().all(|w| w.abs() < 1e-6));
    /// # Ok::<(), wannier_interp::Error>(())
    /// ```
    pub fn phonons(&mut self, model: &WannierModel, mesh: &UniformMesh) -> Result<Phonons, Error> {
        let nm = model.crystal.nmodes();
        let inv_sqrt_mass: Vec<f64> = (0..nm)
            .map(|mu| 1.0 / model.crystal.atom_mass(mu / 3).sqrt())
            .collect();
        let mass_factor: Vec<C> = (0..nm * nm)
            .map(|i| C::new(inv_sqrt_mass[i % nm] * inv_sqrt_mass[i / nm], 0.0))
            .collect();
        let mass_factor = TypedTensor::from_vec_col_major(vec![nm, nm], mass_factor)?;
        let dq = self.sum_onto_mesh(&model.force_constants, nm * nm, &model.ws_q, mesh)?;
        let dq = self.reshape(dq, vec![nm, nm, mesh.len()])?;
        let dq = [&dq, &mass_factor].einsum("abq,ab->abq", &mut self.backend)?;
        let (omega2, modes) = self.hermitian_eigen(&dq)?;
        let frequencies = omega2
            .iter()
            .map(|&w2| w2.signum() * w2.abs().sqrt())
            .collect();
        let inv_sqrt_mass = TypedTensor::from_vec_col_major(
            vec![nm],
            inv_sqrt_mass.iter().map(|&m| C::new(m, 0.0)).collect(),
        )?;
        let displacements = [&modes, &inv_sqrt_mass].einsum("avq,a->avq", &mut self.backend)?;
        Ok(Phonons {
            nmodes: nm,
            frequencies,
            displacements,
        })
    }

    fn reshape(&mut self, t: TypedTensor<C>, shape: Vec<usize>) -> Result<TypedTensor<C>, Error> {
        let (_, data) = t.into_vec_col_major().map_err(|f| f.into_parts().1)?;
        Ok(TypedTensor::from_vec_col_major(shape, data)?)
    }

    /// Transform g(R_e, R_p) to g(R_e, q) for a block of q-points (crystal coordinates).
    ///
    /// Reads the vertex in blocks of `rp_block` R_p vectors (`EphMatrixSource`)
    /// and contracts each block with its phase matrix; blocks are accumulated in
    /// fixed R_p order. Returns shape `(nbnd, nbnd, nrr_k, nmodes, nq)`, divided by
    /// `N_{R_p}` but not yet by `N_{R_e}`.
    ///
    /// # Errors
    ///
    /// [`Error::Vertex`] if reading fails, tensor errors otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    ///     .join("../../tests/fixtures/pb_epw_metal/wannier");
    /// let model = eph_io_epw::read_wannier_model(&dir).unwrap();
    /// let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
    /// let geq = wannier_interp::Interpolator::new().vertex_rp_to_q(&g, &model.ws_g, &[[0.0; 3]], 8)?;
    /// assert_eq!(geq.shape(), &[4, 4, 43, 3, 1]);
    /// # Ok::<(), wannier_interp::Error>(())
    /// ```
    pub fn vertex_rp_to_q(
        &mut self,
        source: &dyn EphMatrixSource,
        ws_g: &WsVectors,
        q_points: &[[f64; 3]],
        rp_block: usize,
    ) -> Result<TypedTensor<C>, Error> {
        let dims = source.dims();
        let block_len = dims.block_len();
        let nq = q_points.len();
        let mut total: Option<TypedTensor<C>> = None;
        let mut buffer: Vec<C> = Vec::new();
        let mut start = 0;
        while start < dims.nrr_g {
            let range: Range<usize> = start..(start + rp_block.max(1)).min(dims.nrr_g);
            buffer.resize(block_len * range.len(), C::default());
            source.read_rp_block(range.clone(), &mut buffer)?;
            let phases: Vec<C> = q_points
                .iter()
                .flat_map(|q| {
                    range.clone().map(move |ir| {
                        let r = ws_g.vectors[ir];
                        let arg = phys_units::TWO_PI
                            * (q[0] * r[0] as f64 + q[1] * r[1] as f64 + q[2] * r[2] as f64);
                        C::from_polar(1.0 / f64::from(ws_g.degeneracy[ir]), arg)
                    })
                })
                .collect();
            let g = TypedTensor::from_vec_col_major(
                vec![block_len, range.len()],
                std::mem::take(&mut buffer),
            )?;
            let p = TypedTensor::from_vec_col_major(vec![range.len(), nq], phases)?;
            let partial = [&g, &p].einsum("xr,rq->xq", &mut self.backend)?;
            total = Some(match total {
                None => partial,
                Some(acc) => acc.add(&partial, &mut self.backend)?,
            });
            buffer = g.into_vec_col_major().map_err(|f| f.into_parts().1)?.1;
            start = range.end;
        }
        let total = total.unwrap_or(TypedTensor::from_vec_col_major(
            vec![block_len, nq],
            vec![C::default(); block_len * nq],
        )?);
        self.reshape(
            total,
            vec![dims.nbnd, dims.nbnd, dims.nrr_k, dims.nmodes, nq],
        )
    }

    /// Complete the vertex: R_e -> every k of `k_mesh`, rotate to the Bloch and phonon-mode bases.
    ///
    /// `g_eq` is the output of [`Interpolator::vertex_rp_to_q`] for the q-points
    /// `q_indices` of `q_mesh`; `bands` are the electrons on `k_mesh`;
    /// `displacements` has shape `(nmodes, nmodes, nq)` for those q-points, in the same order.
    /// Returns `g[m, n, k, nu, iq]` (see the crate docs for the convention).
    ///
    /// # Errors
    ///
    /// [`Error::Mesh`] if `q_mesh` does not divide `k_mesh`, tensor errors otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use bz_mesh::UniformMesh;
    /// let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    ///     .join("../../tests/fixtures/pb_epw_metal/wannier");
    /// let model = eph_io_epw::read_wannier_model(&dir).unwrap();
    /// let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
    /// let (kmesh, qmesh) = (UniformMesh::new([2, 2, 2])?, UniformMesh::new([1, 1, 1])?);
    /// let mut interp = wannier_interp::Interpolator::new();
    /// let bands = interp.electrons(&model, &kmesh)?;
    /// let ph = interp.phonons(&model, &qmesh)?;
    /// let geq = interp.vertex_rp_to_q(&g, &model.ws_g, &[qmesh.point(0)], 43)?;
    /// let gkq = interp.vertex_to_bloch(&geq, &model.ws_k, &kmesh, &qmesh, &[0], &bands, &ph.displacements)?;
    /// assert_eq!(gkq.shape(), &[4, 4, 8, 3, 1]);
    /// # Ok::<(), wannier_interp::Error>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    // INVARIANT: the eight inputs are the independent pieces of one interpolation step
    // (vertex, WS set, two meshes, q selection, bands, modes); bundling them would only rename them.
    pub fn vertex_to_bloch(
        &mut self,
        g_eq: &TypedTensor<C>,
        ws_k: &WsVectors,
        k_mesh: &UniformMesh,
        q_mesh: &UniformMesh,
        q_indices: &[usize],
        bands: &ElectronBands,
        displacements: &TypedTensor<C>,
    ) -> Result<TypedTensor<C>, Error> {
        let shape = g_eq.shape().to_vec();
        let (nb, nm, nq) = (shape[0], shape[3], shape[4]);
        let nk = k_mesh.len();
        let shift = k_mesh.shift_table(q_mesh)?;

        // Bring R_e last so it can be summed onto the k mesh: (a, b, mu, q, R_e).
        let moved = [g_eq].einsum("abrmq->abmqr", &mut self.backend)?;
        let (_, data) = moved.into_vec_col_major().map_err(|f| f.into_parts().1)?;
        let gk = self.sum_onto_mesh(&data, nb * nb * nm * nq, ws_k, k_mesh)?;
        let gk = self.reshape(gk, vec![nb, nb, nm, nq, nk])?;

        // U_{k+q} for every (k, q) of the block, conjugated: (a, m, k, q).
        let u = bands.rotations.as_slice()?;
        let mut ukq = Vec::with_capacity(nb * nb * nk * nq);
        for &iq in q_indices {
            for ik in 0..nk {
                let ikq = shift[ik + nk * iq];
                ukq.extend(
                    u[ikq * nb * nb..(ikq + 1) * nb * nb]
                        .iter()
                        .map(|z| z.conj()),
                );
            }
        }
        let ukq = TypedTensor::from_vec_col_major(vec![nb, nb, nk, nq], ukq)?;

        // Contract pairwise in a fixed order. A single four-operand einsum lets the
        // planner choose the path, and its choice was observed to vary between calls,
        // which changes results at the last-bit level.
        let s = &mut self.backend;
        let right = [&gk, &bands.rotations].einsum("abuqk,bnk->anuqk", s)?;
        let both = [&ukq, &right].einsum("amkq,anuqk->mnukq", s)?;
        Ok([&both, displacements].einsum("mnukq,uvq->mnkvq", s)?)
    }
}
