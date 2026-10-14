//! Uniform Brillouin-zone meshes in crystal coordinates.
//!
//! A point `(i, j, l)` of an `n1 x n2 x n3` mesh has crystal coordinates
//! `(i/n1, j/n2, l/n3)` and the linear index `l + n3 * (j + n2 * i)` (last
//! component fastest). This is the order in which EPW enumerates uniform fine
//! meshes, and it is also the column-major index of an array of shape
//! `[n3, n2, n1]`. Fourier sums onto the mesh therefore use FFT grids of shape
//! [`UniformMesh::fft_shape`], with lattice vectors folded by
//! [`UniformMesh::fold`].
//!
//! # Examples
//!
//! ```
//! let mesh = bz_mesh::UniformMesh::new([2, 3, 4])?;
//! assert_eq!(mesh.len(), 24);
//! assert_eq!(mesh.point(1), [0.0, 0.0, 0.25]);
//! assert_eq!(mesh.fft_shape(), [4, 3, 2]);
//! # Ok::<(), bz_mesh::Error>(())
//! ```

/// Mesh construction and compatibility errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A mesh dimension is zero.
    #[error("mesh dimensions must be positive, got {0:?}")]
    EmptyMesh([usize; 3]),
    /// The q mesh does not divide the k mesh, so k+q is not on the k mesh.
    #[error("q mesh {q:?} does not divide k mesh {k:?}; choose a k mesh that is a multiple of the q mesh")]
    Incommensurate {
        /// k mesh.
        k: [usize; 3],
        /// q mesh.
        q: [usize; 3],
    },
}

/// Uniform `n1 x n2 x n3` mesh including the origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformMesh {
    dims: [usize; 3],
}

impl UniformMesh {
    /// Create a mesh.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyMesh`] if a dimension is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(bz_mesh::UniformMesh::new([0, 1, 1]).is_err());
    /// ```
    pub fn new(dims: [usize; 3]) -> Result<Self, Error> {
        if dims.contains(&0) {
            return Err(Error::EmptyMesh(dims));
        }
        Ok(Self { dims })
    }

    /// Mesh dimensions `[n1, n2, n3]`.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(bz_mesh::UniformMesh::new([6, 6, 6])?.dims(), [6, 6, 6]);
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Number of points.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(bz_mesh::UniformMesh::new([6, 6, 6])?.len(), 216);
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Always false: meshes have at least one point.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(!bz_mesh::UniformMesh::new([1, 1, 1])?.is_empty());
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Shape of the FFT grid whose column-major index equals the mesh index.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(bz_mesh::UniformMesh::new([1, 2, 3])?.fft_shape(), [3, 2, 1]);
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn fft_shape(&self) -> [usize; 3] {
        [self.dims[2], self.dims[1], self.dims[0]]
    }

    /// Integer coordinates `(i, j, l)` of a linear index.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(bz_mesh::UniformMesh::new([2, 3, 4])?.coords(5), [0, 1, 1]);
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn coords(&self, index: usize) -> [usize; 3] {
        let [_, n2, n3] = self.dims;
        [index / (n2 * n3), (index / n3) % n2, index % n3]
    }

    /// Linear index of integer coordinates (taken modulo the mesh).
    ///
    /// # Examples
    ///
    /// ```
    /// let mesh = bz_mesh::UniformMesh::new([2, 3, 4])?;
    /// assert_eq!(mesh.index([0, 1, 1]), 5);
    /// assert_eq!(mesh.index([2, 4, 5]), 5);
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn index(&self, coords: [usize; 3]) -> usize {
        let [n1, n2, n3] = self.dims;
        (coords[2] % n3) + n3 * ((coords[1] % n2) + n2 * (coords[0] % n1))
    }

    /// Crystal coordinates of a point.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(bz_mesh::UniformMesh::new([2, 2, 2])?.point(7), [0.5, 0.5, 0.5]);
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn point(&self, index: usize) -> [f64; 3] {
        let c = self.coords(index);
        std::array::from_fn(|a| c[a] as f64 / self.dims[a] as f64)
    }

    /// Mesh index that a lattice vector `R` (crystal units) occupies on the FFT grid.
    ///
    /// For every mesh point `k`, `exp(2 pi i k.R)` equals the phase of the folded
    /// position, so summing `f(R)` into folded bins and transforming the grid is
    /// exact.
    ///
    /// # Examples
    ///
    /// ```
    /// let mesh = bz_mesh::UniformMesh::new([6, 6, 6])?;
    /// assert_eq!(mesh.fold([-1, 0, 2]), mesh.index([5, 0, 2]));
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn fold(&self, r: [i64; 3]) -> usize {
        let c: [usize; 3] = std::array::from_fn(|a| r[a].rem_euclid(self.dims[a] as i64) as usize);
        self.index(c)
    }

    /// For a q mesh dividing this mesh, the table `kq[k + nk * q] = index(k + q)`.
    ///
    /// # Errors
    ///
    /// [`Error::Incommensurate`] if some `n_k` is not a multiple of `n_q`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bz_mesh::UniformMesh;
    /// let k = UniformMesh::new([4, 4, 4])?;
    /// let q = UniformMesh::new([2, 2, 2])?;
    /// let table = k.shift_table(&q)?;
    /// // q = (1/2, 0, 0) moves k = 0 to k = (2/4, 0, 0)
    /// assert_eq!(table[k.len() * q.index([1, 0, 0])], k.index([2, 0, 0]));
    /// # Ok::<(), bz_mesh::Error>(())
    /// ```
    pub fn shift_table(&self, q_mesh: &UniformMesh) -> Result<Vec<usize>, Error> {
        if (0..3).any(|a| !self.dims[a].is_multiple_of(q_mesh.dims[a])) {
            return Err(Error::Incommensurate {
                k: self.dims,
                q: q_mesh.dims,
            });
        }
        let ratio: [usize; 3] = std::array::from_fn(|a| self.dims[a] / q_mesh.dims[a]);
        let mut table = Vec::with_capacity(self.len() * q_mesh.len());
        for iq in 0..q_mesh.len() {
            let q = q_mesh.coords(iq);
            for ik in 0..self.len() {
                let k = self.coords(ik);
                table.push(self.index(std::array::from_fn(|a| k[a] + q[a] * ratio[a])));
            }
        }
        Ok(table)
    }
}
