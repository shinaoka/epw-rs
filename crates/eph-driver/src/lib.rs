//! q-loop driver for the phonon self-energy and alpha^2F (the Pb minimal example).
//!
//! Steps: interpolate electrons on the fine k mesh, determine E_F and the
//! Fermi window, then distribute blocks of the fine q mesh over ranks
//! ([`eph_comm::Comm`]). Each rank interpolates phonons and g(k, q) for the
//! q-blocks it owns and evaluates the linewidths; the per-q results are
//! gathered in global order and alpha^2F is formed from them. Blocks are fixed
//! globally (`Config::q_block`), so the result is bit-identical for any number
//! of ranks.
//!
//! # Examples
//!
//! ```
//! let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
//!     .join("../../tests/fixtures/pb_epw_metal/wannier");
//! let model = eph_io_epw::read_wannier_model(&dir).unwrap();
//! let g = eph_io_epw::read_epmatwp(&dir.join("pb.epmatwp"), eph_io_epw::eph_dims(&model)).unwrap();
//! let config = eph_driver::Config { k_mesh: [2, 2, 2], q_mesh: [2, 2, 2], ..eph_driver::Config::epw_metal_test() };
//! let result = eph_driver::run(&model, &g, &config, &eph_comm::SelfComm)?;
//! assert_eq!(result.linewidths.len(), 8);
//! # Ok::<(), eph_driver::Error>(())
//! ```

pub mod output;

use bz_mesh::UniformMesh;
use eph_comm::{Comm, Partition};
use eph_selfen::{
    EliashbergFunction, EliashbergParams, FermiSurface, Linewidth, LinewidthParams, QPointData,
};
use eph_wannier_data::{EphMatrixSource, WannierModel};
use smearing::{BandSet, FermiLevel, Smearing};
use std::ops::Range;
use tenferro_tensor::TypedTensor;
use wannier_interp::{HermitianPart, Interpolator};

/// Driver errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Interpolation failed.
    #[error("interpolation failed")]
    Interp(#[from] wannier_interp::Error),
    /// Self-energy evaluation failed.
    #[error("self-energy evaluation failed")]
    SelfEnergy(#[from] eph_selfen::Error),
    /// Fermi-level search failed.
    #[error("Fermi level search failed")]
    Fermi(#[from] smearing::Error),
    /// Mesh error.
    #[error("invalid mesh")]
    Mesh(#[from] bz_mesh::Error),
    /// Distribution error.
    #[error("distribution failed")]
    Comm(#[from] eph_comm::Error),
    /// The CPU backend could not be created.
    #[error("cannot create the CPU backend")]
    Backend(#[source] Box<tenferro_cpu::CpuBackendError>),
    /// Unsupported setting.
    #[error("unsupported setting: {0}")]
    Unsupported(String),
    /// Placeholder for the remaining q-points of a rank that already reported an error.
    #[error("rank aborted after an earlier error")]
    RankAborted,
    /// No band lies within the Fermi window.
    #[error("no band within fsthick = {fsthick_ev} eV of E_F; increase fsthick")]
    EmptyWindow {
        /// Window half-width in eV.
        fsthick_ev: f64,
    },
}

impl From<tenferro_cpu::CpuBackendError> for Error {
    fn from(e: tenferro_cpu::CpuBackendError) -> Self {
        Self::Backend(Box::new(e))
    }
}

/// Calculation settings in EPW input units (eV, meV, K, cm^-1).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Fine k mesh (`nkf1..3`).
    pub k_mesh: [usize; 3],
    /// Fine q mesh (`nqf1..3`); must divide the k mesh.
    pub q_mesh: [usize; 3],
    /// Fermi window half-width (`fsthick`, eV).
    pub fsthick_ev: f64,
    /// Electron smearing (`degaussw`, eV).
    pub degaussw_ev: f64,
    /// Smearing type for E_F and N_F (`ngaussw`).
    pub ngaussw: i32,
    /// Temperature of the occupations (`temps`, K).
    pub temperature_k: f64,
    /// Phonon smearing of alpha^2F (`degaussq`, meV).
    pub degaussq_mev: f64,
    /// Number of phonon smearings (`nqsmear`).
    pub nqsmear: usize,
    /// Phonon smearing increment (`delta_qsmear`, meV).
    pub delta_qsmear_mev: f64,
    /// Frequency points of alpha^2F (`nqstep`).
    pub nqstep: usize,
    /// Acoustic cut-off (`eps_acoustic`, cm^-1).
    pub eps_acoustic_cmm1: f64,
    /// q-points interpolated together.
    pub q_block: usize,
    /// R_p vectors read from the vertex source at once.
    pub rp_block: usize,
    /// CPU threads per rank. Results are reproducible for a fixed thread count
    /// but change at the last-bit level with it (tenferro splits reductions by
    /// thread), so keep this fixed when comparing runs. The default is 1.
    pub threads_per_rank: usize,
    /// Hermitian part of H(k) used inside the q loop. E_F, N_F and the Fermi
    /// window always use the symmetrised H(k).
    pub q_loop_hamiltonian: HermitianPart,
}

impl Config {
    /// Settings of QE `test-suite/epw_metal/epw1.in`, other values at EPW defaults.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(eph_driver::Config::epw_metal_test().k_mesh, [6, 6, 6]);
    /// ```
    pub fn epw_metal_test() -> Self {
        Self {
            k_mesh: [6, 6, 6],
            q_mesh: [6, 6, 6],
            fsthick_ev: 6.0,
            degaussw_ev: 0.1,
            ngaussw: 1,
            temperature_k: 300.0,
            degaussq_mev: 0.05,
            nqsmear: 10,
            delta_qsmear_mev: 0.05,
            nqstep: 500,
            eps_acoustic_cmm1: 0.1,
            q_block: 16,
            rp_block: 16,
            threads_per_rank: 1,
            // EPW 6.1 convention for the q loop (see `HermitianPart`).
            q_loop_hamiltonian: HermitianPart::LowerTriangle,
        }
    }
}

/// Results of [`run`].
#[derive(Debug, Clone, PartialEq)]
pub struct Results {
    /// Fermi level found on the fine k mesh.
    pub fermi: FermiLevel,
    /// Fermi level of the coarse DFT calculation (Ry).
    pub fermi_energy_coarse: f64,
    /// DOS per spin at E_F (states / Ry / cell).
    pub dos_per_spin: f64,
    /// Bands within the Fermi window (0-based).
    pub band_window: Range<usize>,
    /// Smearing used for E_F and N_F.
    pub smearing: Smearing,
    /// q-points (crystal coordinates) in global order.
    pub q_points: Vec<[f64; 3]>,
    /// Weight of each q-point.
    pub q_weight: f64,
    /// Linewidths per q-point, global order.
    pub linewidths: Vec<Linewidth>,
    /// Eliashberg function.
    pub eliashberg: EliashbergFunction,
}

/// Run the phonon self-energy and alpha^2F calculation.
///
/// # Errors
///
/// [`Error::Unsupported`] for non-collinear data or an unknown smearing code,
/// [`Error::EmptyWindow`] if no band is near E_F, and the errors of the
/// underlying crates.
///
/// # Examples
///
/// See the crate-level example.
pub fn run<Cm: Comm>(
    model: &WannierModel,
    vertex: &dyn EphMatrixSource,
    config: &Config,
    comm: &Cm,
) -> Result<Results, Error> {
    if model.crystal.noncolin {
        return Err(Error::Unsupported("non-collinear data".into()));
    }
    let smearing = Smearing::from_qe_code(config.ngaussw)
        .ok_or_else(|| Error::Unsupported(format!("ngaussw = {}", config.ngaussw)))?;
    let k_mesh = UniformMesh::new(config.k_mesh)?;
    let q_mesh = UniformMesh::new(config.q_mesh)?;
    let shift = k_mesh.shift_table(&q_mesh)?;
    let nk = k_mesh.len();
    let nb = model.nbnd;

    let sigma = phys_units::ev_to_ry(config.degaussw_ev);
    let fsthick = phys_units::ev_to_ry(config.fsthick_ev);
    let eps_acoustic = phys_units::cmm1_to_ry(config.eps_acoustic_cmm1);

    let backend = || -> Result<tenferro_cpu::CpuBackend, Error> {
        Ok(tenferro_cpu::CpuBackend::with_threads(
            config.threads_per_rank.max(1),
        )?)
    };
    let mut interp = Interpolator::with_backend(backend()?);
    let bands = interp.electrons_with(model, &k_mesh, HermitianPart::Symmetrized)?;
    // Spin-degenerate weights.
    let weights = vec![2.0 / nk as f64; nk];
    let band_set = BandSet::new(&bands.energies, &weights, nb)?;
    let fermi = smearing::fermi_level(
        &band_set,
        model.crystal.wannier_electrons(),
        sigma,
        smearing,
    )?;
    let ef = fermi.energy;
    let dos_per_spin = band_set.dos_at(ef, sigma, smearing) / 2.0;
    let near: Vec<usize> = (0..nb * nk)
        .filter(|&i| (bands.energies[i] - ef).abs() < fsthick)
        .map(|i| i % nb)
        .collect();
    let (lo, hi) = match (near.iter().min(), near.iter().max()) {
        (Some(&lo), Some(&hi)) => (lo, hi),
        _ => {
            return Err(Error::EmptyWindow {
                fsthick_ev: config.fsthick_ev,
            })
        }
    };
    let fs = FermiSurface {
        fermi_energy: ef,
        dos_per_spin,
        band_window: lo..hi + 1,
    };
    let params = LinewidthParams {
        fsthick,
        sigma,
        temperature: phys_units::kelvin_to_ry(config.temperature_k),
        eps_acoustic,
    };

    // The distributed axis is the global q-block: a block is always interpolated as a
    // whole, so its floating-point work does not depend on the number of ranks.
    let loop_bands = match config.q_loop_hamiltonian {
        HermitianPart::Symmetrized => None,
        part => Some(interp.electrons_with(model, &k_mesh, part)?),
    };
    let loop_bands = loop_bands.as_ref().unwrap_or(&bands);

    let nq = q_mesh.len();
    let q_block = config.q_block.max(1);
    let blocks: Vec<Vec<usize>> = (0..nq)
        .collect::<Vec<_>>()
        .chunks(q_block)
        .map(<[usize]>::to_vec)
        .collect();
    let per_block = comm.gather_partitioned(blocks.len(), |part: Partition| {
        linewidths_for_blocks(
            &blocks[part.range()],
            model,
            vertex,
            config,
            &k_mesh,
            &q_mesh,
            &shift,
            loop_bands,
            &weights,
            &fs,
            &params,
        )
    })?;
    let mut linewidths = Vec::with_capacity(nq);
    for block in per_block {
        linewidths.extend(block?);
    }

    let q_weight = 1.0 / nq as f64;
    let a2f_params = EliashbergParams {
        nstep: config.nqstep,
        sigma: phys_units::mev_to_ry(config.degaussq_mev),
        nsmear: config.nqsmear,
        sigma_step: phys_units::mev_to_ry(config.delta_qsmear_mev),
        eps_acoustic,
    };
    let mut backend = backend()?;
    let eliashberg =
        eph_selfen::eliashberg_function(&mut backend, &linewidths, q_weight, &a2f_params)?;

    Ok(Results {
        fermi,
        fermi_energy_coarse: model.fermi_energy_coarse,
        dos_per_spin,
        band_window: fs.band_window.clone(),
        smearing,
        q_points: (0..nq).map(|iq| q_mesh.point(iq)).collect(),
        q_weight,
        linewidths,
        eliashberg,
    })
}

/// Work of one rank: the q-blocks it owns, each returning the linewidths of its q-points.
#[allow(clippy::too_many_arguments)]
// INVARIANT: private helper receiving the loop invariants computed once in `run`.
fn linewidths_for_blocks(
    blocks: &[Vec<usize>],
    model: &WannierModel,
    vertex: &dyn EphMatrixSource,
    config: &Config,
    k_mesh: &UniformMesh,
    q_mesh: &UniformMesh,
    shift: &[usize],
    bands: &wannier_interp::ElectronBands,
    weights: &[f64],
    fs: &FermiSurface,
    params: &LinewidthParams,
) -> Vec<Result<Vec<Linewidth>, Error>> {
    let setup = || -> Result<_, Error> {
        let threads = config.threads_per_rank.max(1);
        let mut interp =
            Interpolator::with_backend(tenferro_cpu::CpuBackend::with_threads(threads)?);
        let backend = tenferro_cpu::CpuBackend::with_threads(threads)?;
        let phonons = interp.phonons(model, q_mesh)?;
        Ok((interp, backend, phonons))
    };
    let (mut interp, mut backend, phonons) = match setup() {
        Ok(p) => p,
        Err(e) => {
            let mut out = vec![Err(e)];
            out.extend((1..blocks.len()).map(|_| Err(Error::RankAborted)));
            return out;
        }
    };
    blocks
        .iter()
        .map(|block| {
            block_linewidths(
                block,
                &mut interp,
                &mut backend,
                &phonons,
                model,
                vertex,
                config,
                k_mesh,
                q_mesh,
                shift,
                bands,
                weights,
                fs,
                params,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
// INVARIANT: private helper; the arguments are the per-rank state plus the loop invariants of `run`.
fn block_linewidths(
    block: &[usize],
    interp: &mut Interpolator,
    backend: &mut tenferro_cpu::CpuBackend,
    phonons: &wannier_interp::Phonons,
    model: &WannierModel,
    vertex: &dyn EphMatrixSource,
    config: &Config,
    k_mesh: &UniformMesh,
    q_mesh: &UniformMesh,
    shift: &[usize],
    bands: &wannier_interp::ElectronBands,
    weights: &[f64],
    fs: &FermiSurface,
    params: &LinewidthParams,
) -> Result<Vec<Linewidth>, Error> {
    let (nb, nk, nm) = (model.nbnd, k_mesh.len(), phonons.nmodes);
    let modes = phonons
        .displacements
        .as_slice()
        .map_err(wannier_interp::Error::from)?;
    let q_points: Vec<[f64; 3]> = block.iter().map(|&iq| q_mesh.point(iq)).collect();
    let block_modes: Vec<_> = block
        .iter()
        .flat_map(|&iq| modes[iq * nm * nm..(iq + 1) * nm * nm].iter().copied())
        .collect();
    let block_modes = TypedTensor::from_vec_col_major(vec![nm, nm, block.len()], block_modes)
        .map_err(wannier_interp::Error::from)?;
    let g_eq = interp.vertex_rp_to_q(vertex, &model.ws_g, &q_points, config.rp_block)?;
    let g = interp.vertex_to_bloch(
        &g_eq,
        &model.ws_k,
        k_mesh,
        q_mesh,
        block,
        bands,
        &block_modes,
    )?;
    let g = g.as_slice().map_err(wannier_interp::Error::from)?;
    let per_q = nb * nb * nk * nm;
    let mut out = Vec::with_capacity(block.len());
    for (slot, &iq) in block.iter().enumerate() {
        let energies_kq: Vec<f64> = (0..nk)
            .flat_map(|ik| {
                let ikq = shift[ik + nk * iq];
                bands.energies[ikq * nb..(ikq + 1) * nb].iter().copied()
            })
            .collect();
        let data = QPointData {
            nbnd: nb,
            omega: &phonons.frequencies[iq * nm..(iq + 1) * nm],
            g: &g[slot * per_q..(slot + 1) * per_q],
            energies_k: &bands.energies,
            energies_kq: &energies_kq,
            weights,
        };
        out.push(eph_selfen::phonon_linewidth(backend, &data, fs, params)?);
    }
    Ok(out)
}
