//! Distribution vocabulary for the downstream e-ph calculation.
//!
//! The physics crates never talk to a communicator directly. They work on the
//! [`Partition`] they are handed (a contiguous range of global indices) and
//! return results tagged by global index. A [`Comm`] implementation runs the
//! per-rank work and assembles the results in global-index order. Sums over a
//! distributed axis are formed afterwards from the assembled array, so the
//! result is independent of the number of ranks and threads (see
//! `docs/crate_design_ja.md`, section 2.1).
//!
//! Implementations:
//! - [`SelfComm`]: one rank owning the full range.
//! - [`ThreadComm`]: `n` ranks emulated by scoped threads, used to test that a
//!   partitioned run is bit-identical to a single-rank run before MPI exists.
//!
//! # Examples
//!
//! ```
//! use eph_comm::{Comm, SelfComm, ThreadComm};
//!
//! let square = |p: eph_comm::Partition| p.range().map(|i| i * i).collect::<Vec<_>>();
//! let one = SelfComm.gather_partitioned(10, square)?;
//! let four = ThreadComm::new(4)?.gather_partitioned(10, square)?;
//! assert_eq!(one, four);
//! # Ok::<(), eph_comm::Error>(())
//! ```

use std::ops::Range;

/// Errors raised while distributing work.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A communicator was requested with zero ranks.
    #[error("a communicator needs at least one rank; use SelfComm for serial runs")]
    NoRanks,
    /// A rank returned a number of results different from the size of its partition.
    #[error("rank {rank} returned {got} results for a partition of length {expected}")]
    PartitionLength {
        /// Rank that misbehaved.
        rank: usize,
        /// Partition length.
        expected: usize,
        /// Number of results returned.
        got: usize,
    },
    /// A worker thread panicked.
    #[error("worker thread for rank {rank} panicked")]
    WorkerPanicked {
        /// Rank of the thread.
        rank: usize,
    },
}

/// Contiguous block of a global index range owned by one rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Partition {
    global_len: usize,
    start: usize,
    end: usize,
}

impl Partition {
    /// The whole range `0..global_len`.
    ///
    /// # Examples
    ///
    /// ```
    /// let p = eph_comm::Partition::full(5);
    /// assert_eq!(p.range(), 0..5);
    /// ```
    pub fn full(global_len: usize) -> Self {
        Self {
            global_len,
            start: 0,
            end: global_len,
        }
    }

    /// Balanced block partition: the first `global_len % nranks` ranks get one extra index.
    ///
    /// # Examples
    ///
    /// ```
    /// use eph_comm::Partition;
    /// assert_eq!(Partition::block(10, 0, 4).range(), 0..3);
    /// assert_eq!(Partition::block(10, 3, 4).range(), 8..10);
    /// ```
    pub fn block(global_len: usize, rank: usize, nranks: usize) -> Self {
        let base = global_len / nranks;
        let extra = global_len % nranks;
        let start = rank * base + rank.min(extra);
        let len = base + usize::from(rank < extra);
        Self {
            global_len,
            start,
            end: start + len,
        }
    }

    /// Owned global indices.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(eph_comm::Partition::full(3).range().count(), 3);
    /// ```
    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Number of owned indices.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(eph_comm::Partition::block(7, 0, 2).len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the partition owns no index.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!(eph_comm::Partition::block(1, 1, 2).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Length of the global range.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(eph_comm::Partition::block(7, 1, 2).global_len(), 7);
    /// ```
    pub fn global_len(&self) -> usize {
        self.global_len
    }
}

/// How partial results of an accumulator are combined across ranks.
///
/// Accumulators declare their kind; the driver applies it. `Sum` reductions are
/// evaluated from per-index results gathered in global order, which keeps them
/// deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    /// Sum over the distributed axis (alpha^2F, lambda, DOS).
    Sum,
    /// One value per global index, owned by exactly one rank (gamma_q, lambda_q, Sigma_k).
    GatherByIndex,
}

/// Runs per-rank work over a distributed axis and gathers the results.
pub trait Comm {
    /// Number of ranks.
    fn size(&self) -> usize;

    /// Split `0..global_len` into one [`Partition`] per rank, run `work` on each
    /// and return all results in global-index order.
    ///
    /// # Errors
    ///
    /// [`Error::PartitionLength`] if `work` returns a vector whose length
    /// differs from its partition, [`Error::WorkerPanicked`] if a thread panics.
    fn gather_partitioned<T, F>(&self, global_len: usize, work: F) -> Result<Vec<T>, Error>
    where
        T: Send,
        F: Fn(Partition) -> Vec<T> + Sync;
}

/// Single-rank communicator.
#[derive(Debug, Clone, Copy, Default)]
pub struct SelfComm;

impl Comm for SelfComm {
    fn size(&self) -> usize {
        1
    }

    fn gather_partitioned<T, F>(&self, global_len: usize, work: F) -> Result<Vec<T>, Error>
    where
        T: Send,
        F: Fn(Partition) -> Vec<T> + Sync,
    {
        let out = work(Partition::full(global_len));
        check_len(0, global_len, out.len())?;
        Ok(out)
    }
}

/// Emulates `n` ranks with scoped threads (test vehicle for partition invariance).
#[derive(Debug, Clone, Copy)]
pub struct ThreadComm {
    nranks: usize,
}

impl ThreadComm {
    /// Create a communicator with `nranks` emulated ranks.
    ///
    /// # Errors
    ///
    /// [`Error::NoRanks`] when `nranks == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use eph_comm::Comm;
    /// assert_eq!(eph_comm::ThreadComm::new(3)?.size(), 3);
    /// # Ok::<(), eph_comm::Error>(())
    /// ```
    pub fn new(nranks: usize) -> Result<Self, Error> {
        if nranks == 0 {
            return Err(Error::NoRanks);
        }
        Ok(Self { nranks })
    }
}

impl Comm for ThreadComm {
    fn size(&self) -> usize {
        self.nranks
    }

    fn gather_partitioned<T, F>(&self, global_len: usize, work: F) -> Result<Vec<T>, Error>
    where
        T: Send,
        F: Fn(Partition) -> Vec<T> + Sync,
    {
        let work = &work;
        let per_rank: Vec<Result<Vec<T>, Error>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..self.nranks)
                .map(|rank| {
                    let part = Partition::block(global_len, rank, self.nranks);
                    scope.spawn(move || (part, work(part)))
                })
                .collect();
            handles
                .into_iter()
                .enumerate()
                .map(|(rank, h)| match h.join() {
                    Ok((part, out)) => check_len(rank, part.len(), out.len()).map(|()| out),
                    Err(_) => Err(Error::WorkerPanicked { rank }),
                })
                .collect()
        });
        // Blocks are contiguous and ordered by rank, so concatenation is global order.
        let mut out = Vec::with_capacity(global_len);
        for chunk in per_rank {
            out.extend(chunk?);
        }
        Ok(out)
    }
}

fn check_len(rank: usize, expected: usize, got: usize) -> Result<(), Error> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::PartitionLength {
            rank,
            expected,
            got,
        })
    }
}
