//! L0 checks: partitions tile the range and gathers do not depend on the rank count.

use eph_comm::{Comm, Partition, SelfComm, ThreadComm};

#[test]
fn block_partitions_tile_the_range() {
    for len in [0, 1, 5, 17, 216] {
        for nranks in 1..=9 {
            let mut next = 0;
            for rank in 0..nranks {
                let p = Partition::block(len, rank, nranks);
                assert_eq!(
                    p.range().start,
                    next,
                    "len {len}, {nranks} ranks, rank {rank}"
                );
                next = p.range().end;
            }
            assert_eq!(next, len);
        }
    }
}

#[test]
fn gather_is_independent_of_rank_count() {
    let work = |p: Partition| p.range().map(|i| (i as f64).sin()).collect::<Vec<_>>();
    let reference = SelfComm.gather_partitioned(37, work).unwrap();
    for nranks in 1..=8 {
        let gathered = ThreadComm::new(nranks)
            .unwrap()
            .gather_partitioned(37, work)
            .unwrap();
        assert_eq!(gathered, reference, "{nranks} ranks");
    }
}

#[test]
fn wrong_result_length_is_reported() {
    let err = ThreadComm::new(2)
        .unwrap()
        .gather_partitioned(4, |_| vec![0.0])
        .unwrap_err();
    assert!(matches!(err, eph_comm::Error::PartitionLength { .. }));
}
