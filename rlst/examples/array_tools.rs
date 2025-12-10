//! Test and example scripts for array tools.
use itertools::{Itertools, izip};
use mpi::traits::Communicator;
use rand::prelude::*;
use rlst::distributed_tools::{
    array_tools::{
        communicate_back, gather_to_all, gather_to_rank, global_inclusive_cumsum, global_max,
        global_min, is_sorted_array,
    },
    scatterv, scatterv_root,
};

/// Test the global exclusive cumsum
fn test_exclusive_cumsum<C: Communicator>(comm: &C) {
    let n = 10;

    let rank = comm.rank();

    let arr = (0..n).map(|elem| rank * n + elem).collect_vec();

    let cumsum = global_inclusive_cumsum(&arr, comm);

    let mut expected = rank * n * (rank * n - 1) / 2;

    for (index, elem) in cumsum.iter().enumerate() {
        expected += rank * n + index as i32;
        assert_eq!(*elem, expected);
    }
}

/// Test scatter and gather data
fn test_scatter_gather<C: Communicator, R: Rng>(comm: &C, rng: &mut R) {
    let n = 1000;

    let root = comm.size() - 1;

    let rank = comm.rank();

    let dist_arr;

    if rank == root {
        let size = comm.size() as usize;

        let mut counts = vec![n / size; size];
        counts[0] += n % size;

        assert_eq!(counts.iter().sum::<usize>(), n);

        let arr = (0..n).map(|_| rng.random::<f64>()).collect_vec();
        dist_arr = scatterv_root(comm, &counts, &arr);

        let actual = gather_to_rank(&dist_arr, root as usize, comm).unwrap();

        for (a, e) in izip!(actual, arr) {
            assert_eq!(a, e);
        }
    } else {
        dist_arr = scatterv(comm, root as usize);
        gather_to_rank(&dist_arr, root as usize, comm);
    }
}

fn test_gather_to_all<C: Communicator>(comm: &C) {
    let rank = comm.rank();

    let n = 100;

    let arr = (0..n).map(|elem| rank * n + elem).collect_vec();

    let out = gather_to_all(&arr, comm);

    for (index, o) in out.iter().enumerate() {
        assert_eq!(index as i32, *o);
    }
}

fn test_communicate_back<C: Communicator>(comm: &C) {
    let rank = comm.rank();

    let n = 100;

    let arr = (0..n).map(|elem| rank * n + elem).collect_vec();

    let actual = communicate_back(&arr, comm);

    if rank != comm.size() - 1 {
        assert_eq!(actual.unwrap(), (1 + rank) * n);
    }
}

fn test_is_sorted_array<C: Communicator>(comm: &C) {
    let rank = comm.rank();

    let n = 100;

    let mut arr = (0..n).map(|elem| rank * n + elem).collect_vec();

    assert!(is_sorted_array(&arr, comm));

    arr[0] += 2;

    assert!(!is_sorted_array(&arr, comm));

    arr[0] -= 2;

    arr[1] += 2;

    assert!(!is_sorted_array(&arr, comm));
}

fn test_global_min_max<C: Communicator>(comm: &C) {
    let rank = comm.rank();
    let size = comm.size();

    let n = 100;

    let arr = (0..n).map(|elem| rank * n + elem).collect_vec();

    assert_eq!(global_max(&arr, comm), n * size - 1);
    assert_eq!(global_min(&arr, comm), 0);
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    test_exclusive_cumsum(&world);
    test_scatter_gather(&world, &mut rng);
    test_gather_to_all(&world);
    test_communicate_back(&world);
    test_is_sorted_array(&world);
    test_global_min_max(&world);

    if world.rank() == 0 {
        println!("All tests passed.");
    }
}
