//! Test of the distributed parallel sort.
use itertools::Itertools;
use rlst::distributed_tools::{array_tools::is_sorted_array, parallel_sort::parsort};

use mpi::traits::Communicator;
use rand::prelude::*;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let n_per_rank = 1000;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    let arr = (0..n_per_rank).map(|_| rng.random::<f64>()).collect_vec();

    let sorted = parsort(&arr, &world, &mut rng).unwrap();

    assert!(is_sorted_array(&sorted, &world));

    if world.rank() == 0 {
        println!("Array is sorted.");
    }
}
