//! Gather a vector on the root rank.

use approx::assert_relative_eq;

use mpi::traits::Communicator;

use rlst::prelude::*;

const ROOT: usize = 0;
const NDIM: usize = 20;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank() as usize;

    let index_layout = EquiDistributedIndexLayout::new(NDIM, 1, &world);

    let vec = DistributedVector::<f64, _>::new(&index_layout);

    let local_index_range = vec.index_layout().index_range(rank).unwrap();

    for (index, var_count) in (local_index_range.0..local_index_range.1).enumerate() {
        vec.local_mut()[[index]] = var_count as f64;
    }

    if rank == ROOT {
        let mut arr = rlst_dynamic_array1!(f64, [NDIM]);
        vec.gather_to_rank_root(arr.r_mut());

        for index in 0..NDIM {
            assert_relative_eq!(index as f64, arr[[index]], epsilon = 1E-12);
        }
    } else {
        vec.gather_to_rank(ROOT);
    }
}
