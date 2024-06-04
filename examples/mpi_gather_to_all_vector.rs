//! Gather a vector to all ranks.

use approx::assert_relative_eq;

use mpi::traits::*;

use rlst::prelude::*;

const NDIM: usize = 20;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank() as usize;

    let index_layout = DefaultMpiIndexLayout::new(NDIM, 1, &world);

    let vec = DistributedVector::<f64, _>::new(&index_layout);

    let local_index_range = vec.index_layout().index_range(rank).unwrap();

    for (index, var_count) in (local_index_range.0..local_index_range.1).enumerate() {
        vec.local_mut()[[index]] = var_count as f64;
    }

    let mut arr = rlst_dynamic_array1!(f64, [NDIM]);

    vec.gather_to_all(arr.view_mut());

    for (index, elem) in arr.iter().enumerate() {
        assert_relative_eq!(elem, index as f64, epsilon = 1E-12);
    }
}
