//! Gather a vector to all ranks.

mod approx_inv_sqrt_accuracy;

use std::rc::Rc;

use approx::assert_relative_eq;

use mpi::traits::Communicator;

use rlst::prelude::*;

const NDIM: usize = 20;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank() as usize;

    let index_layout = Rc::new(IndexLayout::from_equidistributed_chunks(NDIM, 1, &world));

    let vec = DistributedVector::<_, f64>::new(index_layout.clone());

    let local_index_range = vec.index_layout().index_range(rank).unwrap();

    for (index, var_count) in (local_index_range.0..local_index_range.1).enumerate() {
        vec.local_mut()[[index]] = var_count as f64;
    }

    let mut arr = rlst_dynamic_array1!(f64, [NDIM]);

    vec.gather_to_all(arr.r_mut());

    for (index, elem) in arr.iter().enumerate() {
        assert_relative_eq!(elem, index as f64, epsilon = 1E-12);
    }
}
