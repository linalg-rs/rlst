//! Scatter a vector on the root rank.
//? mpirun -n {{NPROCESSES}} --features "mpi"

use std::rc::Rc;

use approx::assert_relative_eq;

use mpi::traits::Communicator;

use rlst::prelude::*;

const ROOT: usize = 0;
const NDIM: usize = 20;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank() as usize;

    let index_layout = Rc::new(IndexLayout::from_equidistributed_chunks(NDIM, 1, &world));

    let mut vec = DistributedVector::<_, f64>::new(index_layout);

    let local_index_range = vec.index_layout().index_range(rank).unwrap();

    if rank == ROOT {
        let mut arr = rlst_dynamic_array1!(f64, [NDIM]);
        for (index, elem) in arr.iter_mut().enumerate() {
            *elem = index as f64;
        }
        vec.scatter_from_root(arr.r_mut());
    } else {
        vec.scatter_from(ROOT);
    }

    for (actual, expected) in
        itertools::izip!(vec.local().iter(), local_index_range.0..local_index_range.1)
    {
        assert_relative_eq!(actual, expected as f64, epsilon = 1E-12);
    }
}
