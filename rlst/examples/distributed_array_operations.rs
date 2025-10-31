//! Demonstrate operations on a distributed array.

#[cfg(not(feature = "mpi"))]
fn main() {
    println!("WARNING: MPI not enabled.");
}

#[cfg(feature = "mpi")]
pub fn main() {
    use std::rc::Rc;

    use approx::assert_relative_eq;
    use mpi::traits::Communicator;
    use rlst::dense::array::DynArray;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank() as usize;

    let ndofs_per_rank = 1000;

    // We create an index layout.
    let index_layout = Rc::new(rlst::distributed_tools::IndexLayout::from_local_counts(
        ndofs_per_rank,
        &world,
    ));

    // We create a distributed array from an array on root.
    let dist_arr = if rank == 0 {
        // Create a local array on root
        let mut arr = rlst::rlst_dynamic_array!(f64, [index_layout.number_of_global_indices()]);
        arr.fill_from_seed_normally_distributed(1);
        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, index_layout.clone())
    };

    // We gather the distributed array to all processes.
    let arr = dist_arr.gather_to_all();

    // We can now perform operations on the distributed array.
    let dist_norm = (5.0 * dist_arr.sin()).norm_2().unwrap();

    // Let's do the same operation on the gathered array for comparison.
    let expected = arr.sin().scalar_mul(5.0).norm_2().unwrap();

    assert_relative_eq!(dist_norm, expected, epsilon = 1E-10);
}
