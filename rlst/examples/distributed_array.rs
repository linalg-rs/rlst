//! Test distributed array functionality.

/// Test three dimensional distributed arrays.
fn test_scatter_dim3<C: mpi::traits::Communicator>(comm: &C) {
    use rlst::{DynArray, assert_array_relative_eq, distributed_tools::IndexLayout};
    use std::rc::Rc;

    let rank = comm.rank() as usize;

    // The leading dofs per rank. Every rank has a different number of dofs.
    let ndofs_per_rank = 10 + 2 * rank;

    // First we create an index layout;
    let index_layout = Rc::new(IndexLayout::from_local_counts(ndofs_per_rank, comm));

    // Let's create an array on the root process and scatter it to all processes.

    // We create the array on each process from the same seed. This allows us to compare
    // the results after scatter.
    let mut arr = DynArray::<f64, 3>::from_shape([index_layout.number_of_global_indices(), 12, 13]);
    arr.fill_from_seed_normally_distributed(1);

    let dist_array = if rank == 0 {
        // Create a local array on root

        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 3>::scatter_from_one(0, index_layout.clone())
    };

    assert_array_relative_eq!(
        dist_array.local,
        arr.r()
            .into_subview([index_layout.scan()[rank], 0, 0], [ndofs_per_rank, 12, 13]),
        1E-10
    );
}

/// Test one dimensional distributed arrays.
fn test_scatter_dim1<C: mpi::traits::Communicator>(comm: &C) {
    use rlst::{DynArray, assert_array_relative_eq, distributed_tools::IndexLayout};
    use std::rc::Rc;

    let rank = comm.rank() as usize;

    // The leading dofs per rank. Every rank has a different number of dofs.
    let ndofs_per_rank = 10 + 2 * rank;

    // First we create an index layout;
    let index_layout = Rc::new(IndexLayout::from_local_counts(ndofs_per_rank, comm));

    // Let's create an array on the root process and scatter it to all processes.

    // We create the array on each process from the same seed. This allows us to compare
    // the results after scatter.
    let mut arr = DynArray::<f64, 1>::from_shape([index_layout.number_of_global_indices()]);
    arr.fill_from_seed_normally_distributed(1);

    let dist_array = if rank == 0 {
        // Create a local array on root

        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, index_layout.clone())
    };

    assert_array_relative_eq!(
        dist_array.local,
        arr.r()
            .into_subview([index_layout.scan()[rank]], [ndofs_per_rank]),
        1E-10
    );
}

fn test_gather_dim3<C: mpi::traits::Communicator>(comm: &C) {
    use rlst::{DynArray, assert_array_relative_eq, distributed_tools::IndexLayout};
    use std::rc::Rc;

    let rank = comm.rank() as usize;

    // The leading dofs per rank. Every rank has a different number of dofs.
    let ndofs_per_rank = 10 + 2 * rank;

    // First we create an index layout;
    let index_layout = Rc::new(IndexLayout::from_local_counts(ndofs_per_rank, comm));

    // Let's create an array on the root process and scatter it to all processes.

    // We create the array on each process from the same seed. This allows us to compare
    // the results after scatter.
    let mut arr = rlst::DynArray::<f64, 1>::from_shape([index_layout.number_of_global_indices()]);
    arr.fill_from_seed_normally_distributed(1);

    let dist_array = if rank == 0 {
        // Create a local array on root

        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, index_layout.clone())
    };

    if rank == 0 {
        let gathered_array = dist_array.gather_to_one_root();

        assert_array_relative_eq!(gathered_array, arr, 1E-10);
    } else {
        dist_array.gather_to_one(0);
    };
}

/// Test one dimensional distributed arrays.
fn test_gather_dim1<C: mpi::traits::Communicator>(comm: &C) {
    use rlst::{DynArray, assert_array_relative_eq, distributed_tools::IndexLayout};
    use std::rc::Rc;

    let rank = comm.rank() as usize;

    // The leading dofs per rank. Every rank has a different number of dofs.
    let ndofs_per_rank = 10 + 2 * rank;

    // First we create an index layout;
    let index_layout = Rc::new(IndexLayout::from_local_counts(ndofs_per_rank, comm));

    // Let's create an array on the root process and scatter it to all processes.

    // We create the array on each process from the same seed. This allows us to compare
    // the results after scatter.
    let mut arr = DynArray::<f64, 1>::from_shape([index_layout.number_of_global_indices()]);
    arr.fill_from_seed_normally_distributed(1);

    let dist_array = if rank == 0 {
        // Create a local array on root

        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, index_layout.clone())
    };

    if rank == 0 {
        let gathered_array = dist_array.gather_to_one_root();

        assert_array_relative_eq!(gathered_array, arr, 1E-10);
    } else {
        dist_array.gather_to_one(0);
    };
}

fn test_gather_to_all_dim3<C: mpi::traits::Communicator>(comm: &C) {
    use rlst::{DynArray, assert_array_relative_eq, distributed_tools::IndexLayout};
    use std::rc::Rc;

    let rank = comm.rank() as usize;

    // The leading dofs per rank. Every rank has a different number of dofs.
    let ndofs_per_rank = 10 + 2 * rank;

    // First we create an index layout;
    let index_layout = Rc::new(IndexLayout::from_local_counts(ndofs_per_rank, comm));

    // Let's create an array on the root process and scatter it to all processes.

    // We create the array on each process from the same seed. This allows us to compare
    // the results after scatter.
    let mut arr = DynArray::<f64, 1>::from_shape([index_layout.number_of_global_indices()]);
    arr.fill_from_seed_normally_distributed(1);

    let dist_array = if rank == 0 {
        // Create a local array on root

        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, index_layout.clone())
    };

    let gathered_array = dist_array.gather_to_all();

    assert_array_relative_eq!(gathered_array, arr, 1E-10);
}

/// Test one dimensional distributed arrays.
fn test_gather_to_all_dim1<C: mpi::traits::Communicator>(comm: &C) {
    use rlst::{DynArray, assert_array_relative_eq, distributed_tools::IndexLayout};
    use std::rc::Rc;

    let rank = comm.rank() as usize;

    // The leading dofs per rank. Every rank has a different number of dofs.
    let ndofs_per_rank = 10 + 2 * rank;

    // First we create an index layout;
    let index_layout = Rc::new(IndexLayout::from_local_counts(ndofs_per_rank, comm));

    // Let's create an array on the root process and scatter it to all processes.

    // We create the array on each process from the same seed. This allows us to compare
    // the results after scatter.
    let mut arr = DynArray::<f64, 1>::from_shape([index_layout.number_of_global_indices()]);
    arr.fill_from_seed_normally_distributed(1);

    let dist_array = if rank == 0 {
        // Create a local array on root

        arr.scatter_from_one_root(index_layout.clone())
    } else {
        DynArray::<f64, 1>::scatter_from_one(0, index_layout.clone())
    };

    let gathered_array = dist_array.gather_to_all();

    assert_array_relative_eq!(gathered_array, arr, 1E-10);
}

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    test_scatter_dim1(&world);
    test_scatter_dim3(&world);
    test_gather_dim1(&world);
    test_gather_dim3(&world);
    test_gather_to_all_dim1(&world);
    test_gather_to_all_dim3(&world);
}
