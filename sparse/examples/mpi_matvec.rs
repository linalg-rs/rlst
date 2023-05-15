//? mpirun -n {{NPROCESSES}} --features "mpi"

use approx::assert_ulps_eq;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use mpi::traits::*;
use rlst_common::traits::*;
use rlst_dense::*;
use rlst_io::matrix_market::read_coordinate_mm;
use rlst_sparse::{
    distributed_vector::DistributedVector, index_layout::DefaultMpiIndexLayout,
    sparse::mpi_csr_mat::MpiCsrMatrix,
};

macro_rules! some_on_root {
    ($code:tt, $rank:expr) => {
        match $rank {
            0 => Some($code),
            _ => None,
        }
    };
}

pub fn main() {
    // Intialize the MPI environment
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let root_process = world.process_at_rank(0);

    // Read in the matrix. The matrix only lives on root.
    // Every other Process stores a None.
    let csr_mat = some_on_root!(
        {
            let pathname = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/",
                "test_mat_53_27.mtx"
            );
            read_coordinate_mm(pathname).unwrap()
        },
        rank
    );

    // Define the serial vector on root
    let serial_vec = some_on_root!(
        {
            let mut vec_t = rlst_col_vec![f64, csr_mat.as_ref().unwrap().shape().1];
            let mut rng = ChaCha8Rng::seed_from_u64(0);
            vec_t.fill_from_standard_normal(&mut rng);
            vec_t
        },
        rank
    );

    // Define the expected result vector
    let expected_result = some_on_root!(
        {
            let mut vec_t = rlst_col_vec![f64, csr_mat.as_ref().unwrap().shape().0];
            csr_mat.as_ref().unwrap().matmul(
                1.0,
                serial_vec.as_ref().unwrap().data(),
                1.0,
                vec_t.data_mut(),
            );
            vec_t
        },
        rank
    );

    // Communicate the shape of the matrix to all processes.

    let mut shape: Vec<usize> = vec![0, 0];
    if rank == 0 {
        shape[0] = csr_mat.as_ref().unwrap().shape().0;
        shape[1] = csr_mat.as_ref().unwrap().shape().1;
    }
    root_process.broadcast_into(shape.as_mut_slice());

    // Create the distributed layout structure
    let domain_layout = DefaultMpiIndexLayout::new(shape[1], &world);
    let range_layout = DefaultMpiIndexLayout::new(shape[0], &world);

    // Scatter the sparse matrix from root to all processes
    let dist_mat = MpiCsrMatrix::from_root(csr_mat, &domain_layout, &range_layout, &world);

    // Scatter the serial vector from root to all processes
    let dist_vec = DistributedVector::from_root(&domain_layout, &serial_vec).unwrap();

    // Create a distributed result vector
    let mut result_vec = DistributedVector::<f64, _>::new(&range_layout);

    // Multiply the distributed matrix with the distributed vector
    dist_mat.matmul(1.0, &dist_vec, 1.0, &mut result_vec);

    // Send the distributed vector back to root
    let serial_result = result_vec.to_root();

    // On root check that we have the correct result
    if rank == 0 {
        // Check that the result dimension is correct
        assert_eq!(expected_result.as_ref().unwrap().shape().0, shape[0]);

        // Check that the individual entries are correct
        for (expected, actual) in expected_result
            .as_ref()
            .unwrap()
            .data()
            .iter()
            .zip(serial_result.as_ref().unwrap().data().iter())
        {
            assert_ulps_eq!(expected, actual, max_ulps = 10);
        }
        println!("Parallel and serial results agree.");
    }
}
