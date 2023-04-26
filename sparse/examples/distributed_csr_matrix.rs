use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rlst_sparse::{
    traits::{
        index_layout::IndexLayout,
        indexable_vector::{IndexableVector, IndexableVectorView, IndexableVectorViewMut},
    },
    vector::{DefaultMpiVector, DefaultSerialVector},
};

use rlst_io::matrix_market::read_coordinate_mm;

pub fn main() {
    pub use mpi::traits::*;
    pub use rlst_sparse::ghost_communicator::GhostCommunicator;
    pub use rlst_sparse::index_layout::DefaultMpiIndexLayout;
    pub use rlst_sparse::sparse::csr_mat::CsrMatrix;
    pub use rlst_sparse::sparse::mpi_csr_mat::MpiCsrMatrix;
    pub use rlst_sparse::traits::index_layout;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let n_domain = 27;
    let n_range = 53;

    let csr_mat = match rank {
        0 => {
            let pathname = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/",
                "test_mat_53_27.mtx"
            );
            Some(read_coordinate_mm(pathname).unwrap())
        }
        _ => None,
    };

    let serial_vec = match rank {
        0 => {
            let mut vec_t = DefaultSerialVector::<f64>::new(csr_mat.as_ref().unwrap().shape().1);
            let mut rng = ChaCha8Rng::seed_from_u64(0);
            vec_t.fill_from_rand_standard_normal(&mut rng);
            Some(vec_t)
        }
        _ => None,
    };

    let expected_result = match rank {
        0 => {
            let mut vec_t = DefaultSerialVector::<f64>::new(csr_mat.as_ref().unwrap().shape().0);
            csr_mat.as_ref().unwrap().matmul(
                1.0,
                serial_vec.as_ref().unwrap().view().unwrap().data(),
                1.0,
                vec_t.view_mut().unwrap().data_mut(),
            );

            Some(vec_t)
        }
        _ => None,
    };

    let domain_layout = DefaultMpiIndexLayout::new(n_domain, &world);
    let range_layout = DefaultMpiIndexLayout::new(n_range, &world);

    let dist_mat = MpiCsrMatrix::from_csr(csr_mat, &domain_layout, &range_layout, &world);

    let mut distributed_vec = DefaultMpiVector::<f64, _>::new(&domain_layout);
    distributed_vec.fill_from_root(&serial_vec).unwrap();

    let mut result_vec = DefaultMpiVector::<f64, _>::new(&range_layout);

    dist_mat.matmul(1.0, &distributed_vec, 1.0, &mut result_vec);

    let serial_result = result_vec.to_root();

    if rank == 0 {}
}
