fn main() {
    pub use mpi::traits::*;
    pub use rlst_sparse::ghost_communicator::GhostCommunicator;
    pub use rlst_sparse::index_layout::DefaultMpiIndexLayout;
    pub use rlst_sparse::sparse::csr_mat::CsrMatrix;
    pub use rlst_sparse::sparse::mpi_csr_mat::MpiCsrMatrix;
    pub use rlst_sparse::traits::index_layout;

    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let n_domain = 5;
    let n_range = 2;

    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![0, 1, 2, 0, 1, 2];
    let indptr = vec![0, 3, 6];

    let csr_mat: Option<CsrMatrix<f64>>;
    if rank == 0 {
        csr_mat = Some(CsrMatrix::new((2, 3), indices, indptr, values));
    } else {
        csr_mat = None;
    }

    let domain_layout = DefaultMpiIndexLayout::new(n_domain, &world);
    let range_layout = DefaultMpiIndexLayout::new(n_range, &world);

    let dist_mat = MpiCsrMatrix::from_csr(csr_mat, &domain_layout, &range_layout, &world);

    if rank == 1 {
        println!("Indices: {:#?}", dist_mat.indices());
        println!("Data: {:#?}", dist_mat.data());
        println!("Indptr: {:#?}", dist_mat.indptr());
        println!("Shape: {:#?}", dist_mat.shape());
        println!("Local Shape {:#?}", dist_mat.local_shape());
    }
}
