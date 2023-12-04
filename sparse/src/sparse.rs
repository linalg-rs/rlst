pub mod csc_mat;
pub mod csr_mat;
pub mod mpi_csr_mat;
pub mod tools;
pub mod umfpack;

#[derive(Copy, Clone)]
pub enum SparseMatType {
    Csr,
    Csc,
}
