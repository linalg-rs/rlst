pub mod csc_mat;
pub mod csr_mat;
pub mod mpi_csr_mat;
pub mod tools;

pub enum SparseMatType {
    Csr,
    Csc,
}
