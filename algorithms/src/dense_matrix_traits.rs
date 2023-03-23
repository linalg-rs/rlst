pub use rlst_common::types::{IndexType, Scalar};

pub trait DenseMatrixInterface {
    type T: Scalar;

    fn dim() -> (IndexType, IndexType);
    fn stride() -> (IndexType, IndexType);

    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &Self::T;
    fn get(&self, row: IndexType, col: IndexType) -> Option<&Self::T>;

    fn data(&self) -> &[Self::T];
}

// struct DenseMatrix<'a, MatImpl: DenseMatrixInterface> {

//     mat: &'a MatImpl

// }

// mat.as_algorithms().as_lapack().u

// pub trait UpperTriangularSolve<Rhs> {
//     type Output;

//     fn solve_upper_triangular(&self, rhs: &Rhs) -> Self::Output;
// }
