//! An implementation independent trait to provide access to linear algebra routines.
use std::marker::PhantomData;

use rlst_common::types::Scalar;
use rlst_dense::{Matrix, MatrixImplTrait, SizeIdentifier};
use rlst_sparse::sparse::{csc_mat::CscMatrix, csr_mat::CsrMatrix};

pub trait LinAlg {
    type T: Scalar;
    type Out<'a>
    where
        Self: 'a;

    fn linalg(&self) -> Self::Out<'_>;
}

pub struct DenseMatrixLinAlgBuilder<'a, T: Scalar, Mat> {
    pub(crate) mat: &'a Mat,
    _marker: std::marker::PhantomData<T>,
}

pub struct SparseMatrixLinalgBuilder<'a, T: Scalar, Mat> {
    #[allow(dead_code)]
    pub(crate) mat: &'a Mat,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar, Mat: MatrixImplTrait<T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> LinAlg
    for Matrix<T, Mat, RS, CS>
{
    type T = T;
    type Out<'a> = DenseMatrixLinAlgBuilder<'a, Self::T, Self> where Self: 'a;
    fn linalg(&self) -> Self::Out<'_> {
        DenseMatrixLinAlgBuilder {
            mat: self,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar> LinAlg for CsrMatrix<T> {
    type T = T;
    type Out<'a> = SparseMatrixLinalgBuilder<'a, Self::T, CsrMatrix<T>> where Self: 'a;
    fn linalg(&self) -> Self::Out<'_> {
        SparseMatrixLinalgBuilder {
            mat: self,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar> LinAlg for CscMatrix<T> {
    type T = T;
    type Out<'a> = SparseMatrixLinalgBuilder<'a, Self::T, CscMatrix<T>> where Self: 'a;
    fn linalg(&self) -> Self::Out<'_> {
        SparseMatrixLinalgBuilder {
            mat: self,
            _marker: PhantomData,
        }
    }
}
