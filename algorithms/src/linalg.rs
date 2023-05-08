//! An implementation independent trait to provide access to linear algebra routines.
use std::marker::PhantomData;

use rlst_common::types::Scalar;
use rlst_dense::{Matrix, MatrixImplTrait, SizeIdentifier};

pub struct DenseMatrixLinAlgBuilder<'a, T: Scalar, Mat> {
    pub(crate) mat: &'a Mat,
    _marker: std::marker::PhantomData<T>,
}

pub trait LinAlg {
    type T: Scalar;
    type Out<'a>
    where
        Self: 'a;

    fn linalg<'a>(&'a self) -> Self::Out<'a>;
}

impl<T: Scalar, Mat: MatrixImplTrait<T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> LinAlg
    for Matrix<T, Mat, RS, CS>
{
    type T = T;
    type Out<'a> = DenseMatrixLinAlgBuilder<'a, Self::T, Self> where Self: 'a;
    fn linalg<'a>(&'a self) -> Self::Out<'a> {
        DenseMatrixLinAlgBuilder {
            mat: &self,
            _marker: PhantomData,
        }
    }
}
