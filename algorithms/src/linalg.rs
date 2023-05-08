//! An implementation independent trait to provide access to linear algebra routines.
use std::marker::PhantomData;

use rlst_common::types::Scalar;
use rlst_dense::{Matrix, MatrixImplTrait, SizeIdentifier};

pub struct LinAlgBuilder<'a, T: Scalar, Mat> {
    pub(crate) mat: &'a Mat,
    _marker: std::marker::PhantomData<T>,
}

pub trait LinAlg {
    type T: Scalar;
    type Out;

    fn linalg(&self) -> LinAlgBuilder<Self::T, Self::Out>;
}

impl<T: Scalar, Mat: MatrixImplTrait<T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> LinAlg
    for Matrix<T, Mat, RS, CS>
{
    type T = T;
    type Out = Self;
    fn linalg(&self) -> LinAlgBuilder<Self::T, Self::Out> {
        LinAlgBuilder {
            mat: &self,
            _marker: PhantomData,
        }
    }
}
