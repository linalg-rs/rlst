//! An implementation independent trait to provide access to linear algebra routines.
use rlst_common::types::Scalar;
use rlst_dense::{Matrix, MatrixImplTrait, SizeIdentifier};

pub struct LinAlgBuilder<'a, Mat> {
    pub(crate) mat: &'a Mat,
}

pub trait LinAlg {
    type Out;

    fn linalg(&self) -> LinAlgBuilder<Self::Out>;
}

impl<T: Scalar, Mat: MatrixImplTrait<T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> LinAlg
    for Matrix<T, Mat, RS, CS>
{
    type Out = Self;
    fn linalg(&self) -> LinAlgBuilder<Self::Out> {
        LinAlgBuilder { mat: &self }
    }
}
