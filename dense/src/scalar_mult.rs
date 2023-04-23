//! Multiplication of a matrix with a scalar
//!
//! This module implements the multiplication of a matrix with a scalar. Instead
//! of immediately executing the product a new matrix is created whose implementation
//! is a struct that holds the original operator and the scalar. On element access the
//! multiplication is performed.

use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::{c32, c64, Scalar};
use crate::{matrix::*, DefaultLayout};
use std::marker::PhantomData;

/// This type represents the multiplication of a matrix with a scalar.
pub type ScalarProdMat<Item, MatImpl, RS, CS> =
    Matrix<Item, ScalarMult<Item, MatImpl, RS, CS>, RS, CS>;

/// A structure holding a reference to the matrix and the scalar to be multiplied
/// with it. This struct implements [MatrixTrait] and acts like a matrix.
/// However, random access returns the corresponding matrix entry multiplied with
/// the scalar.
pub struct ScalarMult<Item, MatImpl, RS, CS>(
    Matrix<Item, MatImpl, RS, CS>,
    Item,
    PhantomData<Item>,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixTrait<Item, RS, CS>;

impl<Item: Scalar, MatImpl: MatrixTrait<Item, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
    ScalarMult<Item, MatImpl, RS, CS>
{
    pub fn new(mat: Matrix<Item, MatImpl, RS, CS>, scalar: Item) -> Self {
        Self(mat, scalar, PhantomData, PhantomData, PhantomData)
    }
}

impl<Item: Scalar, MatImpl: MatrixTrait<Item, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
    SizeType for ScalarMult<Item, MatImpl, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<Item: Scalar, MatImpl: MatrixTrait<Item, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
    Layout for ScalarMult<Item, MatImpl, RS, CS>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<Item: Scalar, MatImpl: MatrixTrait<Item, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
    UnsafeRandomAccessByValue for ScalarMult<Item, MatImpl, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.1 * self.0.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.1 * self.0.get1d_value_unchecked(index)
    }
}

macro_rules! scalar_mult_impl {
    ($Scalar:ty) => {
        impl<MatImpl: MatrixTrait<$Scalar, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
            std::ops::Mul<Matrix<$Scalar, MatImpl, RS, CS>> for $Scalar
        {
            type Output = ScalarProdMat<$Scalar, MatImpl, RS, CS>;

            fn mul(self, rhs: Matrix<$Scalar, MatImpl, RS, CS>) -> Self::Output {
                Matrix::new(ScalarMult::new(rhs, self))
            }
        }

        impl<'a, MatImpl: MatrixTrait<$Scalar, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
            std::ops::Mul<&'a Matrix<$Scalar, MatImpl, RS, CS>> for $Scalar
        {
            type Output = ScalarProdMat<$Scalar, MatrixRef<'a, $Scalar, MatImpl, RS, CS>, RS, CS>;

            fn mul(self, rhs: &'a Matrix<$Scalar, MatImpl, RS, CS>) -> Self::Output {
                ScalarProdMat::new(ScalarMult::new(Matrix::from_ref(rhs), self))
            }
        }

        impl<MatImpl: MatrixTrait<$Scalar, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
            std::ops::Mul<$Scalar> for Matrix<$Scalar, MatImpl, RS, CS>
        {
            type Output = ScalarProdMat<$Scalar, MatImpl, RS, CS>;

            fn mul(self, rhs: $Scalar) -> Self::Output {
                Matrix::new(ScalarMult::new(self, rhs))
            }
        }

        impl<'a, MatImpl: MatrixTrait<$Scalar, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>
            std::ops::Mul<$Scalar> for &'a Matrix<$Scalar, MatImpl, RS, CS>
        {
            type Output = ScalarProdMat<$Scalar, MatrixRef<'a, $Scalar, MatImpl, RS, CS>, RS, CS>;

            fn mul(self, rhs: $Scalar) -> Self::Output {
                ScalarProdMat::new(ScalarMult::new(Matrix::from_ref(self), rhs))
            }
        }
    };
}

scalar_mult_impl!(f32);
scalar_mult_impl!(f64);
scalar_mult_impl!(c32);
scalar_mult_impl!(c64);

#[cfg(test)]

mod test {

    use super::*;

    #[test]
    fn scalar_mult() {
        let mut mat = MatrixD::<f64>::zeros_from_dim(2, 3);

        *mat.get_mut(1, 2).unwrap() = 5.0;

        let res = 2.0 * mat;
        let res = res.eval();

        assert_eq!(res.get_value(1, 2).unwrap(), 10.0);
    }
}
