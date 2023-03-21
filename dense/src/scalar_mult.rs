//! Multiplication of a matrix with a scalar
//!
//! This module implements the multiplication of a matrix with a scalar. Instead
//! of immediately executing the product a new matrix is created whose implementation
//! is a struct that holds the original operator and the scalar. On element access the
//! multiplication is performed.

use crate::matrix::*;
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::{c32, c64, IndexType, Scalar};
use std::marker::PhantomData;

/// This type represents the multiplication of a matrix with a scalar.
pub type ScalarProdMat<Item, MatImpl, L, RS, CS> =
    Matrix<Item, ScalarMult<Item, MatImpl, L, RS, CS>, L, RS, CS>;

/// A structure holding a reference to the matrix and the scalar to be multiplied
/// with it. This struct implements [MatrixTrait] and acts like a matrix.
/// However, random access returns the corresponding matrix entry multiplied with
/// the scalar.
pub struct ScalarMult<Item, MatImpl, L, RS, CS>(
    Matrix<Item, MatImpl, L, RS, CS>,
    Item,
    PhantomData<Item>,
    PhantomData<L>,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: Scalar,
    L: LayoutType,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixTrait<Item, L, RS, CS>;

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > ScalarMult<Item, MatImpl, L, RS, CS>
{
    pub fn new(mat: Matrix<Item, MatImpl, L, RS, CS>, scalar: Item) -> Self {
        Self(
            mat,
            scalar,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
        )
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for ScalarMult<Item, MatImpl, L, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for ScalarMult<Item, MatImpl, L, RS, CS>
{
    type Impl = L;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccess for ScalarMult<Item, MatImpl, L, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> Self::Item {
        self.1 * self.0.get_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: IndexType) -> Self::Item {
        self.1 * self.0.get1d_unchecked(index)
    }
}

macro_rules! scalar_mult_impl {
    ($Scalar:ty) => {
        impl<
                MatImpl: MatrixTrait<$Scalar, L, RS, CS>,
                L: LayoutType,
                RS: SizeIdentifier,
                CS: SizeIdentifier,
            > std::ops::Mul<Matrix<$Scalar, MatImpl, L, RS, CS>> for $Scalar
        {
            type Output = ScalarProdMat<$Scalar, MatImpl, L, RS, CS>;

            fn mul(self, rhs: Matrix<$Scalar, MatImpl, L, RS, CS>) -> Self::Output {
                Matrix::new(ScalarMult::new(rhs, self))
            }
        }

        impl<
                'a,
                MatImpl: MatrixTrait<$Scalar, L, RS, CS>,
                L: LayoutType,
                RS: SizeIdentifier,
                CS: SizeIdentifier,
            > std::ops::Mul<&'a Matrix<$Scalar, MatImpl, L, RS, CS>> for $Scalar
        {
            type Output =
                ScalarProdMat<$Scalar, MatrixRef<'a, $Scalar, MatImpl, L, RS, CS>, L, RS, CS>;

            fn mul(self, rhs: &'a Matrix<$Scalar, MatImpl, L, RS, CS>) -> Self::Output {
                ScalarProdMat::new(ScalarMult::new(Matrix::from_ref(rhs), self))
            }
        }

        impl<
                MatImpl: MatrixTrait<$Scalar, L, RS, CS>,
                L: LayoutType,
                RS: SizeIdentifier,
                CS: SizeIdentifier,
            > std::ops::Mul<$Scalar> for Matrix<$Scalar, MatImpl, L, RS, CS>
        {
            type Output = ScalarProdMat<$Scalar, MatImpl, L, RS, CS>;

            fn mul(self, rhs: $Scalar) -> Self::Output {
                Matrix::new(ScalarMult::new(self, rhs))
            }
        }

        impl<
                'a,
                MatImpl: MatrixTrait<$Scalar, L, RS, CS>,
                L: LayoutType,
                RS: SizeIdentifier,
                CS: SizeIdentifier,
            > std::ops::Mul<$Scalar> for &'a Matrix<$Scalar, MatImpl, L, RS, CS>
        {
            type Output =
                ScalarProdMat<$Scalar, MatrixRef<'a, $Scalar, MatImpl, L, RS, CS>, L, RS, CS>;

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

    use crate::layouts::RowMajor;

    use super::*;

    #[test]
    fn scalar_mult() {
        let mut mat = MatrixD::<f64, RowMajor>::zeros_from_dim(2, 3);

        *mat.get_mut(1, 2) = 5.0;

        let res = 2.0 * mat;
        let res = res.eval();

        assert_eq!(res.get(1, 2), 10.0);
    }
}
