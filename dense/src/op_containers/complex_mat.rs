//! Convert a matrix to complex numbers.

use crate::traits::*;
use crate::types::{c32, c64, Scalar};
use crate::{matrix::*, DefaultLayout};
use std::marker::PhantomData;

/// This type represents the Complex of a matrix.
pub type ComplexMat<Item, MatImpl, S> = Matrix<Item, ComplexContainer<Item, MatImpl, S>, S>;

pub struct ComplexContainer<Item, MatImpl, S>(
    Matrix<<Item as Scalar>::Real, MatImpl, S>,
    PhantomData<S>,
)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl: MatrixImplTrait<<Item as Scalar>::Real, S>;

macro_rules! complex_container_impl {
    ($scalar:ty) => {
        /// A structure holding a reference to the matrix.
        /// This struct implements [MatrixImplTrait] and acts like a matrix.
        /// However, random access returns the corresponding complex entries.

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier>
            ComplexContainer<$scalar, MatImpl, S>
        {
            pub fn new(mat: Matrix<<$scalar as Scalar>::Real, MatImpl, S>) -> Self {
                Self(mat, PhantomData)
            }
        }

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier> Size
            for ComplexContainer<$scalar, MatImpl, S>
        {
            type S = S;
        }

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier>
            MatrixImplIdentifier for ComplexContainer<$scalar, MatImpl, S>
        {
            const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
        }

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier> Layout
            for ComplexContainer<$scalar, MatImpl, S>
        {
            type Impl = DefaultLayout;

            #[inline]
            fn layout(&self) -> &Self::Impl {
                self.0.layout()
            }
        }

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier>
            UnsafeRandomAccessByValue for ComplexContainer<$scalar, MatImpl, S>
        {
            type Item = $scalar;

            #[inline]
            unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
                <$scalar>::from_real(self.0.get_value_unchecked(row, col))
            }

            #[inline]
            unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
                <$scalar>::from_real(self.0.get1d_value_unchecked(index))
            }
        }

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier> RawAccess
            for ComplexContainer<$scalar, MatImpl, S>
        {
            type T = $scalar;

            #[inline]
            fn data(&self) -> &[Self::T] {
                std::unimplemented!();
            }

            #[inline]
            fn get_pointer(&self) -> *const Self::T {
                std::unimplemented!();
            }

            #[inline]
            fn get_slice(&self, _first: usize, _last: usize) -> &[Self::T] {
                std::unimplemented!()
            }
        }

        impl<MatImpl: MatrixImplTrait<<$scalar as Scalar>::Real, S>, S: SizeIdentifier> RawAccessMut
            for ComplexContainer<$scalar, MatImpl, S>
        {
            #[inline]
            fn data_mut(&mut self) -> &mut [Self::T] {
                std::unimplemented!();
            }

            #[inline]
            fn get_pointer_mut(&mut self) -> *mut Self::T {
                std::unimplemented!()
            }

            #[inline]
            fn get_slice_mut(&mut self, _first: usize, _last: usize) -> &mut [Self::T] {
                std::unimplemented!()
            }
        }
    };
}

complex_container_impl!(c32);
complex_container_impl!(c64);
