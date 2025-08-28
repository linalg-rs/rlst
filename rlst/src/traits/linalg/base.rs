//! Basic traits for linear algebra operations.

use crate::{base_types::TransMode, traits::base_operations::BaseItem};

pub use num::traits::MulAdd;
pub use std::ops::{Add, Mul, Sub};

/// Multiply into
pub trait MultInto<First, Second>: BaseItem {
    /// Multiply First * Second and sum into Self
    fn simple_mult_into(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
        Self::Item: num::One,
        Self::Item: num::Zero,
    {
        self.mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            <Self::Item as num::One>::one(),
            arr_a,
            arr_b,
            <Self::Item as num::Zero>::zero(),
        )
    }
    /// Multiply into
    fn mult_into(
        self,
        transa: TransMode,
        transb: TransMode,
        alpha: Self::Item,
        arr_a: First,
        arr_b: Second,
        beta: Self::Item,
    ) -> Self;
}

/// Gemm
pub trait Gemm: Sized {
    /// Gemm
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        transa: TransMode,
        transb: TransMode,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        rsa: usize,
        csa: usize,
        b: &[Self],
        rsb: usize,
        csb: usize,
        beta: Self,
        c: &mut [Self],
        rsc: usize,
        csc: usize,
    );
}

/// Multiply into with resize.
/// This trait allows to resize the target array if necessary.
/// It is used for matrix multiplication where the result array may need to be resized.
/// It is not meant to be used outside the library. use the `dot` macro instead.
pub trait MultIntoResize<First, Second>: BaseItem {
    /// Multiply First * Second and sum into Self. Allow to resize Self if necessary
    fn simple_mult_into_resize(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
        Self::Item: num::One,
        Self::Item: num::Zero,
    {
        self.mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            <Self::Item as num::One>::one(),
            arr_a,
            arr_b,
            <Self::Item as num::Zero>::zero(),
        )
    }
    /// Multiply into with resize
    fn mult_into_resize(
        self,
        transa: TransMode,
        transb: TransMode,
        alpha: Self::Item,
        arr_a: First,
        arr_b: Second,
        beta: Self::Item,
    ) -> Self;
}

/// A helper trait to implement generic operators over matrices.
pub trait AsMatrixApply<OtherX, OtherY>: BaseItem {
    ///  Compute the matvec `y -> alpha * self * x  + beta * y` with `self` a matrix.
    fn apply(&self, alpha: Self::Item, x: &OtherX, beta: Self::Item, y: &mut OtherY);
}

/// Compute the inner product with another vector.
pub trait Inner<Other = Self> {
    /// The Item type of the inner product.
    type Output;
    /// Return the inner product of `Self` with `Other`.
    fn inner(&self, other: &Other) -> Self::Output;
}

/// Compute the norm of an entity.
pub trait Norm {
    /// The output of the norm.
    type Output;

    /// Compute the norm of an object.
    fn norm(&self) -> Self::Output;
}

/// Return the supremum norm of an array.
pub trait NormSup {
    /// The Item type of the norm.
    type Output;

    /// Return the supremum norm.
    fn norm_sup(&self) -> Self::Output;
}

/// Return the 1-norm of an array.
pub trait NormOne {
    /// The Item type of the norm.
    type Output;

    /// Return the 1-norm.
    fn norm_1(&self) -> Self::Output;
}

/// Return the 2-norm of an array.
pub trait NormTwo {
    /// The Item type of the norm.
    type Output;

    /// Return the 2-norm.
    fn norm_2(&self) -> Self::Output;
}
