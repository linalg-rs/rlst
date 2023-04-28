pub trait Sum {
    fn sum(&self, other: &Self) -> Self;
}

pub trait Sub {
    fn sub(&self, other: &Self) -> Self;
}

pub trait Apply<Domain> {
    type T: Scalar;
    type Range;

    /// Compute y -> alpha A x + beta y
    fn apply(&self, alpha: Self::T, x: &Domain, y: &mut Self::Range, beta: Self::T);
}

pub trait MultSomeInto {
    type T: Scalar;

    // self -> ax + self;
    fn mult_sum_into(&self, alpha: Self::T, x: &Self);
}

pub trait Norm2 {
    type T: Scalar;

    fn norm2(&self) -> <Self::T as Scalar>::Real;
}

pub trait Norm1 {
    type T: Scalar;

    fn norm1(&self) -> <Self::T as Scalar>::Real;
}

use crate::types::Scalar;

pub trait NormInf {
    type T: Scalar;

    fn norm_inf(&self) -> <Self::T as Scalar>::Real;
}

pub trait Inner {
    type T: Scalar;

    fn inner(&self, other: &Self) -> Self::T;
}

pub trait Dual {
    type T: Scalar;
    type Other;

    fn dual(&self, other: &Self::Other) -> Self::T;
}
