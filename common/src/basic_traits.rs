pub use crate::types::Scalar;

pub trait Scale {
    type T: Scalar;

    fn scale(&mut self, alpha: Self::T);
}

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

pub trait Inner {
    type T: Scalar;

    fn inner(&self, other: &Self) -> Self::T;
}

impl<Obj: Inner> Norm2 for Obj {
    type T = <Self as Inner>::T;

    fn norm2(&self) -> <Self::T as Scalar>::Real {
        self.inner(self).re().sqrt()
    }
}

pub trait NewFromZero {
    fn new_from_zero(&self) -> Self;
}

pub trait FillFrom {
    fn fill_from(&mut self, other: &Self);
}

pub trait MultSomeInto {
    type T: Scalar;

    // self -> ax + self;
    fn mult_sum_into(&self, alpha: Self::T, x: &Self);
}

pub trait Duplicate {
    // Duplicate an object

    fn duplicate(&self) -> Self;
}

impl<Obj: NewFromZero + FillFrom> Duplicate for Obj {
    fn duplicate(&self) -> Self {
        let mut new_obj = self.new_from_zero();
        new_obj.fill_from(self);
        new_obj
    }
}

pub trait Dual {
    type T: Scalar;
    type Other;

    fn dual(&self, other: &Self::Other) -> Self::T;
}

pub trait Norm2 {
    type T: Scalar;

    fn norm2(&self) -> <Self::T as Scalar>::Real;
}

pub trait Norm1 {
    type T: Scalar;

    fn norm1(&self) -> <Self::T as Scalar>::Real;
}

pub trait NormInf {
    type T: Scalar;

    fn norm_inf(&self) -> <Self::T as Scalar>::Real;
}

pub trait Dimension {
    fn dim(&self) -> (usize, usize);
}

pub trait RandomAccess {
    type T: Scalar;

    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::T;

    unsafe fn get_unchecked_value(&self, row: usize, col: usize) -> Self::T {
        *self.get_unchecked(row, col)
    }

    fn get(&self, row: usize, col: usize) -> Option<&Self::T>;

    fn get_value(&self, row: usize, col: usize) -> Option<Self::T> {
        if let Some(reference) = self.get(row, col) {
            Some(*reference)
        } else {
            None
        }
    }
}
