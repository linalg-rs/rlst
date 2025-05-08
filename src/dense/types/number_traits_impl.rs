use crate::dense::traits::number_traits::*;

use super::RlstScalar;

impl<T> Conj for T
where
    T: RlstScalar,
{
    type Output = Self;
    fn conj(&self) -> Self {
        RlstScalar::conj(self)
    }
}

macro_rules! impl_conj {
    ($dtype:ty) => {
        impl Conj for $dtype {
            type Output = $dtype;
            fn conj(&self) -> $dtype {
                self
            }
        }
    };
}

impl_conj!(i8);
impl_conj!(i16);
impl_conj!(i32);
impl_conj!(i64);
impl_conj!(i128);
impl_conj!(isize);
impl_conj!(u8);
impl_conj!(u16);
impl_conj!(u32);
impl_conj!(u64);
impl_conj!(u128);
impl_conj!(usize);

impl Max for f32 {
    type Output = f32;

    fn max(&self, other: &Self) -> Self::Output {
        f32::max(*self, *other)
    }
}

impl Max for f64 {
    type Output = f64;

    fn max(&self, other: &Self) -> Self::Output {
        f64::max(*self, *other)
    }
}

impl Min for f32 {
    type Output = f32;

    fn min(&self, other: &Self) -> Self::Output {
        f32::min(*self, *other)
    }
}

impl Min for f64 {
    type Output = f64;

    fn min(&self, other: &Self) -> Self::Output {
        f64::min(*self, *other)
    }
}

macro_rules! impl_max {
    ($dtype:ty) => {
        impl Max for $dtype {
            type Output = $dtype;
            fn max(&self, other: &$dtype) -> $dtype {
                std::cmp::max(self, other)
            }
        }
    };
}

impl_max!(i8);
impl_max!(i16);
impl_max!(i32);
impl_max!(i64);
impl_max!(i128);
impl_max!(isize);
impl_max!(u8);
impl_max!(u16);
impl_max!(u32);
impl_max!(u64);
impl_max!(u128);
impl_max!(usize);
