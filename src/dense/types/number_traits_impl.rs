use std::ops::Mul;

use crate::dense::traits::{Abs, AbsSquare, Conj, Max, Min, Sqrt};
use crate::{
    Acos, Acosh, Asin, Asinh, Atan, Atanh, Cos, Cosh, Exp, Ln, Recip, Sin, Sinh, Square, Tan, Tanh,
};

use super::RlstScalar;
use super::{c32, c64};

impl<T> Conj for T
where
    T: RlstScalar,
{
    type Output = Self;
    fn conj(&self) -> Self {
        RlstScalar::conj(&self)
    }
}

macro_rules! impl_conj {
    ($dtype:ty) => {
        impl Conj for $dtype {
            type Output = $dtype;
            fn conj(&self) -> $dtype {
                *self
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
                std::cmp::max(*self, *other)
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

macro_rules! impl_min {
    ($dtype:ty) => {
        impl Min for $dtype {
            type Output = $dtype;
            fn min(&self, other: &$dtype) -> $dtype {
                std::cmp::min(*self, *other)
            }
        }
    };
}

impl_min!(i8);
impl_min!(i16);
impl_min!(i32);
impl_min!(i64);
impl_min!(i128);
impl_min!(isize);
impl_min!(u8);
impl_min!(u16);
impl_min!(u32);
impl_min!(u64);
impl_min!(u128);
impl_min!(usize);

impl Abs for c32 {
    type Output = f32;

    fn abs(&self) -> Self::Output {
        RlstScalar::abs(*self)
    }
}

impl Abs for c64 {
    type Output = f64;

    fn abs(&self) -> Self::Output {
        RlstScalar::abs(*self)
    }
}

macro_rules! impl_abs {
    ($dtype:ty) => {
        impl Abs for $dtype {
            type Output = $dtype;
            fn abs(&self) -> $dtype {
                <$dtype>::abs(*self)
            }
        }
    };
}

macro_rules! impl_abs_ident {
    ($dtype:ty) => {
        impl Abs for $dtype {
            type Output = $dtype;
            fn abs(&self) -> $dtype {
                *self
            }
        }
    };
}

impl_abs!(i8);
impl_abs!(i16);
impl_abs!(i32);
impl_abs!(i64);
impl_abs!(i128);
impl_abs!(isize);
impl_abs!(f32);
impl_abs!(f64);

impl_abs_ident!(u8);
impl_abs_ident!(u16);
impl_abs_ident!(u32);
impl_abs_ident!(u64);
impl_abs_ident!(u128);
impl_abs_ident!(usize);

impl AbsSquare for c32 {
    type Output = f32;

    fn abs_square(&self) -> Self::Output {
        <c32 as RlstScalar>::square(*self)
    }
}

impl AbsSquare for c64 {
    type Output = f64;

    fn abs_square(&self) -> Self::Output {
        <c64 as RlstScalar>::square(*self)
    }
}

macro_rules! impl_abs_square {
    ($dtype:ty) => {
        impl AbsSquare for $dtype {
            type Output = $dtype;

            fn abs_square(&self) -> $dtype {
                *self * *self
            }
        }
    };
}

impl_abs_square!(i8);
impl_abs_square!(i16);
impl_abs_square!(i32);
impl_abs_square!(i64);
impl_abs_square!(i128);
impl_abs_square!(isize);
impl_abs_square!(f32);
impl_abs_square!(f64);
impl_abs_square!(u8);
impl_abs_square!(u16);
impl_abs_square!(u32);
impl_abs_square!(u64);
impl_abs_square!(u128);
impl_abs_square!(usize);

impl<T: Mul<Output = T> + Copy> Square for T {
    type Output = T;

    fn square(&self) -> Self::Output {
        *self * *self
    }
}

macro_rules! impl_unary_op {
    ($trait_name:ident, $method_name:ident) => {
        impl<T: RlstScalar> $trait_name for T {
            type Output = T;

            fn $method_name(&self) -> Self::Output {
                <T as RlstScalar>::$method_name(*self)
            }
        }
    };
}

impl_unary_op!(Sqrt, sqrt);
impl_unary_op!(Exp, exp);
impl_unary_op!(Ln, ln);
impl_unary_op!(Recip, recip);
impl_unary_op!(Sin, sin);
impl_unary_op!(Cos, cos);
impl_unary_op!(Tan, tan);
impl_unary_op!(Asin, asin);
impl_unary_op!(Acos, acos);
impl_unary_op!(Atan, atan);
impl_unary_op!(Sinh, sinh);
impl_unary_op!(Cosh, cosh);
impl_unary_op!(Tanh, tanh);
impl_unary_op!(Asinh, asinh);
impl_unary_op!(Acosh, acosh);
impl_unary_op!(Atanh, atanh);
