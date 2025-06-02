//! Implementation of number traits.

use num::complex::Complex;
use num::traits::{Float, NumCast, ToPrimitive, Zero};
use rand::{distributions::Standard, prelude::*};

pub use num::complex::Complex32 as c32;
pub use num::complex::Complex64 as c64;

use super::rlst_num::RlstScalar;

macro_rules! impl_float {
    ($name:ident) => {
        #[inline]
        fn $name(self) -> Self {
            Float::$name(self)
        }
    };
}

macro_rules! impl_complex {
    ($name:ident) => {
        #[inline]
        fn $name(self) -> Self {
            Complex::$name(self)
        }
    };
}

macro_rules! impl_with_real {
    ($name:ident, $op:tt) => {
        #[inline]
        fn $name(self, re: Self::Real) -> Self {
            self $op re
        }
    }
}

macro_rules! impl_with_complex {
    ($name:ident, $op:tt) => {
        #[inline]
        fn $name(self, im: Self::Complex) -> Self::Complex {
            self $op im
        }
    }
}

macro_rules! impl_scalar {
    ($real:ty, $complex:ty) => {
        impl RlstScalar for $real {
            type Real = $real;
            type Complex = $complex;

            #[inline]
            fn re(&self) -> Self::Real {
                *self
            }
            #[inline]
            fn im(&self) -> Self::Real {
                0.0
            }

            #[inline]
            fn from_real(re: Self::Real) -> Self {
                re
            }

            fn pow(self, n: Self) -> Self {
                self.powf(n)
            }
            fn powi(self, n: i32) -> Self {
                Float::powi(self, n)
            }
            fn powf(self, n: Self::Real) -> Self {
                Float::powf(self, n)
            }
            fn powc(self, n: Self::Complex) -> Self::Complex {
                self.as_c().powc(n)
            }

            #[inline]
            fn real<T: ToPrimitive>(re: T) -> Self::Real {
                NumCast::from(re).unwrap()
            }
            #[inline]
            fn complex<T: ToPrimitive>(re: T, im: T) -> Self::Complex {
                Complex {
                    re: NumCast::from(re).unwrap(),
                    im: NumCast::from(im).unwrap(),
                }
            }
            #[inline]
            fn as_c(&self) -> Self::Complex {
                Complex::new(*self, 0.0)
            }
            #[inline]
            fn conj(&self) -> Self {
                *self
            }
            #[inline]
            fn square(self) -> Self::Real {
                self * self
            }

            fn rand(rng: &mut impl Rng) -> Self {
                rng.sample(Standard)
            }

            impl_with_real!(add_real, +);
            impl_with_real!(sub_real, -);
            impl_with_real!(mul_real, *);
            impl_with_real!(div_real, /);
            impl_with_complex!(add_complex, +);
            impl_with_complex!(sub_complex, -);
            impl_with_complex!(mul_complex, *);
            impl_with_complex!(div_complex, /);

            impl_float!(sqrt);
            impl_float!(abs);
            impl_float!(exp);
            impl_float!(ln);
            impl_float!(sin);
            impl_float!(cos);
            impl_float!(tan);
            impl_float!(sinh);
            impl_float!(cosh);
            impl_float!(tanh);
            impl_float!(asin);
            impl_float!(acos);
            impl_float!(atan);
            impl_float!(asinh);
            impl_float!(acosh);
            impl_float!(atanh);
        }

        impl RlstScalar for $complex {
            type Real = $real;
            type Complex = $complex;

            #[inline]
            fn re(&self) -> Self::Real {
                self.re
            }
            #[inline]
            fn im(&self) -> Self::Real {
                self.im
            }

            #[inline]
            fn from_real(re: Self::Real) -> Self {
                Self::new(re, Zero::zero())
            }

            fn pow(self, n: Self) -> Self {
                self.powc(n)
            }
            fn powi(self, n: i32) -> Self {
                self.powf(n as Self::Real)
            }
            fn powf(self, n: Self::Real) -> Self {
                self.powf(n)
            }
            fn powc(self, n: Self::Complex) -> Self::Complex {
                self.powc(n)
            }

            #[inline]
            fn real<T: ToPrimitive>(re: T) -> Self::Real {
                NumCast::from(re).unwrap()
            }
            #[inline]
            fn complex<T: ToPrimitive>(re: T, im: T) -> Self::Complex {
                Complex {
                    re: NumCast::from(re).unwrap(),
                    im: NumCast::from(im).unwrap(),
                }
            }
            #[inline]
            fn as_c(&self) -> Self::Complex {
                *self
            }
            #[inline]
            fn conj(&self) -> Self {
                Complex::conj(self)
            }
            #[inline]
            fn square(self) -> Self::Real {
                Complex::norm_sqr(&self)
            }
            #[inline]
            fn abs(self) -> Self::Real {
                Complex::norm(self)
            }

            fn rand(rng: &mut impl Rng) -> Self {
                rng.sample(Standard)
            }

            impl_with_real!(add_real, +);
            impl_with_real!(sub_real, -);
            impl_with_real!(mul_real, *);
            impl_with_real!(div_real, /);
            impl_with_complex!(add_complex, +);
            impl_with_complex!(sub_complex, -);
            impl_with_complex!(mul_complex, *);
            impl_with_complex!(div_complex, /);

            impl_complex!(sqrt);
            impl_complex!(exp);
            impl_complex!(ln);
            impl_complex!(sin);
            impl_complex!(cos);
            impl_complex!(tan);
            impl_complex!(sinh);
            impl_complex!(cosh);
            impl_complex!(tanh);
            impl_complex!(asin);
            impl_complex!(acos);
            impl_complex!(atan);
            impl_complex!(asinh);
            impl_complex!(acosh);
            impl_complex!(atanh);
        }
    }
}

impl_scalar!(f32, c32);
impl_scalar!(f64, c64);
