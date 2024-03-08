//! Basic types.

use thiserror::Error;

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum RlstError {
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    #[error("Matrix has empty dimension {0:#?}.")]
    MatrixIsEmpty((usize, usize)),
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError { expected: usize, actual: usize },
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
    #[error("Incompatible stride for Lapack.")]
    IncompatibleStride,
    #[error("Lapack error: {0}")]
    LapackError(i32),
    #[error("{0}")]
    GeneralError(String),
    #[error("I/O Error: {0}")]
    IoError(String),
    #[error("Umfpack Error Code: {0}")]
    UmfpackError(i32),
    #[error("Matrix is not square. Dimension: {0}x{1}")]
    MatrixNotSquare(usize, usize),
    #[error("Matrix is not Hermitian (complex conjugate symmetric).")]
    MatrixNotHermitian,
}

/// Alias for an Rlst Result type.
pub type RlstResult<T> = std::result::Result<T, RlstError>;

/// Data chunk of fixed size N.
/// The field `valid_entries` stores how many entries of the chunk
/// contain valid data.
pub struct DataChunk<Item: Scalar, const N: usize> {
    pub data: [Item; N],
    pub start_index: usize,
    pub valid_entries: usize,
}

use num::complex::Complex;
use num::traits::{Float, FromPrimitive, NumAssign, NumCast, NumOps, ToPrimitive, Zero};
use rand::{distributions::Standard, prelude::*};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, LowerExp, UpperExp};
use std::iter::{Product, Sum};
use std::ops::Neg;

pub use num::complex::Complex32 as c32;
pub use num::complex::Complex64 as c64;

use crate::gemm::Gemm;

pub trait Scalar:
    NumAssign
    + FromPrimitive
    + NumCast
    + Neg<Output = Self>
    + Copy
    + Clone
    + Display
    + Debug
    + LowerExp
    + UpperExp
    + Sum
    + Product
    + Serialize
    + Send
    + Sync
    + Gemm
    + for<'de> Deserialize<'de>
    + 'static
{
    type Real: Scalar<Real = Self::Real, Complex = Self::Complex>
        + NumOps<Self::Real, Self::Real>
        + Float;
    type Complex: Scalar<Real = Self::Real, Complex = Self::Complex>
        + NumOps<Self::Real, Self::Complex>
        + NumOps<Self::Complex, Self::Complex>;

    /// Create a new real number
    fn real<T: ToPrimitive>(re: T) -> Self::Real;
    /// Create a new complex number
    fn complex<T: ToPrimitive>(re: T, im: T) -> Self::Complex;

    fn from_real(re: Self::Real) -> Self;

    fn add_real(self, re: Self::Real) -> Self;
    fn sub_real(self, re: Self::Real) -> Self;
    fn mul_real(self, re: Self::Real) -> Self;
    fn div_real(self, re: Self::Real) -> Self;

    fn add_complex(self, im: Self::Complex) -> Self::Complex;
    fn sub_complex(self, im: Self::Complex) -> Self::Complex;
    fn mul_complex(self, im: Self::Complex) -> Self::Complex;
    fn div_complex(self, im: Self::Complex) -> Self::Complex;

    fn pow(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self::Real) -> Self;
    fn powc(self, n: Self::Complex) -> Self::Complex;

    /// Real part
    fn re(&self) -> Self::Real;
    /// Imaginary part
    fn im(&self) -> Self::Real;
    /// As a complex number
    fn as_c(&self) -> Self::Complex;
    /// Complex conjugate
    fn conj(&self) -> Self;

    /// Absolute value
    fn abs(self) -> Self::Real;
    /// Sqaure of absolute value
    fn square(self) -> Self::Real;

    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;

    /// Generate an random number from
    /// [rand::distributions::Standard](https://docs.rs/rand/0.7.2/rand/distributions/struct.Standard.html)
    fn rand(rng: &mut impl Rng) -> Self;
}

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
        impl Scalar for $real {
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

        impl Scalar for $complex {
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

/// Transposition Mode.
#[derive(Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum TransMode {
    /// Complex conjugate of matrix.
    ConjNoTrans,
    /// No modification of matrix.
    NoTrans,
    /// Transposition of matrix.
    Trans,
    /// Conjugate transpose of matrix.
    ConjTrans,
}
