//! Basic types.

use bytemuck::Pod;
use thiserror::Error;

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum RlstError {
    /// Not implemented
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    /// Operation failed
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    /// Matrix is empty
    #[error("Matrix has empty dimension {0:#?}.")]
    MatrixIsEmpty((usize, usize)),
    /// Dimension mismatch
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },
    /// Index layout error
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    /// MPI rank error
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
    /// Incompatible stride for Lapack
    #[error("Incompatible stride for Lapack.")]
    IncompatibleStride,
    /// Lapack error
    #[error("Lapack error: {0}")]
    LapackError(i32),
    /// General error
    #[error("{0}")]
    GeneralError(String),
    /// I/O error
    #[error("I/O Error: {0}")]
    IoError(String),
    /// Umfpack error
    #[error("Umfpack Error Code: {0}")]
    UmfpackError(i32),
    /// Matrix is not square
    #[error("Matrix is not square. Dimension: {0}x{1}")]
    MatrixNotSquare(usize, usize),
    /// Matrix is not Hermitian
    #[error("Matrix is not Hermitian (complex conjugate symmetric).")]
    MatrixNotHermitian,
}

/// Alias for an Rlst Result type.
pub type RlstResult<T> = std::result::Result<T, RlstError>;

/// Data chunk of fixed size N.
pub struct DataChunk<Item: RlstBase, const N: usize> {
    /// The data
    pub data: [Item; N],
    /// The start index
    pub start_index: usize,
    /// How many entries of the chunk contain valid data.
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

use super::traits::Conj;

/// Base trait for Rlst admissible objects
pub trait RlstBase:
    Default + Sized + Copy + Clone + 'static + Display + Send + Sync + Debug
{
}

impl<T: Default + Sized + Copy + Clone + Display + Send + Sync + Debug + 'static> RlstBase for T {}

/// Trait representing numbers
pub trait RlstNum:
    RlstBase + NumAssign + NumCast + Sum + Product + NumCast + FromPrimitive
{
}

impl<T: RlstBase + NumAssign + NumCast + Sum + Product + NumCast + FromPrimitive> RlstNum for T {}

/// Rlst scalar
///
/// The following RlstScalar trait and is implementation for f32, f64, c32, c64
/// is a modifed version of the Scalar trait from the Rust Cauchy package
/// (<https://github.com/rust-math/cauchy>), which is MIT licensed. For the full license text see
/// <https://github.com/rust-math/cauchy/blob/master/LICENSE>.
pub trait RlstScalar:
    RlstNum
    + Neg<Output = Self>
    + LowerExp
    + UpperExp
    + Serialize
    + Pod
    // + Gemm
    + for<'de> Deserialize<'de>
    + 'static
{
    /// Real type
    type Real: RlstScalar<Real = Self::Real, Complex = Self::Complex>
        + NumOps<Self::Real, Self::Real>
        + Float;
    /// Complex type
    type Complex: RlstScalar<Real = Self::Real, Complex = Self::Complex>
        + NumOps<Self::Real, Self::Complex>
        + NumOps<Self::Complex, Self::Complex>;

    /// Create a new real number
    fn real<T: ToPrimitive>(re: T) -> Self::Real;
    /// Create a new complex number
    fn complex<T: ToPrimitive>(re: T, im: T) -> Self::Complex;
    /// Create from a real number
    fn from_real(re: Self::Real) -> Self;
    /// Add a real number
    fn add_real(self, re: Self::Real) -> Self;
    /// Subtract a real number
    fn sub_real(self, re: Self::Real) -> Self;
    /// Multiply by a real number
    fn mul_real(self, re: Self::Real) -> Self;
    /// Divide by a real number
    fn div_real(self, re: Self::Real) -> Self;
    /// Add a complex number
    fn add_complex(self, im: Self::Complex) -> Self::Complex;
    /// Subtract a complex number
    fn sub_complex(self, im: Self::Complex) -> Self::Complex;
    /// Multiply by a complex number
    fn mul_complex(self, im: Self::Complex) -> Self::Complex;
    /// Divide by a complex number
    fn div_complex(self, im: Self::Complex) -> Self::Complex;
    /// Raise to a power
    fn pow(self, n: Self) -> Self;
    /// Raise to an integer power
    fn powi(self, n: i32) -> Self;
    /// Raise to a real power
    fn powf(self, n: Self::Real) -> Self;
    /// Raise to a complex power
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

    /// Square root
    fn sqrt(self) -> Self;
    /// Exponential
    fn exp(self) -> Self;
    /// Natural logarithm
    fn ln(self) -> Self;
    /// Sine
    fn sin(self) -> Self;
    /// Cosine
    fn cos(self) -> Self;
    /// Tangeng
    fn tan(self) -> Self;
    /// Inverse sine
    fn asin(self) -> Self;
    /// Inverse cosine
    fn acos(self) -> Self;
    /// Inverse tangent
    fn atan(self) -> Self;
    /// Hyperbolic sine
    fn sinh(self) -> Self;
    /// Hyperbolic cosine
    fn cosh(self) -> Self;
    /// Hyperbolic tangent
    fn tanh(self) -> Self;
    /// Inverse hyperbolic sine
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent
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

/// Transposition Mode.
#[derive(Clone, Copy, PartialEq)]
pub enum TransMode {
    /// No modification of matrix.
    NoTrans,
    /// Complex conjugate of matrix.
    ConjNoTrans,
    /// Transposition of matrix.
    Trans,
    /// Conjugate transpose of matrix.
    ConjTrans,
}

impl<T> Conj for T
where
    T: RlstScalar,
{
    type Output = Self;
    fn conj(&self) -> Self {
        RlstScalar::conj(self)
    }
}
