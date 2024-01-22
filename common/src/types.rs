//! Basic types.

use cauchy::Scalar;
use num::traits::{Float, FromPrimitive, NumAssign, NumCast, NumOps, ToPrimitive};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, LowerExp, UpperExp};
use std::iter::{Product, Sum};
use std::ops::Neg;

pub use cauchy::{c32, c64};
use thiserror::Error;

pub trait RlstScalar:
    NumAssign
    + FromPrimitive
    + NumCast
    + Neg<Output = Self>
    + Copy
    + Clone
    + std::fmt::Display
    + Debug
    + LowerExp
    + UpperExp
    + Sum
    + Product
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
{
    type Real: RlstScalar<Real = Self::Real, Complex = Self::Complex>
        + NumOps<Self::Real, Self::Real>
        + Float;
    type Complex: RlstScalar<Real = Self::Real, Complex = Self::Complex>
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

impl<Item: Scalar> RlstScalar for Item {
    type Real = <Self as Scalar>::Real;
    type Complex = <Self as Scalar>::Complex;

    /// Create a new real number
    fn real<T: ToPrimitive>(re: T) -> Self::Real {
        <Self as Scalar>::real(re)
    }
    /// Create a new complex number
    fn complex<T: ToPrimitive>(re: T, im: T) -> Self::Complex {
        <Self as Scalar>::complex(re, im)
    }

    fn from_real(re: Self::Real) -> Self {
        <Self as Scalar>::from_real(re)
    }

    fn add_real(self, re: Self::Real) -> Self {
        <Self as Scalar>::add_real(self, re)
    }
    fn sub_real(self, re: Self::Real) -> Self {
        <Self as Scalar>::sub_real(self, re)
    }
    fn mul_real(self, re: Self::Real) -> Self {
        <Self as Scalar>::mul_real(self, re)
    }
    fn div_real(self, re: Self::Real) -> Self {
        <Self as Scalar>::div_real(self, re)
    }

    fn add_complex(self, im: Self::Complex) -> Self::Complex {
        <Self as Scalar>::add_complex(self, im)
    }
    fn sub_complex(self, im: Self::Complex) -> Self::Complex {
        <Self as Scalar>::sub_complex(self, im)
    }
    fn mul_complex(self, im: Self::Complex) -> Self::Complex {
        <Self as Scalar>::mul_complex(self, im)
    }
    fn div_complex(self, im: Self::Complex) -> Self::Complex {
        <Self as Scalar>::div_complex(self, im)
    }

    fn pow(self, n: Self) -> Self {
        <Self as Scalar>::pow(self, n)
    }
    fn powi(self, n: i32) -> Self {
        <Self as Scalar>::powi(self, n)
    }
    fn powf(self, n: Self::Real) -> Self {
        <Self as Scalar>::powf(self, n)
    }
    fn powc(self, n: Self::Complex) -> Self::Complex {
        <Self as Scalar>::powc(self, n)
    }

    /// Real part
    fn re(&self) -> Self::Real {
        <Self as Scalar>::re(self)
    }
    /// Imaginary part
    fn im(&self) -> Self::Real {
        <Self as Scalar>::im(self)
    }
    /// As a complex number
    fn as_c(&self) -> Self::Complex {
        <Self as Scalar>::as_c(self)
    }
    /// Complex conjugate
    fn conj(&self) -> Self {
        <Self as Scalar>::conj(self)
    }

    /// Absolute value
    fn abs(self) -> Self::Real {
        <Self as Scalar>::abs(self)
    }
    /// Sqaure of absolute value
    fn square(self) -> Self::Real {
        <Self as Scalar>::square(self)
    }

    fn sqrt(self) -> Self {
        <Self as Scalar>::sqrt(self)
    }
    fn exp(self) -> Self {
        <Self as Scalar>::exp(self)
    }
    fn ln(self) -> Self {
        <Self as Scalar>::ln(self)
    }
    fn sin(self) -> Self {
        <Self as Scalar>::sin(self)
    }
    fn cos(self) -> Self {
        <Self as Scalar>::cos(self)
    }
    fn tan(self) -> Self {
        <Self as Scalar>::tan(self)
    }
    fn asin(self) -> Self {
        <Self as Scalar>::asin(self)
    }
    fn acos(self) -> Self {
        <Self as Scalar>::acos(self)
    }
    fn atan(self) -> Self {
        <Self as Scalar>::atan(self)
    }
    fn sinh(self) -> Self {
        <Self as Scalar>::sinh(self)
    }
    fn cosh(self) -> Self {
        <Self as Scalar>::cosh(self)
    }
    fn tanh(self) -> Self {
        <Self as Scalar>::tanh(self)
    }
    fn asinh(self) -> Self {
        <Self as Scalar>::asinh(self)
    }
    fn acosh(self) -> Self {
        <Self as Scalar>::acosh(self)
    }
    fn atanh(self) -> Self {
        <Self as Scalar>::atanh(self)
    }

    fn rand(rng: &mut impl Rng) -> Self {
        <Self as Scalar>::rand(rng)
    }
}

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
pub struct DataChunk<Item: RlstScalar, const N: usize> {
    pub data: [Item; N],
    pub start_index: usize,
    pub valid_entries: usize,
}
