//! Basic number traits.

use bytemuck::Pod;

use num::traits::{Float, FromPrimitive, NumAssign, NumCast, NumOps, ToPrimitive};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, LowerExp, UpperExp};
use std::iter::{Product, Sum};
use std::ops::Neg;

/// Base trait for Rlst number types
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

    /// Reciprocal
    fn recip(self) -> Self;
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

/// This trait implements a simple convenient function to return random scalars
/// from a given random number generator and distribution. For complex types the
/// generator and distribution are separately applied to obtain the real and imaginary
/// part of the random number.
pub trait RandScalar: RlstScalar {
    /// Returns a random number from a given random number generator `rng` and associated
    /// distribution `dist`.
    fn random_scalar<R: Rng, D: Distribution<<Self as RlstScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self;
}
