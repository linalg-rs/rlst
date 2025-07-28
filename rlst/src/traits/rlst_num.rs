//! Basic number traits.

use bytemuck::Pod;

use num::traits::{Float, FromPrimitive, NumAssign, NumCast, NumOps, ToPrimitive};
use pulp::Simd;
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

/// [RlstScalar](crate::RlstScalar) extension trait for SIMD operations.
#[allow(dead_code)]
pub trait RlstSimd: Pod + Send + Sync + num::Zero + 'static {
    /// Simd register that has the layout `[Self; N]` for some `N > 0`.
    type Scalars<S: Simd>: Pod + Copy + Send + Sync + Debug + 'static;
    /// Simd mask register that has the layout `[Self; N]` for some `N > 0`.
    type Mask<S: Simd>: Copy + Send + Sync + Debug + 'static;

    /// Splits the slice into a vector and scalar part.
    fn as_simd_slice<S: Simd>(slice: &[Self]) -> (&[Self::Scalars<S>], &[Self]);

    /// Splits an array of arrays into vector and scalar part.
    ///
    /// Consider an array of the form [[x1, y1, z1], [x2, y2, z2], ...] and
    /// a Simd vector length of 4. This function returns a slice, where each
    /// element is an array of length 12, containing 4 points and a tail containing
    /// the remainder points. The elements of the head can then be processed with the
    /// corresponding `deinterleave` function so as to obtain elements of the form
    /// [[x1, x2, x3, x4], [y1, y2, y3, y4], [z1, z2, z3, z4]].
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    fn as_simd_slice_from_vec<S: Simd, const N: usize>(
        vec_slice: &[[Self; N]],
    ) -> (&[[Self::Scalars<S>; N]], &[[Self; N]]) {
        assert_eq!(
            core::mem::align_of::<[Self; N]>(),
            core::mem::align_of::<[Self::Scalars<S>; N]>()
        );
        let chunk_size = core::mem::size_of::<Self::Scalars<S>>() / core::mem::size_of::<Self>();
        let len = vec_slice.len();
        let data = vec_slice.as_ptr();
        let div = len / chunk_size;
        let rem = len % chunk_size;

        unsafe {
            (
                std::slice::from_raw_parts(data as *const [Self::Scalars<S>; N], div),
                std::slice::from_raw_parts(data.add(len - rem), rem),
            )
        }
    }

    /// Splits a mutable array of arrays into vector and scalar part.
    ///
    /// Consider an array of the form [[x1, y1, z1], [x2, y2, z2], ...] and
    /// a Simd vector length of 4. This function returns a slice, where each
    /// element is an array of length 12, containing 4 points and a tail containing
    /// the remainder points. The elements of the head can then be processed with the
    /// [deinterleave](SimdFor::deinterleave) function so as to obtain elements of the form
    /// [[x1, x2, x3, x4], [y1, y2, y3, y4], [z1, z2, z3, z4]].
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    fn as_simd_slice_from_vec_mut<S: Simd, const N: usize>(
        vec_slice: &mut [[Self; N]],
    ) -> (&mut [[Self::Scalars<S>; N]], &mut [[Self; N]]) {
        assert_eq!(
            core::mem::align_of::<[Self; N]>(),
            core::mem::align_of::<[Self::Scalars<S>; N]>()
        );
        let chunk_size = core::mem::size_of::<Self::Scalars<S>>() / core::mem::size_of::<Self>();
        let len = vec_slice.len();
        let data = vec_slice.as_mut_ptr();
        let div = len / chunk_size;
        let rem = len % chunk_size;

        unsafe {
            (
                std::slice::from_raw_parts_mut(data as *mut [Self::Scalars<S>; N], div),
                std::slice::from_raw_parts_mut(data.add(len - rem), rem),
            )
        }
    }

    /// Splits the mutable slice into a vector and scalar part.
    fn as_simd_slice_mut<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Scalars<S>], &mut [Self]);

    /// Compare two SIMD registers for equality.
    fn simd_cmp_eq<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Mask<S>;
    /// Compare two SIMD registers for less-than.
    fn simd_cmp_lt<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Mask<S>;
    /// Compare two SIMD registers for less-than-or-equal.
    fn simd_cmp_le<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Mask<S>;
    /// Select from two simd registers depending on whether the mask is set.
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::Mask<S>,
        if_true: Self::Scalars<S>,
        if_false: Self::Scalars<S>,
    ) -> Self::Scalars<S>;

    /// Broadcasts the value to each element in the output simd register.
    fn simd_splat<S: Simd>(simd: S, value: Self) -> Self::Scalars<S>;

    /// Add two SIMD registers.
    fn simd_neg<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S>;

    /// Add two SIMD registers.
    fn simd_add<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Scalars<S>;

    /// Subtract two SIMD registers.
    fn simd_sub<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Scalars<S>;

    /// Multiply two SIMD registers.
    fn simd_mul<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Scalars<S>;

    /// Multiply two SIMD registers.
    fn simd_mul_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
        acc: Self::Scalars<S>,
    ) -> Self::Scalars<S>;

    /// Divide two SIMD registers.
    fn simd_div<S: Simd>(simd: S, lhs: Self::Scalars<S>, rhs: Self::Scalars<S>)
        -> Self::Scalars<S>;

    /// Compute the sine and cosine of each element in the register.
    fn simd_sin_cos<S: Simd>(
        simd: S,
        value: Self::Scalars<S>,
    ) -> (Self::Scalars<S>, Self::Scalars<S>);

    // /// Compute the sine and cosine of each element in the register,
    // /// assuming that its absolute value is smaller than or equal to `pi / 2`.
    // fn simd_sin_cos_quarter_circle<S: Simd>(
    //     simd: S,
    //     value: Self::Scalars<S>,
    // ) -> (Self::Scalars<S>, Self::Scalars<S>);

    /// Compute the base e exponential of a Simd vector.
    fn simd_exp<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S>;

    /// Compute the square root of each element in the register.
    fn simd_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S>;

    /// Compute the approximate reciprocal of each element in the register.
    fn simd_approx_recip<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S>;

    /// Compute the approximate reciprocal square root of each element in the register.
    fn simd_approx_recip_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S>;

    /// Compute the horizontal sum of the given value.
    fn simd_reduce_add<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self;

    /// Deinterleaves a register of values `[x0, y0, x1, y1, ...]` to
    /// `[x0, x1, ... y0, y1, ...]`.
    fn simd_deinterleave_2<S: Simd>(simd: S, value: [Self::Scalars<S>; 2])
        -> [Self::Scalars<S>; 2];

    /// Deinterleaves a register of values `[x0, y0, z0, x1, y1, z1, ...]` to
    /// `[x0, x1, ... y0, y1, ..., z0, z1, ...]`.
    fn simd_deinterleave_3<S: Simd>(simd: S, value: [Self::Scalars<S>; 3])
        -> [Self::Scalars<S>; 3];

    /// Deinterleaves a register of values `[x0, y0, z0, w0, x1, y1, z1, w1, ...]` to
    /// `[x0, x1, ... y0, y1, ..., z0, z1, ..., w0, w1, ...]`.
    fn simd_deinterleave_4<S: Simd>(simd: S, value: [Self::Scalars<S>; 4])
        -> [Self::Scalars<S>; 4];

    /// Inverse of [deinterleave_2](RlstSimd::simd_deinterleave_2).
    fn simd_interleave_2<S: Simd>(simd: S, value: [Self::Scalars<S>; 2]) -> [Self::Scalars<S>; 2] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 2];
        {
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[2 * i] = x[i];
                out[2 * i + 1] = x[n + i];
            }
        }
        out
    }

    /// Inverse of [deinterleave_3](RlstSimd::simd_deinterleave_3).
    fn simd_interleave_3<S: Simd>(simd: S, value: [Self::Scalars<S>; 3]) -> [Self::Scalars<S>; 3] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 3];
        {
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[3 * i] = x[i];
                out[3 * i + 1] = x[n + i];
                out[3 * i + 2] = x[2 * n + i];
            }
        }
        out
    }

    /// Inverse of [deinterleave_4](RlstSimd::simd_deinterleave_4).
    fn simd_interleave_4<S: Simd>(simd: S, value: [Self::Scalars<S>; 4]) -> [Self::Scalars<S>; 4] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 4];
        {
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[4 * i] = x[i];
                out[4 * i + 1] = x[n + i];
                out[4 * i + 2] = x[2 * n + i];
                out[4 * i + 3] = x[3 * n + i];
            }
        }
        out
    }
}
