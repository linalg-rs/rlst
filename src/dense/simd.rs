//! Basic traits for SIMD Operations

use bytemuck::Pod;
use num::Zero;
use pulp::Simd;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// A simplified wrapper to call into Simd operations in a type
/// and architecture independent way.
#[derive(Copy, Clone, Debug)]
pub struct SimdFor<T, S> {
    /// Architecture specific Simd information.
    pub simd: S,
    __marker: PhantomData<T>,
}

#[allow(dead_code)]
impl<T: RlstSimd, S: Simd> SimdFor<T, S> {
    #[inline]
    /// Instantiate a new Simd object.
    pub fn new(simd: S) -> Self {
        Self {
            simd,
            __marker: PhantomData,
        }
    }

    /// Copy a scalar across all Simd lanes.
    #[inline(always)]
    pub fn splat(self, value: T) -> T::Scalars<S> {
        T::simd_splat(self.simd, value)
    }

    /// Compare for equality.
    #[inline(always)]
    pub fn cmp_eq(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Mask<S> {
        T::simd_cmp_eq(self.simd, lhs, rhs)
    }

    /// Compare for less than.
    #[inline(always)]
    pub fn cmp_lt(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Mask<S> {
        T::simd_cmp_lt(self.simd, lhs, rhs)
    }

    /// Compare for less equal.
    #[inline(always)]
    pub fn cmp_le(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Mask<S> {
        T::simd_cmp_le(self.simd, lhs, rhs)
    }

    /// Select based on bitmask.
    #[inline(always)]
    pub fn select(
        self,
        mask: T::Mask<S>,
        if_true: T::Scalars<S>,
        if_false: T::Scalars<S>,
    ) -> T::Scalars<S> {
        T::simd_select(self.simd, mask, if_true, if_false)
    }

    /// Negate input.
    #[inline(always)]
    pub fn neg(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_neg(self.simd, value)
    }

    /// Add two Simd vectors.
    #[inline(always)]
    pub fn add(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_add(self.simd, lhs, rhs)
    }

    /// Subtract two Simd vectors.
    #[inline(always)]
    pub fn sub(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_sub(self.simd, lhs, rhs)
    }

    /// Multiply two Simd vectors.
    #[inline(always)]
    pub fn mul(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_mul(self.simd, lhs, rhs)
    }

    /// Multiply `lhs` and `rhs` and add to `acc`.
    #[inline(always)]
    pub fn mul_add(
        self,
        lhs: T::Scalars<S>,
        rhs: T::Scalars<S>,
        acc: T::Scalars<S>,
    ) -> T::Scalars<S> {
        T::simd_mul_add(self.simd, lhs, rhs, acc)
    }

    /// Divide two Simd vectors.
    #[inline(always)]
    pub fn div(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_div(self.simd, lhs, rhs)
    }

    /// Compute the sine and cosine of two Simd vectors.
    #[inline(always)]
    pub fn sin_cos(self, value: T::Scalars<S>) -> (T::Scalars<S>, T::Scalars<S>) {
        T::simd_sin_cos(self.simd, value)
    }
    // #[inline(always)]
    // pub fn sin_cos_quarter_circle(self, value: T::Scalars<S>) -> (T::Scalars<S>, T::Scalars<S>) {
    //     T::simd_sin_cos_quarter_circle(self.simd, value)
    // }

    /// Compute the square root of a Simd vector.
    #[inline(always)]
    pub fn sqrt(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_sqrt(self.simd, value)
    }

    /// Compute an approximate inverse of a Simd vector.
    #[inline(always)]
    pub fn approx_recip(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_approx_recip(self.simd, value)
    }

    /// Compute an approximate inverse of a Simd vector.
    #[inline(always)]
    pub fn approx_recip_sqrt(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_approx_recip_sqrt(self.simd, value)
    }

    /// Sum the elements of a Simd vector.
    #[inline(always)]
    pub fn reduce_add(self, value: T::Scalars<S>) -> T {
        T::simd_reduce_add(self.simd, value)
    }

    /// Deinterleave N Simd vectors.
    #[inline(always)]
    pub fn deinterleave<const N: usize>(self, value: [T::Scalars<S>; N]) -> [T::Scalars<S>; N] {
        use coe::coerce_static as to;
        match N {
            2 => to(T::simd_deinterleave_2(self.simd, to(value))),
            3 => to(T::simd_deinterleave_3(self.simd, to(value))),
            4 => to(T::simd_deinterleave_4(self.simd, to(value))),
            _ => panic!("unsupported size"),
        }
    }

    /// Interleave N Simd vectors.
    #[allow(clippy::type_complexity)]
    #[inline(always)]
    pub fn interleave<const N: usize>(self, value: [T::Scalars<S>; N]) -> [T::Scalars<S>; N] {
        use coe::coerce_static as to;
        match N {
            2 => to(T::simd_interleave_2(self.simd, to(value))),
            3 => to(T::simd_interleave_3(self.simd, to(value))),
            4 => to(T::simd_interleave_4(self.simd, to(value))),
            _ => panic!("unsupported size"),
        }
    }
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
                from_raw_parts(data as *const [Self::Scalars<S>; N], div),
                from_raw_parts(data.add(len - rem), rem),
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
                from_raw_parts_mut(data as *mut [Self::Scalars<S>; N], div),
                from_raw_parts_mut(data.add(len - rem), rem),
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

    /// Compute the sine and cosine of each element in the register,
    /// assuming that its absolute value is smaller than or equal to `pi / 2`.
    // fn simd_sin_cos_quarter_circle<S: Simd>(
    //     simd: S,
    //     value: Self::Scalars<S>,
    // ) -> (Self::Scalars<S>, Self::Scalars<S>);

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

impl RlstSimd for f32 {
    type Scalars<S: Simd> = S::f32s;
    type Mask<S: Simd> = S::m32s;

    #[inline(always)]
    fn simd_cmp_eq<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.f32s_equal(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_lt<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.f32s_less_than(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_le<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.f32s_less_than_or_equal(lhs, rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::Mask<S>,
        if_true: Self::Scalars<S>,
        if_false: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.m32s_select_f32s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn as_simd_slice<S: Simd>(slice: &[Self]) -> (&[Self::Scalars<S>], &[Self]) {
        S::f32s_as_simd(slice)
    }

    #[inline(always)]
    fn as_simd_slice_mut<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Scalars<S>], &mut [Self]) {
        S::f32s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, value: Self) -> Self::Scalars<S> {
        simd.f32s_splat(value)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        simd.f32s_neg(value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f32s_add(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f32s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f32s_mul(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
        acc: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f32s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f32s_div(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sin_cos<S: Simd>(
        simd: S,
        value: Self::Scalars<S>,
    ) -> (Self::Scalars<S>, Self::Scalars<S>) {
        let mut s_out = Self::simd_splat(simd, Self::zero());
        let mut c_out = Self::simd_splat(simd, Self::zero());
        {
            let value: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));

            let c_out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut c_out));
            let s_out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut s_out));

            for (value, s_out, c_out) in
                itertools::izip!(value.iter(), &mut s_out.iter_mut(), &mut c_out.iter_mut())
            {
                *s_out = Self::sin(*value);
                *c_out = Self::cos(*value)
            }
        }

        (s_out, c_out)
    }

    #[inline(always)]
    fn simd_deinterleave_2<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 2],
    ) -> [Self::Scalars<S>; 2] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 2];
        {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                if coe::is_same::<S, pulp::aarch64::Neon>() {
                    let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                    return unsafe {
                        bytemuck::cast(simd.neon.vld2q_f32(value.as_ptr() as *const f32))
                    };
                }
            }
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[i] = x[2 * i];
                out[n + i] = x[2 * i + 1];
            }
        }
        out
    }

    #[inline(always)]
    fn simd_deinterleave_3<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 3],
    ) -> [Self::Scalars<S>; 3] {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                return unsafe {
                    bytemuck::cast(simd.neon.vld3q_f32(value.as_ptr() as *const f32))
                };
            }
        }

        let mut out = [Self::simd_splat(simd, Self::zero()); 3];
        {
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[i] = x[3 * i];
                out[n + i] = x[3 * i + 1];
                out[2 * n + i] = x[3 * i + 2];
            }
        }
        out
    }

    #[inline(always)]
    fn simd_deinterleave_4<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 4],
    ) -> [Self::Scalars<S>; 4] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 4];
        {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                if coe::is_same::<S, pulp::aarch64::Neon>() {
                    let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                    return unsafe {
                        bytemuck::cast(simd.neon.vld4q_f32(value.as_ptr() as *const f32))
                    };
                }
            }
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[i] = x[4 * i];
                out[n + i] = x[4 * i + 1];
                out[2 * n + i] = x[4 * i + 2];
                out[3 * n + i] = x[4 * i + 3];
            }
        }
        out
    }

    #[inline(always)]
    fn simd_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        #[cfg(target_arch = "x86_64")]
        {
            use coe::coerce_static as to;
            #[cfg(feature = "nightly")]
            if coe::is_same::<S, pulp::x86::V4>() {
                let simd: pulp::x86::V4 = to(simd);
                return to(simd.sqrt_f32x16(to(value)));
            }
            if coe::is_same::<S, pulp::x86::V3>() {
                let simd: pulp::x86::V3 = to(simd);
                return to(simd.sqrt_f32x8(to(value)));
            }
        }
        let mut out = simd.f32s_splat(0.0);
        {
            let out: &mut [f32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for (out, x) in itertools::izip!(out, x) {
                *out = x.sqrt();
            }
        }
        out
    }

    #[inline(always)]
    fn simd_approx_recip<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        #[cfg(target_arch = "x86_64")]
        {
            use coe::coerce_static as to;
            #[cfg(feature = "nightly")]
            if coe::is_same::<S, pulp::x86::V4>() {
                let simd: pulp::x86::V4 = to(simd);
                return to(simd.avx512f._mm512_rcp14_ps(to(value)));
            }
            if coe::is_same::<S, pulp::x86::V3>() {
                let simd: pulp::x86::V3 = to(simd);
                return to(simd.approx_reciprocal_f32x8(to(value)));
            }
        }
        simd.f32s_div(simd.f32s_splat(1.0), value)
    }

    #[inline(always)]
    fn simd_approx_recip_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        #[cfg(target_arch = "x86_64")]
        {
            use coe::coerce_static as to;
            #[cfg(feature = "nightly")]
            if coe::is_same::<S, pulp::x86::V4>() {
                let simd: pulp::x86::V4 = to(simd);
                return to(simd.avx512f._mm512_rsqrt14_ps(to(value)));
            }
            if coe::is_same::<S, pulp::x86::V3>() {
                let simd: pulp::x86::V3 = to(simd);
                return to(simd.approx_reciprocal_sqrt_f32x8(to(value)));
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use coe::coerce_static as to;
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                let value: pulp::f32x4 = to(value);
                let mut res: pulp::f32x4 = unsafe {
                    std::mem::transmute(simd.neon.vrsqrteq_f32(std::mem::transmute(value)))
                };
                for _ in 0..2 {
                    res = simd.mul_f32x4(res, unsafe {
                        std::mem::transmute(simd.neon.vrsqrtsq_f32(
                            std::mem::transmute(value),
                            std::mem::transmute(simd.mul_f32x4(res, res)),
                        ))
                    });
                }
                return to(res);
            }
        }

        Self::simd_div(
            simd,
            Self::simd_splat(simd, 1.0),
            Self::simd_sqrt(simd, value),
        )
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use coe::coerce_static as to;
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                return simd
                    .neon
                    .vaddvq_f32(unsafe { std::mem::transmute(to::<_, pulp::f32x4>(value)) });
            }
        }

        simd.f32s_reduce_sum(value)
    }
}

impl RlstSimd for f64 {
    type Scalars<S: Simd> = S::f64s;
    type Mask<S: Simd> = S::m64s;

    #[inline(always)]
    fn simd_cmp_eq<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.f64s_equal(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_lt<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.f64s_less_than(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_le<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.f64s_less_than_or_equal(lhs, rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::Mask<S>,
        if_true: Self::Scalars<S>,
        if_false: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.m64s_select_f64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn as_simd_slice<S: Simd>(slice: &[Self]) -> (&[Self::Scalars<S>], &[Self]) {
        S::f64s_as_simd(slice)
    }

    #[inline(always)]
    fn as_simd_slice_mut<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Scalars<S>], &mut [Self]) {
        S::f64s_as_mut_simd(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, value: Self) -> Self::Scalars<S> {
        simd.f64s_splat(value)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        simd.f64s_neg(value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f64s_add(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f64s_sub(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f64s_mul(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
        acc: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f64s_mul_add_e(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.f64s_div(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sin_cos<S: Simd>(
        simd: S,
        value: Self::Scalars<S>,
    ) -> (Self::Scalars<S>, Self::Scalars<S>) {
        let mut s_out = Self::simd_splat(simd, Self::zero());
        let mut c_out = Self::simd_splat(simd, Self::zero());
        {
            let value: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));

            let c_out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut c_out));
            let s_out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut s_out));

            for (value, s_out, c_out) in
                itertools::izip!(value.iter(), &mut s_out.iter_mut(), &mut c_out.iter_mut())
            {
                *s_out = Self::sin(*value);
                *c_out = Self::cos(*value)
            }
        }

        (s_out, c_out)
    }

    #[inline(always)]
    fn simd_deinterleave_2<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 2],
    ) -> [Self::Scalars<S>; 2] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 2];
        {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                if coe::is_same::<S, pulp::aarch64::Neon>() {
                    let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                    return unsafe {
                        bytemuck::cast(simd.neon.vld2q_f64(value.as_ptr() as *const f64))
                    };
                }
            }
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[i] = x[2 * i];
                out[n + i] = x[2 * i + 1];
            }
        }
        out
    }

    #[inline(always)]
    fn simd_deinterleave_3<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 3],
    ) -> [Self::Scalars<S>; 3] {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                return unsafe {
                    bytemuck::cast(simd.neon.vld3q_f64(value.as_ptr() as *const f64))
                };
            }
        }

        let mut out = [Self::simd_splat(simd, Self::zero()); 3];
        {
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[i] = x[3 * i];
                out[n + i] = x[3 * i + 1];
                out[2 * n + i] = x[3 * i + 2];
            }
        }
        out
    }

    #[inline(always)]
    fn simd_deinterleave_4<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 4],
    ) -> [Self::Scalars<S>; 4] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 4];
        {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                if coe::is_same::<S, pulp::aarch64::Neon>() {
                    let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                    return unsafe {
                        bytemuck::cast(simd.neon.vld4q_f64(value.as_ptr() as *const f64))
                    };
                }
            }
            let n = std::mem::size_of::<Self::Scalars<S>>() / std::mem::size_of::<Self>();

            let out: &mut [Self] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[Self] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for i in 0..n {
                out[i] = x[4 * i];
                out[n + i] = x[4 * i + 1];
                out[2 * n + i] = x[4 * i + 2];
                out[3 * n + i] = x[4 * i + 3];
            }
        }
        out
    }

    #[inline(always)]
    fn simd_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        #[cfg(target_arch = "x86_64")]
        {
            use coe::coerce_static as to;
            #[cfg(feature = "nightly")]
            if coe::is_same::<S, pulp::x86::V4>() {
                let simd: pulp::x86::V4 = to(simd);
                return to(simd.sqrt_f64x8(to(value)));
            }
            if coe::is_same::<S, pulp::x86::V3>() {
                let simd: pulp::x86::V3 = to(simd);
                return to(simd.sqrt_f64x4(to(value)));
            }
        }
        let mut out = simd.f64s_splat(0.0);
        {
            let out: &mut [f64] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[f64] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for (out, x) in itertools::izip!(out, x) {
                *out = x.sqrt();
            }
        }
        out
    }

    #[inline(always)]
    fn simd_approx_recip<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        simd.f64s_div(simd.f64s_splat(1.0), value)
    }

    #[inline(always)]
    fn simd_approx_recip_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use coe::coerce_static as to;
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                let value: pulp::f64x2 = to(value);
                let mut res: pulp::f64x2 = unsafe {
                    std::mem::transmute(simd.neon.vrsqrteq_f64(std::mem::transmute(value)))
                };
                for _ in 0..3 {
                    res = simd.mul_f64x2(res, unsafe {
                        std::mem::transmute(simd.neon.vrsqrtsq_f64(
                            std::mem::transmute(value),
                            std::mem::transmute(simd.mul_f64x2(res, res)),
                        ))
                    });
                }
                return to(res);
            }
        }

        Self::simd_approx_recip(simd, Self::simd_sqrt(simd, value))
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use coe::coerce_static as to;
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                return simd
                    .neon
                    .vaddvq_f64(unsafe { std::mem::transmute(to::<_, pulp::f64x2>(value)) });
            }
        }
        simd.f64s_reduce_sum(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;

    #[test]
    fn test_sin_cos() {
        let rng = &mut StdRng::seed_from_u64(0);
        let mut max_err_c = 0.0;
        let mut max_err_s = 0.0;
        for _ in 0..1000 {
            let x = rng.gen::<f32>() * 1000.0;
            let (s, c) = f32::simd_sin_cos(pulp::Scalar, x);
            let (s_target, c_target) = x.sin_cos();
            max_err_c = f32::max(max_err_c, (c - c_target).abs() / c_target.abs());
            max_err_s = f32::max(max_err_s, (s - s_target).abs() / s_target.abs());
        }

        assert!(max_err_c <= 1E-4);
        assert!(max_err_s <= 1E-4);

        let mut max_err_c = 0.0;
        let mut max_err_s = 0.0;

        for _ in 0..1000 {
            let x = rng.gen::<f64>() * 1000.0;
            let (s, c) = f64::simd_sin_cos(pulp::Scalar, x);
            let (s_target, c_target) = x.sin_cos();
            max_err_c = f64::max(max_err_c, (c - c_target).abs() / c_target.abs());
            max_err_s = f64::max(max_err_s, (s - s_target).abs() / s_target.abs());
        }

        assert!(max_err_c <= 1E-12);
        assert!(max_err_s <= 1E-12);
    }

    #[test]
    fn test_approx_inv_sqrt() {
        let nsamples = 10000;
        let eps_f32 = 1E-6;
        let eps_f64 = 1E-14;

        let mut rng = StdRng::seed_from_u64(0);

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        for _ in 0..nsamples {
            let sample_f32: [f32; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];
            let sample_f64: [f64; 2] = [rng.gen(), rng.gen()];

            let simd_for = SimdFor::<f32, _>::new(pulp::aarch64::Neon::try_new().unwrap());
            let res_f32 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f32));

            let simd_for = SimdFor::<f64, _>::new(pulp::aarch64::Neon::try_new().unwrap());
            let res_f64 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f64));

            assert_relative_eq!(
                res_f32.0,
                1.0 / sample_f32[0].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.1,
                1.0 / sample_f32[1].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.2,
                1.0 / sample_f32[2].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.3,
                1.0 / sample_f32[3].sqrt(),
                max_relative = eps_f32
            );

            assert_relative_eq!(
                res_f64.0,
                1.0 / sample_f64[0].sqrt(),
                max_relative = eps_f64
            );
            assert_relative_eq!(
                res_f64.1,
                1.0 / sample_f64[1].sqrt(),
                max_relative = eps_f64
            );
        }
        #[cfg(target_arch = "x86_64")]
        for _ in 0..nsamples {
            let sample_f32: [f32; 8] = [
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];
            let sample_f64: [f64; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];

            let simd_for = SimdFor::<f32, _>::new(pulp::x86::V3::try_new().unwrap());
            let res_f32 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f32));

            let simd_for = SimdFor::<f64, _>::new(pulp::x86::V3::try_new().unwrap());
            let res_f64 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f64));

            assert_relative_eq!(
                res_f32.0,
                1.0 / sample_f32[0].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.1,
                1.0 / sample_f32[1].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.2,
                1.0 / sample_f32[2].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.3,
                1.0 / sample_f32[3].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.4,
                1.0 / sample_f32[4].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.5,
                1.0 / sample_f32[5].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.6,
                1.0 / sample_f32[6].sqrt(),
                max_relative = eps_f32
            );
            assert_relative_eq!(
                res_f32.7,
                1.0 / sample_f32[7].sqrt(),
                max_relative = eps_f32
            );

            assert_relative_eq!(
                res_f64.0,
                1.0 / sample_f64[0].sqrt(),
                max_relative = eps_f64
            );
            assert_relative_eq!(
                res_f64.1,
                1.0 / sample_f64[1].sqrt(),
                max_relative = eps_f64
            );
            assert_relative_eq!(
                res_f64.2,
                1.0 / sample_f64[2].sqrt(),
                max_relative = eps_f64
            );
            assert_relative_eq!(
                res_f64.3,
                1.0 / sample_f64[3].sqrt(),
                max_relative = eps_f64
            );
        }
    }

    #[test]
    fn test_approx_inv() {
        let nsamples = 10000;
        let eps_f32 = 1E-6;
        let eps_f64 = 1E-14;

        let rng = &mut StdRng::seed_from_u64(0);

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        for _ in 0..nsamples {
            let sample_f32: [f32; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];
            let sample_f64: [f64; 2] = [rng.gen(), rng.gen()];

            let simd_for = SimdFor::<f32, _>::new(pulp::aarch64::Neon::try_new().unwrap());
            let res_f32 = simd_for.approx_recip(bytemuck::cast(sample_f32));

            let simd_for = SimdFor::<f64, _>::new(pulp::aarch64::Neon::try_new().unwrap());
            let res_f64 = simd_for.approx_recip(bytemuck::cast(sample_f64));

            assert_relative_eq!(res_f32.0, 1.0 / sample_f32[0], max_relative = eps_f32);
            assert_relative_eq!(res_f32.1, 1.0 / sample_f32[1], max_relative = eps_f32);
            assert_relative_eq!(res_f32.2, 1.0 / sample_f32[2], max_relative = eps_f32);
            assert_relative_eq!(res_f32.3, 1.0 / sample_f32[3], max_relative = eps_f32);

            assert_relative_eq!(res_f64.0, 1.0 / sample_f64[0], max_relative = eps_f64);
            assert_relative_eq!(res_f64.1, 1.0 / sample_f64[1], max_relative = eps_f64);
        }
        #[cfg(target_arch = "x86_64")]
        for _ in 0..nsamples {
            let sample_f32: [f32; 8] = [
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];
            let sample_f64: [f64; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];

            let simd_for = SimdFor::<f32, _>::new(pulp::x86::V3::try_new().unwrap());
            let res_f32 = simd_for.approx_recip(bytemuck::cast(sample_f32));

            let simd_for = SimdFor::<f64, _>::new(pulp::x86::V3::try_new().unwrap());
            let res_f64 = simd_for.approx_recip(bytemuck::cast(sample_f64));

            assert_relative_eq!(res_f32.0, 1.0 / sample_f32[0], max_relative = eps_f32);
            assert_relative_eq!(res_f32.1, 1.0 / sample_f32[1], max_relative = eps_f32);
            assert_relative_eq!(res_f32.2, 1.0 / sample_f32[2], max_relative = eps_f32);
            assert_relative_eq!(res_f32.3, 1.0 / sample_f32[3], max_relative = eps_f32);
            assert_relative_eq!(res_f32.4, 1.0 / sample_f32[4], max_relative = eps_f32);
            assert_relative_eq!(res_f32.5, 1.0 / sample_f32[5], max_relative = eps_f32);
            assert_relative_eq!(res_f32.6, 1.0 / sample_f32[6], max_relative = eps_f32);
            assert_relative_eq!(res_f32.7, 1.0 / sample_f32[7], max_relative = eps_f32);

            assert_relative_eq!(res_f64.0, 1.0 / sample_f64[0], max_relative = eps_f64);
            assert_relative_eq!(res_f64.1, 1.0 / sample_f64[1], max_relative = eps_f64);
            assert_relative_eq!(res_f64.2, 1.0 / sample_f64[2], max_relative = eps_f64);
            assert_relative_eq!(res_f64.3, 1.0 / sample_f64[3], max_relative = eps_f64);
        }
    }
}
