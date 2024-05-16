//! Basic traits for SIMD Operations

use bytemuck::Pod;
use pulp::Simd;
use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Copy, Clone, Debug)]
pub struct SimdFor<T, S> {
    pub simd: S,
    __marker: PhantomData<T>,
}

#[allow(dead_code)]
impl<T: RlstSimd, S: Simd> SimdFor<T, S> {
    #[inline]
    pub fn new(simd: S) -> Self {
        Self {
            simd,
            __marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn splat(self, value: T) -> T::Scalars<S> {
        T::simd_splat(self.simd, value)
    }

    #[inline(always)]
    pub fn cmp_eq(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Mask<S> {
        T::simd_cmp_eq(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn cmp_lt(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Mask<S> {
        T::simd_cmp_lt(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn cmp_le(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Mask<S> {
        T::simd_cmp_le(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn select(
        self,
        mask: T::Mask<S>,
        if_true: T::Scalars<S>,
        if_false: T::Scalars<S>,
    ) -> T::Scalars<S> {
        T::simd_select(self.simd, mask, if_true, if_false)
    }

    #[inline(always)]
    pub fn neg(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_neg(self.simd, value)
    }
    #[inline(always)]
    pub fn add(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_add(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn sub(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_sub(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn mul(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_mul(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn mul_add(
        self,
        lhs: T::Scalars<S>,
        rhs: T::Scalars<S>,
        acc: T::Scalars<S>,
    ) -> T::Scalars<S> {
        T::simd_mul_add(self.simd, lhs, rhs, acc)
    }
    #[inline(always)]
    pub fn div(self, lhs: T::Scalars<S>, rhs: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_div(self.simd, lhs, rhs)
    }
    #[inline(always)]
    pub fn sin_cos(self, value: T::Scalars<S>) -> (T::Scalars<S>, T::Scalars<S>) {
        T::simd_sin_cos(self.simd, value)
    }
    #[inline(always)]
    pub fn sin_cos_quarter_circle(self, value: T::Scalars<S>) -> (T::Scalars<S>, T::Scalars<S>) {
        T::simd_sin_cos_quarter_circle(self.simd, value)
    }
    #[inline(always)]
    pub fn sqrt(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_sqrt(self.simd, value)
    }
    #[inline(always)]
    pub fn approx_recip(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_approx_recip(self.simd, value)
    }
    #[inline(always)]
    pub fn approx_recip_sqrt(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_approx_recip_sqrt(self.simd, value)
    }
    #[inline(always)]
    pub fn reduce_add(self, value: T::Scalars<S>) -> T {
        T::simd_reduce_add(self.simd, value)
    }

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

/// [`rlst::RlstScalar`] extension trait for SIMD operations.
#[allow(dead_code)]
pub trait RlstSimd: Pod + Send + Sync + num::Zero + 'static {
    /// Simd register that has the layout `[Self; N]` for some `N > 0`.
    type Scalars<S: Simd>: Pod + Copy + Send + Sync + Debug + 'static;
    /// Simd mask register that has the layout `[Self; N]` for some `N > 0`.
    type Mask<S: Simd>: Copy + Send + Sync + Debug + 'static;

    /// Splits the slice into a vector and scalar part.
    fn as_simd_slice<S: Simd>(slice: &[Self]) -> (&[Self::Scalars<S>], &[Self]);

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
    fn simd_sin_cos_quarter_circle<S: Simd>(
        simd: S,
        value: Self::Scalars<S>,
    ) -> (Self::Scalars<S>, Self::Scalars<S>);

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
    fn simd_deinterleave_2<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 2],
    ) -> [Self::Scalars<S>; 2] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 2];
        {
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

    /// Deinterleaves a register of values `[x0, y0, z0, x1, y1, z1, ...]` to
    /// `[x0, x1, ... y0, y1, ..., z0, z1, ...]`.
    fn simd_deinterleave_3<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 3],
    ) -> [Self::Scalars<S>; 3] {
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

    /// Deinterleaves a register of values `[x0, y0, z0, w0, x1, y1, z1, w1, ...]` to
    /// `[x0, x1, ... y0, y1, ..., z0, z1, ..., w0, w1, ...]`.
    fn simd_deinterleave_4<S: Simd>(
        simd: S,
        value: [Self::Scalars<S>; 4],
    ) -> [Self::Scalars<S>; 4] {
        let mut out = [Self::simd_splat(simd, Self::zero()); 4];
        {
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

    /// Inverse of [`RealScalar::deinterleave_2`].
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

    /// Inverse of [`RealScalar::deinterleave_3`].
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

    /// Inverse of [`RealScalar::deinterleave_4`].
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
        #[inline(always)]
        fn div_2pi_mod_1<S: Simd>(simd: S, x: S::f32s) -> S::f32s {
            #[cfg(target_arch = "x86_64")]
            {
                use coe::coerce_static as to;

                #[cfg(feature = "nightly")]
                if coe::is_same::<S, pulp::x86::V4>() {
                    let simd: pulp::x86::V4 = to(simd);
                    let x = to(x);
                    let div = simd.div_f32x16(x, simd.splat_f32x16(2.0 * std::f32::consts::PI));
                    return to(simd.sub_f32x16(div, simd.floor_f32x16(div)));
                }
                if coe::is_same::<S, pulp::x86::V3>() {
                    let simd: pulp::x86::V3 = to(simd);
                    let x = to(x);

                    let div = simd.div_f32x8(x, simd.splat_f32x8(2.0 * std::f32::consts::PI));
                    return to(simd.sub_f32x8(div, simd.floor_f32x8(div)));
                }
            }

            let mut out = simd.f32s_splat(0.0);
            {
                let out: &mut [f32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
                let x: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&x));

                for (out, x) in itertools::izip!(out, x) {
                    let div = x / (2.0 * std::f32::consts::PI);
                    *out = div - div.floor();
                }
            }
            out
        }

        let x_o2pi = div_2pi_mod_1(simd, value);
        let flip_y = simd.f32s_greater_than_or_equal(x_o2pi, simd.f32s_splat(0.5));
        let x_o2pi =
            simd.m32s_select_f32s(flip_y, simd.f32s_sub(simd.f32s_splat(1.0), x_o2pi), x_o2pi);
        let flip_x = simd.f32s_greater_than_or_equal(x_o2pi, simd.f32s_splat(0.25));
        let x_o2pi =
            simd.m32s_select_f32s(flip_x, simd.f32s_sub(simd.f32s_splat(0.5), x_o2pi), x_o2pi);
        let swap = simd.f32s_greater_than_or_equal(x_o2pi, simd.f32s_splat(0.125));
        let x_o2pi =
            simd.m32s_select_f32s(swap, simd.f32s_sub(simd.f32s_splat(0.25), x_o2pi), x_o2pi);

        let x = simd.f32s_mul(x_o2pi, simd.f32s_splat(2.0 * std::f32::consts::PI));
        let (sin, cos) = Self::simd_sin_cos_quarter_circle(simd, x);
        let (sin, cos) = (
            simd.m32s_select_f32s(swap, cos, sin),
            simd.m32s_select_f32s(swap, sin, cos),
        );

        (
            simd.m32s_select_f32s(flip_y, simd.f32s_neg(sin), sin),
            simd.m32s_select_f32s(flip_x, simd.f32s_neg(cos), cos),
        )
    }

    #[inline(always)]
    fn simd_sin_cos_quarter_circle<S: Simd>(
        simd: S,
        value: Self::Scalars<S>,
    ) -> (Self::Scalars<S>, Self::Scalars<S>) {
        // constants taken from eve (c++ simd library)
        const C0: f32 = hexf::hexf32!("0x1.55554ap-5");
        const C1: f32 = hexf::hexf32!("-0x1.6c0c32p-10");
        const C2: f32 = hexf::hexf32!("0x1.99eb9cp-16");

        const S0: f32 = hexf::hexf32!("-0x1.555544p-3");
        const S1: f32 = hexf::hexf32!("0x1.11073ap-7");
        const S2: f32 = hexf::hexf32!("-0x1.9943f2p-13");

        let x = value;
        let z = simd.f32s_mul(x, x);

        let c = {
            let y = simd.f32s_mul_add(
                simd.f32s_mul_add(simd.f32s_splat(C2), z, simd.f32s_splat(C1)),
                z,
                simd.f32s_splat(C0),
            );
            simd.f32s_mul_add(
                simd.f32s_mul_add(y, z, simd.f32s_splat(-0.5)),
                z,
                simd.f32s_splat(1.0),
            )
        };
        let s = {
            let y = simd.f32s_mul_add(
                simd.f32s_mul_add(simd.f32s_splat(S2), z, simd.f32s_splat(S1)),
                z,
                simd.f32s_splat(S0),
            );
            simd.f32s_mul_add(simd.f32s_mul(y, z), x, x)
        };

        (s, c)
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
        Self::simd_approx_recip(simd, Self::simd_sqrt(simd, value))
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self {
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
        #[inline(always)]
        fn div_2pi_mod_1<S: Simd>(simd: S, x: S::f64s) -> S::f64s {
            #[cfg(target_arch = "x86_64")]
            {
                use coe::coerce_static as to;

                #[cfg(feature = "nightly")]
                if coe::is_same::<S, pulp::x86::V4>() {
                    let simd: pulp::x86::V4 = to(simd);
                    let x = to(x);
                    let div = simd.div_f64x8(x, simd.splat_f64x8(2.0 * std::f64::consts::PI));
                    return to(simd.sub_f64x8(div, simd.floor_f64x8(div)));
                }
                if coe::is_same::<S, pulp::x86::V3>() {
                    let simd: pulp::x86::V3 = to(simd);
                    let x = to(x);

                    let div = simd.div_f64x4(x, simd.splat_f64x4(2.0 * std::f64::consts::PI));
                    return to(simd.sub_f64x4(div, simd.floor_f64x4(div)));
                }
            }

            let mut out = simd.f64s_splat(0.0);
            {
                let out: &mut [f64] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
                let x: &[f64] = bytemuck::cast_slice(std::slice::from_ref(&x));

                for (out, x) in itertools::izip!(out, x) {
                    let div = x / (2.0 * std::f64::consts::PI);
                    *out = div - div.floor();
                }
            }
            out
        }

        let x_o2pi = div_2pi_mod_1(simd, value);
        let flip_y = simd.f64s_greater_than_or_equal(x_o2pi, simd.f64s_splat(0.5));
        let x_o2pi =
            simd.m64s_select_f64s(flip_y, simd.f64s_sub(simd.f64s_splat(1.0), x_o2pi), x_o2pi);
        let flip_x = simd.f64s_greater_than_or_equal(x_o2pi, simd.f64s_splat(0.25));
        let x_o2pi =
            simd.m64s_select_f64s(flip_x, simd.f64s_sub(simd.f64s_splat(0.5), x_o2pi), x_o2pi);
        let swap = simd.f64s_greater_than_or_equal(x_o2pi, simd.f64s_splat(0.125));
        let x_o2pi =
            simd.m64s_select_f64s(swap, simd.f64s_sub(simd.f64s_splat(0.25), x_o2pi), x_o2pi);

        let x = simd.f64s_mul(x_o2pi, simd.f64s_splat(2.0 * std::f64::consts::PI));
        let (sin, cos) = Self::simd_sin_cos_quarter_circle(simd, x);
        let (sin, cos) = (
            simd.m64s_select_f64s(swap, cos, sin),
            simd.m64s_select_f64s(swap, sin, cos),
        );
        (
            simd.m64s_select_f64s(flip_y, simd.f64s_neg(sin), sin),
            simd.m64s_select_f64s(flip_x, simd.f64s_neg(cos), cos),
        )
    }

    #[inline(always)]
    fn simd_sin_cos_quarter_circle<S: Simd>(
        simd: S,
        value: Self::Scalars<S>,
    ) -> (Self::Scalars<S>, Self::Scalars<S>) {
        const C0: f64 = hexf::hexf64!("0x1.0000000000000p-1");
        const C1: f64 = hexf::hexf64!("-0x1.5555555555551p-5");
        const C2: f64 = hexf::hexf64!("0x1.6c16c16c15d47p-10");
        const C3: f64 = hexf::hexf64!("-0x1.a01a019ddbcd9p-16");
        const C4: f64 = hexf::hexf64!("0x1.27e4f8e06d9a5p-22");
        const C5: f64 = hexf::hexf64!("-0x1.1eea7c1e514d4p-29");
        const C6: f64 = hexf::hexf64!("0x1.8ff831ad9b219p-37");

        const S0: f64 = hexf::hexf64!("-0x1.5555555555548p-3");
        const S1: f64 = hexf::hexf64!("0x1.111111110f7d0p-7");
        const S2: f64 = hexf::hexf64!("-0x1.a01a019bfdf03p-13");
        const S3: f64 = hexf::hexf64!("0x1.71de3567d4896p-19");
        const S4: f64 = hexf::hexf64!("-0x1.ae5e5a9291691p-26");
        const S5: f64 = hexf::hexf64!("0x1.5d8fd1fcf0ec1p-33");

        let x = value;
        let z = simd.f64s_mul(x, x);

        let c = {
            let y = simd.f64s_splat(C6);
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(C5));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(C4));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(C3));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(C2));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(C1));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(C0));

            simd.f64s_mul_add(simd.f64s_neg(y), z, simd.f64s_splat(1.0))
        };
        let s = {
            let y = simd.f64s_splat(S5);
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(S4));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(S3));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(S2));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(S1));
            let y = simd.f64s_mul_add(z, y, simd.f64s_splat(S0));

            simd.f64s_mul_add(simd.f64s_mul(y, z), x, x)
        };

        (s, c)
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
        Self::simd_approx_recip(simd, Self::simd_sqrt(simd, value))
    }

    #[inline(always)]
    fn simd_reduce_add<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self {
        simd.f64s_reduce_sum(value)
    }
}
