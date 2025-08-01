//! Basic traits for SIMD Operations

use crate::traits::rlst_num::RlstSimd;
use coe;
use num::Zero;
use pulp::Simd;
use std::fmt::Debug;
use std::marker::PhantomData;
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

    /// Length of a Simd vector.
    pub fn simd_vector_width(self) -> usize {
        std::mem::size_of::<<T as RlstSimd>::Scalars<S>>() / std::mem::size_of::<T>()
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

    /// Compute the base e exponential of a Simd vector.
    #[inline(always)]
    pub fn exp(self, value: T::Scalars<S>) -> T::Scalars<S> {
        T::simd_exp(self.simd, value)
    }

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

impl RlstSimd for f32 {
    type Scalars<S: Simd> = S::f32s;
    type Mask<S: Simd> = S::m32s;

    #[inline(always)]
    fn simd_cmp_eq<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.equal_f32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_lt<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.less_than_f32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_le<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.less_than_or_equal_f32s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::Mask<S>,
        if_true: Self::Scalars<S>,
        if_false: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.select_f32s_m32s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn as_simd_slice<S: Simd>(slice: &[Self]) -> (&[Self::Scalars<S>], &[Self]) {
        S::as_simd_f32s(slice)
    }

    #[inline(always)]
    fn as_simd_slice_mut<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Scalars<S>], &mut [Self]) {
        S::as_mut_simd_f32s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, value: Self) -> Self::Scalars<S> {
        simd.splat_f32s(value)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        simd.neg_f32s(value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.add_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.sub_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.mul_f32s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
        acc: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.mul_add_e_f32s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.div_f32s(lhs, rhs)
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
                itertools::izip!(value.iter(), s_out.iter_mut(), c_out.iter_mut())
            {
                *s_out = Self::sin(*value);
                *c_out = Self::cos(*value);
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
                    return bytemuck::cast(unsafe {
                        simd.neon.vld2q_f32(value.as_ptr() as *const f32)
                    });
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
    fn simd_exp<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        let mut out = simd.splat_f32s(0.0);
        {
            let out: &mut [f32] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[f32] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for (out, x) in itertools::izip!(out, x) {
                *out = x.exp();
            }
        }
        out
    }

    #[inline(always)]
    fn simd_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        let mut out = simd.splat_f32s(0.0);
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
        // #[cfg(target_arch = "x86_64")]
        // {
        //     use coe::coerce_static as to;
        //     #[cfg(feature = "nightly")]
        //     if coe::is_same::<S, pulp::x86::V4>() {
        //         let simd: pulp::x86::V4 = to(simd);
        //         return to(simd.avx512f._mm512_rcp14_ps(to(value)));
        //     }
        //     if coe::is_same::<S, pulp::x86::V3>() {
        //         let simd: pulp::x86::V3 = to(simd);
        //         return to(simd.approx_reciprocal_f32x8(to(value)));
        //     }
        // }
        simd.div_f32s(simd.splat_f32s(1.0), value)
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
                // Initial approximation
                let value = to(value);
                let mut x = simd.approx_reciprocal_sqrt_f32x8(value);
                let minus_half_value = simd.mul_f32x8(simd.splat_f32x8(-0.5), value);
                let three_half = simd.splat_f32x8(1.5);

                for _ in 0..1 {
                    x = simd.mul_f32x8(
                        x,
                        simd.mul_add_f32x8(minus_half_value, simd.mul_f32x8(x, x), three_half),
                    );
                }
                return to(x);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use coe::coerce_static as to;
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                let mut res: pulp::f32x4 =
                    bytemuck::cast(simd.neon.vrsqrteq_f32(bytemuck::cast(value)));
                for _ in 0..2 {
                    res = simd.mul_f32x4(
                        res,
                        bytemuck::cast(simd.neon.vrsqrtsq_f32(
                            bytemuck::cast(value),
                            bytemuck::cast(simd.mul_f32x4(res, res)),
                        )),
                    );
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
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                return simd.neon.vaddvq_f32(bytemuck::cast(value));
            }
        }

        simd.reduce_sum_f32s(value)
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
        simd.equal_f64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_lt<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.less_than_f64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_cmp_le<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Mask<S> {
        simd.less_than_or_equal_f64s(lhs, rhs)
    }
    #[inline(always)]
    fn simd_select<S: Simd>(
        simd: S,
        mask: Self::Mask<S>,
        if_true: Self::Scalars<S>,
        if_false: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.select_f64s_m64s(mask, if_true, if_false)
    }

    #[inline(always)]
    fn as_simd_slice<S: Simd>(slice: &[Self]) -> (&[Self::Scalars<S>], &[Self]) {
        S::as_simd_f64s(slice)
    }

    #[inline(always)]
    fn as_simd_slice_mut<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Scalars<S>], &mut [Self]) {
        S::as_mut_simd_f64s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, value: Self) -> Self::Scalars<S> {
        simd.splat_f64s(value)
    }

    #[inline(always)]
    fn simd_neg<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        simd.neg_f64s(value)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.add_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.sub_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.mul_f64s(lhs, rhs)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
        acc: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.mul_add_e_f64s(lhs, rhs, acc)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(
        simd: S,
        lhs: Self::Scalars<S>,
        rhs: Self::Scalars<S>,
    ) -> Self::Scalars<S> {
        simd.div_f64s(lhs, rhs)
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
                itertools::izip!(value.iter(), s_out.iter_mut(), c_out.iter_mut())
            {
                *s_out = Self::sin(*value);
                *c_out = Self::cos(*value);
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
                return bytemuck::cast(unsafe {
                    simd.neon.vld3q_f64(value.as_ptr() as *const f64)
                });
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
    fn simd_exp<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        let mut out = simd.splat_f64s(0.0);
        {
            let out: &mut [f64] = bytemuck::cast_slice_mut(std::slice::from_mut(&mut out));
            let x: &[f64] = bytemuck::cast_slice(std::slice::from_ref(&value));
            for (out, x) in itertools::izip!(out, x) {
                *out = x.exp();
            }
        }
        out
    }

    #[inline(always)]
    fn simd_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        let mut out = simd.splat_f64s(0.0);
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
        simd.div_f64s(simd.splat_f64s(1.0), value)
    }

    #[inline(always)]
    fn simd_approx_recip_sqrt<S: Simd>(simd: S, value: Self::Scalars<S>) -> Self::Scalars<S> {
        #[cfg(target_arch = "x86_64")]
        if coe::is_same::<S, pulp::x86::V3>() {
            use coe::coerce_static as to;
            let simd: pulp::x86::V3 = to(simd);

            // For the initial approximation convert to f32 and use the f32 routine
            let value_f32 = bytemuck::cast::<_, [f64; 4]>(value);
            let value_f32: pulp::f32x4 = bytemuck::cast([
                value_f32[0] as f32,
                value_f32[1] as f32,
                value_f32[2] as f32,
                value_f32[3] as f32,
            ]);

            let x = simd.approx_reciprocal_sqrt_f32x4(value_f32);
            let mut x: pulp::f64x4 =
                bytemuck::cast([x.0 as f64, x.1 as f64, x.2 as f64, x.3 as f64]);

            let minus_half_value = simd.mul_f64x4(simd.splat_f64x4(-0.5), bytemuck::cast(value));
            let three_half = simd.splat_f64x4(1.5);

            for _ in 0..2 {
                x = simd.mul_f64x4(
                    x,
                    simd.mul_add_f64x4(minus_half_value, simd.mul_f64x4(x, x), three_half),
                );
            }
            return to(x);
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use coe::coerce_static as to;
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                let value: pulp::f64x2 = to(value);
                let mut res: pulp::f64x2 =
                    bytemuck::cast(simd.neon.vrsqrteq_f64(bytemuck::cast(value)));
                for _ in 0..3 {
                    res = simd.mul_f64x2(
                        res,
                        bytemuck::cast(simd.neon.vrsqrtsq_f64(
                            bytemuck::cast(value),
                            bytemuck::cast(simd.mul_f64x2(res, res)),
                        )),
                    );
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
            if coe::is_same::<S, pulp::aarch64::Neon>() {
                let simd: pulp::aarch64::Neon = coe::coerce_static(simd);
                return simd.neon.vaddvq_f64(bytemuck::cast(value));
            }
        }
        simd.reduce_sum_f64s(value)
    }
}

#[cfg(test)]
mod tests {

    use crate::traits::rlst_num::RlstScalar;

    use super::*;
    use approx::assert_relative_eq;
    use coe::{coerce_static, Coerce};
    use paste;
    use rand::prelude::*;

    #[test]
    fn test_simd_approx_recip() {
        let nsamples = 1001;

        trait Tolerance: RlstScalar<Real = Self> {
            const TOL: Self;
        }

        impl Tolerance for f32 {
            const TOL: f32 = 1E-6;
        }

        impl Tolerance for f64 {
            const TOL: f64 = 1E-13;
        }

        struct Impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> {
            rng: StdRng,
            nsamples: usize,
            _marker: PhantomData<T>,
        }

        impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> pulp::WithSimd for Impl<T> {
            type Output = ();

            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let Self {
                    mut rng, nsamples, ..
                } = self;

                let samples = (0..nsamples)
                    .map(|_| num::cast::<_, T>(rng.random::<f64>()).unwrap())
                    .collect::<Vec<_>>();

                let mut actual = vec![T::default(); nsamples];

                let (value_head, value_tail) =
                    <T as RlstSimd>::as_simd_slice::<S>(samples.as_slice());
                let (actual_head, actual_tail) =
                    <T as RlstSimd>::as_simd_slice_mut::<S>(actual.as_mut_slice());

                fn impl_slice<T: RlstScalar<Real = T> + RlstSimd + Tolerance, S: Simd>(
                    simd: S,
                    values: &[<T as RlstSimd>::Scalars<S>],
                    actual: &mut [<T as RlstSimd>::Scalars<S>],
                ) {
                    let simd = SimdFor::<T, S>::new(simd);

                    for (&v, a) in itertools::izip!(values, actual) {
                        *a = simd.approx_recip(v);
                    }
                }

                impl_slice::<T, _>(simd, value_head, actual_head);
                impl_slice::<T, _>(
                    pulp::Scalar::new(),
                    value_tail.coerce(),
                    actual_tail.coerce(),
                );

                if coe::is_same::<T, f32>() {
                    for (v, a) in itertools::izip!(samples, actual) {
                        let v: f32 = coerce_static(v);
                        let a: f32 = coerce_static(a);
                        assert_relative_eq!(1.0 / v, a, max_relative = f32::TOL);
                    }
                } else if coe::is_same::<T, f64>() {
                    for (v, a) in itertools::izip!(samples, actual) {
                        let v: f64 = coerce_static(v);
                        let a: f64 = coerce_static(a);
                        assert_relative_eq!(1.0 / v, a, max_relative = f64::TOL);
                    }
                } else {
                    panic!("Unsupported type");
                }
            }
        }

        pulp::Arch::new().dispatch(Impl::<f32> {
            rng: StdRng::seed_from_u64(0),
            nsamples,
            _marker: PhantomData,
        });

        pulp::Arch::new().dispatch(Impl::<f64> {
            rng: StdRng::seed_from_u64(0),
            nsamples,
            _marker: PhantomData,
        });
    }

    #[test]
    fn test_simd_approx_recip_sqrt() {
        let nsamples = 1001;

        trait Tolerance: RlstScalar<Real = Self> {
            const TOL: Self;
        }

        impl Tolerance for f32 {
            const TOL: f32 = 1E-6;
        }

        impl Tolerance for f64 {
            const TOL: f64 = 1E-13;
        }

        struct Impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> {
            rng: StdRng,
            nsamples: usize,
            _marker: PhantomData<T>,
        }

        impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> pulp::WithSimd for Impl<T> {
            type Output = ();

            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let Self {
                    mut rng, nsamples, ..
                } = self;

                let samples = (0..nsamples)
                    .map(|_| num::cast::<_, T>(rng.random::<f64>()).unwrap())
                    .collect::<Vec<_>>();

                let mut actual = vec![T::default(); nsamples];

                let (value_head, value_tail) =
                    <T as RlstSimd>::as_simd_slice::<S>(samples.as_slice());
                let (actual_head, actual_tail) =
                    <T as RlstSimd>::as_simd_slice_mut::<S>(actual.as_mut_slice());

                fn impl_slice<T: RlstScalar<Real = T> + RlstSimd + Tolerance, S: Simd>(
                    simd: S,
                    values: &[<T as RlstSimd>::Scalars<S>],
                    actual: &mut [<T as RlstSimd>::Scalars<S>],
                ) {
                    let simd = SimdFor::<T, S>::new(simd);

                    for (&v, a) in itertools::izip!(values, actual) {
                        *a = simd.approx_recip_sqrt(v);
                    }
                }

                impl_slice::<T, _>(simd, value_head, actual_head);
                impl_slice::<T, _>(
                    pulp::Scalar::new(),
                    value_tail.coerce(),
                    actual_tail.coerce(),
                );

                if coe::is_same::<T, f32>() {
                    for (v, a) in itertools::izip!(samples, actual) {
                        let v: f32 = coerce_static(v);
                        let a: f32 = coerce_static(a);
                        assert_relative_eq!(1.0 / v.sqrt(), a, max_relative = f32::TOL);
                    }
                } else if coe::is_same::<T, f64>() {
                    for (v, a) in itertools::izip!(samples, actual) {
                        let v: f64 = coerce_static(v);
                        let a: f64 = coerce_static(a);
                        assert_relative_eq!(1.0 / v.sqrt(), a, max_relative = f64::TOL);
                    }
                } else {
                    panic!("Unsupported type");
                }
            }
        }

        pulp::Arch::new().dispatch(Impl::<f32> {
            rng: StdRng::seed_from_u64(0),
            nsamples,
            _marker: PhantomData,
        });

        pulp::Arch::new().dispatch(Impl::<f64> {
            rng: StdRng::seed_from_u64(0),
            nsamples,
            _marker: PhantomData,
        });
    }

    #[test]
    fn test_simd_sin_cos() {
        let nsamples = 1001;

        trait Tolerance: RlstScalar<Real = Self> {
            const TOL: Self;
        }

        impl Tolerance for f64 {
            const TOL: f64 = 1E-13;
        }

        impl Tolerance for f32 {
            const TOL: f32 = 1E-6;
        }

        struct Impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> {
            rng: StdRng,
            nsamples: usize,
            _marker: PhantomData<T>,
        }

        impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> pulp::WithSimd for Impl<T> {
            type Output = ();

            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let Self {
                    mut rng, nsamples, ..
                } = self;

                let samples = (0..nsamples)
                    .map(|_| num::cast::<_, T>(rng.random::<f64>()).unwrap())
                    .collect::<Vec<_>>();

                let mut actual_sin = vec![T::default(); nsamples];
                let mut actual_cos = vec![T::default(); nsamples];

                let (value_head, value_tail) =
                    <T as RlstSimd>::as_simd_slice::<S>(samples.as_slice());
                let (actual_sin_head, actual_sin_tail) =
                    <T as RlstSimd>::as_simd_slice_mut::<S>(actual_sin.as_mut_slice());
                let (actual_cos_head, actual_cos_tail) =
                    <T as RlstSimd>::as_simd_slice_mut::<S>(actual_cos.as_mut_slice());

                fn impl_slice<T: RlstScalar<Real = T> + RlstSimd + Tolerance, S: Simd>(
                    simd: S,
                    values: &[<T as RlstSimd>::Scalars<S>],
                    actual_sin: &mut [<T as RlstSimd>::Scalars<S>],
                    actual_cos: &mut [<T as RlstSimd>::Scalars<S>],
                ) {
                    let simd = SimdFor::<T, S>::new(simd);

                    for (&v, asin, acos) in itertools::izip!(values, actual_sin, actual_cos) {
                        (*asin, *acos) = simd.sin_cos(v);
                    }
                }

                impl_slice::<T, _>(simd, value_head, actual_sin_head, actual_cos_head);
                impl_slice::<T, _>(
                    pulp::Scalar::new(),
                    value_tail.coerce(),
                    actual_sin_tail.coerce(),
                    actual_cos_tail.coerce(),
                );

                if coe::is_same::<T, f32>() {
                    for (v, s, c) in itertools::izip!(samples, actual_sin, actual_cos) {
                        let v: f32 = coerce_static(v);
                        let s: f32 = coerce_static(s);
                        let c: f32 = coerce_static(c);
                        assert_relative_eq!(v.sin(), s, max_relative = f32::TOL);
                        assert_relative_eq!(v.cos(), c, max_relative = f32::TOL);
                    }
                } else if coe::is_same::<T, f64>() {
                    for (v, s, c) in itertools::izip!(samples, actual_sin, actual_cos) {
                        let v: f64 = coerce_static(v);
                        let s: f64 = coerce_static(s);
                        let c: f64 = coerce_static(c);
                        assert_relative_eq!(v.sin(), s, max_relative = f64::TOL);
                        assert_relative_eq!(v.cos(), c, max_relative = f64::TOL);
                    }
                } else {
                    panic!("Unsupported type");
                }
            }
        }

        pulp::Arch::new().dispatch(Impl::<f32> {
            rng: StdRng::seed_from_u64(0),
            nsamples,
            _marker: PhantomData,
        });

        pulp::Arch::new().dispatch(Impl::<f64> {
            rng: StdRng::seed_from_u64(0),
            nsamples,
            _marker: PhantomData,
        });
    }

    macro_rules! impl_unary_simd_test {
        ($name:expr, $tol_f32:expr, $tol_f64:expr) => {
            paste::paste! {
            #[test]
            fn [<test_simd_ $name>]() {
                let nsamples = 1001;

                trait Tolerance: RlstScalar<Real = Self> {
                    const TOL: Self;
                }

                impl Tolerance for f32 {
                    const TOL: f32 = $tol_f32;
                }

                impl Tolerance for f64 {
                    const TOL: f64 = $tol_f64;
                }


                struct Impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> {
                    rng: StdRng,
                    nsamples: usize,
                    _marker: PhantomData<T>,
                }

                impl<T: RlstScalar<Real = T> + RlstSimd + Tolerance> pulp::WithSimd for Impl<T> {
                    type Output = ();

                    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                        let Self {
                            mut rng, nsamples, ..
                        } = self;

                        let samples = (0..nsamples)
                            .map(|_| num::cast::<_, T>(rng.random::<f64>()).unwrap())
                            .collect::<Vec<_>>();

                        let mut actual = vec![T::default(); nsamples];

                        let (value_head, value_tail) =
                            <T as RlstSimd>::as_simd_slice::<S>(samples.as_slice());
                        let (actual_head, actual_tail) =
                            <T as RlstSimd>::as_simd_slice_mut::<S>(actual.as_mut_slice());

                        fn impl_slice<T: RlstScalar<Real = T> + RlstSimd + Tolerance, S: Simd>(
                            simd: S,
                            values: &[<T as RlstSimd>::Scalars<S>],
                            actual: &mut [<T as RlstSimd>::Scalars<S>],
                        ) {
                            let simd = SimdFor::<T, S>::new(simd);

                            for (&v, a) in itertools::izip!(values, actual) {
                                *a = simd.[<$name>](v);
                            }
                        }

                        impl_slice::<T, _>(simd, value_head, actual_head);
                        impl_slice::<T, _>(
                            pulp::Scalar::new(),
                            value_tail.coerce(),
                            actual_tail.coerce(),
                        );

                        if coe::is_same::<T, f32>() {
                            for (v, a) in itertools::izip!(samples, actual) {
                                let v: f32 = coerce_static(v);
                                let a: f32 = coerce_static(a);
                                assert_relative_eq!(v.[<$name>](), a, max_relative = f32::TOL);
                            }
                        } else if coe::is_same::<T, f64>() {
                            for (v, a) in itertools::izip!(samples, actual) {
                                let v: f64 = coerce_static(v);
                                let a: f64 = coerce_static(a);
                                assert_relative_eq!(v.[<$name>](), a, max_relative = f64::TOL);
                            }
                        } else {
                            panic!("Unsupported type");
                        }
                    }
                }

                pulp::Arch::new().dispatch(Impl::<f32> {
                    rng: StdRng::seed_from_u64(0),
                    nsamples,
                    _marker: PhantomData,
                });

                pulp::Arch::new().dispatch(Impl::<f64> {
                    rng: StdRng::seed_from_u64(0),
                    nsamples,
                    _marker: PhantomData,
                });
            }
        }
        };

    }

    impl_unary_simd_test!(exp, 1E-6, 1E-13);
    impl_unary_simd_test!(sqrt, 1E-6, 1E-13);
}
