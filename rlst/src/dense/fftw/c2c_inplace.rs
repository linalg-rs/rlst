//! Definition of a complex 2 complex inplace FFT plan

use std::{ffi::c_void, marker::PhantomData, os::raw::c_int};

use fftw_sys::{FFTW_BACKWARD, FFTW_FORWARD};
use itertools::{Itertools, izip};

use crate::{
    Array, BaseItem, RawAccess, RawAccessMut, RlstScalar, Shape, Stride, UnsafeRandom1DAccessByRef,
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut, c32, c64,
    dense::fftw::{
        FFTW_PLAN_INTERFACE, FftPlan, FftwPlanFlags, FftwPlanPtr, FftwPlanPtrType,
        FftwPlanPtrTypeTrait, PlanInterfaceTrait,
    },
};

/// Trait for types and dimensions that implement the C2C FFT
#[allow(private_bounds)]
pub trait C2CInplaceFft<Item, const NDIM: usize>: Sealed
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
{
    /// The implementation type of the Array
    type ArrayImpl: Shape<NDIM> + Stride<NDIM> + RawAccessMut<Item = Item>;

    /// Return a C2C FFT Instance
    fn into_c2c_fft(
        self,
        flags: FftwPlanFlags,
    ) -> Option<Array<C2CInplace<Item, Self::ArrayImpl, NDIM>, NDIM>>;
}

/// A complex 2 complex inplace plan
#[allow(private_bounds)]
pub struct C2CInplace<Item, ArrayImpl, const NDIM: usize>
where
    Item: RlstScalar + 'static,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
{
    arr: Array<ArrayImpl, NDIM>,
    plan_forward: FftwPlanPtr<Item::Real>,
    plan_backward: FftwPlanPtr<Item::Real>,
    _marker: std::marker::PhantomData<Item>,
}

macro_rules! impl_c2c_inplace {
    ($ty: ty, $precision: ident, $execute: ident) => {
        impl<ArrayImpl, const NDIM: usize> C2CInplaceFft<$ty, NDIM> for Array<ArrayImpl, NDIM>
        where
            ArrayImpl: Shape<NDIM> + Stride<NDIM> + RawAccessMut<Item = $ty>,
        {
            type ArrayImpl = ArrayImpl;
            /// Create a new plan wrapped inside an array structure. Returns None if the Plan cannot be created.
            fn into_c2c_fft(
                mut self,
                flags: FftwPlanFlags,
            ) -> Option<Array<C2CInplace<$ty, Self::ArrayImpl, NDIM>, NDIM>> {
                // Return None if NDIM > 3 as FFTW only supports up to three dimensions
                if NDIM > 3 {
                    return None;
                }
                // Return None if the array is not contiguous or is empty
                if !self.is_contiguous() || self.is_empty() {
                    return None;
                }

                let mut shape = self
                    .shape()
                    .iter()
                    .map(|&elem| elem as i32)
                    .collect_array()
                    .unwrap();

                // If the array is column-major reverse the order of the shape as
                // FFTW uses row-major order

                if matches!(self.memory_layout(), crate::MemoryLayout::ColumnMajor) {
                    shape = {
                        let mut rev_shape = [0_i32; NDIM];

                        for (&orig_index, rev_index) in
                            izip!(shape.iter().rev(), rev_shape.iter_mut())
                        {
                            *rev_index = orig_index;
                        }
                        rev_shape
                    };
                }

                let data = self.data_mut()?.as_mut_ptr();

                let plan_ptr_forward = unsafe {
                    FFTW_PLAN_INTERFACE.lock().unwrap().$precision.create_c2c(
                        NDIM as c_int,
                        shape.as_ptr(),
                        data,
                        data,
                        FFTW_FORWARD,
                        flags as u32,
                    )
                };

                let plan_ptr_backward = unsafe {
                    FFTW_PLAN_INTERFACE.lock().unwrap().$precision.create_c2c(
                        NDIM as c_int,
                        shape.as_ptr(),
                        data,
                        data,
                        FFTW_BACKWARD as i32,
                        flags as u32,
                    )
                };

                if plan_ptr_forward.is_null() || plan_ptr_backward.is_null() {
                    None
                } else {
                    Some(Array::new(C2CInplace {
                        arr: self,
                        plan_forward: FftwPlanPtr {
                            ptr: plan_ptr_forward as *mut c_void,
                            _marker: PhantomData,
                        },
                        plan_backward: FftwPlanPtr {
                            ptr: plan_ptr_backward as *mut c_void,
                            _marker: PhantomData,
                        },
                        _marker: PhantomData,
                    }))
                }
            }
        }

        impl<ArrayImpl, const NDIM: usize> FftPlan<NDIM> for C2CInplace<$ty, ArrayImpl, NDIM>
        where
            ArrayImpl: BaseItem<Item = $ty>,
        {
            type Item = $ty;

            type ArrayImpl = ArrayImpl;

            fn execute_forward(&mut self) {
                unsafe {
                    fftw_sys::$execute(
                        <FftwPlanPtrType<<$ty as RlstScalar>::Real> as FftwPlanPtrTypeTrait>::get_fftw_ptr(
                            self.plan_forward.ptr,
                        ),
                    );
                }
            }

            fn execute_backwawrd(&mut self) {
                unsafe {
                    fftw_sys::$execute(
                        <FftwPlanPtrType<<$ty as RlstScalar>::Real> as FftwPlanPtrTypeTrait>::get_fftw_ptr(
                            self.plan_backward.ptr,
                        ),
                    );
                }
            }

            fn into_imp(self) -> Array<Self::ArrayImpl, NDIM> {
                self.arr
            }
        }
    };
}

impl_c2c_inplace!(c64, fftw_double, fftw_execute);
impl_c2c_inplace!(c32, fftw_single, fftwf_execute);

impl<Item, ArrayImpl, const NDIM: usize> BaseItem for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<Item, ArrayImpl, const NDIM: usize> Shape<NDIM> for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: Shape<NDIM>,
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Stride<NDIM> for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: Stride<NDIM>,
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe { self.arr.get_value_unchecked(multi_index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: UnsafeRandomAccessByRef<NDIM, Item = Item>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        unsafe { self.arr.get_unchecked(multi_index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: UnsafeRandomAccessMut<NDIM, Item = Item>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        unsafe { self.arr.get_unchecked_mut(multi_index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.arr.imp().get_value_1d_unchecked(index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByRef
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: UnsafeRandom1DAccessByRef<Item = Item>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.arr.imp().get_1d_unchecked(index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessMut
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.arr.imp_mut().get_1d_unchecked_mut(index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> RawAccess for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: RawAccess<Item = Item>,
{
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.arr.data()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> RawAccessMut for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: RlstScalar,
    FftwPlanPtrType<Item::Real>: FftwPlanPtrTypeTrait,
    ArrayImpl: RawAccessMut<Item = Item>,
{
    #[inline(always)]
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.arr.data_mut()
    }
}

trait Sealed {}

impl<ArrayImpl, const NDIM: usize> Sealed for Array<ArrayImpl, NDIM> {}

#[cfg(test)]
mod test {

    use num::FromPrimitive;

    use crate::{EvaluateObject, c32, c64};

    use crate::DynArray;
    use crate::dense::fftw::FftwPlanFlags;

    use super::C2CInplaceFft;

    #[test]
    fn test_c2c_1d_f64() {
        let n = 10;
        let mut arr = DynArray::<c64, _>::from_shape([10]);
        arr.fill_from_seed_equally_distributed(0);

        let expected = arr.eval();

        let sum = arr
            .eval()
            .r_mut()
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .iter_value()
            .sum::<c64>();

        let actual = arr
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .ifft()
            .fft_into_imp();

        // Check that the zeroth frequency is the sum of all input elements.
        approx::assert_relative_eq!(
            sum,
            c64::from_usize(n).unwrap() * expected[[0]],
            epsilon = 1E-10
        );
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(actual, c64::from_usize(n).unwrap() * expected.r(), 1E-10);
    }

    #[test]
    fn test_c2c_1d_f32() {
        let n = 10;
        let mut arr = DynArray::<c32, _>::from_shape([n]);
        arr.fill_from_seed_equally_distributed(0);
        let expected = arr.eval();

        let sum = arr
            .eval()
            .r_mut()
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .iter_value()
            .sum::<c32>();

        let actual = arr
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .ifft()
            .fft_into_imp();

        // Check that the zeroth frequency is the sum of all input elements.
        approx::assert_relative_eq!(
            sum,
            c32::from_usize(n).unwrap() * expected[[0]],
            epsilon = 1E-6
        );
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(actual, c32::from_usize(n).unwrap() * expected.r(), 1E-6);
    }

    #[test]
    fn test_c2c_2d_f64() {
        let shape = [5, 2];
        let n = shape.iter().product();
        let mut arr = DynArray::<c64, _>::from_shape(shape);
        arr.fill_from_seed_equally_distributed(0);

        let expected = arr.eval();

        let sum = arr
            .eval()
            .r_mut()
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .iter_value()
            .sum::<c64>();

        let actual = arr
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .ifft()
            .fft_into_imp();

        // Check that the zeroth frequency is the sum of all input elements.
        approx::assert_relative_eq!(
            sum,
            c64::from_usize(n).unwrap() * expected[[0, 0]],
            epsilon = 1E-10
        );
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(actual, c64::from_usize(n).unwrap() * expected.r(), 1E-10);
    }

    #[test]
    fn test_c2c_2d_f32() {
        let shape = [5, 2];
        let n = shape.iter().product();
        let mut arr = DynArray::<c32, _>::from_shape(shape);
        arr.fill_from_seed_equally_distributed(0);

        let expected = arr.eval();

        let sum = arr
            .eval()
            .r_mut()
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .iter_value()
            .sum::<c32>();

        let actual = arr
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .ifft()
            .fft_into_imp();

        // Check that the zeroth frequency is the sum of all input elements.
        approx::assert_relative_eq!(
            sum,
            c32::from_usize(n).unwrap() * expected[[0, 0]],
            epsilon = 1E-6
        );
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(actual, c32::from_usize(n).unwrap() * expected.r(), 1E-6);
    }

    #[test]
    fn test_c2c_3d_f64() {
        let shape = [5, 2, 4];
        let n = shape.iter().product();
        let mut arr = DynArray::<c64, _>::from_shape(shape);
        arr.fill_from_seed_equally_distributed(0);

        let expected = arr.eval();

        let sum = arr
            .eval()
            .r_mut()
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .iter_value()
            .sum::<c64>();

        let actual = arr
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .ifft()
            .fft_into_imp();

        // Check that the zeroth frequency is the sum of all input elements.
        approx::assert_relative_eq!(
            sum,
            c64::from_usize(n).unwrap() * expected[[0, 0, 0]],
            epsilon = 1E-10
        );
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(actual, c64::from_usize(n).unwrap() * expected.r(), 1E-10);
    }

    #[test]
    fn test_c2c_3d_f32() {
        let shape = [5, 2, 4];
        let n = shape.iter().product();
        let mut arr = DynArray::<c32, _>::from_shape(shape);
        arr.fill_from_seed_equally_distributed(0);

        let expected = arr.eval();

        let sum = arr
            .eval()
            .r_mut()
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .iter_value()
            .sum::<c32>();

        let actual = arr
            .into_c2c_fft(FftwPlanFlags::Estimate)
            .unwrap()
            .fft()
            .ifft()
            .fft_into_imp();

        // Check that the zeroth frequency is the sum of all input elements.
        approx::assert_relative_eq!(
            sum,
            c32::from_usize(n).unwrap() * expected[[0, 0, 0]],
            epsilon = 1E-5
        );
        // Compute the forward and inverse FFT and check that the result is still the same.
        crate::assert_array_relative_eq!(actual, c32::from_usize(n).unwrap() * expected.r(), 1E-5);
    }
}
