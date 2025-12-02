//! Definition of a complex 2 complex inplace FFT plan

use std::{ffi::c_void, marker::PhantomData, os::raw::c_int};

use fftw_sys::{FFTW_BACKWARD, FFTW_FORWARD};
use itertools::{Itertools, izip};

use crate::{
    Array, BaseItem, RawAccess, RawAccessMut, Shape, Stride, UnsafeRandom1DAccessByRef,
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut, c64,
    dense::fftw::{FftPlan, FftPlanFlags, FftPlanPtr, FftPrecision},
};

/// A complex 2 complex inplace plan
pub struct C2CInplace<Item, ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    plan_forward: FftPlanPtr,
    plan_backward: FftPlanPtr,
    _marker: std::marker::PhantomData<Item>,
}

impl<ArrayImpl, const NDIM: usize> C2CInplace<c64, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM> + Stride<NDIM> + RawAccessMut<Item = c64>,
{
    /// Create a new plan wrapped inside an array structure. Returns None if the Plan cannot be created.
    pub fn from_arr(
        mut arr: Array<ArrayImpl, NDIM>,
        flags: FftPlanFlags,
    ) -> Option<Array<C2CInplace<c64, ArrayImpl, NDIM>, NDIM>> {
        // Return None if NDIM > 3 as FFTW only supports up to three dimensions
        if NDIM > 3 {
            return None;
        }
        // Return None if the array is not contiguous or is empty
        if !arr.is_contiguous() || arr.is_empty() {
            return None;
        }

        let mut shape = arr
            .shape()
            .iter()
            .map(|&elem| elem as i32)
            .collect_array()
            .unwrap();

        // If the array is column-major reverse the order of the shape as
        // FFTW uses row-major order

        if matches!(arr.memory_layout(), crate::MemoryLayout::ColumnMajor) {
            shape = {
                let mut rev_shape = [0_i32; NDIM];

                for (&orig_index, rev_index) in izip!(shape.iter().rev(), rev_shape.iter_mut()) {
                    *rev_index = orig_index;
                }
                rev_shape
            };
        }

        let data = arr.data_mut()?.as_mut_ptr();

        let plan_ptr_forward = unsafe {
            fftw_sys::fftw_plan_dft(
                NDIM as c_int,
                shape.as_ptr(),
                data,
                data,
                FFTW_FORWARD,
                flags as u32,
            )
        };

        let plan_ptr_backward = unsafe {
            fftw_sys::fftw_plan_dft(
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
                arr,
                plan_forward: FftPlanPtr {
                    precision: FftPrecision::Double,
                    ptr: plan_ptr_forward as *mut c_void,
                },
                plan_backward: FftPlanPtr {
                    precision: FftPrecision::Double,
                    ptr: plan_ptr_backward as *mut c_void,
                },
                _marker: PhantomData,
            }))
        }
    }
}

impl<ArrayImpl, const NDIM: usize> FftPlan<NDIM> for C2CInplace<c64, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = c64>,
{
    type ItemIn = c64;

    type ItemOut = c64;

    fn execute_forward(&mut self) {
        unsafe {
            fftw_sys::fftw_execute(self.plan_forward.ptr as fftw_sys::fftw_plan);
        }
    }

    fn execute_backwawrd(&mut self) {
        unsafe {
            fftw_sys::fftw_execute(self.plan_backward.ptr as fftw_sys::fftw_plan);
        }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> BaseItem for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: Default + Copy,
    ArrayImpl: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<Item, ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: Default + Copy,
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
    Item: Default + Copy,
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
    Item: Default + Copy,
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
    Item: Default + Copy,
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
    Item: Default + Copy,
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
    Item: Default + Copy,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.arr.imp_mut().get_1d_unchecked_mut(index) }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> RawAccess for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: Default + Copy,
    ArrayImpl: RawAccess<Item = Item>,
{
    #[inline(always)]
    fn data(&self) -> Option<&[Self::Item]> {
        self.arr.data()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> RawAccessMut for C2CInplace<Item, ArrayImpl, NDIM>
where
    Item: Default + Copy,
    ArrayImpl: RawAccessMut<Item = Item>,
{
    #[inline(always)]
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        self.arr.data_mut()
    }
}
