//! Implementation of array addition

use std::ops::Div;

use crate::dense::{
    array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::UnsafeRandom1DAccessByValue,
};

/// Component-wise division
pub struct CmpWiseDivision<ArrayImpl1, ArrayImpl2, const NDIM: usize> {
    operator1: Array<ArrayImpl1, NDIM>,
    operator2: Array<ArrayImpl2, NDIM>,
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> CmpWiseDivision<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
{
    /// Create new
    pub fn new(operator1: Array<ArrayImpl1, NDIM>, operator2: Array<ArrayImpl2, NDIM>) -> Self {
        assert_eq!(
            operator1.shape(),
            operator2.shape(),
            "In op1 / op2 shapes not identical. op1.shape = {:#?}, op2.shape = {:#?}",
            operator1.shape(),
            operator2.shape()
        );
        Self {
            operator1,
            operator2,
        }
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for CmpWiseDivision<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1::Item: Div<ArrayImpl2::Item>,
    ArrayImpl1: UnsafeRandomAccessByValue<NDIM>,
    ArrayImpl2: UnsafeRandomAccessByValue<NDIM>,
{
    type Item = <ArrayImpl1::Item as Div<ArrayImpl2::Item>>::Output;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.operator1.get_value_unchecked(multi_index)
            / self.operator2.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> UnsafeRandom1DAccessByValue
    for CmpWiseDivision<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1::Item: Div<ArrayImpl2::Item>,
    ArrayImpl1: UnsafeRandom1DAccessByValue,
    ArrayImpl2: UnsafeRandom1DAccessByValue,
{
    type Item = <ArrayImpl1::Item as Div<ArrayImpl2::Item>>::Output;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.operator1.get_value_1d_unchecked(index) / self.operator2.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> Shape<NDIM>
    for CmpWiseDivision<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.operator1.shape()
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::Div<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
{
    type Output = Array<CmpWiseDivision<ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn div(self, rhs: Array<ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(CmpWiseDivision {
            operator1: self,
            operator2: rhs,
        })
    }
}
