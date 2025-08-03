//! Container representing application of a unary operator

use crate::{
    dense::array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::{accessors::UnsafeRandom1DAccessByValue, base_operations::BaseItem},
    ContainerType,
};
use paste::paste;

/// Application of a unitary Operator
pub struct ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, const NDIM: usize>
where
    Op: Fn(OpItem) -> OpTarget,
{
    arr: Array<ArrayImpl, NDIM>,
    op: Op,
    _marker: std::marker::PhantomData<(OpItem, OpTarget)>,
}

impl<OpItem, OpTarget, ArrayImpl, Op: Fn(OpItem) -> OpTarget, const NDIM: usize>
    ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
{
    /// Create new
    pub fn new(arr: Array<ArrayImpl, NDIM>, op: Op) -> Self {
        Self {
            arr,
            op,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<OpItem, OpTarget, ArrayImpl, Op: Fn(OpItem) -> OpTarget, const NDIM: usize> ContainerType
    for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<OpItem, OpTarget, ArrayImpl, Op, const NDIM: usize> BaseItem
    for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: BaseItem<Item = OpItem>,
    OpTarget: Copy + Default,
    Op: Fn(OpItem) -> OpTarget,
{
    type Item = OpTarget;
}

impl<OpItem, OpTarget, ArrayImpl, Op, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = OpItem>,
    OpTarget: Copy + Default,
    Op: Fn(OpItem) -> OpTarget,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        (self.op)(self.arr.get_value_unchecked(multi_index))
    }
}

impl<OpItem, OpTarget, ArrayImpl, Op, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = OpItem>,
    OpTarget: Copy + Default,
    Op: Fn(OpItem) -> OpTarget,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        (self.op)(self.arr.imp().get_value_1d_unchecked(index))
    }
}

impl<OpItem, OpTarget, ArrayImpl, Op: Fn(OpItem) -> OpTarget, const NDIM: usize> Shape<NDIM>
    for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}
