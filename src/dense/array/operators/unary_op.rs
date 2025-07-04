//! Container representing application of a unary operator

use crate::{
    dense::array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::{accessors::UnsafeRandom1DAccessByValue, array::BaseItem},
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

impl<OpItem, OpTarget, ArrayImpl, Op: Fn(OpItem) -> OpTarget, const NDIM: usize> BaseItem
    for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: BaseItem<Item = OpItem>,
{
    type Item = OpTarget;
}

impl<OpItem, OpTarget, ArrayImpl, Op: Fn(OpItem) -> OpTarget, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = OpItem>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        (self.op)(self.arr.get_value_unchecked(multi_index))
    }
}

impl<OpItem, OpTarget, ArrayImpl, Op: Fn(OpItem) -> OpTarget, const NDIM: usize>
    UnsafeRandom1DAccessByValue for ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = OpItem>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        (self.op)(self.arr.get_value_1d_unchecked(index))
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

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Create a new array by applying the unitary operator `op` to each element of `self`.
    pub fn apply_unary_op<OpItem, OpTarget, Op: Fn(OpItem) -> OpTarget>(
        self,
        op: Op,
    ) -> Array<ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>, NDIM> {
        Array::new(ArrayUnaryOperator::new(self, op))
    }
}

macro_rules! impl_unary_op_trait {
    ($name:ident, $method_name:ident) => {
        paste! {

        use crate::traits::array::[<ArrayOp $name>];
        use crate::traits::number_traits::$name;
        impl<Item: $name, ArrayImpl, const NDIM: usize> [<ArrayOp $name>] for Array<ArrayImpl, NDIM>
            where
                ArrayImpl: BaseItem<Item = Item>,
            {
                type Output = Array<ArrayUnaryOperator<Item, <Item as $name>::Output, ArrayImpl, fn(Item) -> <Item as $name>::Output, NDIM>, NDIM>;

                #[inline(always)]
                fn $method_name(self) -> Self::Output {
                    self.apply_unary_op(|x| x.$method_name())
                }
            }
        }

    };
}

impl_unary_op_trait!(Abs, abs);
impl_unary_op_trait!(Square, square);
impl_unary_op_trait!(AbsSquare, abs_square);
impl_unary_op_trait!(Sqrt, sqrt);
impl_unary_op_trait!(Exp, exp);
impl_unary_op_trait!(Ln, ln);
impl_unary_op_trait!(Recip, recip);
impl_unary_op_trait!(Sin, sin);
impl_unary_op_trait!(Cos, cos);
impl_unary_op_trait!(Tan, tan);
impl_unary_op_trait!(Asin, asin);
impl_unary_op_trait!(Acos, acos);
impl_unary_op_trait!(Atan, atan);
impl_unary_op_trait!(Sinh, sinh);
impl_unary_op_trait!(Cosh, cosh);
impl_unary_op_trait!(Tanh, tanh);
impl_unary_op_trait!(Asinh, asinh);
impl_unary_op_trait!(Acosh, acosh);
impl_unary_op_trait!(Atanh, atanh);
