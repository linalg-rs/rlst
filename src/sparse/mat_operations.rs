//! Operations for sparse matrices

use std::ops::{Add, AddAssign, Div, Mul, Sub};

use crate::base_types::{c32, c64};
use crate::{BaseItem, FromAij, Shape};

use super::{
    binary_operator::BinaryAijOperator, csr_mat::CsrMatrix, unary_aij_operator::UnaryAijOperator,
};

/// Matrix structure defined through an iterator.
pub struct SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    iter: I,
    shape: [usize; 2],
}

/// Iterator that multiplies each entry  by a scalar.
/// This is necessary to avoid issues with variable capturing in the closure for the iterator.
/// Basically, since we use a closure of type  `fn(Item) -> Out`, we cannot capture the scalar directly
/// as this is not allowed for `fn` style closures.
pub struct ScalarMulIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    iter: I,
    scalar: Item,
}

impl<Item, I> ScalarMulIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    /// Create a new scalar multiplication iterator.
    pub fn new(iter: I, scalar: Item) -> Self {
        Self { iter, scalar }
    }
}

impl<Item, I> Iterator for ScalarMulIterator<Item, I>
where
    Item: Copy + Default + Mul<Item, Output = Item>,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Item = ([usize; 2], Item);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|(index, value)| (index, value * self.scalar))
    }
}

impl<Item, I> SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    /// Create a new sparse matrix operation.
    pub fn new(iter: I, shape: [usize; 2]) -> Self {
        Self { iter, shape }
    }

    /// Get the iterator over the sparse matrix entries.
    pub fn iter(&self) -> &I {
        &self.iter
    }

    /// Convert into a Csr matrix.
    pub fn into_csr(self) -> CsrMatrix<Item>
    where
        Item: Copy + Default + PartialEq + AddAssign,
    {
        let shape = self.shape();

        CsrMatrix::from_aij_iter(shape, self.iter)
    }
}

impl<Item, I> Shape<2> for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl<Item, I> BaseItem for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Item = Item;
}

macro_rules! impl_unary_op {
    ($trait_name:ident, $method_name:ident) => {
        impl<Item, I> crate::traits::number_traits::$trait_name for SparseMatOpIterator<Item, I>
        where
            Item: Copy + Default + crate::traits::number_traits::$trait_name,
            <Item as crate::traits::number_traits::$trait_name>::Output: Copy + Default + PartialEq,
            I: Iterator<Item = ([usize; 2], Item)>,
        {
            type Output = SparseMatOpIterator<
                <Item as crate::traits::number_traits::$trait_name>::Output,
                UnaryAijOperator<
                    Item,
                    I,
                    <Item as crate::traits::number_traits::$trait_name>::Output,
                    fn(Item) -> <Item as crate::traits::number_traits::$trait_name>::Output,
                >,
            >;

            fn $method_name(self) -> Self::Output {
                let shape = self.shape();
                SparseMatOpIterator::new(
                    UnaryAijOperator::new(self.iter, |val: Item| val.$method_name()),
                    shape,
                )
            }
        }
    };
}

impl_unary_op!(Abs, abs);
impl_unary_op!(Square, square);
impl_unary_op!(AbsSquare, abs_square);
impl_unary_op!(Sqrt, sqrt);
impl_unary_op!(Exp, exp);
impl_unary_op!(Ln, ln);
impl_unary_op!(Recip, recip);
impl_unary_op!(Sin, sin);
impl_unary_op!(Cos, cos);
impl_unary_op!(Tan, tan);
impl_unary_op!(Asin, asin);
impl_unary_op!(Acos, acos);
impl_unary_op!(Atan, atan);
impl_unary_op!(Sinh, sinh);
impl_unary_op!(Cosh, cosh);
impl_unary_op!(Tanh, tanh);
impl_unary_op!(Asinh, asinh);
impl_unary_op!(Acosh, acosh);
impl_unary_op!(Atanh, atanh);

impl<Item1, Item2, Out, I1, I2> std::ops::Add<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Add<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn add(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a + b),
            shape,
        )
    }
}

impl<Item1, Item2, Out, I1, I2> std::ops::Sub<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Sub<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn sub(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a - b),
            shape,
        )
    }
}

macro_rules! impl_scalar_mult {
    ($scalar_type:ty) => {
        impl<I> std::ops::Mul<SparseMatOpIterator<$scalar_type, I>> for $scalar_type
        where
            I: Iterator<Item = ([usize; 2], $scalar_type)>,
        {
            type Output = SparseMatOpIterator<$scalar_type, ScalarMulIterator<$scalar_type, I>>;

            fn mul(self, other: SparseMatOpIterator<$scalar_type, I>) -> Self::Output {
                let shape = other.shape();
                let new_iter = ScalarMulIterator::new(other.iter, self);
                SparseMatOpIterator::new(new_iter, shape)
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);

impl<Item, I> std::ops::Mul<Item> for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default + Mul<Item, Output = Item>,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Output = SparseMatOpIterator<Item, ScalarMulIterator<Item, I>>;

    fn mul(self, scalar: Item) -> Self::Output {
        let shape = self.shape();
        let new_iter = ScalarMulIterator::new(self.iter, scalar);
        SparseMatOpIterator::new(new_iter, shape)
    }
}

impl<Item1, Item2, Out, I1, I2> std::ops::Mul<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Mul<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn mul(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a * b),
            shape,
        )
    }
}

impl<Item1, Item2, Out, I1, I2> std::ops::Div<SparseMatOpIterator<Item2, I2>>
    for SparseMatOpIterator<Item1, I1>
where
    Item1: Div<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
{
    type Output = SparseMatOpIterator<
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
    >;

    fn div(self, other: SparseMatOpIterator<Item2, I2>) -> Self::Output {
        let shape = self.shape();
        SparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a / b),
            shape,
        )
    }
}
