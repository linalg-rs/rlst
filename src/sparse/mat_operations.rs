//! Operations for sparse matrices

use std::ops::{Add, AddAssign, Mul, Sub};

use num::Zero;

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

// /// Iterator that multiplies each entry  by a scalar.
// /// This is
// pub struct ScalarMulIterator<Item, I>
// where
//     Item: Copy + Default,
//     I: Iterator<Item = ([usize; 2], Item)>,
// {
//     iter: I,
//     scalar: Item,
// }

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
        Item: Copy + Default + PartialEq + AddAssign + Zero,
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

impl<Item, Out, I> std::ops::Mul<SparseMatOpIterator<Item, I>> for f64
where
    f64: Mul<Item, Output = Out>,
    Item: Copy + Default,
    Out: PartialEq + Default + Copy,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Output = SparseMatOpIterator<Out, UnaryAijOperator<Item, I, Out, fn(Item) -> Out>>;

    fn mul(self, other: SparseMatOpIterator<Item, I>) -> Self::Output {
        let shape = other.shape();
        SparseMatOpIterator::new(UnaryAijOperator::new(other.iter, |val| self * val), shape)
    }
}
