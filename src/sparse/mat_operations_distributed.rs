//! Operations for sparse matrices

use std::ops::{Add, AddAssign, Div, Mul, Sub};
use std::rc::Rc;

use mpi::traits::{Communicator, Equivalence};

use crate::base_types::{c32, c64};
use crate::distributed_tools::IndexLayout;
use crate::{BaseItem, FromAijDistributed, Shape};

use super::distributed_csr_mat::DistributedCsrMatrix;
use super::{
    binary_operator::BinaryAijOperator, csr_mat::CsrMatrix, unary_aij_operator::UnaryAijOperator,
};

use super::mat_operations::ScalarMulIterator;

/// Matrix structure defined through an iterator.
pub struct DistributedSparseMatOpIterator<'a, Item, I, C>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
    C: Communicator,
{
    iter: I,
    domain_layout: Rc<IndexLayout<'a, C>>,
    range_layout: Rc<IndexLayout<'a, C>>,
}

impl<'a, Item, I, C> DistributedSparseMatOpIterator<'a, Item, I, C>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
    C: Communicator,
{
    /// Create a new sparse matrix operation.
    pub fn new(
        iter: I,
        domain_layout: Rc<IndexLayout<'a, C>>,
        range_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self {
        Self {
            iter,
            domain_layout,
            range_layout,
        }
    }

    /// Get the iterator over the sparse matrix entries.
    pub fn iter(&self) -> &I {
        &self.iter
    }

    pub fn domain_layout(&self) -> &Rc<IndexLayout<'a, C>> {
        &self.domain_layout
    }

    pub fn range_layout(&self) -> &Rc<IndexLayout<'a, C>> {
        &self.range_layout
    }

    /// Convert into a Csr matrix.
    pub fn into_distributed_csr(self) -> DistributedCsrMatrix<'a, Item, C>
    where
        Item: Copy + Default + PartialEq + AddAssign + Equivalence,
    {
        DistributedCsrMatrix::from_aij_iter(
            self.domain_layout.clone(),
            self.range_layout.clone(),
            self.iter,
        )
    }
}

impl<'a, Item, I, C> Shape<2> for DistributedSparseMatOpIterator<'a, Item, I, C>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
    C: Communicator,
{
    fn shape(&self) -> [usize; 2] {
        [
            self.range_layout().number_of_global_indices(),
            self.domain_layout().number_of_global_indices(),
        ]
    }
}

impl<'a, Item, I, C> BaseItem for DistributedSparseMatOpIterator<'a, Item, I, C>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
    C: Communicator,
{
    type Item = Item;
}

macro_rules! impl_unary_op {
    ($trait_name:ident, $method_name:ident) => {
        impl<'a, Item, I, C> crate::traits::number_traits::$trait_name
            for DistributedSparseMatOpIterator<'a, Item, I, C>
        where
            Item: Copy + Default + crate::traits::number_traits::$trait_name,
            <Item as crate::traits::number_traits::$trait_name>::Output: Copy + Default + PartialEq,
            I: Iterator<Item = ([usize; 2], Item)>,
            C: Communicator,
        {
            type Output = DistributedSparseMatOpIterator<
                'a,
                <Item as crate::traits::number_traits::$trait_name>::Output,
                UnaryAijOperator<
                    Item,
                    I,
                    <Item as crate::traits::number_traits::$trait_name>::Output,
                    fn(Item) -> <Item as crate::traits::number_traits::$trait_name>::Output,
                >,
                C,
            >;

            fn $method_name(self) -> Self::Output {
                DistributedSparseMatOpIterator::new(
                    UnaryAijOperator::new(self.iter, |val: Item| val.$method_name()),
                    self.domain_layout.clone(),
                    self.range_layout.clone(),
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

impl<'a, Item1, Item2, Out, I1, I2, C>
    std::ops::Add<DistributedSparseMatOpIterator<'a, Item2, I2, C>>
    for DistributedSparseMatOpIterator<'a, Item1, I1, C>
where
    Item1: Add<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
    C: Communicator,
{
    type Output = DistributedSparseMatOpIterator<
        'a,
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
        C,
    >;

    fn add(self, other: DistributedSparseMatOpIterator<'a, Item2, I2, C>) -> Self::Output {
        assert!(self.domain_layout().is_same(other.domain_layout()));
        assert!(self.range_layout().is_same(other.range_layout()));
        DistributedSparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a + b),
            self.domain_layout.clone(),
            self.range_layout.clone(),
        )
    }
}

impl<'a, Item1, Item2, Out, I1, I2, C>
    std::ops::Sub<DistributedSparseMatOpIterator<'a, Item2, I2, C>>
    for DistributedSparseMatOpIterator<'a, Item1, I1, C>
where
    Item1: Sub<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
    C: Communicator,
{
    type Output = DistributedSparseMatOpIterator<
        'a,
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
        C,
    >;

    fn sub(self, other: DistributedSparseMatOpIterator<'a, Item2, I2, C>) -> Self::Output {
        assert!(self.domain_layout().is_same(other.domain_layout()));
        assert!(self.range_layout().is_same(other.range_layout()));
        DistributedSparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a - b),
            self.domain_layout.clone(),
            self.range_layout.clone(),
        )
    }
}

macro_rules! impl_scalar_mult {
    ($scalar_type:ty) => {
        impl<'a, I, C> std::ops::Mul<DistributedSparseMatOpIterator<'a, $scalar_type, I, C>>
            for $scalar_type
        where
            I: Iterator<Item = ([usize; 2], $scalar_type)>,
            C: Communicator,
        {
            type Output = DistributedSparseMatOpIterator<
                'a,
                $scalar_type,
                ScalarMulIterator<$scalar_type, I>,
                C,
            >;

            fn mul(
                self,
                other: DistributedSparseMatOpIterator<'a, $scalar_type, I, C>,
            ) -> Self::Output {
                let new_iter = ScalarMulIterator::new(other.iter, self);
                DistributedSparseMatOpIterator::new(
                    new_iter,
                    other.domain_layout.clone(),
                    other.range_layout.clone(),
                )
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

impl<'a, Item, I, C> std::ops::Mul<Item> for DistributedSparseMatOpIterator<'a, Item, I, C>
where
    Item: Copy + Default + Mul<Item, Output = Item>,
    I: Iterator<Item = ([usize; 2], Item)>,
    C: Communicator,
{
    type Output = DistributedSparseMatOpIterator<'a, Item, ScalarMulIterator<Item, I>, C>;

    fn mul(self, scalar: Item) -> Self::Output {
        let new_iter = ScalarMulIterator::new(self.iter, scalar);
        DistributedSparseMatOpIterator::new(
            new_iter,
            self.domain_layout.clone(),
            self.range_layout.clone(),
        )
    }
}

impl<'a, Item1, Item2, Out, I1, I2, C>
    std::ops::Mul<DistributedSparseMatOpIterator<'a, Item2, I2, C>>
    for DistributedSparseMatOpIterator<'a, Item1, I1, C>
where
    Item1: Mul<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
    C: Communicator,
{
    type Output = DistributedSparseMatOpIterator<
        'a,
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
        C,
    >;

    fn mul(self, other: DistributedSparseMatOpIterator<'a, Item2, I2, C>) -> Self::Output {
        assert!(self.domain_layout().is_same(other.domain_layout()));
        assert!(self.range_layout().is_same(other.range_layout()));
        DistributedSparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a * b),
            self.domain_layout.clone(),
            self.range_layout.clone(),
        )
    }
}

impl<'a, Item1, Item2, Out, I1, I2, C>
    std::ops::Div<DistributedSparseMatOpIterator<'a, Item2, I2, C>>
    for DistributedSparseMatOpIterator<'a, Item1, I1, C>
where
    Item1: Div<Item2, Output = Out> + Copy + Default,
    Item2: Default + Copy,
    Out: PartialEq + Default + Copy,
    I1: Iterator<Item = ([usize; 2], Item1)>,
    I2: Iterator<Item = ([usize; 2], Item2)>,
    C: Communicator,
{
    type Output = DistributedSparseMatOpIterator<
        'a,
        Out,
        BinaryAijOperator<Item1, Item2, I1, I2, Out, fn(Item1, Item2) -> Out>,
        C,
    >;

    fn div(self, other: DistributedSparseMatOpIterator<'a, Item2, I2, C>) -> Self::Output {
        assert!(self.domain_layout().is_same(other.domain_layout()));
        assert!(self.range_layout().is_same(other.range_layout()));
        DistributedSparseMatOpIterator::new(
            BinaryAijOperator::new(self.iter, other.iter, |a, b| a / b),
            self.domain_layout.clone(),
            self.range_layout.clone(),
        )
    }
}
