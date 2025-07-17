//! Operations for sparse matrices

use std::ops::AddAssign;

use num::Zero;

use crate::{Abs, ArrayOpAbs, BaseItem, FromAij, Shape};

use super::{csr_mat::CsrMatrix, unary_aij_operator::UnaryAijOperator};

/// Matrix structure defined through an iterator.
pub struct SparseMatOpIterator<Item, I>
where
    Item: Copy + Default,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    iter: I,
    shape: [usize; 2],
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

impl<Item, I> ArrayOpAbs for SparseMatOpIterator<Item, I>
where
    Item: Copy + Default + Abs,
    <Item as Abs>::Output: Copy + Default + PartialEq,
    I: Iterator<Item = ([usize; 2], Item)>,
{
    type Output = SparseMatOpIterator<
        <Item as Abs>::Output,
        UnaryAijOperator<Item, I, <Item as Abs>::Output, fn(Item) -> <Item as Abs>::Output>,
    >;

    fn abs(self) -> Self::Output {
        let shape = self.shape();
        SparseMatOpIterator::new(
            UnaryAijOperator::new(self.iter, |val: Item| val.abs()),
            shape,
        )
    }
}
