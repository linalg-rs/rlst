//! A unary operator that takes an iterator over sparse matrix entries and applies a function to them.

use crate::{Abs, ArrayOpAbs};

/// Apply a unary operation to each entry in a sparse matrix represented as an iterator of (i, j, value) tuples.
pub struct UnaryAijOperator<Item, Iter, Out, F>
where
    Item: Copy,
    Out: Copy,
    F: Fn(Item) -> Out,
    Iter: Iterator<Item = ([usize; 2], Item)>,
{
    iter: Iter,
    func: F,
}

impl<Item, Iter, Out, F> UnaryAijOperator<Item, Iter, Out, F>
where
    Item: Copy,
    Out: Copy,
    F: Fn(Item) -> Out,
    Iter: Iterator<Item = ([usize; 2], Item)>,
{
    /// Create a new unary operator.
    pub fn new(iter: Iter, func: F) -> Self {
        Self { iter, func }
    }
}

impl<Item, Iter, Out, F> Iterator for UnaryAijOperator<Item, Iter, Out, F>
where
    Item: Copy + Default,
    Out: Copy + Default + PartialEq,
    F: Fn(Item) -> Out,
    Iter: Iterator<Item = ([usize; 2], Item)>,
{
    type Item = ([usize; 2], Out);

    fn next(&mut self) -> Option<Self::Item> {
        let candidate = self
            .iter
            .next()
            .map(|(index, value)| (index, (self.func)(value)));

        // Filter out zero values.
        if let Some((index, value)) = candidate {
            if value != Out::default() {
                Some((index, value))
            } else {
                self.next()
            }
        } else {
            None
        }
    }
}
