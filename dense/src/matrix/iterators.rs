//! Various iterator implementations

use crate::{traits::*, Matrix};
use crate::{RandomAccessByValue, SizeIdentifier};
use rlst_common::types::Scalar;

pub struct MatrixColumnMajorIterator<
    'a,
    Item: Scalar,
    MatImpl: MatrixImplTrait<Item, S>,
    S: SizeIdentifier,
> {
    mat: &'a Matrix<Item, MatImpl, S>,
    pos: usize,
}

pub struct MatrixColumnMajorIteratorMut<
    'a,
    Item: Scalar,
    MatImpl: MatrixImplTraitMut<Item, S>,
    S: SizeIdentifier,
> {
    mat: &'a mut Matrix<Item, MatImpl, S>,
    pos: usize,
}

pub struct DiagonalIterator<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
{
    mat: &'a Matrix<Item, MatImpl, S>,
    pos: usize,
}

pub struct DiagonalIteratorMut<
    'a,
    Item: Scalar,
    MatImpl: MatrixImplTraitMut<Item, S>,
    S: SizeIdentifier,
> {
    mat: &'a mut Matrix<Item, MatImpl, S>,
    pos: usize,
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    MatrixColumnMajorIterator<'a, Item, MatImpl, S>
{
    pub fn new(mat: &'a Matrix<Item, MatImpl, S>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    MatrixColumnMajorIteratorMut<'a, Item, MatImpl, S>
{
    pub fn new(mat: &'a mut Matrix<Item, MatImpl, S>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    DiagonalIterator<'a, Item, MatImpl, S>
{
    pub fn new(mat: &'a Matrix<Item, MatImpl, S>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    DiagonalIteratorMut<'a, Item, MatImpl, S>
{
    pub fn new(mat: &'a mut Matrix<Item, MatImpl, S>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> std::iter::Iterator
    for MatrixColumnMajorIterator<'a, Item, MatImpl, S>
{
    type Item = Item;
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.mat.get1d_value(self.pos);
        self.pos += 1;
        elem
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> std::iter::Iterator
    for DiagonalIterator<'a, Item, MatImpl, S>
{
    type Item = Item;
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.mat.get_value(self.pos, self.pos);
        self.pos += 1;
        elem
    }
}

// In the following have to use transmute to manually change the lifetime of the data
// obtained by `get_mut` to the lifetime 'a of the matrix. The borrow checker cannot see
// that the reference obtained by get_mut is bound to the lifetime of the iterator due
// to the mutable reference in its initialization.
// See also: https://stackoverflow.com/questions/62361624/lifetime-parameter-problem-in-custom-iterator-over-mutable-references
// And also: https://users.rust-lang.org/t/when-is-transmuting-lifetimes-useful/56140

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> std::iter::Iterator
    for MatrixColumnMajorIteratorMut<'a, Item, MatImpl, S>
{
    type Item = &'a mut Item;
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.mat.get1d_mut(self.pos);
        self.pos += 1;
        match elem {
            None => None,
            Some(inner) => Some(unsafe { std::mem::transmute::<&mut Item, &'a mut Item>(inner) }),
        }
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> std::iter::Iterator
    for DiagonalIteratorMut<'a, Item, MatImpl, S>
{
    type Item = &'a mut Item;
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.mat.get_mut(self.pos, self.pos);
        self.pos += 1;
        match elem {
            None => None,
            Some(inner) => Some(unsafe { std::mem::transmute::<&mut Item, &'a mut Item>(inner) }),
        }
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    rlst_common::traits::iterators::ColumnMajorIterator for Matrix<Item, MatImpl, S>
{
    type T = Item;
    type Iter<'a> = MatrixColumnMajorIterator<'a, Item, MatImpl, S> where Self: 'a;

    fn iter_col_major(&self) -> Self::Iter<'_> {
        MatrixColumnMajorIterator::new(self)
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    rlst_common::traits::iterators::ColumnMajorIteratorMut for Matrix<Item, MatImpl, S>
{
    type T = Item;
    type IterMut<'a> = MatrixColumnMajorIteratorMut<'a, Item, MatImpl, S> where Self: 'a;

    fn iter_col_major_mut(&mut self) -> Self::IterMut<'_> {
        MatrixColumnMajorIteratorMut::new(self)
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    rlst_common::traits::iterators::DiagonalIterator for Matrix<Item, MatImpl, S>
{
    type T = Item;
    type Iter<'a> = DiagonalIterator<'a, Item, MatImpl, S> where Self: 'a;

    fn iter_diag(&self) -> Self::Iter<'_> {
        DiagonalIterator::new(self)
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    rlst_common::traits::iterators::DiagonalIteratorMut for Matrix<Item, MatImpl, S>
{
    type T = Item;
    type IterMut<'a> = DiagonalIteratorMut<'a, Item, MatImpl, S> where Self: 'a;

    fn iter_diag_mut(&mut self) -> Self::IterMut<'_> {
        DiagonalIteratorMut::new(self)
    }
}

#[cfg(test)]
mod test {

    use rlst_common::traits::*;

    #[test]
    fn test_col_major_mut() {
        let mut mat = crate::rlst_mat![f64, (2, 2)];

        for (index, item) in mat.iter_col_major_mut().enumerate() {
            *item = index as f64;
        }

        assert_eq!(mat[[0, 0]], 0.0);
        assert_eq!(mat[[1, 0]], 1.0);
        assert_eq!(mat[[0, 1]], 2.0);
        assert_eq!(mat[[1, 1]], 3.0);
    }
}
