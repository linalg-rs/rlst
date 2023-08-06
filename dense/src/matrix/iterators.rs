//! Various iterator implementations

use crate::{matrix::GenericBaseMatrix, DataContainerMut, RandomAccessByValue, SizeIdentifier};
use crate::{traits::*, Matrix};
use rlst_common::types::Scalar;

pub struct MatrixColumnMajorIterator<
    'a,
    Item: Scalar,
    MatImpl: MatrixImplTrait<Item, RS, CS>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
> {
    mat: &'a Matrix<Item, MatImpl, RS, CS>,
    pos: usize,
}

pub struct MatrixColumnMajorIteratorMut<
    'a,
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
> {
    mat: &'a mut GenericBaseMatrix<Item, Data, RS, CS>,
    pos: usize,
}

pub struct DiagonalIterator<
    'a,
    Item: Scalar,
    MatImpl: MatrixImplTrait<Item, RS, CS>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
> {
    mat: &'a Matrix<Item, MatImpl, RS, CS>,
    pos: usize,
}

pub struct DiagonalIteratorMut<
    'a,
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
> {
    mat: &'a mut GenericBaseMatrix<Item, Data, RS, CS>,
    pos: usize,
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixColumnMajorIterator<'a, Item, MatImpl, RS, CS>
{
    pub fn new(mat: &'a Matrix<Item, MatImpl, RS, CS>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<
        'a,
        Item: Scalar,
        Data: DataContainerMut<Item = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixColumnMajorIteratorMut<'a, Item, Data, RS, CS>
{
    pub fn new(mat: &'a mut GenericBaseMatrix<Item, Data, RS, CS>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > DiagonalIterator<'a, Item, MatImpl, RS, CS>
{
    pub fn new(mat: &'a Matrix<Item, MatImpl, RS, CS>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<
        'a,
        Item: Scalar,
        Data: DataContainerMut<Item = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > DiagonalIteratorMut<'a, Item, Data, RS, CS>
{
    pub fn new(mat: &'a mut GenericBaseMatrix<Item, Data, RS, CS>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::iter::Iterator for MatrixColumnMajorIterator<'a, Item, MatImpl, RS, CS>
{
    type Item = Item;
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self.mat.get1d_value(self.pos);
        self.pos += 1;
        elem
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::iter::Iterator for DiagonalIterator<'a, Item, MatImpl, RS, CS>
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

impl<
        'a,
        Item: Scalar,
        Data: DataContainerMut<Item = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::iter::Iterator for MatrixColumnMajorIteratorMut<'a, Item, Data, RS, CS>
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

impl<
        'a,
        Item: Scalar,
        Data: DataContainerMut<Item = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::iter::Iterator for DiagonalIteratorMut<'a, Item, Data, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > rlst_common::traits::iterators::ColumnMajorIterator for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;
    type Iter<'a> = MatrixColumnMajorIterator<'a, Item, MatImpl, RS, CS> where Self: 'a;

    fn iter_col_major(&self) -> Self::Iter<'_> {
        MatrixColumnMajorIterator::new(self)
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    rlst_common::traits::iterators::ColumnMajorIteratorMut
    for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;
    type IterMut<'a> = MatrixColumnMajorIteratorMut<'a, Item, Data, RS, CS> where Self: 'a;

    fn iter_col_major_mut(&mut self) -> Self::IterMut<'_> {
        MatrixColumnMajorIteratorMut::new(self)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > rlst_common::traits::iterators::DiagonalIterator for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;
    type Iter<'a> = DiagonalIterator<'a, Item, MatImpl, RS, CS> where Self: 'a;

    fn iter_diag(&self) -> Self::Iter<'_> {
        DiagonalIterator::new(self)
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    rlst_common::traits::iterators::DiagonalIteratorMut for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;
    type IterMut<'a> = DiagonalIteratorMut<'a, Item, Data, RS, CS> where Self: 'a;

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
