//! Various iterator implementations

use crate::{matrix::GenericBaseMatrix, DataContainer, RandomAccessByValue, SizeIdentifier};
use rlst_common::types::Scalar;

pub struct MatrixColumnMajorIterator<
    'a,
    Item: Scalar,
    Data: DataContainer<Item = Item>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
> {
    mat: &'a GenericBaseMatrix<Item, Data, RS, CS>,
    pos: usize,
}

impl<
        'a,
        Item: Scalar,
        Data: DataContainer<Item = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixColumnMajorIterator<'a, Item, Data, RS, CS>
{
    pub fn new(mat: &'a GenericBaseMatrix<Item, Data, RS, CS>) -> Self {
        Self { mat, pos: 0 }
    }
}

impl<
        'a,
        Item: Scalar,
        Data: DataContainer<Item = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::iter::Iterator for MatrixColumnMajorIterator<'a, Item, Data, RS, CS>
{
    type Item = Item;
    fn next(&mut self) -> Option<Self::Item> {
        let elem = self
            .mat
            .get_value(self.pos % self.mat.dim().0, self.pos / self.mat.dim().0);
        self.pos += 1;
        elem
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    rlst_common::basic_traits::ColumnMajorIterator for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;
    type Iter<'a> = MatrixColumnMajorIterator<'a, Item, Data, RS, CS> where Self: 'a;

    fn iter_col_major<'a>(&'a self) -> Self::Iter<'a> {
        MatrixColumnMajorIterator::new(self)
    }
}
