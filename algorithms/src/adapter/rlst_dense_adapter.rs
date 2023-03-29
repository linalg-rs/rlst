//! Convert an RLST dense matrix type into an ```rlst-algorithms``` dense matrix type.
use super::adapter_traits::AlgorithmsAdapter;
use rlst_common::types::Scalar;
use rlst_dense::{
    DataContainer, DataContainerMut, GenericBaseMatrix, GenericBaseMatrixMut, Layout, LayoutType,
    RandomAccessByRef, RandomAccessMut, SizeIdentifier, UnsafeRandomAccessByRef,
    UnsafeRandomAccessMut,
};

use crate::adapter::dense_matrix::DenseContainer;

pub struct RlstMatrixAdapter<'a, Item, Data, L, RS, CS>
where
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    L: LayoutType,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
{
    mat: &'a mut GenericBaseMatrixMut<Item, L, Data, RS, CS>,
}

impl<'a, Item, Data, L, RS, CS> super::dense_matrix::DenseContainerInterface
    for RlstMatrixAdapter<'a, Item, Data, L, RS, CS>
where
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    L: LayoutType,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
{
    type T = Item;

    fn data(&self) -> &[Self::T] {
        self.mat.data()
    }

    fn dim(&self) -> (rlst_common::types::IndexType, rlst_common::types::IndexType) {
        self.mat.dim()
    }

    fn number_of_elements(&self) -> rlst_common::types::IndexType {
        self.mat.layout().number_of_elements()
    }

    unsafe fn get1d_unchecked(&self, elem: rlst_common::types::IndexType) -> &Self::T {
        self.mat.get1d_unchecked(elem)
    }

    fn get1d(&self, elem: rlst_common::types::IndexType) -> Option<&Self::T> {
        self.mat.get1d(elem)
    }

    fn get(
        &self,
        row: rlst_common::types::IndexType,
        col: rlst_common::types::IndexType,
    ) -> Option<&Self::T> {
        self.mat.get(row, col)
    }

    unsafe fn get_unchecked(
        &self,
        row: rlst_common::types::IndexType,
        col: rlst_common::types::IndexType,
    ) -> &Self::T {
        self.mat.get_unchecked(row, col)
    }

    fn stride(&self) -> (rlst_common::types::IndexType, rlst_common::types::IndexType) {
        self.mat.layout().stride()
    }
}

impl<Item, Data, L, RS, CS> AlgorithmsAdapter for GenericBaseMatrixMut<Item, L, Data, RS, CS>
where
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    L: LayoutType,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
{
    type T = Item;
    type OutputAdapter<'a> = RlstMatrixAdapter<'a, Item, Data, L, RS, CS> where Self: 'a;

    fn algorithms_mut<'a>(&'a mut self) -> DenseContainer<Self::OutputAdapter<'a>> {
        DenseContainer::new(RlstMatrixAdapter { mat: self })
    }
}

impl<'a, Item, Data, L, RS, CS> super::dense_matrix::DenseContainerInterfaceMut
    for RlstMatrixAdapter<'a, Item, Data, L, RS, CS>
where
    Item: Scalar,
    Data: DataContainerMut<Item = Item>,
    L: LayoutType,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
{
    unsafe fn get_unchecked_mut(
        &mut self,
        row: rlst_common::types::IndexType,
        col: rlst_common::types::IndexType,
    ) -> &mut Self::T {
        self.mat.get_unchecked_mut(row, col)
    }

    fn data_mut(&mut self) -> &mut [Self::T] {
        self.mat.data_mut()
    }

    fn get_mut(
        &mut self,
        row: rlst_common::types::IndexType,
        col: rlst_common::types::IndexType,
    ) -> Option<&mut Self::T> {
        self.mat.get_mut(row, col)
    }

    fn get1d_mut(&mut self, elem: rlst_common::types::IndexType) -> Option<&mut Self::T> {
        self.mat.get1d_mut(elem)
    }

    unsafe fn get1d_unchecked(&mut self, elem: rlst_common::types::IndexType) -> &mut Self::T {
        self.mat.get1d_unchecked_mut(elem)
    }
}

#[cfg(test)]
pub mod test {

    use super::*;
    use rlst_dense::*;

    // #[test]
    // fn test_conversion() {
    //     let mut rlst_mat = rlst_rand_mat![f64, (5, 5)];

    //     rlst_mat[[2, 2]] = 1.0;

    //     assert_eq!(1.0, *rlst_mat.algorithms().lapack().get(2, 2).unwrap());
    //     assert_eq!(rlst_mat[[2, 2]], rlst_mat.algorithms().to_rlst()[[2, 2]]);

    //     assert_eq!(1.0, *rlst_mat.algorithms().lapack().get_mut(2, 2).unwrap());
    //     //assert_eq!(1.0, rlst_mat.algorithms_mut().to_rlst_mut()[[2, 2]]);
    // }
}
