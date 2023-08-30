//! Creation of subblocks of matrices.

use super::{GenericBaseMatrix, Matrix, SliceMatrix, SliceMatrixMut};
use crate::base_matrix::BaseMatrix;
use crate::data_container::{DataContainer, DataContainerMut};
use crate::traits::*;
use crate::types::Scalar;

impl<Item: Scalar, Data: DataContainer<Item = Item>>
    Matrix<Item, BaseMatrix<Item, Data, Dynamic>, Dynamic>
{
    /// Return a new matrix that is a subblock of another matrix.
    ///
    /// The block is specified by giving the (row, column) index of the
    /// top-left element of the block and its dimension.
    pub fn block<'a>(
        &'a self,
        top_left: (usize, usize),
        dim: (usize, usize),
    ) -> SliceMatrix<'a, Item, Dynamic> {
        assert!(
            (top_left.0 + dim.0 <= self.layout().dim().0)
                & (top_left.1 + dim.1 <= self.layout().dim().1),
            "Lower right corner {:?} out of bounds for matrix with dim {:?}",
            (top_left.0 + dim.0 - 1, top_left.1 + dim.1 - 1),
            self.layout().dim()
        );
        let start_index = self.layout().convert_2d_raw(top_left.0, top_left.1);
        unsafe {
            crate::rlst_pointer_mat!('a, Item, self.get_pointer().add(start_index), dim, self.layout().stride())
        }
    }
}
impl<Item: Scalar, Data: DataContainerMut<Item = Item>>
    Matrix<Item, BaseMatrix<Item, Data, Dynamic>, Dynamic>
{
    /// Return a new matrix that is a mutable subblock of another matrix.
    ///
    /// The block is specified by giving the (row, column) index of the
    /// top-left element of the block and its dimension.
    pub fn block_mut<'a>(
        &'a mut self,
        top_left: (usize, usize),
        dim: (usize, usize),
    ) -> SliceMatrixMut<'a, Item, Dynamic> {
        assert!(
            (top_left.0 + dim.0 <= self.layout().dim().0)
                & (top_left.1 + dim.1 <= self.layout().dim().1),
            "Lower right corner {:?} out of bounds for matrix with dim {:?}",
            (top_left.0 + dim.0 - 1, top_left.1 + dim.1 - 1),
            self.layout().dim()
        );
        let start_index = self.layout().convert_2d_raw(top_left.0, top_left.1);

        unsafe {
            crate::rlst_mut_pointer_mat!('a, Item, self.get_pointer_mut().add(start_index), dim, self.layout().stride())
        }
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>> GenericBaseMatrix<Item, Data, Dynamic> {
    #[allow(clippy::type_complexity)]
    /// Split a mutable matrix into four mutable subblocks.
    ///
    /// This method splits the matrix into 4 blocks according to the
    /// following pattern.
    ///
    /// |0|1|
    /// |2|3|
    ///
    /// The first block starts at position (0, 0). The last block
    /// starts at the position given by the tuple `split_at`.
    pub fn split_in_four_mut<'a>(
        &'a mut self,
        split_at: (usize, usize),
    ) -> (
        SliceMatrixMut<'a, Item, Dynamic>,
        SliceMatrixMut<'a, Item, Dynamic>,
        SliceMatrixMut<'a, Item, Dynamic>,
        SliceMatrixMut<'a, Item, Dynamic>,
    ) {
        let dim = self.layout().dim();
        let stride = self.layout().stride();
        let ptr = self.get_pointer_mut();
        let dim0 = split_at;
        let dim1 = (split_at.0, dim.1 - split_at.1);
        let dim2 = (dim.0 - split_at.0, split_at.1);
        let dim3 = (dim.0 - split_at.0, dim.1 - split_at.1);

        let origin0 = (0, 0);
        let origin1 = (0, split_at.1);
        let origin2 = (split_at.0, 0);
        let origin3 = split_at;

        let start0 = self.layout().convert_2d_raw(origin0.0, origin0.1) as isize;
        let start1 = self.layout().convert_2d_raw(origin1.0, origin1.1) as isize;
        let start2 = self.layout().convert_2d_raw(origin2.0, origin2.1) as isize;
        let start3 = self.layout().convert_2d_raw(origin3.0, origin3.1) as isize;

        unsafe {
            (
                crate::rlst_mut_pointer_mat!['a, Item, ptr.offset(start0), dim0, stride],
                crate::rlst_mut_pointer_mat!['a, Item, ptr.offset(start1), dim1, stride],
                crate::rlst_mut_pointer_mat!['a, Item, ptr.offset(start2), dim2, stride],
                crate::rlst_mut_pointer_mat!['a, Item, ptr.offset(start3), dim3, stride],
            )
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::rlst_rand_mat;

    #[test]
    fn test_simple_slice() {
        let mut mat = crate::rlst_dynamic_mat![f64, (3, 4)];
        *mat.get_mut(1, 2).unwrap() = 1.0;

        let slice = mat.block((0, 1), (2, 2));

        assert_eq!(slice.get_value(1, 1).unwrap(), 1.0);
        assert_eq!(slice.get1d_value(3).unwrap(), 1.0);
    }

    #[test]
    fn test_double_slice() {
        let mut mat = crate::rlst_dynamic_mat![f64, (3, 4)];
        *mat.get_mut(1, 2).unwrap() = 1.0;

        let slice1 = mat.block((0, 1), (3, 3));
        let slice2 = slice1.block((1, 0), (2, 2));

        assert_eq!(slice1.get_value(1, 1).unwrap(), 1.0);
        assert_eq!(slice2.get_value(0, 1).unwrap(), 1.0);
    }

    #[test]
    fn test_disjoint_slices() {
        let mut mat = rlst_rand_mat![f64, (10, 10)];
        let (mut m1, mut m2, mut m3, mut m4) = mat.split_in_four_mut((5, 5));
        *m1.get_mut(1, 0).unwrap() = 2.0;
        *m2.get_mut(3, 4).unwrap() = 3.0;
        *m3.get_mut(2, 1).unwrap() = 4.0;
        *m4.get_mut(4, 2).unwrap() = 5.0;
        assert_eq!(mat.get_value(1, 0).unwrap(), 2.0);
        assert_eq!(mat.get_value(3, 9).unwrap(), 3.0);
        assert_eq!(mat.get_value(7, 1).unwrap(), 4.0);
        assert_eq!(mat.get_value(9, 7).unwrap(), 5.0);
    }
}
