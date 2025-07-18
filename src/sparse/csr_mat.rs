//! Definition of CSR matrices.

use std::ops::{Add, AddAssign, Mul};

use crate::sparse::tools::normalize_aij;
use crate::traits::ArrayIteratorByValue;
use crate::{dense::array::DynArray, sparse::SparseMatType, AijIteratorByValue, BaseItem, Shape};
use crate::{
    AijIteratorMut, Array, ArrayIteratorMut, AsMatrixApply, ColumnIterator, ColumnIteratorMut,
    FromAij, Len, Nonzeros, RandomAccessByRef, RawAccess, RawAccessMut, SparseMatrixType,
};
use itertools::{izip, Itertools};
use num::One;

use super::mat_operations::SparseMatOpIterator;

/// A CSR matrix
pub struct CsrMatrix<Item> {
    mat_type: SparseMatType,
    shape: [usize; 2],
    indices: DynArray<usize, 1>,
    indptr: DynArray<usize, 1>,
    data: DynArray<Item, 1>,
}

impl<Item> CsrMatrix<Item> {
    /// Create a new CSR matrix
    pub fn new(
        shape: [usize; 2],
        indices: DynArray<usize, 1>,
        indptr: DynArray<usize, 1>,
        data: DynArray<Item, 1>,
    ) -> Self {
        Self {
            mat_type: SparseMatType::Csr,
            shape,
            indices,
            indptr,
            data,
        }
    }
}

impl<Item: AddAssign + PartialEq + Copy + Default> FromAij for CsrMatrix<Item> {
    fn from_aij(shape: [usize; 2], rows: &[usize], cols: &[usize], data: &[Item]) -> Self {
        let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);

        let max_col = if let Some(col) = cols.iter().max() {
            *col
        } else {
            0
        };
        let max_row = if let Some(row) = rows.last() { *row } else { 0 };

        assert!(
            max_col < shape[1],
            "Maximum column {} must be smaller than `shape.1` {}",
            max_col,
            shape[1]
        );

        assert!(
            max_row < shape[0],
            "Maximum row {} must be smaller than `shape.0` {}",
            max_row,
            shape[0]
        );

        let nelems = data.len();

        let mut indptr = Vec::<usize>::with_capacity(1 + shape[0]);

        let mut count: usize = 0;
        for row in 0..(shape[0]) {
            indptr.push(count);
            while count < nelems && row == rows[count] {
                count += 1;
            }
        }
        indptr.push(count);

        let indptr = DynArray::from_shape_and_vec([1 + shape[0]], indptr);
        let indices = DynArray::from_shape_and_vec([nelems], cols);
        let data = DynArray::from_shape_and_vec([nelems], data);

        Self::new(shape, indices, indptr, data)
    }
}

impl<Item> Shape<2> for CsrMatrix<Item> {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl<Item> BaseItem for CsrMatrix<Item> {
    type Item = Item;
}

impl<Item> Nonzeros for CsrMatrix<Item> {
    fn nnz(&self) -> usize {
        self.data.len()
    }
}

impl<Item> SparseMatrixType for CsrMatrix<Item> {
    fn mat_type(&self) -> SparseMatType {
        self.mat_type
    }
}

impl<Item> AijIteratorByValue for CsrMatrix<Item>
where
    Item: Copy,
{
    fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_ {
        self.indptr
            .iter_value()
            .tuple_windows::<(usize, usize)>()
            .enumerate()
            .flat_map(|(row, (start, end))| {
                izip!(
                    self.indices.data()[start..end].iter(),
                    self.data.data()[start..end].iter()
                )
                .map(|(col, value)| ([row, *col], *value))
                .collect::<Vec<_>>()
            })
    }
}

impl<Item> AijIteratorMut for CsrMatrix<Item>
where
    Item: Copy,
{
    fn iter_aij_mut(&mut self) -> impl Iterator<Item = ([usize; 2], &mut Self::Item)> + '_ {
        self.indptr
            .iter_value()
            .tuple_windows::<(usize, usize)>()
            .enumerate()
            .flat_map(|(row, (start, end))| {
                izip!(
                    self.indices.data()[start..end].iter(),
                    self.data.data_mut()[start..end]
                        .iter_mut()
                        // Need to convert the mutable reference to the raw pointer
                        // as borrow checker does not allow the mutable reference to leak from FnMut.
                        .map(|v| v as *mut Item)
                )
                .map(|(col, value)| ([row, *col], value))
                .collect::<Vec<_>>()
            })
            .map(|(idx, value)| (idx, unsafe { &mut *value }))
    }
}

impl<Item: Copy + Default> CsrMatrix<Item> {
    /// Return as sparse matrix in iterator form.
    pub fn op(&self) -> SparseMatOpIterator<Item, impl Iterator<Item = ([usize; 2], Item)> + '_> {
        SparseMatOpIterator::new(self.iter_aij_value(), self.shape())
    }
}

impl<Item, ArrayImplX, ArrayImplY> AsMatrixApply<Array<ArrayImplX, 1>, Array<ArrayImplY, 1>, 1>
    for CsrMatrix<Item>
where
    Item: Default + Mul<Output = Item> + AddAssign<Item> + Add<Output = Item> + Copy + One,
    Self: BaseItem<Item = Item>,
    ArrayImplX: RandomAccessByRef<1, Item = Item>,
    ArrayImplY: BaseItem<Item = Item>,
    Array<ArrayImplY, 1>: ArrayIteratorMut<Item = Item>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &crate::Array<ArrayImplX, 1>,
        beta: Self::Item,
        y: &mut crate::Array<ArrayImplY, 1>,
    ) {
        for (row, out) in y.iter_mut().enumerate() {
            *out = beta * *out
                + alpha * {
                    let c1 = self.indptr[[row]];
                    let c2 = self.indptr[[1 + row]];
                    let mut acc = Item::default();

                    for index in c1..c2 {
                        let col = self.indices[[index]];
                        acc += self.data[[index]] * x[[col]];
                    }
                    acc
                }
        }
    }
}

impl<Item, ArrayImplX, ArrayImplY> AsMatrixApply<Array<ArrayImplX, 2>, Array<ArrayImplY, 2>, 2>
    for CsrMatrix<Item>
where
    Item: Copy,
    Self: BaseItem<Item = Item>,
    Array<ArrayImplX, 2>: ColumnIterator<Item = Array<ArrayImplX, 1>>,
    Array<ArrayImplY, 2>: ColumnIteratorMut<Item = Array<ArrayImplY, 1>>,
    for<'b> Self: AsMatrixApply<
        <Array<ArrayImplX, 2> as ColumnIterator>::Col<'b>,
        <Array<ArrayImplY, 2> as ColumnIteratorMut>::Col<'b>,
        1,
    >,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &crate::Array<ArrayImplX, 2>,
        beta: Self::Item,
        y: &mut crate::Array<ArrayImplY, 2>,
    ) {
        for (colx, mut coly) in izip!(x.col_iter(), y.col_iter_mut()) {
            self.apply(alpha, &colx, beta, &mut coly)
        }
    }
}

// /// Convert to CSC matrix
// pub fn into_csc(self) -> CscMatrix<Item> {
//     let mut rows = Vec::<usize>::with_capacity(self.nelems());
//     let mut cols = Vec::<usize>::with_capacity(self.nelems());
//     let mut data = Vec::<Item>::with_capacity(self.nelems());
//
//     for (row, col, elem) in self.iter_aij() {
//         rows.push(row);
//         cols.push(col);
//         data.push(elem);
//     }
//
//     CscMatrix::from_aij(self.shape(), &rows, &cols, &data).unwrap()
// }
//
// /// Create CSR matrix from rows, columns and data
// pub fn from_aij(
//     shape: [usize; 2],
//     rows: &[usize],
//     cols: &[usize],
//     data: &[Item],
// ) -> RlstResult<Self> {
//     let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);
//
//     let max_col = if let Some(col) = cols.iter().max() {
//         *col
//     } else {
//         0
//     };
//     let max_row = if let Some(row) = rows.last() { *row } else { 0 };
//
//     assert!(
//         max_col < shape[1],
//         "Maximum column {} must be smaller than `shape.1` {}",
//         max_col,
//         shape[1]
//     );
//
//     assert!(
//         max_row < shape[0],
//         "Maximum row {} must be smaller than `shape.0` {}",
//         max_row,
//         shape[0]
//     );
//
//     let nelems = data.len();
//
//     let mut indptr = Vec::<usize>::with_capacity(1 + shape[0]);
//
//     let mut count: usize = 0;
//     for row in 0..(shape[0]) {
//         indptr.push(count);
//         while count < nelems && row == rows[count] {
//             count += 1;
//         }
//     }
//     indptr.push(count);
//
//     Ok(Self::new(shape, cols, indptr, data))
// }
// }
