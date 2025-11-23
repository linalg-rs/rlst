//! Definition of CSR matrices.
//!
//! A CSR matrix consists of three arrays.
//! - `data` - Stores all entries of the CSR matrix.
//! - `indices` - The column indices associated with each entry in `data`.
//! - `indptr` - An arry of pointers. The data entries for row `i` are contained in `data[indptr[i]]..data[indptr[i + 1]]`.
//!
//! The last entry of `indptr` is the number of nonzero elements of the sparse matrix.

use std::ops::{Add, AddAssign, Mul};

use crate::dense::array::reference::{ArrayRef, ArrayRefMut};
use crate::dense::array::slice::ArraySlice;
use crate::sparse::tools::normalize_aij;
use crate::{AijIteratorByValue, BaseItem, Shape, dense::array::DynArray, sparse::SparseMatType};
use crate::{
    AijIteratorMut, Array, AsMatrixApply, FromAij, Nonzeros, SparseMatrixType,
    UnsafeRandom1DAccessMut, UnsafeRandomAccessByValue, UnsafeRandomAccessMut, empty_array,
};
use itertools::{Itertools, izip};
use num::One;

use super::mat_operations::SparseMatOpIterator;

/// A CSR matrix
pub struct CsrMatrix<Item> {
    /// The `mat_type` denotes the storage type for the sparse matrix.
    mat_type: SparseMatType,
    /// The shape of the sparse matrix.
    shape: [usize; 2],
    /// The array of column indices.
    indices: DynArray<usize, 1>,
    /// The array of index pointers.
    indptr: DynArray<usize, 1>,
    /// The entries of the sparse matrix.
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
        // Check that the indices cannot be out of bounds.
        // This is because `apply` uses unsafe access to the entries.

        assert_eq!(indptr.len(), 1 + shape[0]);
        assert_eq!(data.len(), indices.len());
        assert_eq!(*indptr.data().unwrap().last().unwrap(), data.len());

        // Check that the indices in indptr are monotonically increasing and
        // are smaller or equal to the overall length of `indices`. This
        // guarantees that there cannot be a memory error in the unsafe
        // access in `apply`.
        for (first, second) in indptr.iter_value().tuple_windows() {
            assert!(
                first <= second,
                "Elements of indptr not in increasing order {first} > {second}."
            );
        }
        // Check that the last element in indptr is the length of the `indices` array.
        assert_eq!(*indptr.data().unwrap().last().unwrap(), indices.len());

        // Check that the column indices in `indices` are smaller than `shape[1]`.

        if let Some(&max_col_index) = indices.data().unwrap().iter().max() {
            assert!(max_col_index < shape[1]);
        }

        Self {
            mat_type: SparseMatType::Csr,
            shape,
            indices,
            indptr,
            data,
        }
    }

    /// Return the index pointer.
    pub fn indptr(&self) -> &DynArray<usize, 1> {
        &self.indptr
    }

    /// Return the indices.
    pub fn indices(&self) -> &DynArray<usize, 1> {
        &self.indices
    }

    /// Return the data.
    pub fn data(&self) -> &DynArray<Item, 1> {
        &self.data
    }
}

impl<Item> FromAij for CsrMatrix<Item>
where
    Item: AddAssign + PartialEq + Copy + Default,
{
    /// Create a new CSR matrix from arrays `row`, `cols` and `data` that store for
    /// each nonzero entry the associated row, column, and value.
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

impl<Item> BaseItem for CsrMatrix<Item>
where
    Item: Copy + Default,
{
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
    Item: Copy + Default,
{
    fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_ {
        self.indptr
            .iter_value()
            .tuple_windows::<(usize, usize)>()
            .enumerate()
            .flat_map(|(row, (start, end))| {
                izip!(
                    self.indices.data().unwrap()[start..end].iter(),
                    self.data.data().unwrap()[start..end].iter()
                )
                .map(|(col, value)| ([row, *col], *value))
                .collect::<Vec<_>>()
            })
    }
}

impl<Item> AijIteratorMut for CsrMatrix<Item>
where
    Item: Copy + Default,
{
    fn iter_aij_mut(&mut self) -> impl Iterator<Item = ([usize; 2], &mut Self::Item)> + '_ {
        self.indptr
            .iter_value()
            .tuple_windows::<(usize, usize)>()
            .enumerate()
            .flat_map(|(row, (start, end))| {
                izip!(
                    self.indices.data().unwrap()[start..end].iter(),
                    self.data.data_mut().unwrap()[start..end]
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

impl<Item: Copy + Default> CsrMatrix<Item> {
    /// Convert to a dense matrix.
    pub fn todense(&self) -> DynArray<Item, 2> {
        DynArray::from_iter_aij(self.shape(), self.iter_aij_value())
    }
}

impl<Item: Default + Mul<Output = Item> + AddAssign<Item> + Add<Output = Item> + Copy + One>
    CsrMatrix<Item>
{
    /// Apply the matrix to a vector or dense matrix.
    pub fn dot<ArrayImpl, const NDIM: usize>(
        &self,
        other: &Array<ArrayImpl, NDIM>,
    ) -> DynArray<Item, NDIM>
    where
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    {
        let mut out = empty_array::<Item, NDIM>();

        if NDIM == 1 {
            let mut out = out.r_mut().coerce_dim::<1>().unwrap();
            let other = other.r().coerce_dim::<1>().unwrap();
            out.resize_in_place([self.shape()[0]]);
            self.apply(One::one(), &other, Default::default(), &mut out);
        } else if NDIM == 2 {
            let mut out = out.r_mut().coerce_dim::<2>().unwrap();
            let other = other.r().coerce_dim::<2>().unwrap();
            out.resize_in_place([self.shape()[0], other.shape()[1]]);
            self.apply(One::one(), &other, Default::default(), &mut out);
        } else {
            panic!(
                "Unsupported number of dimensions NDIM = {NDIM}. Only NDIM=1 or NDIM=2 supported."
            );
        }

        out
    }
}

impl<Item, ArrayImplX, ArrayImplY> AsMatrixApply<Array<ArrayImplX, 1>, Array<ArrayImplY, 1>>
    for CsrMatrix<Item>
where
    Item: Default + Mul<Output = Item> + AddAssign<Item> + Add<Output = Item> + Copy + One,
    ArrayImplX: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1>,
    ArrayImplY: UnsafeRandom1DAccessMut<Item = Item> + Shape<1>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &crate::Array<ArrayImplX, 1>,
        beta: Self::Item,
        y: &mut crate::Array<ArrayImplY, 1>,
    ) {
        assert_eq!(y.len(), self.shape()[0]);
        assert_eq!(x.len(), self.shape()[1]);
        for (row, out) in y.iter_mut().enumerate() {
            *out = beta * *out
                + alpha * {
                    let c1 = unsafe { self.indptr.get_value_unchecked([row]) };
                    let c2 = unsafe { self.indptr.get_value_unchecked([1 + row]) };
                    let mut acc = Item::default();

                    for index in c1..c2 {
                        let col = unsafe { self.indices.get_value_unchecked([index]) };
                        acc += unsafe {
                            self.data.get_value_unchecked([index]) * x.get_value_unchecked([col])
                        };
                    }
                    acc
                }
        }
    }
}

impl<Item, ArrayImplX, ArrayImplY> AsMatrixApply<Array<ArrayImplX, 2>, Array<ArrayImplY, 2>>
    for CsrMatrix<Item>
where
    Item: Copy,
    Self: BaseItem<Item = Item>,
    ArrayImplX: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>,
    ArrayImplY: UnsafeRandomAccessMut<2, Item = Item> + Shape<2>,
    for<'b> Self: AsMatrixApply<
            Array<ArraySlice<ArrayRef<'b, ArrayImplX, 2>, 2, 1>, 1>,
            Array<ArraySlice<ArrayRefMut<'b, ArrayImplY, 2>, 2, 1>, 1>,
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

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_csr() {
        // We create a simple CSR matrix.
        let rows: Vec<usize> = vec![1, 4, 4];
        let cols: Vec<usize> = vec![2, 5, 6];
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];

        let shape = [8, 13];
        let sparse_mat = CsrMatrix::from_aij(shape, &rows, &cols, &data);

        let mut x = DynArray::<f64, 1>::from_shape([shape[1]]);
        x.fill_from_seed_equally_distributed(0);

        let y = crate::dot!(sparse_mat, x);
        let expected = crate::dot!(sparse_mat.todense(), x);

        crate::assert_array_relative_eq!(y, expected, 1E-10);
    }
}
