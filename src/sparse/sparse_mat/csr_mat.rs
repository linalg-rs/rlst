//! Definition of CSR matrices.

use crate::dense::traits::AijIterator;
use crate::dense::types::RlstResult;
use crate::sparse::sparse_mat::SparseMatType;

use crate::dense::traits::Shape;
use crate::dense::types::RlstScalar;
use crate::sparse::sparse_mat::tools::normalize_aij;

use super::csc_mat::CscMatrix;

/// A CSR matrix
#[derive(Clone)]
pub struct CsrMatrix<T: RlstScalar> {
    mat_type: SparseMatType,
    shape: [usize; 2],
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<T>,
}

impl<Item: RlstScalar> CsrMatrix<Item> {
    /// Create a new CSR matrix
    pub fn new(
        shape: [usize; 2],
        indices: Vec<usize>,
        indptr: Vec<usize>,
        data: Vec<Item>,
    ) -> Self {
        Self {
            mat_type: SparseMatType::Csr,
            shape,
            indices,
            indptr,
            data,
        }
    }

    /// Number of element
    pub fn nelems(&self) -> usize {
        self.data.len()
    }

    /// Matrix type
    pub fn mat_type(&self) -> &SparseMatType {
        &self.mat_type
    }

    /// Indices of items
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Indices at which each row starts
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Entries of the matrix
    pub fn data(&self) -> &[Item] {
        &self.data
    }

    /// Matrix multiplication
    pub fn matmul(&self, alpha: Item, x: &[Item], beta: Item, y: &mut [Item]) {
        assert_eq!(self.shape()[0], y.len());
        assert_eq!(self.shape()[1], x.len());
        for (row, out) in y.iter_mut().enumerate() {
            *out = beta * *out
                + alpha * {
                    let c1 = self.indptr()[row];
                    let c2 = self.indptr()[1 + row];
                    let mut acc = Item::zero();

                    for index in c1..c2 {
                        let col = self.indices()[index];
                        acc += self.data()[index] * x[col];
                    }
                    acc
                }
        }
    }

    /// Convert to CSC matrix
    pub fn into_csc(self) -> CscMatrix<Item> {
        let mut rows = Vec::<usize>::with_capacity(self.nelems());
        let mut cols = Vec::<usize>::with_capacity(self.nelems());
        let mut data = Vec::<Item>::with_capacity(self.nelems());

        for (row, col, elem) in self.iter_aij() {
            rows.push(row);
            cols.push(col);
            data.push(elem);
        }

        CscMatrix::from_aij(self.shape(), &rows, &cols, &data).unwrap()
    }

    /// Create CSR matrix from rows, columns and data
    pub fn from_aij(
        shape: [usize; 2],
        rows: &[usize],
        cols: &[usize],
        data: &[Item],
    ) -> RlstResult<Self> {
        let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csr);

        let max_col = cols.iter().max().unwrap();
        let max_row = rows.last().unwrap();

        assert!(
            *max_col < shape[1],
            "Maximum column {} must be smaller than `shape.1` {}",
            max_col,
            shape[1]
        );

        assert!(
            *max_row < shape[0],
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

        Ok(Self::new(shape, cols, indptr, data))
    }
}

/// CSR iterator
pub struct CsrAijIterator<'a, Item: RlstScalar> {
    mat: &'a CsrMatrix<Item>,
    row: usize,
    pos: usize,
}

impl<'a, Item: RlstScalar> CsrAijIterator<'a, Item> {
    /// Create a new iterator
    pub fn new(mat: &'a CsrMatrix<Item>) -> Self {
        // We need to move the row pointer to the first row that has at least one element.

        let mut row: usize = 0;

        while row < mat.shape()[0] && mat.indptr[row] == mat.indptr[1 + row] {
            row += 1;
        }

        Self { mat, row, pos: 0 }
    }
}

impl<'a, Item: RlstScalar> std::iter::Iterator for CsrAijIterator<'a, Item> {
    type Item = (usize, usize, Item);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.mat.data().len() {
            return None;
        }

        let result = Some((
            self.row,
            *self.mat.indices().get(self.pos).unwrap(),
            *self.mat.data().get(self.pos).unwrap(),
        ));

        self.pos += 1;

        // The following jumps over all zero rows to the next relevant row
        while self.row < self.mat.shape()[0] && self.mat.indptr()[1 + self.row] <= self.pos {
            self.row += 1;
        }

        result
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.mat.data().len()
    }
}

impl<Item: RlstScalar> AijIterator for CsrMatrix<Item> {
    type Item = Item;
    type Iter<'a> = CsrAijIterator<'a, Item> where Self: 'a;

    fn iter_aij(&self) -> Self::Iter<'_> {
        CsrAijIterator::new(self)
    }
}

impl<Item: RlstScalar> Shape<2> for CsrMatrix<Item> {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
