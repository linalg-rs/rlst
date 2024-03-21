//! Definition of CSC matrices.

use crate::sparse::SparseMatType;
use rlst_dense::types::RlstResult;

use crate::sparse::csr_mat::CsrMatrix;
use crate::sparse::tools::normalize_aij;
use itertools::Itertools;
use rlst_dense::traits::AijIterator;
use rlst_dense::traits::Shape;
use rlst_dense::types::RlstScalar;

/// A CSC matrix
#[derive(Clone)]
pub struct CscMatrix<Item: RlstScalar> {
    mat_type: SparseMatType,
    shape: [usize; 2],
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<Item>,
}

impl<Item: RlstScalar> CscMatrix<Item> {
    /// Create a new CSC matrix
    pub fn new(
        shape: [usize; 2],
        indices: Vec<usize>,
        indptr: Vec<usize>,
        data: Vec<Item>,
    ) -> Self {
        Self {
            mat_type: SparseMatType::Csc,
            shape,
            indices,
            indptr,
            data,
        }
    }

    /// Number of elements
    pub fn nelems(&self) -> usize {
        self.data.len()
    }

    /// Matrix type
    pub fn mat_type(&self) -> &SparseMatType {
        &self.mat_type
    }

    /// Row indices of items
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Indices at which each column starts
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Entries of the matrix
    pub fn data(&self) -> &[Item] {
        &self.data
    }

    /// Matrix multiplication
    pub fn matmul(&self, alpha: Item, x: &[Item], beta: Item, y: &mut [Item]) {
        y.iter_mut().for_each(|elem| *elem = beta * *elem);

        for (col, (&col_start, &col_end)) in self.indptr().iter().tuple_windows().enumerate() {
            let x_elem = x[col];
            for (&row, &elem) in self.indices[col_start..col_end]
                .iter()
                .zip(self.data[col_start..col_end].iter())
            {
                y[row] += alpha * elem * x_elem;
            }
        }
    }

    /// Converts the matrix into a tuple (shape, indices, indptr, data)
    pub fn into_tuple(self) -> ([usize; 2], Vec<usize>, Vec<usize>, Vec<Item>) {
        (self.shape, self.indices, self.indptr, self.data)
    }

    /// Convert to CSR matrix
    pub fn into_csr(self) -> CsrMatrix<Item> {
        let mut rows = Vec::<usize>::with_capacity(self.nelems());
        let mut cols = Vec::<usize>::with_capacity(self.nelems());
        let mut data = Vec::<Item>::with_capacity(self.nelems());

        for (row, col, elem) in self.iter_aij() {
            rows.push(row);
            cols.push(col);
            data.push(elem);
        }

        CsrMatrix::from_aij(self.shape(), &rows, &cols, &data).unwrap()
    }

    /// Create CSC matrix from rows, columns and data
    pub fn from_aij(
        shape: [usize; 2],
        rows: &[usize],
        cols: &[usize],
        data: &[Item],
    ) -> RlstResult<Self> {
        let (rows, cols, data) = normalize_aij(rows, cols, data, SparseMatType::Csc);

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
        for col in 0..(shape[1]) {
            indptr.push(count);
            while count < nelems && col == cols[count] {
                count += 1;
            }
        }
        indptr.push(count);

        Ok(Self::new(shape, rows, indptr, data))
    }
}

/// CSC iterator
pub struct CscAijIterator<'a, Item: RlstScalar> {
    mat: &'a CscMatrix<Item>,
    col: usize,
    pos: usize,
}

impl<'a, Item: RlstScalar> CscAijIterator<'a, Item> {
    /// Create a new CSC iterator
    pub fn new(mat: &'a CscMatrix<Item>) -> Self {
        // We need to move the col pointer to the first col that has at least one element.

        let mut col: usize = 0;

        while col < mat.shape()[1] && mat.indptr[col] == mat.indptr[1 + col] {
            col += 1;
        }

        Self { mat, col, pos: 0 }
    }
}

impl<'a, Item: RlstScalar> std::iter::Iterator for CscAijIterator<'a, Item> {
    type Item = (usize, usize, Item);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.mat.data().len() {
            return None;
        }

        let result = Some((
            self.mat.indices[self.pos],
            self.col,
            self.mat.data[self.pos],
        ));

        self.pos += 1;

        // The following jumps over all zero cols to the next relevant col
        // It needs a <= comparison since self.pos has already been increased but
        // indptr[1+self.col] may be the old value (in the case we encounter a zero column).
        while self.col < self.mat.shape()[1] && self.mat.indptr()[1 + self.col] <= self.pos {
            self.col += 1;
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

impl<Item: RlstScalar> AijIterator for CscMatrix<Item> {
    type Item = Item;
    type Iter<'a> = CscAijIterator<'a, Item> where Self: 'a;

    fn iter_aij(&self) -> Self::Iter<'_> {
        CscAijIterator::new(self)
    }
}

impl<Item: RlstScalar> rlst_dense::traits::Shape<2> for CscMatrix<Item> {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
