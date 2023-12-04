//! Definition of CSC matrices.

use crate::sparse::SparseMatType;
use rlst_common::types::RlstResult;

use crate::sparse::csr_mat::CsrMatrix;
use crate::sparse::tools::normalize_aij;
use itertools::Itertools;
use rlst_common::types::Scalar;
use rlst_dense::traits::AijIterator;
use rlst_dense::traits::Shape;

#[derive(Clone)]
pub struct CscMatrix<Item: Scalar> {
    mat_type: SparseMatType,
    shape: [usize; 2],
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<Item>,
}

impl<Item: Scalar> CscMatrix<Item> {
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

    pub fn nelems(&self) -> usize {
        self.data.len()
    }

    pub fn mat_type(&self) -> &SparseMatType {
        &self.mat_type
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn data(&self) -> &[Item] {
        &self.data
    }

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

pub struct CscAijIterator<'a, Item: Scalar> {
    mat: &'a CscMatrix<Item>,
    col: usize,
    pos: usize,
}

impl<'a, Item: Scalar> CscAijIterator<'a, Item> {
    pub fn new(mat: &'a CscMatrix<Item>) -> Self {
        // We need to move the col pointer to the first col that has at least one element.

        let mut col: usize = 0;

        while col < mat.shape()[1] && mat.indptr[col] == mat.indptr[1 + col] {
            col += 1;
        }

        Self { mat, col, pos: 0 }
    }
}

impl<'a, Item: Scalar> std::iter::Iterator for CscAijIterator<'a, Item> {
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

impl<Item: Scalar> AijIterator for CscMatrix<Item> {
    type Item = Item;
    type Iter<'a> = CscAijIterator<'a, Item> where Self: 'a;

    fn iter_aij(&self) -> Self::Iter<'_> {
        CscAijIterator::new(self)
    }
}

impl<Item: Scalar> rlst_dense::traits::Shape<2> for CscMatrix<Item> {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg(test)]
mod test {

    use rlst_dense::traits::AijIterator;

    use super::*;

    #[test]
    fn test_csc_from_aij() {
        // Test the matrix [[1, 2], [3, 4]]
        let rows = vec![0, 0, 1, 1, 0];
        let cols = vec![0, 1, 0, 1, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 6.0];

        let csc = CscMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

        assert_eq!(csc.data().len(), 4);
        assert_eq!(csc.indices().len(), 4);
        assert_eq!(csc.indptr().len(), 3);
        assert_eq!(csc.data()[2], 8.0);

        // Test the matrix [[0, 2.0, 0.0], [0, 0, 0], [0, 0, 0]]
        let rows = vec![0, 2, 0];
        let cols = vec![1, 2, 1];
        let data = vec![2.0, 0.0, 3.0];

        let csc = CscMatrix::from_aij([3, 3], &rows, &cols, &data).unwrap();

        assert_eq!(csc.indptr()[0], 0);
        assert_eq!(csc.indptr()[1], 0);
        assert_eq!(csc.indptr()[2], 1);
        assert_eq!(csc.indptr()[3], 1);
        assert_eq!(csc.data()[0], 5.0);
    }

    #[test]
    fn test_csc_matmul() {
        // Test the matrix [[1, 2], [3, 4]]
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let csc = CscMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

        // Execute 2 * [1, 2] + 3 * A*x with x = [3, 4];
        // Expected result is [35, 79].

        let x = vec![3.0, 4.0];
        let mut res = vec![1.0, 2.0];

        csc.matmul(3.0, &x, 2.0, &mut res);

        assert_eq!(res[0], 35.0);
        assert_eq!(res[1], 79.0);
    }

    #[test]
    fn test_aij_iterator() {
        let rows = vec![2, 3, 4, 4, 6];
        let cols = vec![1, 1, 3, 3, 4];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // This matrix has a zero col at the beginning one in between and several zero cols at the end.
        let csr = CscMatrix::from_aij([10, 20], &rows, &cols, &data).unwrap();

        let aij_data: Vec<(usize, usize, f64)> = csr.iter_aij().collect();

        assert_eq!(aij_data.len(), 4);

        assert_eq!(aij_data[0], (2, 1, 1.0));
        assert_eq!(aij_data[1], (3, 1, 2.0));
        assert_eq!(aij_data[2], (4, 3, 7.0));
        assert_eq!(aij_data[3], (6, 4, 5.0));
    }
}
