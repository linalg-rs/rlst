//! Definition of CSR matrices.

use crate::sparse::SparseMatType;
use rlst_dense::traits::AijIterator;
use rlst_dense::types::RlstResult;

use crate::sparse::tools::normalize_aij;
use rlst_dense::traits::Shape;
use rlst_dense::types::RlstScalar;

use super::csc_mat::CscMatrix;

#[derive(Clone)]
pub struct CsrMatrix<T: RlstScalar> {
    mat_type: SparseMatType,
    shape: [usize; 2],
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<T>,
}

impl<Item: RlstScalar> CsrMatrix<Item> {
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

pub struct CsrAijIterator<'a, Item: RlstScalar> {
    mat: &'a CsrMatrix<Item>,
    row: usize,
    pos: usize,
}

impl<'a, Item: RlstScalar> CsrAijIterator<'a, Item> {
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

#[cfg(test)]
mod test {

    use rlst_dense::traits::AijIterator;

    use super::*;

    #[test]
    fn test_csr_from_aij() {
        // Test the matrix [[1, 2], [3, 4]]
        let rows = vec![0, 0, 1, 1, 0];
        let cols = vec![0, 1, 0, 1, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 6.0];

        let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

        assert_eq!(csr.data().len(), 4);
        assert_eq!(csr.indices().len(), 4);
        assert_eq!(csr.indptr().len(), 3);
        assert_eq!(csr.data()[1], 8.0);

        //Test the matrix [[0, 0, 0], [2.0, 0, 0], [0, 0, 0]]
        let rows = vec![1, 2, 1];
        let cols = vec![0, 2, 0];
        let data = vec![2.0, 0.0, 3.0];

        let csr = CsrMatrix::from_aij([3, 3], &rows, &cols, &data).unwrap();

        assert_eq!(csr.indptr()[0], 0);
        assert_eq!(csr.indptr()[1], 0);
        assert_eq!(csr.indptr()[2], 1);
        assert_eq!(csr.indptr()[3], 1);
        assert_eq!(csr.data()[0], 5.0);
    }

    #[test]
    fn test_csr_matmul() {
        // Test the matrix [[1, 2], [3, 4]]
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();

        // Execute 2 * [1, 2] + 3 * A*x with x = [3, 4];
        // Expected result is [35, 79].

        let x = vec![3.0, 4.0];
        let mut res = vec![1.0, 2.0];

        csr.matmul(3.0, &x, 2.0, &mut res);

        assert_eq!(res[0], 35.0);
        assert_eq!(res[1], 79.0);
    }

    #[test]
    fn test_aij_iterator() {
        let rows = vec![2, 3, 4, 4, 6];
        let cols = vec![0, 1, 0, 2, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // This matrix has a zero row at the beginning one in between and several zero rows at the end.
        let csr = CsrMatrix::from_aij([10, 3], &rows, &cols, &data).unwrap();

        let aij_data: Vec<(usize, usize, f64)> = csr.iter_aij().collect();

        assert_eq!(aij_data.len(), 5);

        assert_eq!(aij_data[0], (2, 0, 1.0));
        assert_eq!(aij_data[1], (3, 1, 2.0));
        assert_eq!(aij_data[2], (4, 0, 3.0));
        assert_eq!(aij_data[3], (4, 2, 4.0));
        assert_eq!(aij_data[4], (6, 1, 5.0));
    }
}
