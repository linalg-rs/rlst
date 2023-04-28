//! Definition of CSR matrices.

use crate::sparse::SparseMatType;
use rlst_common::types::RlstResult;

use rlst_common::types::Scalar;

pub struct CsrMatrix<T: Scalar> {
    mat_type: SparseMatType,
    shape: (usize, usize),
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn new(
        shape: (usize, usize),
        indices: Vec<usize>,
        indptr: Vec<usize>,
        data: Vec<T>,
    ) -> Self {
        Self {
            mat_type: SparseMatType::Csr,
            shape,
            indices,
            indptr,
            data,
        }
    }

    pub fn mat_type(&self) -> &SparseMatType {
        &self.mat_type
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn matmul(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {
        for (row, out) in y.iter_mut().enumerate() {
            *out = beta * *out
                + alpha * {
                    let c1 = self.indptr()[row];
                    let c2 = self.indptr()[1 + row];
                    let mut acc = T::zero();

                    for index in c1..c2 {
                        unsafe {
                            let col = *self.indices().get_unchecked(index);
                            acc += *self.data().get_unchecked(index) * *x.get_unchecked(col);
                        }
                    }
                    acc
                }
        }
    }

    pub fn from_aij(
        shape: (usize, usize),
        rows: &[usize],
        cols: &[usize],
        data: &[T],
    ) -> RlstResult<Self> {
        let nelems = data.len();
        let mut sorted: Vec<usize> = (0..nelems).collect();
        // Sorts first by column, then by row. Ensures that
        // elements with consecutive columns are next to each other.
        sorted.sort_by_key(|&idx| cols[idx]);
        sorted.sort_by_key(|&idx| rows[idx]);

        // Now merge consecutive elements together and filter out zeros

        let mut rows_t = Vec::<usize>::with_capacity(nelems);
        let mut cols_t = Vec::<usize>::with_capacity(nelems);
        let mut data_t = Vec::<T>::with_capacity(nelems);

        let mut count: usize = 0;
        while count < nelems {
            let current_row = rows[sorted[count]];
            let current_col = cols[sorted[count]];
            let mut current_data = T::zero();
            while count < nelems
                && rows[sorted[count]] == current_row
                && cols[sorted[count]] == current_col
            {
                current_data += data[sorted[count]];
                count += 1;
            }
            if current_data != T::zero() {
                rows_t.push(current_row);
                cols_t.push(current_col);
                data_t.push(current_data);
            }
        }

        let nelems = data_t.len();

        let mut indptr = Vec::<usize>::with_capacity(1 + shape.0);
        let mut indices = Vec::<usize>::with_capacity(nelems);
        let mut new_data = Vec::<T>::with_capacity(nelems);

        let mut count: usize = 0;
        for row in 0..(shape.0) {
            indptr.push(count);
            while count < nelems && row == rows[sorted[count]] {
                count += 1;
            }
        }
        indptr.push(count);

        for index in 0..nelems {
            indices.push(cols_t[index]);
            new_data.push(data_t[index]);
        }

        Ok(Self::new(shape, indices, indptr, new_data))
    }
}

pub struct CsrAijIterator<'a, T: Scalar> {
    mat: &'a CsrMatrix<T>,
    row: usize,
    pos: usize,
}

impl<'a, T: Scalar> CsrAijIterator<'a, T> {
    pub fn new(mat: &'a CsrMatrix<T>) -> Self {
        // We need to move the row pointer to the first row that has at least one element.

        let mut row: usize = 0;

        while row < mat.shape().0 && mat.indptr[row] == mat.indptr[1 + row] {
            row += 1;
        }

        Self { mat, row, pos: 0 }
    }
}

impl<'a, T: Scalar> std::iter::Iterator for CsrAijIterator<'a, T> {
    type Item = (usize, usize, T);

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
        while self.row < self.mat.shape().0 && self.mat.indptr()[1 + self.row] <= self.pos {
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

impl<T: Scalar> rlst_common::basic_traits::AijIterator for CsrMatrix<T> {
    type T = T;
    type Iter<'a> = CsrAijIterator<'a, T> where Self: 'a;

    fn iter_aij<'a>(&'a self) -> Self::Iter<'a> {
        CsrAijIterator::new(self)
    }
}

impl<T: Scalar> rlst_common::basic_traits::Dimension for CsrMatrix<T> {
    fn dim(&self) -> (usize, usize) {
        self.shape()
    }
}

#[cfg(test)]
mod test {

    use rlst_common::basic_traits::AijIterator;

    use super::*;

    #[test]
    fn test_csr_from_aij() {
        // Test the matrix [[1, 2], [3, 4]]
        let rows = vec![0, 0, 1, 1, 0];
        let cols = vec![0, 1, 0, 1, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 6.0];

        let csr = CsrMatrix::from_aij((2, 2), &rows, &cols, &data).unwrap();

        assert_eq!(csr.data().len(), 4);
        assert_eq!(csr.indices().len(), 4);
        assert_eq!(csr.indptr().len(), 3);
        assert_eq!(csr.data()[1], 8.0);

        //Test the matrix [[0, 0, 0], [2.0, 0, 0], [0, 0, 0]]
        let rows = vec![1, 2, 1];
        let cols = vec![0, 2, 0];
        let data = vec![2.0, 0.0, 3.0];

        let csr = CsrMatrix::from_aij((3, 3), &rows, &cols, &data).unwrap();

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

        let csr = CsrMatrix::from_aij((2, 2), &rows, &cols, &data).unwrap();

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
        let csr = CsrMatrix::from_aij((10, 2), &rows, &cols, &data).unwrap();

        let aij_data: Vec<(usize, usize, f64)> = csr.iter_aij().collect();

        assert_eq!(aij_data.len(), 5);

        assert_eq!(aij_data[0], (2, 0, 1.0));
        assert_eq!(aij_data[1], (3, 1, 2.0));
        assert_eq!(aij_data[2], (4, 0, 3.0));
        assert_eq!(aij_data[3], (4, 2, 4.0));
        assert_eq!(aij_data[4], (6, 1, 5.0));
    }
}
