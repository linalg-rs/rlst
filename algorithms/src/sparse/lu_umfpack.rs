//! Interface for sparse matrix LU via UMFPACK

use crate::traits::{
    lu_decomp::{LUDecomp, LU},
    types::TransposeMode,
};
use rlst_common::types::{c64, RlstError, Scalar};
use rlst_dense::{rlst_mat, MatrixD, RawAccessMut, Shape};
use rlst_sparse::sparse::csc_mat::CscMatrix;
use rlst_umfpack as umfpack;
use std::ffi::c_void;

pub struct UmfpackLu<T: Scalar> {
    shape: (usize, usize),
    indices: Vec<i32>,
    indptr: Vec<i32>,
    data: Vec<T>,
    numeric: *mut c_void,
}

impl<T: Scalar> Drop for UmfpackLu<T> {
    fn drop(&mut self) {
        unsafe { umfpack::umfpack_di_free_numeric(&mut self.numeric) };
    }
}

impl<'a> LU for crate::linalg::SparseMatrixLinalgBuilder<'a, f64, CscMatrix<f64>> {
    type Out = UmfpackLu<f64>;
    type T = f64;

    fn lu(self) -> rlst_common::types::RlstResult<Self::Out> {
        let shape = self.mat.shape();

        if shape.0 != shape.1 {
            return Err(RlstError::GeneralError(format!(
                "Matrix is not square. rows: {}, cols: {}",
                shape.0, shape.1,
            )));
        }

        let n = shape.0 as i32;

        let mut symbolic = std::ptr::null_mut::<c_void>();

        let mut umfpack_lu = UmfpackLu::<f64> {
            shape,
            indices: self.mat.indices().iter().map(|&item| item as i32).collect(),
            indptr: self.mat.indptr().iter().map(|&item| item as i32).collect(),
            data: self.mat.data().to_vec(),
            numeric: std::ptr::null_mut::<c_void>(),
        };

        let info = unsafe {
            umfpack::umfpack_di_symbolic(
                n,
                n,
                umfpack_lu.indptr.as_ptr(),
                umfpack_lu.indices.as_ptr(),
                umfpack_lu.data.as_ptr(),
                &mut symbolic,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        if info != 0 {
            return Err(RlstError::UmfpackError(info));
        }

        let info = unsafe {
            umfpack::umfpack_di_numeric(
                umfpack_lu.indptr.as_ptr(),
                umfpack_lu.indices.as_ptr(),
                umfpack_lu.data.as_ptr(),
                symbolic,
                &mut umfpack_lu.numeric,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        if info != 0 {
            return Err(RlstError::UmfpackError(info));
        }

        unsafe { umfpack::umfpack_di_free_symbolic(&mut symbolic) };

        Ok(umfpack_lu)
    }
}

impl<'a> LU for crate::linalg::SparseMatrixLinalgBuilder<'a, c64, CscMatrix<c64>> {
    type Out = UmfpackLu<c64>;
    type T = c64;

    fn lu(self) -> rlst_common::types::RlstResult<Self::Out> {
        let shape = self.mat.shape();

        if shape.0 != shape.1 {
            return Err(RlstError::GeneralError(format!(
                "Matrix is not square. rows: {}, cols: {}",
                shape.0, shape.1,
            )));
        }

        let n = shape.0 as i32;

        let mut symbolic = std::ptr::null_mut::<c_void>();

        let mut umfpack_lu = UmfpackLu::<c64> {
            shape,
            indices: self.mat.indices().iter().map(|&item| item as i32).collect(),
            indptr: self.mat.indptr().iter().map(|&item| item as i32).collect(),
            data: self.mat.data().to_vec(),
            numeric: std::ptr::null_mut::<c_void>(),
        };

        let info = unsafe {
            // We use packed format for complex numbers. In that case
            // Umfpack expects the complex numbers as real array (of twice the length)
            // and the imaginary part array as null pointer.
            umfpack::umfpack_zi_symbolic(
                n,
                n,
                umfpack_lu.indptr.as_ptr(),
                umfpack_lu.indices.as_ptr(),
                umfpack_lu.data.as_ptr() as *const f64,
                std::ptr::null(),
                &mut symbolic,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        if info != 0 {
            return Err(RlstError::UmfpackError(info));
        }

        let info = unsafe {
            umfpack::umfpack_zi_numeric(
                umfpack_lu.indptr.as_ptr(),
                umfpack_lu.indices.as_ptr(),
                umfpack_lu.data.as_ptr() as *const f64,
                std::ptr::null(),
                symbolic,
                &mut umfpack_lu.numeric,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        if info != 0 {
            return Err(RlstError::UmfpackError(info));
        }

        unsafe { umfpack::umfpack_di_free_symbolic(&mut symbolic) };

        Ok(umfpack_lu)
    }
}

impl LUDecomp for UmfpackLu<f64> {
    type Sol = MatrixD<f64>;

    type T = f64;

    fn data(&self) -> &[Self::T] {
        std::unimplemented!()
    }

    fn get_l(&self) -> MatrixD<Self::T> {
        std::unimplemented!()
    }

    fn get_perm(&self) -> Vec<usize> {
        std::unimplemented!()
    }

    fn get_u(&self) -> MatrixD<Self::T> {
        std::unimplemented!()
    }

    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn solve<Rhs: rlst_dense::RandomAccessByValue<Item = Self::T> + Shape>(
        &self,
        rhs: &Rhs,
        trans: crate::traits::types::TransposeMode,
    ) -> rlst_common::types::RlstResult<Self::Sol> {
        let sys = match trans {
            TransposeMode::NoTrans => umfpack::UMFPACK_A,
            TransposeMode::Trans => umfpack::UMFPACK_Aat,
            TransposeMode::ConjugateTrans => umfpack::UMFPACK_At,
        };

        if rhs.shape().0 != self.shape.1 {
            return Err(RlstError::SingleDimensionError {
                expected: self.shape.1,
                actual: rhs.shape().0,
            });
        }

        let n = self.shape().0;

        let mut sol = rlst_mat![f64, rhs.shape()];
        let mut b = vec![0.0; n];

        for col_index in 0..sol.shape().1 {
            let x = &mut sol.data_mut()[n * col_index..(n + 1) * col_index];
            for (row_index, elem) in b.iter_mut().enumerate() {
                *elem = rhs.get_value(row_index, col_index).unwrap();
            }
            let info = unsafe {
                umfpack::umfpack_di_solve(
                    sys as i32,
                    self.indptr.as_ptr(),
                    self.indices.as_ptr(),
                    self.data.as_ptr(),
                    x.as_mut_ptr(),
                    b.as_ptr(),
                    self.numeric,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };

            if info != 0 {
                return Err(RlstError::UmfpackError(info));
            }
        }

        Ok(sol)
    }
}

impl LUDecomp for UmfpackLu<c64> {
    type Sol = MatrixD<c64>;

    type T = c64;

    fn data(&self) -> &[Self::T] {
        std::unimplemented!()
    }

    fn get_l(&self) -> MatrixD<Self::T> {
        std::unimplemented!()
    }

    fn get_perm(&self) -> Vec<usize> {
        std::unimplemented!()
    }

    fn get_u(&self) -> MatrixD<Self::T> {
        std::unimplemented!()
    }

    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn solve<Rhs: rlst_dense::RandomAccessByValue<Item = Self::T> + Shape>(
        &self,
        rhs: &Rhs,
        trans: crate::traits::types::TransposeMode,
    ) -> rlst_common::types::RlstResult<Self::Sol> {
        let sys = match trans {
            TransposeMode::NoTrans => umfpack::UMFPACK_A,
            TransposeMode::Trans => umfpack::UMFPACK_Aat,
            TransposeMode::ConjugateTrans => umfpack::UMFPACK_At,
        };

        if rhs.shape().0 != self.shape.1 {
            return Err(RlstError::SingleDimensionError {
                expected: self.shape.1,
                actual: rhs.shape().0,
            });
        }

        let n = self.shape().0;

        let mut sol = rlst_mat![c64, rhs.shape()];
        let mut b = vec![c64::new(0.0, 0.0); n];

        for col_index in 0..sol.shape().1 {
            let x = &mut sol.data_mut()[n * col_index..(n + 1) * col_index];
            for (row_index, elem) in b.iter_mut().enumerate() {
                *elem = rhs.get_value(row_index, col_index).unwrap();
            }
            let info = unsafe {
                // We use the packed format, so convert the complex
                // pointer to a real array of twice the length and leave
                // the imaginary parts zero.
                umfpack::umfpack_zi_solve(
                    sys as i32,
                    self.indptr.as_ptr(),
                    self.indices.as_ptr(),
                    self.data.as_ptr() as *const f64,
                    std::ptr::null(),
                    x.as_mut_ptr() as *mut f64,
                    std::ptr::null_mut(),
                    b.as_ptr() as *const f64,
                    std::ptr::null(),
                    self.numeric,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };

            if info != 0 {
                return Err(RlstError::UmfpackError(info));
            }
        }

        Ok(sol)
    }
}

#[cfg(test)]
mod test {
    use rlst_common::assert_matrix_relative_eq;
    use rlst_common::types::c64;
    use rlst_dense::traits::*;
    use rlst_dense::{rlst_rand_mat, Dot};

    use crate::{
        linalg::LinAlg,
        traits::{
            lu_decomp::{LUDecomp, LU},
            types::TransposeMode,
        },
    };

    #[test]
    fn test_umfpack_f64() {
        let n = 5;
        let cols = 3;

        let mat = rlst_rand_mat![f64, (n, n)];
        let x_exact = rlst_rand_mat![f64, (n, cols)];

        let rhs = mat.dot(&x_exact);

        let mut rows = Vec::<usize>::with_capacity(n * n);
        let mut cols = Vec::<usize>::with_capacity(n * n);
        let mut data = Vec::<f64>::with_capacity(n * n);

        for col_index in 0..n {
            for row_index in 0..n {
                rows.push(row_index);
                cols.push(col_index);
                data.push(mat[[row_index, col_index]]);
            }
        }

        let sparse_mat =
            rlst_sparse::sparse::csc_mat::CscMatrix::from_aij((n, n), &rows, &cols, &data).unwrap();

        let x_actual = sparse_mat
            .linalg()
            .lu()
            .unwrap()
            .solve(&rhs, TransposeMode::NoTrans)
            .unwrap();

        assert_matrix_relative_eq!(x_actual, x_exact, 1E-12);
    }

    #[test]
    fn test_umfpack_c64() {
        let n = 5;
        let cols = 3;

        let mat = rlst_rand_mat![c64, (n, n)];
        let x_exact = rlst_rand_mat![c64, (n, cols)];

        let rhs = mat.dot(&x_exact);

        let mut rows = Vec::<usize>::with_capacity(n * n);
        let mut cols = Vec::<usize>::with_capacity(n * n);
        let mut data = Vec::<c64>::with_capacity(n * n);

        for col_index in 0..n {
            for row_index in 0..n {
                rows.push(row_index);
                cols.push(col_index);
                data.push(mat[[row_index, col_index]]);
            }
        }

        let sparse_mat =
            rlst_sparse::sparse::csc_mat::CscMatrix::from_aij((n, n), &rows, &cols, &data).unwrap();

        let x_actual = sparse_mat
            .linalg()
            .lu()
            .unwrap()
            .solve(&rhs, TransposeMode::NoTrans)
            .unwrap();

        assert_matrix_relative_eq!(x_actual, x_exact, 1E-12);
    }
}
