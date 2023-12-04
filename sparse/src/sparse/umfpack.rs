//! Implementation of Umfpack for CSC Matrices

use std::ffi::c_void;

use crate::sparse::csc_mat::CscMatrix;
use rlst_common::types::Scalar;
use rlst_common::types::*;
use rlst_common::types::{RlstError, RlstResult};
use rlst_dense::array::Array;
use rlst_dense::traits::{RawAccess, RawAccessMut, Shape, Stride};
use rlst_umfpack as umfpack;

use super::csr_mat::CsrMatrix;

pub enum TransposeMode {
    NoTrans,
    Trans,
    ConjugateTrans,
}

pub struct UmfpackLu<T: Scalar> {
    shape: [usize; 2],
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

impl UmfpackLu<f64> {
    pub fn solve<
        ArrayImplX: rlst_dense::traits::RandomAccessByValue<1, Item = f64>
            + Shape<1>
            + RawAccessMut<Item = f64>
            + Stride<1>,
        ArrayImplRhs: rlst_dense::traits::RandomAccessByValue<1, Item = f64>
            + Shape<1>
            + RawAccess<Item = f64>
            + Stride<1>,
    >(
        &self,
        rhs: Array<f64, ArrayImplRhs, 1>,
        mut x: Array<f64, ArrayImplX, 1>,
        trans: TransposeMode,
    ) -> rlst_common::types::RlstResult<()> {
        assert_eq!(rhs.stride()[0], 1);
        assert_eq!(x.stride()[0], 1);

        let sys = match trans {
            TransposeMode::NoTrans => umfpack::UMFPACK_A,
            TransposeMode::Trans => umfpack::UMFPACK_Aat,
            TransposeMode::ConjugateTrans => umfpack::UMFPACK_At,
        };

        let n = self.shape[0];
        assert_eq!(x.shape()[0], n);
        assert_eq!(rhs.shape()[0], n);

        let info = unsafe {
            umfpack::umfpack_di_solve(
                sys as i32,
                self.indptr.as_ptr(),
                self.indices.as_ptr(),
                self.data.as_ptr(),
                x.data_mut().as_mut_ptr(),
                rhs.data().as_ptr(),
                self.numeric,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        if info != 0 {
            return Err(RlstError::UmfpackError(info));
        }

        Ok(())
    }
}

impl UmfpackLu<c64> {
    pub fn solve<
        ArrayImplX: rlst_dense::traits::RandomAccessByValue<1, Item = c64>
            + Shape<1>
            + RawAccessMut<Item = c64>
            + Stride<1>,
        ArrayImplRhs: rlst_dense::traits::RandomAccessByValue<1, Item = c64>
            + Shape<1>
            + RawAccess<Item = c64>
            + Stride<1>,
    >(
        &self,
        rhs: Array<c64, ArrayImplRhs, 1>,
        mut x: Array<c64, ArrayImplX, 1>,
        trans: TransposeMode,
    ) -> rlst_common::types::RlstResult<()> {
        assert_eq!(rhs.stride()[0], 1);
        assert_eq!(x.stride()[0], 1);

        let sys = match trans {
            TransposeMode::NoTrans => umfpack::UMFPACK_A,
            TransposeMode::Trans => umfpack::UMFPACK_Aat,
            TransposeMode::ConjugateTrans => umfpack::UMFPACK_At,
        };

        let n = self.shape[0];
        assert_eq!(x.shape()[0], n);
        assert_eq!(rhs.shape()[0], n);

        let info = unsafe {
            umfpack::umfpack_zi_solve(
                sys as i32,
                self.indptr.as_ptr(),
                self.indices.as_ptr(),
                self.data.as_ptr() as *const f64,
                std::ptr::null(),
                x.data_mut().as_mut_ptr() as *mut f64,
                std::ptr::null_mut(),
                rhs.data().as_ptr() as *mut f64,
                std::ptr::null(),
                self.numeric,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        if info != 0 {
            return Err(RlstError::UmfpackError(info));
        }

        Ok(())
    }
}

impl CsrMatrix<f64> {
    pub fn into_lu(self) -> RlstResult<UmfpackLu<f64>> {
        self.into_csc().into_lu()
    }
}

impl CsrMatrix<c64> {
    pub fn into_lu(self) -> RlstResult<UmfpackLu<c64>> {
        self.into_csc().into_lu()
    }
}

impl CscMatrix<f64> {
    pub fn into_lu(self) -> RlstResult<UmfpackLu<f64>> {
        let shape = self.shape();

        if shape[0] != shape[1] {
            return Err(RlstError::GeneralError(format!(
                "Matrix is not square. rows: {}, cols: {}",
                shape[0], shape[1],
            )));
        }

        let n = shape[0] as i32;

        let mut symbolic = std::ptr::null_mut::<c_void>();

        let mut umfpack_lu = UmfpackLu::<f64> {
            shape,
            indices: self.indices().iter().map(|&item| item as i32).collect(),
            indptr: self.indptr().iter().map(|&item| item as i32).collect(),
            data: self.data().to_vec(),
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

impl CscMatrix<c64> {
    pub fn into_lu(self) -> RlstResult<UmfpackLu<c64>> {
        let shape = self.shape();

        if shape[0] != shape[1] {
            return Err(RlstError::GeneralError(format!(
                "Matrix is not square. rows: {}, cols: {}",
                shape[0], shape[1],
            )));
        }

        let n = shape[0] as i32;

        let mut symbolic = std::ptr::null_mut::<c_void>();

        let mut umfpack_lu = UmfpackLu::<c64> {
            shape,
            indices: self.indices().iter().map(|&item| item as i32).collect(),
            indptr: self.indptr().iter().map(|&item| item as i32).collect(),
            data: self.data().to_vec(),
            numeric: std::ptr::null_mut::<c_void>(),
        };

        let info = unsafe {
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

#[cfg(test)]
mod test {

    use rlst_dense::traits::*;

    use rlst_dense::{
        array::empty_array, assert_array_relative_eq, rlst_dynamic_array1, rlst_dynamic_array2,
        traits::MultIntoResize,
    };

    use super::*;

    #[test]
    fn test_csc_umfpack_f64() {
        let n = 5;

        let mut mat = rlst_dynamic_array2!(f64, [n, n]);
        let mut x_exact = rlst_dynamic_array1!(f64, [n]);
        let mut x_actual = rlst_dynamic_array1!(f64, [n]);

        mat.fill_from_seed_equally_distributed(0);
        x_exact.fill_from_seed_equally_distributed(1);

        let rhs = empty_array::<f64, 1>().simple_mult_into_resize(mat.view(), x_exact.view());

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
            crate::sparse::csc_mat::CscMatrix::from_aij([n, n], &rows, &cols, &data).unwrap();

        sparse_mat
            .into_lu()
            .unwrap()
            .solve(rhs.view(), x_actual.view_mut(), TransposeMode::NoTrans)
            .unwrap();

        assert_array_relative_eq!(x_actual, x_exact, 1E-12);
    }

    #[test]
    fn test_csc_umfpack_c64() {
        let n = 5;

        let mut mat = rlst_dynamic_array2!(c64, [n, n]);
        let mut x_exact = rlst_dynamic_array1!(c64, [n]);
        let mut x_actual = rlst_dynamic_array1!(c64, [n]);

        mat.fill_from_seed_equally_distributed(0);
        x_exact.fill_from_seed_equally_distributed(1);

        let rhs = empty_array::<c64, 1>().simple_mult_into_resize(mat.view(), x_exact.view());

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
            crate::sparse::csc_mat::CscMatrix::from_aij([n, n], &rows, &cols, &data).unwrap();

        sparse_mat
            .into_lu()
            .unwrap()
            .solve(rhs.view(), x_actual.view_mut(), TransposeMode::NoTrans)
            .unwrap();

        assert_array_relative_eq!(x_actual, x_exact, 1E-12);
    }
}
