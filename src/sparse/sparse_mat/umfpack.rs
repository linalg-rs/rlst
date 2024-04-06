//! Implementation of Umfpack for CSC Matrices

use std::ffi::c_void;

use crate::dense::array::Array;
use crate::dense::traits::{RawAccess, RawAccessMut, Shape, Stride};
use crate::dense::types::RlstScalar;
use crate::dense::types::TransMode;
use crate::dense::types::*;
use crate::dense::types::{RlstError, RlstResult};
use crate::external::umfpack;
use crate::sparse::sparse_mat::csc_mat::CscMatrix;

use super::csr_mat::CsrMatrix;

/// Holds the Umfpack data structures.
pub struct UmfpackLu<T: RlstScalar> {
    shape: [usize; 2],
    indices: Vec<i32>,
    indptr: Vec<i32>,
    data: Vec<T>,
    numeric: *mut c_void,
}

impl<T: RlstScalar> Drop for UmfpackLu<T> {
    fn drop(&mut self) {
        unsafe { umfpack::raw::umfpack_di_free_numeric(&mut self.numeric) };
    }
}

impl UmfpackLu<f64> {
    /// Solve for a given right-hand side.
    pub fn solve<
        ArrayImplX: crate::dense::traits::RandomAccessByValue<1, Item = f64>
            + Shape<1>
            + RawAccessMut<Item = f64>
            + Stride<1>,
        ArrayImplRhs: crate::dense::traits::RandomAccessByValue<1, Item = f64>
            + Shape<1>
            + RawAccess<Item = f64>
            + Stride<1>,
    >(
        &self,
        rhs: Array<f64, ArrayImplRhs, 1>,
        mut x: Array<f64, ArrayImplX, 1>,
        trans: TransMode,
    ) -> crate::dense::types::RlstResult<()> {
        assert_eq!(rhs.stride()[0], 1);
        assert_eq!(x.stride()[0], 1);

        let sys = match trans {
            TransMode::NoTrans => umfpack::raw::UMFPACK_A,
            TransMode::Trans => umfpack::raw::UMFPACK_Aat,
            TransMode::ConjTrans => umfpack::raw::UMFPACK_At,
            _ => panic!("Transpose mode not supported for Umfpack"),
        };

        let n = self.shape[0];
        assert_eq!(x.shape()[0], n);
        assert_eq!(rhs.shape()[0], n);

        let info = unsafe {
            umfpack::raw::umfpack_di_solve(
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
    /// Solve for a given right-hand side.
    pub fn solve<
        ArrayImplX: crate::dense::traits::RandomAccessByValue<1, Item = c64>
            + Shape<1>
            + RawAccessMut<Item = c64>
            + Stride<1>,
        ArrayImplRhs: crate::dense::traits::RandomAccessByValue<1, Item = c64>
            + Shape<1>
            + RawAccess<Item = c64>
            + Stride<1>,
    >(
        &self,
        rhs: Array<c64, ArrayImplRhs, 1>,
        mut x: Array<c64, ArrayImplX, 1>,
        trans: TransMode,
    ) -> crate::dense::types::RlstResult<()> {
        assert_eq!(rhs.stride()[0], 1);
        assert_eq!(x.stride()[0], 1);

        let sys = match trans {
            TransMode::NoTrans => umfpack::raw::UMFPACK_A,
            TransMode::Trans => umfpack::raw::UMFPACK_Aat,
            TransMode::ConjTrans => umfpack::raw::UMFPACK_At,
            _ => panic!("Transpose mode not supported for Umfpack"),
        };

        let n = self.shape[0];
        assert_eq!(x.shape()[0], n);
        assert_eq!(rhs.shape()[0], n);

        let info = unsafe {
            umfpack::raw::umfpack_zi_solve(
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
    /// Compute the sparse LU decomposition of the matrix.
    ///
    /// Note that the sparse matrix is first converted to
    /// CSC format before computing the LU.
    pub fn into_lu(self) -> RlstResult<UmfpackLu<f64>> {
        self.into_csc().into_lu()
    }
}

impl CsrMatrix<c64> {
    /// Compute the sparse LU decomposition of the matrix.
    ///
    /// Note that the sparse matrix is first converted to
    /// CSC format before computing the LU.
    pub fn into_lu(self) -> RlstResult<UmfpackLu<c64>> {
        self.into_csc().into_lu()
    }
}

impl CscMatrix<f64> {
    /// Compute the sparse LU decomposition of the matrix.
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
            umfpack::raw::umfpack_di_symbolic(
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
            umfpack::raw::umfpack_di_numeric(
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

        unsafe { umfpack::raw::umfpack_di_free_symbolic(&mut symbolic) };

        Ok(umfpack_lu)
    }
}

impl CscMatrix<c64> {
    /// Compute the sparse LU decomposition of the matrix.
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
            umfpack::raw::umfpack_zi_symbolic(
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
            umfpack::raw::umfpack_zi_numeric(
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

        unsafe { umfpack::raw::umfpack_di_free_symbolic(&mut symbolic) };

        Ok(umfpack_lu)
    }
}
