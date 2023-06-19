//! Interface to Lapack routines.
pub mod lu_decomp;
pub mod qr_decomp;
pub mod svd;
pub mod trisolve;
pub use crate::linalg::DenseMatrixLinAlgBuilder;
pub use lapacke::Layout;
use rlst_common::traits::*;
pub use rlst_common::types::{RlstError, RlstResult};

// pub trait LapackCompatible: RawAccessMut + Shape + Stride + Sized {}

// impl<
//         Obj: RawAccessMut + Shape + Stride, //+ std::ops::IndexMut<[usize; 2], Output = <Self as RawAccess>::T>,
//     > LapackCompatible for Obj
// {
// }

/// Transposition mode for Lapack.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum TransposeMode {
    /// No transpose
    NoTrans = b'N',
    /// Transpose
    Trans = b'T',
    /// Conjugate Transpose
    ConjugateTrans = b'C',
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SideMode {
    Left = b'L',
    Right = b'R',
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum TriangularType {
    Upper = b'U',
    Lower = b'L',
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TriangularDiagonal {
    Unit = b'U',
    NonUnit = b'N',
}
/// A simple container to take ownership of a matrix for Lapack operations.
pub struct LapackDataOwned<T: Scalar, Mat: RawAccessMut<T = T> + Shape + Stride> {
    /// The matrix on which to perform a Lapack operation.
    pub mat: Mat,
    /// The Lapack LDA parameter, which is the distance from one column to the next in memory.
    pub lda: i32,
}

/// A simple container that borrows a matrix for Lapack operations.
pub struct LapackDataBorrowed<'a, T: Scalar, Mat: RawAccess<T = T> + Shape + Stride> {
    /// The matrix on which to perform a Lapack operation.
    pub mat: &'a Mat,
    /// The Lapack LDA parameter, which is the distance from one column to the next in memory.
    pub lda: i32,
}

/// Return true if a given stride is Lapack compatible. Otherwise, return false.
pub fn check_lapack_stride(dim: (usize, usize), stride: (usize, usize)) -> bool {
    stride.0 == 1 && stride.1 >= std::cmp::max(1, dim.0)
}

impl<'a, Mat: Copy> DenseMatrixLinAlgBuilder<'a, <<Mat as Copy>::Out as RawAccess>::T, Mat>
where
    <Mat as Copy>::Out: RawAccessMut + Shape + Stride,
{
    /// Take ownership of a matrix and check that its layout is compatible with Lapack.
    pub fn into_lapack(
        self,
    ) -> RlstResult<LapackDataOwned<<<Mat as Copy>::Out as RawAccess>::T, <Mat as Copy>::Out>> {
        let copied = self.mat.copy();
        let shape = copied.shape();
        if check_lapack_stride(shape, copied.stride()) {
            Ok(LapackDataOwned {
                mat: copied,
                lda: shape.0 as i32,
            })
        } else {
            Err(RlstError::IncompatibleStride)
        }
    }
}

impl<'a, T: Scalar, Mat: RawAccess<T = T> + Shape + Stride> DenseMatrixLinAlgBuilder<'a, T, Mat> {
    /// Take ownership of a matrix and check that its layout is compatible with Lapack.
    pub fn borrow_lapack(&'a self) -> RlstResult<LapackDataBorrowed<'a, T, Mat>> {
        let shape = self.mat.shape();
        if check_lapack_stride(shape, self.mat.stride()) {
            Ok(LapackDataBorrowed {
                mat: self.mat,
                lda: shape.0 as i32,
            })
        } else {
            Err(RlstError::IncompatibleStride)
        }
    }
}

//impl<Mat: RawAccessMut + Shape + Stride> AsLapack for Mat {}
