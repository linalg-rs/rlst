//! Interface to Lapack routines.
pub mod lu_decomp;
pub mod qr_decomp;
pub mod svd;
pub mod triangular_solve;
pub use crate::linalg::DenseMatrixLinAlgBuilder;
pub use lapacke::Layout;
pub use rlst_common::types::{RlstError, RlstResult};
pub mod cholesky_decomp;
pub mod evd;
pub mod inverse;
pub use rlst_dense::traits::*;
pub use rlst_dense::{
    DataContainer, GenericBaseMatrix, Matrix, MatrixD, MatrixImplTrait, SizeIdentifier,
};

// // pub trait LapackCompatible: RawAccessMut + Shape + Stride + Sized {}

// // impl<
// //         Obj: RawAccessMut + Shape + Stride, //+ std::ops::IndexMut<[usize; 2], Output = <Self as RawAccess>::T>,
// //     > LapackCompatible for Obj
// // {
// // }

// /// A simple container to take ownership of a matrix for Lapack operations.
// pub struct LapackDataOwned<T: Scalar, Mat: RawAccessMut<T = T> + Shape + Stride> {
//     /// The matrix on which to perform a Lapack operation.
//     pub mat: Mat,
//     /// The Lapack LDA parameter, which is the distance from one column to the next in memory.
//     pub lda: i32,
// }

// //impl<T: Scalar, Mat: MatrixImplTrait<T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier> LinAlg

// /// A simple container that borrows a matrix for Lapack operations.
// pub struct LapackDataBorrowed<
//     'a,
//     T: Scalar,
//     Data: DataContainer<Item = T>,
//     RS: SizeIdentifier,
//     CS: SizeIdentifier,
// > {
//     /// The matrix on which to perform a Lapack operation.
//     pub mat: &'a GenericBaseMatrix<T, Data, RS, CS>,
//     /// The Lapack LDA parameter, which is the distance from one column to the next in memory.
//     pub lda: i32,
// }

// /// Return true if a given stride is Lapack compatible. Otherwise, return false.
// pub fn check_lapack_stride(dim: (usize, usize), stride: (usize, usize)) -> bool {
//     stride.0 == 1 && stride.1 >= std::cmp::max(1, dim.0)
// }

// impl<
//         'a,
//         T: Scalar,
//         MatImpl: MatrixImplTrait<T, RS, CS>,
//         RS: SizeIdentifier,
//         CS: SizeIdentifier,
//     > DenseMatrixLinAlgBuilder<'a, T, Matrix<T, MatImpl, RS, CS>>
// {
//     /// Take ownership of a matrix and check that its layout is compatible with Lapack.
//     pub fn into_lapack(self) -> RlstResult<LapackDataOwned<T, MatrixD<T>>> {
//         let mut copied = rlst_dense::rlst_mat![T, self.mat.shape()];
//         for (elem, old) in copied.iter_col_major_mut().zip(self.mat.iter_col_major()) {
//             *elem = old;
//         }
//         Ok(LapackDataOwned {
//             mat: copied,
//             lda: copied.shape().0 as i32,
//         })
//     }
// }

// impl<'a, T: Scalar, Data: DataContainer<Item = T>, RS: SizeIdentifier, CS: SizeIdentifier>
//     DenseMatrixLinAlgBuilder<'a, T, GenericBaseMatrix<T, Data, RS, CS>>
// {
//     /// Take ownership of a matrix and check that its layout is compatible with Lapack.
//     pub fn borrow_lapack(&'a self) -> RlstResult<LapackDataBorrowed<'a, T, Data, RS, CS>> {
//         let shape = self.mat.shape();
//         if check_lapack_stride(shape, self.mat.stride()) {
//             Ok(LapackDataBorrowed {
//                 mat: self.mat,
//                 lda: shape.0 as i32,
//             })
//         } else {
//             Err(RlstError::IncompatibleStride)
//         }
//     }
// }

// //impl<Mat: RawAccessMut + Shape + Stride> AsLapack for Mat {}
