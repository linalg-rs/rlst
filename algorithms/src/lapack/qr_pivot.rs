
pub struct QRPivotDecompLapack<
T: Scalar,
Mat: RawAccessMut<T = T> + Shape + Stride,
> {
    data: LapackData<T, Mat>,
    tau: Vec<T>,
    jpvt: Vec<i32>,
}

impl<'a, Mat: Copy> QRDecomposableTrait for DenseMatrixLinAlgBuilder<'a, f64, Mat>
where
    <Mat as Copy>::Out: RawAccessMut<T=f64> + Shape + Stride,
{
///// Returns the QR decomposition of the input matrix using column pivoting (LAPACK xGEQP3).
    ///// # Arguments
    ///// * `jpvt'
    /////    On Input:
    /////     - If jpvt(j) ≠ 0, the j-th column of a is permuted to the front of AP (a leading column);
    /////     - If jpvt(j) = 0, the j-th column of a is a free column.
    /////     - Specified as: a one-dimensional array of dimension max(1, n);
    /////    On Output:
    /////     - If jpvt(j) = k, then the j-th column of AP was the k-th column of a.
    /////     - Specified as: a one-dimensional array of dimension max(1, n);
    // fn qr_col_pivot_jpvt(
    //     self,
    //     mut jpvt: Vec<i32>,
    // ) -> RlstResult<Self::Out> {
    //     let mut copied = self.into_lapack()?;
    //     let dim = copied.mat.shape();
    //     let stride = copied.mat.stride();

    //     let m = dim.0 as i32;
    //     let n = dim.1 as i32;
    //     let lda = stride.1 as i32;
    //     let mut tau: Vec<f64> = vec![0.0; std::cmp::min(dim.0, dim.1)];

    //     let info = unsafe {
    //         lapacke::dgeqp3(
    //             lapacke::Layout::ColumnMajor,
    //             m,
    //             n,
    //             copied.mat.data_mut(),
    //             lda,
    //             &mut jpvt,
    //             &mut tau,
    //         )
    //     };

    //     if info == 0 {
    //         return Ok(QRPivotDecompLapack {
    //             data: copied,
    //             tau: tau,
    //             jpvt: jpvt,
    //         });
    //     } else {
    //         return Err(RlstError::LapackError(info));
    //     }
    // }

    // pub fn qr_col_pivot(
    //     mut self,
    // ) -> RlstResult<QRPivotDecompLapack<f64, MatrixD<f64>>> {
    //     let mut jpvt = vec![0, self.mat.shape().1 as i32];
    //     self.qr_col_pivot_jpvt(jpvt)
    // }
    
    // fn solve_qr_pivot(
    //     &mut self,
    //     rhs: &mut MatrixD<f64>,
    //     trans: TransposeMode,
    // ) -> RlstResult<()> {
    //     if !check_lapack_stride(rhs.shape(), rhs.stride()) {
    //         return Err(RlstError::IncompatibleStride);
    //     } else {
    //         //TODO: add a check for m>n?
    //         // let mat = &self.data.mat;
    //         let dim = self.mat.shape();
    //         let stride = self.mat.stride();

    //         let m = dim.0 as i32;
    //         let n = dim.1 as i32;
    //         let lda = stride.1 as i32;
    //         let ldb = rhs.stride().1 as i32;
    //         let nrhs = rhs.shape().1;

    //         let mut jpvt = vec![0;n as usize];
    //         let rcond = 0.0;
    //         let mut rank = 0;

    //         let info = unsafe {
    //             lapacke::dgelsy(
    //                 lapacke::Layout::ColumnMajor,
    //                 m,
    //                 n,
    //                 nrhs as i32,
    //                 self.mat.data_mut(),
    //                 lda,
    //                 rhs.data_mut(),
    //                 ldb,
    //                 &mut jpvt,
    //                 rcond,
    //                 &mut rank,
    //             )
    //         };

    //         if info != 0 {
    //             return Err(RlstError::LapackError(info));
    //         } else {
    //             Ok(())
    //         }
    //     }
    // }
}


// impl QRDecompTrait
//     for QRPivotDecompLapack<f64, MatrixD<f64>>
// {
//     type T = f64;

//     fn data(&self) -> &[Self::T] {
//         self.data.mat.data()
//     }

//     fn shape(&self) -> (usize, usize) {
//         self.data.mat.shape()
//     }

//     /// Returns Q*RHS
//     fn q_x_rhs<
//         RhsData: DataContainerMut<Item = Self::T>,
//         RhsR: SizeIdentifier,
//         RhsC: SizeIdentifier,
//     >(
//         &mut self,
//         rhs: &mut GenericBaseMatrixMut<Self::T, RhsData, RhsR, RhsC>,
//         trans: TransposeMode,
//     ) -> RlstResult<()> {
//         if !check_lapack_stride(rhs.shape(), rhs.stride()) {
//             return Err(RlstError::IncompatibleStride);
//         } else {
//             let rhs_dim = rhs.shape();

//             let m = rhs_dim.0 as i32;
//             let n = rhs_dim.1 as i32;
//             let lda = self.data.mat.stride().1 as i32;
//             let ldc = rhs.stride().1 as i32;

//             let info = unsafe {
//                 lapacke::dormqr(
//                     lapacke::Layout::ColumnMajor,
//                     SideMode::Left as u8,
//                     trans as u8,
//                     m,
//                     n,
//                     self.data.mat.shape().1 as i32,
//                     self.data(),
//                     lda,
//                     &self.tau,
//                     rhs.data_mut(),
//                     ldc,
//                 )
//             };
//             if info != 0 {
//                 return Err(RlstError::LapackError(info));
//             }
//             Ok(())
//         }
//     }

//     fn solve(
//         &mut self,
//         rhs: &mut MatrixD<Self::T>,
//         trans: TransposeMode,
//     ) -> RlstResult<()> {
//         if !check_lapack_stride(rhs.shape(), rhs.stride()) {
//             return Err(RlstError::IncompatibleStride);
//         } else {
//             // // let mat = &self.data.mat;
//             // let dim = self.data.mat.shape();
//             // let stride = self.data.mat.stride();

//             // let m = dim.0 as i32;
//             // let n = dim.1 as i32;
//             // let lda = stride.1 as i32;
//             // let ldb = rhs.stride().1 as i32;
//             // let nrhs = rhs.shape().1 as i32;

//             // self.q_x_rhs(rhs, trans).unwrap();

//             // let info = unsafe {
//             //     lapacke::dtrtrs(
//             //         lapacke::Layout::ColumnMajor,
//             //         TriangularType::Upper as u8,
//             //         trans as u8,
//             //         TriangularDiagonal::NonUnit as u8,
//             //         n,
//             //         nrhs,
//             //         self.data(),
//             //         lda,
//             //         rhs.data_mut(),
//             //         ldb,
//             //     )
//             // };
//             //TODO: Implement
//             let info = 1;
//             if info != 0 {
//                 return Err(RlstError::LapackError(info));
//             } else {
//                 Ok(())
//             }
//         }
//     }
// }

