use crate::lapack::LapackData;
use crate::traits::lu_decomp::LUDecomp;
use lapacke;
use rlst_common::types::{IndexType, RlstError, RlstResult, Scalar};
use rlst_dense::{
    DataContainerMut, DefaultLayout, GenericBaseMatrixMut, Layout, LayoutType, MatrixD,
    MatrixTrait, MatrixTraitMut, SizeIdentifier, UnsafeRandomAccessMut, VectorContainer,
};

use super::{check_lapack_stride, TransposeMode};

pub struct LUDecompLapack<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixTraitMut<Item, RS, CS> + Sized,
> {
    data: LapackData<Item, RS, CS, Mat>,
    ipiv: Vec<i32>,
}

impl<
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainerMut<Item = f64>,
        //Mat: MatrixTraitMut<Item, RS, CS> + Sized,
    > LapackData<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>
{
    pub fn lu(
        mut self,
    ) -> RlstResult<LUDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>> {
        let dim = self.mat.layout().dim();
        let stride = self.mat.layout().stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;
        let lda = stride.1 as i32;

        let mut ipiv: Vec<i32> = vec![0; std::cmp::min(dim.0, dim.1)];
        let info = unsafe {
            lapacke::dgetrf(
                lapacke::Layout::ColumnMajor,
                m,
                n,
                self.mat.data_mut(),
                lda,
                &mut ipiv,
            )
        };
        if info == 0 {
            return Ok(LUDecompLapack { data: self, ipiv });
        } else {
            return Err(RlstError::LapackError(info));
        }
    }
}

impl<Data: DataContainerMut<Item = f64>, RS: SizeIdentifier, CS: SizeIdentifier> LUDecomp
    for LUDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>
{
    type T = f64;

    fn data(&self) -> &[Self::T] {
        self.data.mat.data()
    }

    fn dim(&self) -> (IndexType, IndexType) {
        self.data.mat.dim()
    }

    fn solve<
        RhsData: DataContainerMut<Item = Self::T>,
        Rhs_rs: SizeIdentifier,
        Rhs_cs: SizeIdentifier,
    >(
        &self,
        rhs: &mut GenericBaseMatrixMut<Self::T, RhsData, Rhs_rs, Rhs_cs>,
        trans: TransposeMode,
    ) -> RlstResult<()> {
        if !check_lapack_stride(rhs.layout().dim(), rhs.layout().stride()) {
            return Err(RlstError::IncompatibleStride);
        } else {
            let mat = &self.data.mat;
            let ldb = rhs.layout().stride().1;

            let info = unsafe {
                lapacke::dgetrs(
                    lapacke::Layout::ColumnMajor,
                    trans as u8,
                    mat.layout().dim().1 as i32,
                    mat.layout().dim().1 as i32,
                    mat.data(),
                    mat.layout().stride().1 as i32,
                    &self.ipiv,
                    rhs.data_mut(),
                    ldb as i32,
                )
            };

            if info != 0 {
                return Err(RlstError::LapackError(info));
            } else {
                Ok(())
            }
        }
    }
}

// impl<'a, ContainerImpl: DenseContainerInterfaceMut<T = f64>> LUDecompLapack<ContainerImpl> {
//     pub fn new(mut data: AsLapack<ContainerImpl>) -> RlstResult<Self> {
//         let dim = data.dim();
//         let stride = data.stride();

//         let m = dim.0 as i32;
//         let n = dim.1 as i32;

//         let mut ipiv: Vec<i32> = vec![0; std::cmp::min(dim.0, dim.1)];
//         if let Ok((layout, lda)) = get_lapack_layout(stride) {
//             let info = unsafe { lapacke::dgetrf(layout, m, n, data.data_mut(), lda, &mut ipiv) };
//             if info == 0 {
//                 return Ok(Self {
//                     data,
//                     lda,
//                     dim,
//                     ipiv,
//                 });
//             } else {
//                 return Err(RlstError::LapackError(info));
//             }
//         } else {
//             Err(RlstError::IncompatibleStride)
//         }
//     }
// }

// impl<ContainerImpl: DenseContainerInterfaceMut<T = f64>> LUDecomp
//     for LUDecompLapack<ContainerImpl>
// {
//     type T = ContainerImpl::T;
//     type ContainerImpl = ContainerImpl;

//     fn data(&self) -> &[Self::T] {
//         self.data.data()
//     }

//     fn dim(&self) -> (rlst_common::types::IndexType, rlst_common::types::IndexType) {
//         self.dim
//     }

//     fn solve<VecImpl: DenseContainerInterfaceMut<T = Self::T>>(
//         &self,
//         rhs: &mut DenseContainer<VecImpl>,
//         trans: TransposeMode,
//     ) -> RlstResult<()> {
//         let rhs_stride = rhs.stride();

//         if let Ok((rhs_layout, ldb)) = get_lapack_layout(rhs_stride) {
//             println!("lda: {}", self.lda);
//             println!("ldb: {}", ldb);
//             println!("rhs layout {:?}", rhs_layout);
//             println!("n: {}", self.dim.1);
//             println!("nrhs: {}", rhs.dim().1);
//             println!("ipiv: {}, {}", self.ipiv[0], self.ipiv[1]);
//             println!("a: {}, {}", self.data()[0], self.data()[1]);
//             println!("a: {}, {}", self.data()[2], self.data()[3]);
//             println!("rhs: {}, {}", rhs.data()[0], rhs.data()[1]);
//             let t = trans as u8;
//             let t = t as char;

//             println!("trans {:?}", trans);

//             let info = unsafe {
//                 lapacke::dgetrs(
//                     lapacke::Layout::RowMajor,
//                     trans as u8,
//                     self.dim.1 as i32,
//                     rhs.dim().1 as i32,
//                     self.data(),
//                     self.lda,
//                     &self.ipiv,
//                     rhs.data_mut(),
//                     1,
//                 )
//             };

//             println!("x: {}, {}", rhs.data()[0], rhs.data()[1]);

//             if info == 0 {
//                 return Ok(());
//             } else {
//                 return Err(RlstError::LapackError(info));
//             }
//         } else {
//             Err(RlstError::IncompatibleStride)
//         }
//     }
// }

#[cfg(test)]
use super::*;
use rlst_dense::matrix::*;
use rlst_dense::{rlst_mat, rlst_rand_mat, rlst_rand_vec, rlst_vec, Dot};

#[test]
fn test_lu_decomp_f64() {
    let mut rlst_mat = rlst_mat![f64, (2, 2)];
    let mut rlst_vec = rlst_vec![f64, 2];

    println!(
        "Stride: {}, {}",
        rlst_mat.layout().stride().0,
        rlst_mat.layout().stride().1
    );

    rlst_mat[[0, 0]] = 1.0;
    rlst_mat[[0, 1]] = 1.0;
    rlst_mat[[1, 0]] = 3.0;
    rlst_mat[[1, 1]] = 1.0;

    rlst_vec[[0, 0]] = 2.3;
    rlst_vec[[1, 0]] = 7.1;

    let mut rhs = rlst_mat.dot(&rlst_vec);

    let info = rlst_mat
        .lapack()
        .unwrap()
        .lu()
        .unwrap()
        .solve(&mut rhs, TransposeMode::NoTrans);

    let x = rhs;

    let diff = (&x - &rlst_vec).eval();

    println!("Sol: {}, {}", x[[0, 0]], x[[1, 0]]);

    //let lu_decomp = rlst_mat.algorithms().lapack().lu();
}
