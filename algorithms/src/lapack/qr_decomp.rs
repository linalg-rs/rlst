use std::cmp;

use crate::traits::qr_decomp::QRDecomp;
use crate::{lapack::LapackData, traits::lu_decomp::LUDecomp};
use lapacke;
use rlst_common::types::{c32, c64, IndexType, RlstError, RlstResult, Scalar};
use rlst_dense::{
    DataContainerMut, GenericBaseMatrixMut, Layout, LayoutType, MatrixTraitMut, SizeIdentifier,
};

use super::{check_lapack_stride, TransposeMode, SideMode, TriangularType, TriangularDiagonal};

pub struct QRDecompLapack<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixTraitMut<Item, RS, CS> + Sized,
> {
    data: LapackData<Item, RS, CS, Mat>,
    tau: Vec<f64>,
}

impl<RS: SizeIdentifier, CS: SizeIdentifier, Data: DataContainerMut<Item = f64>>
    LapackData<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>
{
    pub fn qr(
        mut self,
    ) -> RlstResult<QRDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>> {
        let dim = self.mat.layout().dim();
        let stride = self.mat.layout().stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;
        let lda = stride.1 as i32;
        let mut tau: Vec<f64> = vec![0.0; std::cmp::min(dim.0, dim.1)];

        let info = unsafe {
            lapacke::dgeqrf(
                lapacke::Layout::ColumnMajor,
                m,
                n,
                self.mat.data_mut(),
                lda,
                &mut tau,
            )
        };

        if info == 0 {
            return Ok(QRDecompLapack { data: self, tau });
        } else {
            return Err(RlstError::LapackError(info));
        }
    }

    fn solve<
        RhsData: DataContainerMut<Item = f64>,
        RhsR: SizeIdentifier,
        RhsC: SizeIdentifier,
    >(
        &mut self,
        rhs: &mut GenericBaseMatrixMut<f64, RhsData, RhsR, RhsC>,
        trans: TransposeMode,
    ) -> RlstResult<()> {
        if !check_lapack_stride(rhs.layout().dim(), rhs.layout().stride()) {
            return Err(RlstError::IncompatibleStride);
        } else {
            // let mat = &self.data.mat;
            let dim = self.mat.layout().dim();
            let stride = self.mat.layout().stride();

            let m = dim.0 as i32;
            let n = dim.1 as i32;
            let lda = stride.1 as i32;
            let ldb = rhs.layout().stride().1;
            let nrhs = rhs.dim().1;

            let info = unsafe {
                lapacke::dgels(
                    lapacke::Layout::ColumnMajor,
                    trans as u8,
                    m,
                    n,
                    nrhs as i32,
                    self.mat.data_mut(),
                    lda,
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

impl<Data: DataContainerMut<Item = f64>, RS: SizeIdentifier, CS: SizeIdentifier> QRDecomp
    for QRDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>
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
        RhsR: SizeIdentifier,
        RhsC: SizeIdentifier,
    >(
        &mut self,
        rhs: &mut GenericBaseMatrixMut<Self::T, RhsData, RhsR, RhsC>,
        trans: TransposeMode,
    ) -> RlstResult<()> {
        if !check_lapack_stride(rhs.layout().dim(), rhs.layout().stride()) {
            return Err(RlstError::IncompatibleStride);
        } else {
            // let mat = &self.data.mat;
            let dim = self.data.mat.layout().dim();
            let stride = self.data.mat.layout().stride();

            let m = dim.0 as i32;
            let n = dim.1 as i32;
            let lda = stride.1 as i32;
            let ldb = rhs.layout().stride().1 as i32;
            let nrhs = rhs.dim().1 as i32;

            let info = unsafe {
                lapacke::dormqr(
                    lapacke::Layout::ColumnMajor,
                    SideMode::Left as u8,
                    trans as u8,
                    m,
                    n,
                    1,
                    self.data(),
                    lda,
                    &self.tau,
                    rhs.data_mut(),
                    ldb
                )
            };



            if info != 0 {
                return Err(RlstError::LapackError(info));
            }
            let info = unsafe{
                lapacke::dtrtrs(
                    lapacke::Layout::ColumnMajor,
                    TriangularType::Upper as u8,
                    trans as u8,
                    TriangularDiagonal::NonUnit as u8,
                    n,
                    nrhs,
                    self.data(),
                    lda,
                    rhs.data_mut(),
                    ldb
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
