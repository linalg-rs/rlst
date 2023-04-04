use std::cmp;

use crate::traits::qr_decomp_trait::QRDecompTrait;
use crate::{lapack::LapackData, traits::lu_decomp::LUDecomp};
use lapacke;
use rlst_common::types::{c32, c64, IndexType, RlstError, RlstResult, Scalar};
use rlst_dense::{
    DataContainerMut, GenericBaseMatrixMut, Layout, LayoutType, MatrixTraitMut, SizeIdentifier, GenericBaseMatrix,
};

use super::{check_lapack_stride, TransposeMode, SideMode, TriangularType, TriangularDiagonal, AsLapack};

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

    fn solve_qr<
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

impl<Data: DataContainerMut<Item = f64>, RS: SizeIdentifier, CS: SizeIdentifier> QRDecompTrait
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


#[test]
fn test_qr_solve_f64()
{
    let mut rlst_mat = rlst_dense::rlst_mat![f64,(4,3)];
    let mut rlst_vec = rlst_dense::rlst_vec![f64,4];

    rlst_mat[[0, 0]]  =  1.0; rlst_mat[[0, 1]]  =  2.0; rlst_mat[[0, 2]]  =  3.0;
    rlst_mat[[1, 0]]  = -3.0; rlst_mat[[1, 1]]  =  2.0; rlst_mat[[1, 2]]  =  1.0;
    rlst_mat[[2, 0]]  =  2.0; rlst_mat[[2, 1]]  =  0.0; rlst_mat[[2, 2]]  = -1.0;
    rlst_mat[[3, 0]]  =  3.0; rlst_mat[[3, 1]]  = -1.0; rlst_mat[[3, 2]]  =  2.0;

    rlst_vec[[0,0]] = 2.;
    rlst_vec[[1,0]] = 4.;
    rlst_vec[[2,0]] = 6.;
    rlst_vec[[3,0]] = 8.;

    let _ = rlst_mat
        .lapack()
        .unwrap()
        .solve_qr(&mut rlst_vec, TransposeMode::NoTrans);

    print_matrix(rlst_vec)
}

#[test]
fn test_qr_decomp()
{
    let mut rlst_mat = rlst_dense::rlst_mat![f64,(4,3)];
    let mut rlst_vec = rlst_dense::rlst_vec![f64,4];

    rlst_mat[[0, 0]]  =  1.0; rlst_mat[[0, 1]]  =  2.0; rlst_mat[[0, 2]]  =  3.0;
    rlst_mat[[1, 0]]  = -3.0; rlst_mat[[1, 1]]  =  2.0; rlst_mat[[1, 2]]  =  1.0;
    rlst_mat[[2, 0]]  =  2.0; rlst_mat[[2, 1]]  =  0.0; rlst_mat[[2, 2]]  = -1.0;
    rlst_mat[[3, 0]]  =  3.0; rlst_mat[[3, 1]]  = -1.0; rlst_mat[[3, 2]]  =  2.0;

    let _ = rlst_mat
        .lapack()
        .unwrap()
        .qr()
        .unwrap();
    //Q
    /*
⌈	0.209	0.879	0.156	⌉
|	-0.626	0.415	0.146	｜
|	0.417	0.232	-0.767	｜
⌊	0.626	-0.0332	0.605	⌋ */
    //R
    /*
    4.8	-1.46	0.834	⌉
|	0	2.62	2.75	｜
⌊	0	0	2.59	⌋
    */


}

#[test]
fn test_qr_decomp_and_solve()
{
    let mut rlst_mat = rlst_dense::rlst_mat![f64,(4,3)];
    let mut rlst_vec = rlst_dense::rlst_vec![f64,4];

    rlst_mat[[0, 0]]  =  1.0; rlst_mat[[0, 1]]  =  2.0; rlst_mat[[0, 2]]  =  3.0;
    rlst_mat[[1, 0]]  = -3.0; rlst_mat[[1, 1]]  =  2.0; rlst_mat[[1, 2]]  =  1.0;
    rlst_mat[[2, 0]]  =  2.0; rlst_mat[[2, 1]]  =  0.0; rlst_mat[[2, 2]]  = -1.0;
    rlst_mat[[3, 0]]  =  3.0; rlst_mat[[3, 1]]  = -1.0; rlst_mat[[3, 2]]  =  2.0;

    rlst_vec[[0,0]] = 2.;
    rlst_vec[[1,0]] = 4.;
    rlst_vec[[2,0]] = 6.;
    rlst_vec[[3,0]] = 8.;

    let _ = rlst_mat
        .lapack()
        .unwrap()
        .qr()
        .unwrap()
        .solve(&mut rlst_vec, TransposeMode::NoTrans);

}

fn print_matrix<T:Scalar,Data: DataContainerMut<Item = T>, RS: SizeIdentifier, CS: SizeIdentifier>(matrix: GenericBaseMatrix<T,Data,RS,CS>) {
    for row in 0..matrix.dim().0 {
        for col in 0..matrix.dim().1 {
            print!("{:.3}",matrix[[row,col]]);
        }
        println!();
    }
}