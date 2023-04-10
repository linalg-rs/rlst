use crate::lapack::LapackData;
use crate::traits::lu_decomp::LUDecomp;
use lapacke;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{
    DataContainerMut, GenericBaseMatrixMut, Layout, LayoutType, MatrixTraitMut, SizeIdentifier,
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

macro_rules! lu_decomp_impl {
    ($scalar:ty, $lapack_getrf:ident, $lapack_getrs:ident) => {
        impl<
                RS: SizeIdentifier,
                CS: SizeIdentifier,
                Data: DataContainerMut<Item = $scalar>,
                //Mat: MatrixTraitMut<Item, RS, CS> + Sized,
            > LapackData<$scalar, RS, CS, GenericBaseMatrixMut<$scalar, Data, RS, CS>>
        {
            pub fn lu(
                mut self,
            ) -> RlstResult<
                LUDecompLapack<$scalar, RS, CS, GenericBaseMatrixMut<$scalar, Data, RS, CS>>,
            > {
                let dim = self.mat.layout().dim();
                let stride = self.mat.layout().stride();

                let m = dim.0 as i32;
                let n = dim.1 as i32;
                let lda = stride.1 as i32;

                let mut ipiv: Vec<i32> = vec![0; std::cmp::min(dim.0, dim.1)];
                let info = unsafe {
                    lapacke::$lapack_getrf(
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

        impl<Data: DataContainerMut<Item = $scalar>, RS: SizeIdentifier, CS: SizeIdentifier>
            LUDecomp
            for LUDecompLapack<$scalar, RS, CS, GenericBaseMatrixMut<$scalar, Data, RS, CS>>
        {
            type T = $scalar;

            fn data(&self) -> &[Self::T] {
                self.data.mat.data()
            }

            fn dim(&self) -> (usize, usize) {
                self.data.mat.dim()
            }

            fn solve<
                RhsData: DataContainerMut<Item = Self::T>,
                RhsR: SizeIdentifier,
                RhsC: SizeIdentifier,
            >(
                &self,
                rhs: &mut GenericBaseMatrixMut<Self::T, RhsData, RhsR, RhsC>,
                trans: TransposeMode,
            ) -> RlstResult<()> {
                if !check_lapack_stride(rhs.layout().dim(), rhs.layout().stride()) {
                    return Err(RlstError::IncompatibleStride);
                } else {
                    let mat = &self.data.mat;
                    let ldb = rhs.layout().stride().1;

                    let info = unsafe {
                        lapacke::$lapack_getrs(
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
    };
}

lu_decomp_impl!(f64, dgetrf, dgetrs);
lu_decomp_impl!(f32, sgetrf, sgetrs);
lu_decomp_impl!(c32, cgetrf, cgetrs);
lu_decomp_impl!(c64, zgetrf, zgetrs);

#[cfg(test)]
use super::*;

#[test]
fn test_lu_decomp_f64() {
    use rlst_dense::Dot;

    let mut rlst_mat = rlst_dense::rlst_mat![f64, (2, 2)];
    let mut rlst_vec = rlst_dense::rlst_vec![f64, 2];

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

    let _ = rlst_mat
        .lapack()
        .unwrap()
        .lu()
        .unwrap()
        .solve(&mut rhs, TransposeMode::NoTrans);

    let x = rhs;

    println!("Sol: {}, {}", x[[0, 0]], x[[1, 0]]);

    //let lu_decomp = rlst_mat.algorithms().lapack().lu();
}
