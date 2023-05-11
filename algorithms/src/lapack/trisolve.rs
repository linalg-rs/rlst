use rlst_common::types::{RlstResult, RlstError};
use rlst_dense::{RawAccessMut, Shape, Stride, RawAccess};

use crate::traits::trisolve_trait::Trisolve;
use rlst_common::traits::Copy;

use super::{DenseMatrixLinAlgBuilder, TriangularType, TriangularDiagonal, TransposeMode, check_lapack_stride};

impl<'a,Mat: Copy> Trisolve for DenseMatrixLinAlgBuilder<'a, f64, Mat>
where <Mat as Copy>::Out: RawAccessMut<T=f64>+Shape+Stride {
    type T = f64;
    fn trisolve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
            self,
            mut rhs: Rhs,
            tritype: TriangularType,
            tridiag: TriangularDiagonal,
            trans: TransposeMode
        ) -> RlstResult<Rhs> 
        {
            if !check_lapack_stride(rhs.shape(), rhs.stride()) {
                return Err(RlstError::IncompatibleStride);
            } else {
                let copied = self.into_lapack()?;
                // let m = self.mat.shape().0;
                let n = copied.mat.shape().1 as i32;
                let lda = copied.mat.stride().1 as i32;
                let nrhs = rhs.shape().1 as i32;
                let ldb = rhs.stride().1 as i32;
                let info = unsafe {
                    lapacke::dtrtrs(
                        lapacke::Layout::ColumnMajor,
                        tritype as u8,
                        trans as u8,
                        tridiag as u8,
                        n,
                        nrhs,
                        copied.mat.data(),
                        lda,
                        rhs.data_mut(),
                        ldb,
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                } else {
                    Ok(rhs)
                }
            }
    }
}

#[cfg(test)]
mod test {
    use cauchy::Scalar;
    use rlst_dense::{rlst_rand_mat, rlst_rand_col_vec, Dot, DataContainerMut, SizeIdentifier, GenericBaseMatrix};

    use crate::{linalg::LinAlg, assert_approx_matrices};
    use approx::assert_abs_diff_eq;
    use super::*;
    
    #[test]
    fn test_trisolve() {
        let mut mat_a = rlst_rand_mat![f64,(4,4)];
        for row in 0..mat_a.shape().0 {
            for col in 0..row{
                mat_a[[row,col]] = 0.;
            }
        }
        let exp_sol = rlst_rand_col_vec![f64,4];
        let mut actual_sol = mat_a.dot(&exp_sol);
        actual_sol = mat_a.linalg().trisolve(
            actual_sol,
            TriangularType::Upper, 
            TriangularDiagonal::NonUnit, 
            TransposeMode::NoTrans).unwrap();
        
        assert_approx_matrices!(&exp_sol,&actual_sol);
    }
}