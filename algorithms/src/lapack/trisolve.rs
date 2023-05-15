use rlst_common::types::{c32, c64,RlstResult, RlstError};
use rlst_dense::{RawAccessMut, Shape, Stride, RawAccess};

use crate::traits::trisolve_trait::Trisolve;
use rlst_common::traits::Copy;

use super::{DenseMatrixLinAlgBuilder, TriangularType, TriangularDiagonal, TransposeMode, check_lapack_stride};

macro_rules! trisolve_impl {
    ($scalar:ty, $lapack_trisolve:ident) => {
        impl<'a,Mat: Copy> Trisolve for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where <Mat as Copy>::Out: RawAccessMut<T=$scalar>+Shape+Stride {
            type T = $scalar;
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
                            lapacke::$lapack_trisolve(
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
    }
}

trisolve_impl!(f32,strtrs);
trisolve_impl!(f64,dtrtrs);
trisolve_impl!(c32,ctrtrs);
trisolve_impl!(c64,ztrtrs);

#[cfg(test)]
mod test {
    use num::Zero;
    use rlst_dense::{rlst_rand_mat, rlst_rand_col_vec, Dot, DataContainerMut, SizeIdentifier, GenericBaseMatrix};

    use crate::{linalg::LinAlg, assert_approx_matrices};
    use approx::{assert_abs_diff_eq,AbsDiffEq};
    use super::*;

    macro_rules! test_trisolve {
        ($scalar:ty, $name:ident) => {
            #[test]
            fn $name() {
                let mut mat_a = rlst_rand_mat![$scalar,(4,4)];
                for row in 0..mat_a.shape().0 {
                    for col in 0..row{
                        mat_a[[row,col]] = <$scalar as Zero>::zero();
                    }
                }
                let exp_sol = rlst_rand_col_vec![$scalar,4];
                let mut actual_sol = mat_a.dot(&exp_sol);
                actual_sol = mat_a.linalg().trisolve(
                    actual_sol,
                    TriangularType::Upper, 
                    TriangularDiagonal::NonUnit, 
                    TransposeMode::NoTrans).unwrap();
                
                assert_approx_matrices!(&exp_sol,&actual_sol, 1000.*<$scalar as AbsDiffEq>::default_epsilon());
            }
        }
    }

    test_trisolve!(f32, test_trisolve_f32);
    test_trisolve!(f64, test_trisolve_f64);
    test_trisolve!(c32, test_trisolve_c32);
    test_trisolve!(c64, test_trisolve_c64);
}