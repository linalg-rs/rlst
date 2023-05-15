use crate::linalg::LinAlg;
use crate::traits::trisolve_trait::Trisolve;
use crate::{lapack::LapackData, traits::qr_decomp_trait::QRDecomposableTrait};
use crate::traits::qr_decomp_trait::QRTrait;
use lapacke;
use num::One;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{Shape, Stride, RawAccessMut, RawAccess};
use rlst_common::traits::Copy;
use rlst_dense::{
    rlst_mat,rlst_col_vec, ColumnVectorD, DataContainerMut, GenericBaseMatrix, MatrixD, Layout,
    LayoutType, SizeIdentifier,
};

use super::{check_lapack_stride, SideMode, TransposeMode, TriangularDiagonal, TriangularType, DenseMatrixLinAlgBuilder};

pub struct QRDecompLapack<
    T: Scalar,
    Mat: RawAccessMut<T = T> + Shape + Stride,
> {
    data: LapackData<T, Mat>,
    tau: Vec<T>,
}


impl<'a, Mat: Copy> QRDecomposableTrait for DenseMatrixLinAlgBuilder<'a, f64, Mat>
where
    <Mat as Copy>::Out: RawAccessMut<T=f64> + Shape + Stride,
{
    type T = f64;
    type Out = QRDecompLapack<f64, <Mat as Copy>::Out>;
    /// Returns the QR decomposition of the input matrix assuming full rank and using LAPACK xGEQRF
    fn qr(self) -> RlstResult<QRDecompLapack<f64,<Mat as Copy>::Out>> {
        let mut copied = self.into_lapack()?;
        let dim = copied.mat.shape();
        let stride = copied.mat.stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;
        let lda = stride.1 as i32;
        let mut tau: Vec<f64> = vec![0.0; std::cmp::min(dim.0, dim.1)];

        let info = unsafe {
            lapacke::dgeqrf(
                lapacke::Layout::ColumnMajor,
                m,
                n,
                copied.mat.data_mut(),
                lda,
                &mut tau,
            )
        };

        if info == 0 {
            return Ok(QRDecompLapack { data: copied, tau });
        } else {
            return Err(RlstError::LapackError(info));
        }
    }

    

    fn qr_and_solve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        self,
        mut rhs: Rhs,
        trans: TransposeMode,
    ) -> RlstResult<(Rhs)> {
        //TODO: add a check for m>n?
        let mut copied = self.into_lapack()?;
        let dim = copied.mat.shape();
        let stride = copied.mat.stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;
        let lda = stride.1 as i32;
        let ldb = rhs.stride().1;
        let nrhs = rhs.shape().1;

        let info = unsafe {
            lapacke::dgels(
                lapacke::Layout::ColumnMajor,
                trans as u8,
                m,
                n,
                nrhs as i32,
                copied.mat.data_mut(),
                lda,
                rhs.data_mut(),
                ldb as i32,
            )
        };

        if info != 0 {
            return Err(RlstError::LapackError(info));
        } else {
            Ok(rhs)
        }
    }
}

impl<
    Mat: RawAccessMut<T = f64>
        + Shape
        + Stride
        + std::ops::Index<[usize; 2], Output = f64>
        + std::ops::IndexMut<[usize; 2], Output = f64>
    > QRTrait
    for QRDecompLapack<f64, Mat>
{
    type T = f64;

    fn data(&self) -> &[Self::T] {
        self.data.mat.data()
    }

    fn shape(&self) -> (usize, usize) {
        self.data.mat.shape()
    }

    fn stride(&self) -> (usize, usize) {
        self.data.mat.stride()
    }

    /// Returns Q*RHS
    fn q_mult<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        &self,
        mut rhs: Rhs,
        trans: TransposeMode,
    ) -> RlstResult<Rhs> {
        if !check_lapack_stride(rhs.shape(), rhs.stride()) {
            return Err(RlstError::IncompatibleStride);
        } else {
            let m = rhs.shape().0 as i32;
            let n = rhs.shape().1 as i32;
            let lda = self.data.mat.stride().1 as i32;
            let ldc = rhs.stride().1 as i32;

            let info = unsafe {
                lapacke::dormqr(
                    lapacke::Layout::ColumnMajor,
                    SideMode::Left as u8,
                    trans as u8,
                    m,
                    n,
                    self.data.mat.shape().1 as i32,
                    self.data(),
                    lda,
                    &self.tau,
                    rhs.data_mut(),
                    ldc,
                )
            };
            if info != 0 {
                return Err(RlstError::LapackError(info));
            }
            Ok(rhs)
        }
    }

    fn get_q(&self) -> RlstResult<MatrixD<Self::T>> {
        let shape = self.shape();
        let mut mat = rlst_mat!(Self::T, (self.shape().0, self.shape().0));

        for index in 0..self.shape().0 {
            mat[[index, index]] = <Self::T as One>::one();
        }
        self.q_mult(mat, TransposeMode::NoTrans)
    }

    fn get_r(&self) -> RlstResult<MatrixD<Self::T>> {
        let shape = self.shape();
        let dim = std::cmp::min(shape.0, shape.1);
        let mut mat = rlst_mat!(Self::T, (shape.0, shape.1));

        for row in 0..dim {
            for col in row..shape.1 {
                mat[[row, col]] = self.data.mat[[row, col]];
            }
        }
        Ok(mat)
    }
    fn solve_qr<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        &self,
        mut rhs: Rhs,
        trans: TransposeMode,
    ) -> RlstResult<Rhs> {
            rhs = self.q_mult(rhs, TransposeMode::Trans)?;
            self.get_r()?.linalg().trisolve(
                rhs, 
                TriangularType::Upper, 
                TriangularDiagonal::NonUnit, 
                trans)
    }

}
#[cfg(test)]
mod test {
    use crate::linalg::LinAlg;

    use super::*;
    use approx::assert_abs_diff_eq;
    use rlst_dense::{
        rlst_mat, rlst_col_vec, Dot
    };

    #[macro_export]
    macro_rules! assert_approx_matrices {
        ($expected_matrix:expr, $actual_matrix:expr) => {{
            assert_eq!($expected_matrix.shape(), $actual_matrix.shape());
            for row in 0..$expected_matrix.shape().0 {
                for col in 0..$expected_matrix.shape().1 {
                    assert_abs_diff_eq!(
                        $actual_matrix[[row, col]],
                        $expected_matrix[[row, col]],
                        epsilon=1000.*f64::EPSILON
                    );
                }
            }
        }};
    }

    macro_rules! test_qr_solve {
        ($scalar:ty, $solver:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let matrix_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut exp_sol = rlst_dense::rlst_rand_col_vec![$scalar, $n];
                let mut rhs = matrix_a.dot(&exp_sol);

                let rhs = matrix_a
                    .linalg()
                    .$solver(rhs, TransposeMode::NoTrans)
                    .unwrap();

                let mut actual_sol = rlst_col_vec!($scalar, $n);
                actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, $n));

                assert_approx_matrices!(&exp_sol, &actual_sol);
            }
        };
    }

    macro_rules! test_q_unitary {
        ($scalar:ty, $qr:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let rlst_mat = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut qr = rlst_mat.linalg().$qr().unwrap();

                let mut expected_i = rlst_mat!($scalar,($m, $m));
                for i in 0..$m {
                    expected_i[[i, i]] = <$scalar as One>::one();
                }

                let mut matrix_q = qr.get_q().unwrap();
                let mut matrix_q_t = rlst_mat![$scalar, (matrix_q.shape().1, matrix_q.shape().0)];
                for row in 0..matrix_q.shape().0 {
                    for col in 0..matrix_q.shape().1 {
                        matrix_q_t[[col, row]] = matrix_q[[row, col]];
                    }
                }

                let actual_i_t = matrix_q_t.dot(&matrix_q);
                assert_approx_matrices!(&expected_i, &actual_i_t);
            }
        };
    }

    macro_rules! test_qr_is_a {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let expected_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut rlst_mat = rlst_dense::rlst_mat![$scalar, expected_a.shape()];
                rlst_mat.data_mut().copy_from_slice(expected_a.data());

                let mut qr = rlst_mat.linalg().$qr_decomp().unwrap();
                let mut matrix_r = qr.get_r().unwrap();
                let mut matrix_q = qr.get_q().unwrap();
                let actual_a = matrix_q.dot(&matrix_r);

                assert_approx_matrices!(expected_a, actual_a);
            }
        };
    }

    macro_rules! test_q_mult_r_is_a {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let expected_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut rlst_mat = rlst_dense::rlst_mat![$scalar, expected_a.shape()];
                rlst_mat.data_mut().copy_from_slice(expected_a.data());

                let mut qr = rlst_mat.linalg().$qr_decomp().unwrap();
                let mut matrix_r = qr.get_r().unwrap();
                let actual_a = qr.q_mult(matrix_r, TransposeMode::NoTrans).unwrap();

                assert_approx_matrices!(&expected_a, &actual_a);
            }
        };
    }

    macro_rules! test_qr_decomp_and_solve {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let matrix_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let exp_sol = rlst_dense::rlst_rand_col_vec![$scalar, $n];
                let mut rhs = matrix_a.dot(&exp_sol);


                let rhs = matrix_a
                    .linalg()
                    .$qr_decomp()
                    .unwrap()
                    .solve_qr(rhs, TransposeMode::NoTrans).unwrap();

                let mut actual_sol = rlst_col_vec!($scalar, $n);
                actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, $n));

                assert_approx_matrices!(&exp_sol, &actual_sol);
            }
        };
    }
    test_qr_solve!(f64, qr_and_solve, test_solve_ls_qr_f64, 4, 3);
    test_q_unitary!(f64, qr, test_q_unitary_f64, 4, 3);
    test_qr_is_a!(f64, qr, test_qr_decomp_f64, 4, 3);
    test_q_mult_r_is_a!(f64, qr, test_q_mult_r_is_a_f64,4,3);
    test_qr_decomp_and_solve!(f64, qr, test_qr_decomp_and_solve_f64,4,3);




    #[test]
    fn test_qr_decomp_and_solve() {
        let m = 4;
        let n = 3;
        let matrix_a = rlst_dense::rlst_rand_mat![f64, (m, n)];
        let exp_sol = rlst_dense::rlst_rand_col_vec![f64, n];
        let mut rhs = matrix_a.dot(&exp_sol);

        let rhs = matrix_a
            .linalg()
            .qr()
            .unwrap()
            .solve_qr(rhs, TransposeMode::NoTrans).unwrap();

        let mut actual_sol = rlst_col_vec!(f64, n);
        actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, n));
        assert_approx_matrices!(&exp_sol, &actual_sol);
    }

    fn print_matrix<
        T: Scalar,
        Data: DataContainerMut<Item = T>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    >(
        matrix: &GenericBaseMatrix<T, Data, RS, CS>,
    ) {
        for row in 0..matrix.shape().0 {
            for col in 0..matrix.shape().1 {
                print!("{:.3} ", matrix[[row, col]]);
            }
            println!();
        }
    }
}
