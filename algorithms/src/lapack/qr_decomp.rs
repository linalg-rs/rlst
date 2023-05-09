use crate::lapack::LapackData;
use crate::traits::qr_decomp_trait::QRDecompTrait;
use lapacke;
#[allow(unused_imports)]
use rlst_common::types::{c32, c64, IndexType, RlstError, RlstResult, Scalar};
#[allow(unused_imports)]
use rlst_dense::{
    rlst_vec, ColumnVectorD, DataContainerMut, GenericBaseMatrix, GenericBaseMatrixMut, Layout,
    LayoutType, MatrixTraitMut, SizeIdentifier,
};

use super::{check_lapack_stride, SideMode, TransposeMode, TriangularDiagonal, TriangularType};

pub struct QRDecompLapack<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixTraitMut<Item, RS, CS> + Sized,
> {
    data: LapackData<Item, RS, CS, Mat>,
    tau: Vec<Item>,
}

pub struct QRPivotDecompLapack<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixTraitMut<Item, RS, CS> + Sized,
> {
    data: LapackData<Item, RS, CS, Mat>,
    tau: Vec<Item>,
    jpvt: Vec<i32>,
}

impl<RS: SizeIdentifier, CS: SizeIdentifier, Data: DataContainerMut<Item = f64>>
    LapackData<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>
{
    /// Returns the QR decomposition of the input matrix assuming full rank and using LAPACK xGEQRF
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

    /// Returns the QR decomposition of the input matrix using column pivoting (LAPACK xGEQP3).
    /// # Arguments
    /// * `jpvt'
    ///    On Input:
    ///     - If jpvt(j) ≠ 0, the j-th column of a is permuted to the front of AP (a leading column);
    ///     - If jpvt(j) = 0, the j-th column of a is a free column.
    ///     - Specified as: a one-dimensional array of dimension max(1, n);
    ///    On Output:
    ///     - If jpvt(j) = k, then the j-th column of AP was the k-th column of a.
    ///     - Specified as: a one-dimensional array of dimension max(1, n);
    pub fn qr_col_pivot_jpvt(
        mut self,
        mut jpvt: Vec<i32>,
    ) -> RlstResult<QRPivotDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>> {
        let dim = self.mat.layout().dim();
        let stride = self.mat.layout().stride();

        let m = dim.0 as i32;
        let n = dim.1 as i32;
        let lda = stride.1 as i32;
        let mut tau: Vec<f64> = vec![0.0; std::cmp::min(dim.0, dim.1)];

        let info = unsafe {
            lapacke::dgeqp3(
                lapacke::Layout::ColumnMajor,
                m,
                n,
                self.mat.data_mut(),
                lda,
                &mut jpvt,
                &mut tau,
            )
        };

        if info == 0 {
            return Ok(QRPivotDecompLapack {
                data: self,
                tau: tau,
                jpvt: jpvt,
            });
        } else {
            return Err(RlstError::LapackError(info));
        }
    }

    pub fn qr_col_pivot(
        mut self,
    ) -> RlstResult<QRPivotDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>> {
        let mut jpvt = vec![0, self.mat.layout().dim().1 as i32];
        self.qr_col_pivot_jpvt(jpvt)
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
            //TODO: add a check for m>n?
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

    fn solve_qr_pivot<
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
            //TODO: add a check for m>n?
            // let mat = &self.data.mat;
            let dim = self.mat.layout().dim();
            let stride = self.mat.layout().stride();

            let m = dim.0 as i32;
            let n = dim.1 as i32;
            let lda = stride.1 as i32;
            let ldb = rhs.layout().stride().1 as i32;
            let nrhs = rhs.dim().1;

            let mut jpvt = vec![0;n as usize];
            let rcond = 0.0;
            let mut rank = 0;

            let info = unsafe {
                lapacke::dgelsy(
                    lapacke::Layout::ColumnMajor,
                    m,
                    n,
                    nrhs as i32,
                    self.mat.data_mut(),
                    lda,
                    rhs.data_mut(),
                    ldb,
                    &mut jpvt,
                    rcond,
                    &mut rank,
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

    /// Returns Q*RHS
    fn q_x_rhs<
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
            let rhs_dim = rhs.layout().dim();

            let m = rhs_dim.0 as i32;
            let n = rhs_dim.1 as i32;
            let lda = self.data.mat.layout().stride().1 as i32;
            let ldc = rhs.layout().stride().1 as i32;

            let info = unsafe {
                lapacke::dormqr(
                    lapacke::Layout::ColumnMajor,
                    SideMode::Left as u8,
                    trans as u8,
                    m,
                    n,
                    self.data.mat.layout().dim().1 as i32,
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
            Ok(())
        }
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

            self.q_x_rhs(rhs, trans).unwrap();

            let info = unsafe {
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
                    ldb,
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
    for QRPivotDecompLapack<f64, RS, CS, GenericBaseMatrixMut<f64, Data, RS, CS>>
{
    type T = f64;

    fn data(&self) -> &[Self::T] {
        self.data.mat.data()
    }

    fn dim(&self) -> (IndexType, IndexType) {
        self.data.mat.dim()
    }

    /// Returns Q*RHS
    fn q_x_rhs<
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
            let rhs_dim = rhs.layout().dim();

            let m = rhs_dim.0 as i32;
            let n = rhs_dim.1 as i32;
            let lda = self.data.mat.layout().stride().1 as i32;
            let ldc = rhs.layout().stride().1 as i32;

            let info = unsafe {
                lapacke::dormqr(
                    lapacke::Layout::ColumnMajor,
                    SideMode::Left as u8,
                    trans as u8,
                    m,
                    n,
                    self.data.mat.layout().dim().1 as i32,
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
            Ok(())
        }
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
            // // let mat = &self.data.mat;
            // let dim = self.data.mat.layout().dim();
            // let stride = self.data.mat.layout().stride();

            // let m = dim.0 as i32;
            // let n = dim.1 as i32;
            // let lda = stride.1 as i32;
            // let ldb = rhs.layout().stride().1 as i32;
            // let nrhs = rhs.dim().1 as i32;

            // self.q_x_rhs(rhs, trans).unwrap();

            // let info = unsafe {
            //     lapacke::dtrtrs(
            //         lapacke::Layout::ColumnMajor,
            //         TriangularType::Upper as u8,
            //         trans as u8,
            //         TriangularDiagonal::NonUnit as u8,
            //         n,
            //         nrhs,
            //         self.data(),
            //         lda,
            //         rhs.data_mut(),
            //         ldb,
            //     )
            // };
            //TODO: Implement
            let info = 1;
            if info != 0 {
                return Err(RlstError::LapackError(info));
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::lapack::AsLapack;

    use super::*;
    use approx::assert_abs_diff_eq;
    use rlst_dense::{
        matrix, rlst_mat, rlst_vec, ColumnVectorD, Dot, MatrixD, RandomAccessByValue,
    };

    // TODO Why does this need to be 10k ULPS? Should switch to epsilon

    #[macro_export]
    macro_rules! assert_approx_matrices {
        ($expected_matrix:expr, $actual_matrix:expr) => {{
            assert_eq!($expected_matrix.dim(), $actual_matrix.dim());
            for row in 0..$expected_matrix.dim().0 {
                for col in 0..$expected_matrix.dim().1 {
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
                // TODO: this should be rlst_rand_vec but for some reason that doesn't work and I couldn't figure out why
                let mut exp_sol = rlst_dense::rlst_vec![$scalar, $n];
                let mut rng = rand::thread_rng();
                exp_sol.fill_from_rand_standard_normal(&mut rng);
                let mut rhs = matrix_a.dot(&exp_sol);

                let _ = matrix_a
                    .lapack()
                    .unwrap()
                    .$solver(&mut rhs, TransposeMode::NoTrans);

                let mut actual_sol = rlst_vec!($scalar, $n);
                actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, $n));

                // println!("expected sol");
                // print_matrix(&exp_sol);
                // println!("actual sol");
                // print_matrix(&actual_sol);
                assert_approx_matrices!(&exp_sol, &actual_sol);
            }
        };
    }

    macro_rules! test_q_unitary {
        ($scalar:ty, $qr:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let rlst_mat = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];

                println!("A");
                print_matrix(&rlst_mat);
                let mut qr = rlst_mat.lapack().unwrap().$qr().unwrap();

                let mut expected_i = MatrixD::<$scalar>::zeros_from_dim($m, $m);
                for i in 0..$m {
                    expected_i[[i, i]] = 1.;
                }

                // get full Q
                let mut matrix_q = rlst_mat![$scalar, ($m, $m)];
                matrix_q.data_mut().copy_from_slice(expected_i.data());
                qr.q_x_rhs(&mut matrix_q, TransposeMode::NoTrans).unwrap();

                let mut matrix_q_t = rlst_mat![$scalar, (matrix_q.dim().1, matrix_q.dim().0)];
                for row in 0..matrix_q.dim().0 {
                    for col in 0..matrix_q.dim().1 {
                        matrix_q_t[[col, row]] = matrix_q[[row, col]];
                    }
                }
                // println!("Q");
                // print_matrix(&matrix_q);
                // println!("QT");
                // print_matrix(&matrix_q_t);

                let actual_i_t = matrix_q_t.dot(&matrix_q);
                println!("QT*Q");
                print_matrix(&actual_i_t);
                print_matrix(&expected_i);
                assert_approx_matrices!(&expected_i, &actual_i_t);
            }
        };
    }

    macro_rules! test_qr_decomp {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let expected_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut rlst_mat = rlst_dense::rlst_mat![$scalar, expected_a.dim()];
                rlst_mat.data_mut().copy_from_slice(expected_a.data());
                // GenericBaseMatrixMut::new(BaseMatrix::new(DataContainerMut{}, layout))
                let mut qr = rlst_mat.lapack().unwrap().$qr_decomp().unwrap();

                let dim = qr.data.mat.layout().dim();
                let stride = qr.data.mat.layout().stride();
                let m = dim.0 as i32;
                let n = dim.1 as i32;
                let lda = stride.1 as i32;
                let ldb = lda;

                let mut matrix_r = rlst_dense::rlst_mat![$scalar, (m as usize, n as usize)];
                //R=upper triangle of qr
                let info = unsafe {
                    lapacke::dlacpy(
                        lapacke::Layout::ColumnMajor,
                        TriangularType::Upper as u8,
                        m,
                        n,
                        qr.data(),
                        lda,
                        matrix_r.data_mut(),
                        ldb,
                    )
                };
                if info != 0 {
                    panic!("DLACPY failed, code {}", info);
                }
                qr.q_x_rhs(&mut matrix_r, TransposeMode::NoTrans).unwrap();

                assert_approx_matrices!(expected_a, matrix_r);
            }
        };
    }
    //TODO: m and n should probably be in test cases not separate macros
    test_qr_solve!(f64, solve_qr, test_solve_qr_f64, 4, 4);
    test_qr_solve!(f64, solve_qr_pivot, test_solve_qr_pivot_f64, 4, 4);
    test_qr_solve!(f64, solve_qr, test_solve_ls_qr_f64, 4, 3);
    test_qr_solve!(f64, solve_qr_pivot, test_solve_ls_qr_pivot_f64, 4, 3);

    test_q_unitary!(f64, qr, test_q_unitary_f64, 4, 3);
    test_q_unitary!(f64, qr_col_pivot, test_q_pivot_unitary_f64, 4, 3);

    test_qr_decomp!(f64, qr, test_qr_decomp_f64, 4, 3);
    test_qr_decomp!(f64, qr, test_qr_pivot_decomp_f64, 40, 30);

    #[test]
    fn test_q_unitary_from_array() {
        let m = 4;
        let n = 3;
        // let rlst_mat = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
        let mut data = [
            0.364, 0.844, -0.473,
            0.213, -1.568, -1.107, 
            0.327, -1.369, -0.177, 
            0.463, 0.110, -0.805
        ];
        // MatrixD::from_data(data, layout)
        let mut rlst_mat = MatrixD::<f64>::zeros_from_dim(m, n);
        rlst_mat.data_mut().copy_from_slice(&data);
        println!("A");
        print_matrix(&rlst_mat);
        let mut qr = rlst_mat.lapack().unwrap().qr().unwrap();

        let mut expected_i = MatrixD::<f64>::zeros_from_dim(m, m);
        for i in 0..m {
            expected_i[[i, i]] = 1.;
        }

        // get full Q
        let mut matrix_q = rlst_mat![f64, (m, m)];
        matrix_q.data_mut().copy_from_slice(expected_i.data());
        qr.q_x_rhs(&mut matrix_q, TransposeMode::NoTrans).unwrap();

        let mut matrix_q_t = rlst_mat![f64, (matrix_q.dim().1, matrix_q.dim().0)];
        for row in 0..matrix_q.dim().0 {
            for col in 0..matrix_q.dim().1 {
                matrix_q_t[[col, row]] = matrix_q[[row, col]];
            }
        }
        println!("Q");
        print_matrix(&matrix_q);
        println!("QT");
        print_matrix(&matrix_q_t);

        let actual_i_t = matrix_q_t.dot(&matrix_q);
        println!("QT*Q");
        print_matrix(&actual_i_t);
        print_matrix(&expected_i);
        assert_approx_matrices!(&expected_i, &actual_i_t);
    }

    #[test]
    fn test_qr_decomp_and_solve() {
        let m = 4;
        let n = 3;
        let matrix_a = rlst_dense::rlst_rand_mat![f64, (m, n)];
        // TODO: this should be rlst_rand_vec but for some reason that doesn't work and I couldn't figure out why
        let mut exp_sol = rlst_dense::rlst_vec![f64, n];
        let mut rng = rand::thread_rng();
        exp_sol.fill_from_rand_standard_normal(&mut rng);
        let mut rhs = matrix_a.dot(&exp_sol);

        let _ = matrix_a
            .lapack()
            .unwrap()
            .qr()
            .unwrap()
            .solve(&mut rhs, TransposeMode::NoTrans);

        let mut actual_sol = rlst_vec!(f64, exp_sol.dim().0);
        actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, n));

        println!("expected sol");
        print_matrix(&exp_sol);
        println!("actual sol");
        print_matrix(&actual_sol);
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
        for row in 0..matrix.dim().0 {
            for col in 0..matrix.dim().1 {
                print!("{:.3} ", matrix[[row, col]]);
            }
            println!();
        }
    }
}
