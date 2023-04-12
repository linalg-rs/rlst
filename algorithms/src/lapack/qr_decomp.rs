use crate::traits::qr_decomp_trait::QRDecompTrait;
use crate::{lapack::LapackData};
use lapacke;
use rlst_common::types::{c32, c64, IndexType, RlstError, RlstResult, Scalar};
use rlst_dense::{
    DataContainerMut, GenericBaseMatrix,
    GenericBaseMatrixMut, Layout, LayoutType, MatrixTraitMut, SizeIdentifier, 
};

use super::{
    check_lapack_stride, SideMode, TransposeMode, TriangularDiagonal, TriangularType,
};

pub struct QRDecompLapack<
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    Mat: MatrixTraitMut<Item, RS, CS> + Sized,
> {
    data: LapackData<Item, RS, CS, Mat>,
    tau: Vec<Item>,
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

    fn xormqr<
        RhsData: DataContainerMut<Item = Self::T>,
        RhsR: SizeIdentifier,
        RhsC: SizeIdentifier,
    >(
        &mut self,
        rhs: &mut GenericBaseMatrixMut<Self::T, RhsData, RhsR, RhsC>,
        trans: TransposeMode,
    ) -> RlstResult<()> {
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
                m,
                self.data(),
                lda,
                &self.tau,
                rhs.data_mut(),
                ldb,
            )
        };
        if info != 0 {
            return Err(RlstError::LapackError(info));
        }
        Ok(())
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

            self.xormqr(rhs, trans).unwrap();

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

mod test {
    use crate::lapack::AsLapack;

    use super::*;
    use approx::assert_ulps_eq;
    use rlst_dense::{RandomAccessByValue, Dot, rlst_mat, MatrixD, matrix};

    #[test]
    fn test_qr_solve_f64() {
        let orig_mat = rlst_dense::rlst_rand_mat![f64, (4, 4)];
        let mut rlst_mat = rlst_dense::rlst_mat![f64,(4,4)];
        rlst_mat.data_mut().copy_from_slice(orig_mat.data());

        // TODO: this should be rlst_rand_vec but for some reason that doesn't work and I couldn't figure out why
        let mut expected_rhs = rlst_dense::rlst_vec![f64, 4];
        let mut rng = rand::thread_rng();
        expected_rhs.fill_from_rand_standard_normal(&mut rng);
        let mut rlst_vec = rlst_dense::rlst_vec![f64, 4];
        rlst_vec.data_mut().copy_from_slice(expected_rhs.data());

        let _ = rlst_mat
            .lapack()
            .unwrap()
            .solve_qr(&mut rlst_vec, TransposeMode::NoTrans);

        let actual_rhs = orig_mat.dot(&rlst_vec);

        println!("expected rhs");
        print_matrix(&expected_rhs);
        println!("actual rhs");
        print_matrix(&actual_rhs);
        for index in 0..expected_rhs.layout().number_of_elements() {
            let val1 = expected_rhs.get1d_value(index);
            let val2 = actual_rhs.get1d_value(index);
            assert_ulps_eq!(&val1, &val2, max_ulps = 100);
        }
    }

    #[test]
    fn test_qr_ls_f64()
    {
        let m = 4;
        let n = 3;
        let matrix_a = rlst_dense::rlst_rand_mat![f64, (m, n)];
        // TODO: this should be rlst_rand_vec but for some reason that doesn't work and I couldn't figure out why
        let mut exp_sol = rlst_dense::rlst_vec![f64,n];
        let mut rng = rand::thread_rng();
        exp_sol.fill_from_rand_standard_normal(&mut rng);
        let mut rhs = matrix_a.dot(&exp_sol);

        let _ = matrix_a
            .lapack()
            .unwrap()
            .solve_qr(&mut rhs, TransposeMode::NoTrans);

        println!("expected sol");
        print_matrix(&exp_sol);
        println!("actual sol");
        print_matrix(&rhs);
        for index in 0..exp_sol.layout().number_of_elements() {
            let val1 = exp_sol.get1d_value(index);
            let val2 = rhs.get1d_value(index);
            assert_ulps_eq!(&val1, &val2, max_ulps = 100);
        }
    }
    #[test]
    fn test_q_unitary_f64() {
        let m = 4;
        let n = 3;
        let rlst_mat = rlst_dense::rlst_rand_mat![f64, (m, n)];
        let mut qr = rlst_mat.lapack().unwrap().qr().unwrap();

        let mut expected_I = MatrixD::<f64>::zeros_from_dim(m, m);
        for i in 0..m {
            expected_I[[i,i]] = 1.;
        }
        println!("expected_I");
        print_matrix(&expected_I);

        let mut matrix_q = rlst_mat![f64,(m,m)];
        matrix_q.data_mut().copy_from_slice(expected_I.data());

        qr.xormqr(&mut matrix_q, TransposeMode::NoTrans);

        // alternative to get Q_1 (mxn) instead
        // let info = unsafe {
        //     lapacke::dorgqr(
        //         lapacke::Layout::ColumnMajor,
        //         m as i32,
        //         n as i32,
        //         n as i32,
        //         qr.data.mat.data_mut(),
        //         m as i32,
        //         &qr.tau,
        //     )
        // };

        // if info != 0 {
        //     panic!("DORGQR failed, code {}", info)
        // }

        let mut matrix_q_t = rlst_mat![f64,(m,m)];
        for row in 0..m {
            for col in 0..m {
                matrix_q_t[[col,row]] = matrix_q[[row,col]];
            }
        }
        println!("Q");
        print_matrix(&matrix_q);
        println!("QT");
        print_matrix(&matrix_q_t);

        let actual_I = matrix_q.dot(&matrix_q_t);
        expected_I = matrix_q_t.dot(&matrix_q);
        println!("QT*Q");
        print_matrix(&expected_I);
        println!("Q*QT");
        print_matrix(&actual_I);
        for index in 0..expected_I.layout().number_of_elements() {
            let val1 = expected_I.get1d_value(index);
            let val2 = actual_I.get1d_value(index);
            assert_ulps_eq!(&val1, &val2, max_ulps = 100);
        }

    }

    #[test]
    fn test_qr_decomp_f64() {
        let expected_a = rlst_dense::rlst_rand_mat![f64, (4, 3)];
        let mut rlst_mat = rlst_dense::rlst_mat![f64, expected_a.dim()];
        rlst_mat.data_mut().copy_from_slice(expected_a.data());
        // GenericBaseMatrixMut::new(BaseMatrix::new(DataContainerMut{}, layout))
        let mut qr = rlst_mat.lapack().unwrap().qr().unwrap();

        let dim = qr.data.mat.layout().dim();
        let stride = qr.data.mat.layout().stride();
        let m = dim.0 as i32;
        let n = dim.1 as i32;
        let lda = stride.1 as i32;
        let ldb = lda;

        let mut matrix_r = rlst_dense::rlst_mat![f64, (m as usize, n as usize)];
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
        qr.xormqr(&mut matrix_r, TransposeMode::NoTrans).unwrap();

        for index in 0..expected_a.layout().number_of_elements() {
                    let val1 = expected_a.get1d_value(index);
                    let val2 = matrix_r.get1d_value(index);
                    assert_ulps_eq!(&val1, &val2, max_ulps = 100);
                }
    }

    #[test]
    fn test_qr_decomp_and_solve() {
        let m = 4;
        let n = 3;
        let matrix_a = rlst_dense::rlst_rand_mat![f64, (m, n)];
        // TODO: this should be rlst_rand_vec but for some reason that doesn't work and I couldn't figure out why
        let mut exp_sol = rlst_dense::rlst_vec![f64,n];
        let mut rng = rand::thread_rng();
        exp_sol.fill_from_rand_standard_normal(&mut rng);
        let mut rhs = matrix_a.dot(&exp_sol);

        let _ = matrix_a
            .lapack()
            .unwrap()
            .qr()
            .unwrap()
            .solve(&mut rhs, TransposeMode::NoTrans);

        println!("expected sol");
        print_matrix(&exp_sol);
        println!("actual sol");
        print_matrix(&rhs);
        for index in 0..exp_sol.layout().number_of_elements() {
            let val1 = exp_sol.get1d_value(index);
            let val2 = rhs.get1d_value(index);
            assert_ulps_eq!(&val1, &val2, max_ulps = 100);
        }
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
                print!("{:.3}", matrix[[row, col]]);
            }
            println!();
        }
    }
}
