//! Implementation of matrix multiplication.
//!
//! This module implements the matrix multiplication. The current implementation
//! uses the [matrixmultiply] crate. To implement with this crate two traits are
//! provided. A low-level trait [MatMul] matmul provides the method
//! [matmul](MatMul::matmul), which
//! performs `mat_c = alpha * mat_a * mat_b + beta * mat_c`, where `*` is to be
//! understood as matrix multipolitcation. A higher level [Dot] trait implements
//! the operation `mat_c = mat_a.dot(&mat_b)`. The latter allocates new memory,
//! while the former relies on suitable memory being allocated.
//!
//! The [MatMul] trait is currently implemented for the product of two dynamic matrices,
//! the product of a dynamic matrix with a vector, and the product of a row vector
//! with a dynamic matrix.

use crate::data_container::{DataContainer, DataContainerMut, VectorContainer};
use crate::matrix::{GenericBaseMatrix, GenericBaseMatrixMut};
use crate::traits::*;
use crate::types::*;

use matrixmultiply::{cgemm, dgemm, sgemm, zgemm, CGemmOption};
use num;

/// This trait provides a high-level interface for the multiplication of a matrix
/// with another matrix. The result is a new matrix, hence memory allocation takes place.
pub trait Dot<Rhs> {
    type Output;

    /// Return the matrix product with a right-hand side.
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

/// This trait is an interface for the `dgemm` operation `mat_c = alpha * mat_a * mat_b + beta * mat_c`.
pub trait MatMul<
    Item: Scalar,
    Data1: DataContainer<Item = Item>,
    Data2: DataContainer<Item = Item>,
    Data3: DataContainerMut<Item = Item>,
    RS1: SizeIdentifier,
    CS1: SizeIdentifier,
    RS2: SizeIdentifier,
    CS2: SizeIdentifier,
    RS3: SizeIdentifier,
    CS3: SizeIdentifier,
>
{
    /// Perform the operation `mat_c = alpha * mat_a * mat_b + beta * mat_c`.
    fn matmul(
        alpha: Item,
        mat_a: &GenericBaseMatrix<Item, Data1, RS1, CS1>,
        mat_b: &GenericBaseMatrix<Item, Data2, RS2, CS2>,
        beta: Item,
        mat_c: &mut GenericBaseMatrixMut<Item, Data3, RS3, CS3>,
    );
}

macro_rules! dot_impl {
    ($Scalar:ty) => {
        // Matrix x Matrix = Matrix
        impl<Data1: DataContainer<Item = $Scalar>, Data2: DataContainer<Item = $Scalar>>
            Dot<GenericBaseMatrix<$Scalar, Data2, Dynamic, Dynamic>>
            for GenericBaseMatrix<$Scalar, Data1, Dynamic, Dynamic>
        {
            type Output = GenericBaseMatrix<$Scalar, VectorContainer<$Scalar>, Dynamic, Dynamic>;

            fn dot(
                &self,
                rhs: &GenericBaseMatrix<$Scalar, Data2, Dynamic, Dynamic>,
            ) -> Self::Output {
                let mut res =
                    Self::Output::zeros_from_dim(self.layout().dim().0, rhs.layout().dim().1);
                <$Scalar>::matmul(
                    num::cast::<f64, $Scalar>(1.0).unwrap(),
                    &self,
                    rhs,
                    num::cast::<f64, $Scalar>(0.0).unwrap(),
                    &mut res,
                );
                res
            }
        }

        // RowVector x Matrix = RowVector
        impl<Data1: DataContainer<Item = $Scalar>, Data2: DataContainer<Item = $Scalar>>
            Dot<GenericBaseMatrix<$Scalar, Data2, Dynamic, Dynamic>>
            for GenericBaseMatrix<$Scalar, Data1, Fixed1, Dynamic>
        {
            type Output = GenericBaseMatrix<$Scalar, VectorContainer<$Scalar>, Fixed1, Dynamic>;

            fn dot(
                &self,
                rhs: &GenericBaseMatrix<$Scalar, Data2, Dynamic, Dynamic>,
            ) -> Self::Output {
                let mut res = Self::Output::zeros_from_length(rhs.layout().dim().1);
                <$Scalar>::matmul(
                    num::cast::<f64, $Scalar>(1.0).unwrap(),
                    &self,
                    rhs,
                    num::cast::<f64, $Scalar>(0.0).unwrap(),
                    &mut res,
                );
                res
            }
        }

        // Matrix x ColumnVector = ColumnVector
        impl<Data1: DataContainer<Item = $Scalar>, Data2: DataContainer<Item = $Scalar>>
            Dot<GenericBaseMatrix<$Scalar, Data2, Dynamic, Fixed1>>
            for GenericBaseMatrix<$Scalar, Data1, Dynamic, Dynamic>
        {
            type Output = GenericBaseMatrix<$Scalar, VectorContainer<$Scalar>, Dynamic, Fixed1>;

            fn dot(
                &self,
                rhs: &GenericBaseMatrix<$Scalar, Data2, Dynamic, Fixed1>,
            ) -> Self::Output {
                let mut res = Self::Output::zeros_from_length(self.layout().dim().0);
                <$Scalar>::matmul(
                    num::cast::<f64, $Scalar>(1.0).unwrap(),
                    &self,
                    rhs,
                    num::cast::<f64, $Scalar>(0.0).unwrap(),
                    &mut res,
                );
                res
            }
        }
    };
}

macro_rules! matmul_impl {

    ($Scalar:ty, $Blas:ident, $RS1:ty, $CS1:ty, $RS2:ty, $CS2:ty, $RS3:ty, $CS3:ty, real) => {

        impl<
        Data1: DataContainer<Item = $Scalar>,
        Data2: DataContainer<Item = $Scalar>,
        Data3: DataContainerMut<Item = $Scalar>
>


        MatMul<
            $Scalar,
            Data1,
            Data2,
            Data3,
            $RS1,
            $RS2,
            $CS1,
            $CS2,
            $RS3,
            $CS3>


        for $Scalar {

            fn matmul(
                alpha: $Scalar,
                mat_a: &GenericBaseMatrix<$Scalar, Data1, $RS1, $CS1>,
                mat_b: &GenericBaseMatrix<$Scalar, Data2, $RS2, $CS2>,
                beta: $Scalar,
                mat_c: &mut GenericBaseMatrixMut<$Scalar, Data3, $RS3, $CS3>
            ) {
                let dim1 = mat_a.layout().dim();
                let dim2 = mat_b.layout().dim();
                let dim3 = mat_c.layout().dim();

                assert!(
                    (dim1.1 == dim2.0) & (dim3.0 == dim1.0) & (dim3.1 == dim2.1),
                    "Matrix multiply incompatible dimensions for C = A * B: A = {:#?}, B = {:#?}, C = {:#?}",
                    dim1,
                    dim2,
                    dim3
                );

                let m = dim1.0 as usize;
                let k = dim1.1 as usize;
                let n = dim2.1 as usize;
                let rsa = mat_a.layout().stride().0 as isize;
                let csa = mat_a.layout().stride().1 as isize;
                let rsb = mat_b.layout().stride().0 as isize;
                let csb = mat_b.layout().stride().1 as isize;
                let rsc = mat_c.layout().stride().0 as isize;
                let csc = mat_c.layout().stride().1 as isize;

                unsafe {
                    $Blas(
                        m,
                        k,
                        n,
                        alpha,
                        mat_a.get_pointer(),
                        rsa,
                        csa,
                        mat_b.get_pointer(),
                        rsb,
                        csb,
                        beta,
                        mat_c.get_pointer_mut(),
                        rsc,
                        csc,
                    );
                }
            }

            }

        };

        ($Scalar:ty, $Blas:ident, $RS1:ty, $CS1:ty, $RS2:ty, $CS2:ty, $RS3:ty, $CS3:ty, complex) => {

            impl<
            Data1: DataContainer<Item = $Scalar>,
            Data2: DataContainer<Item = $Scalar>,
            Data3: DataContainerMut<Item = $Scalar>
    >


            MatMul<
                $Scalar,
                Data1,
                Data2,
                Data3,
                $RS1,
                $RS2,
                $CS1,
                $CS2,
                $RS3,
                $CS3>


            for $Scalar {

                fn matmul(
                    alpha: $Scalar,
                    mat_a: &GenericBaseMatrix<$Scalar, Data1, $RS1, $CS1>,
                    mat_b: &GenericBaseMatrix<$Scalar, Data2, $RS2, $CS2>,
                    beta: $Scalar,
                    mat_c: &mut GenericBaseMatrixMut<$Scalar, Data3, $RS3, $CS3>
                ) {
                    let dim1 = mat_a.layout().dim();
                    let dim2 = mat_b.layout().dim();
                    let dim3 = mat_c.layout().dim();

                    assert!(
                        (dim1.1 == dim2.0) & (dim3.0 == dim1.0) & (dim3.1 == dim2.1),
                        "Matrix multiply incompatible dimensions for C = A * B: A = {:#?}, B = {:#?}, C = {:#?}",
                        dim1,
                        dim2,
                        dim3
                    );

                    let m = dim1.0 as usize;
                    let k = dim1.1 as usize;
                    let n = dim2.1 as usize;
                    let rsa = mat_a.layout().stride().0 as isize;
                    let csa = mat_a.layout().stride().1 as isize;
                    let rsb = mat_b.layout().stride().0 as isize;
                    let csb = mat_b.layout().stride().1 as isize;
                    let rsc = mat_c.layout().stride().0 as isize;
                    let csc = mat_c.layout().stride().1 as isize;

                    let alpha = [alpha.re(), alpha.im()];
                    let beta = [beta.re(), beta.im()];

                    unsafe {
                        $Blas(
                            CGemmOption::Standard,
                            CGemmOption::Standard,
                            m,
                            k,
                            n,
                            alpha,
                            mat_a.get_pointer() as *const [<$Scalar as Scalar>::Real; 2],
                            rsa,
                            csa,
                            mat_b.get_pointer() as *const [<$Scalar as Scalar>::Real; 2],
                            rsb,
                            csb,
                            beta,
                            mat_c.get_pointer_mut() as *mut [<$Scalar as Scalar>::Real; 2],
                            rsc,
                            csc,
                        );
                    }
                }

                }

            };


}

macro_rules! matmul_over_size_types {
    ($RS1:ty, $CS1:ty, $RS2:ty, $CS2:ty, $RS3:ty, $CS3:ty) => {
        matmul_impl!(f64, dgemm, $RS1, $CS1, $RS2, $CS2, $RS3, $CS3, real);
        matmul_impl!(f32, sgemm, $RS1, $CS1, $RS2, $CS2, $RS3, $CS3, real);
        matmul_impl!(c32, cgemm, $RS1, $CS1, $RS2, $CS2, $RS3, $CS3, complex);
        matmul_impl!(c64, zgemm, $RS1, $CS1, $RS2, $CS2, $RS3, $CS3, complex);
    };
}

// matrix x matrix = matrix
matmul_over_size_types!(Dynamic, Dynamic, Dynamic, Dynamic, Dynamic, Dynamic);

// matrix x col_vector = col_vector
matmul_over_size_types!(Dynamic, Dynamic, Dynamic, Fixed1, Dynamic, Fixed1);

// row_vector x matrix = row_vector
matmul_over_size_types!(Fixed1, Dynamic, Dynamic, Dynamic, Fixed1, Dynamic);

dot_impl!(f64);
dot_impl!(f32);
dot_impl!(c32);
dot_impl!(c64);

#[cfg(test)]
mod test {

    use super::*;
    use crate::matrix::*;
    use approx::assert_ulps_eq;
    use rand_distr::StandardNormal;
    use rlst_common::tools::RandScalar;

    use rand::prelude::*;

    fn matmul_expect<
        Item: Scalar,
        Data1: DataContainer<Item = Item>,
        Data2: DataContainer<Item = Item>,
        Data3: DataContainerMut<Item = Item>,
        RS1: SizeIdentifier,
        CS1: SizeIdentifier,
        RS2: SizeIdentifier,
        CS2: SizeIdentifier,
        RS3: SizeIdentifier,
        CS3: SizeIdentifier,
    >(
        alpha: Item,
        mat_a: &GenericBaseMatrix<Item, Data1, RS1, CS1>,
        mat_b: &GenericBaseMatrix<Item, Data2, RS2, CS2>,
        beta: Item,
        mat_c: &mut GenericBaseMatrix<Item, Data3, RS3, CS3>,
    ) {
        let m = mat_a.layout().dim().0;
        let k = mat_a.layout().dim().1;
        let n = mat_b.layout().dim().1;

        for m_index in 0..m {
            for n_index in 0..n {
                *mat_c.get_mut(m_index, n_index).unwrap() *= beta;
                for k_index in 0..k {
                    *mat_c.get_mut(m_index, n_index).unwrap() += alpha
                        * mat_a.get_value(m_index, k_index).unwrap()
                        * mat_b.get_value(k_index, n_index).unwrap();
                }
            }
        }
    }

    macro_rules! matmul_test {
        ($Scalar:ty, $fname:ident) => {
            #[test]
            fn $fname() {
                let mut mat_a = MatrixD::<$Scalar>::zeros_from_dim(4, 6);
                let mut mat_b = MatrixD::<$Scalar>::zeros_from_dim(6, 5);
                let mut mat_c_actual = MatrixD::<$Scalar>::zeros_from_dim(4, 5);
                let mut mat_c_expect = MatrixD::<$Scalar>::zeros_from_dim(4, 5);

                let dist = StandardNormal;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                mat_a.fill_from_rand_standard_normal(&mut rng);
                mat_b.fill_from_rand_standard_normal(&mut rng);
                mat_c_actual.fill_from_rand_standard_normal(&mut rng);

                for index in 0..mat_c_actual.layout().number_of_elements() {
                    *mat_c_expect.get1d_mut(index).unwrap() =
                        mat_c_actual.get1d_value(index).unwrap();
                }

                let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
                let beta = <$Scalar>::random_scalar(&mut rng, &dist);

                matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
                <$Scalar>::matmul(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

                for index in 0..mat_c_expect.layout().number_of_elements() {
                    let val1 = mat_c_actual.get1d_value(index).unwrap();
                    let val2 = mat_c_expect.get1d_value(index).unwrap();
                    assert_ulps_eq!(&val1, &val2, max_ulps = 100);
                }
            }
        };
    }

    macro_rules! col_matvec_test {
        ($Scalar:ty, $fname:ident) => {
            #[test]
            fn $fname() {
                let mut mat_a = MatrixD::<$Scalar>::zeros_from_dim(4, 6);
                let mut mat_b = ColumnVectorD::<$Scalar>::zeros_from_length(6);
                let mut mat_c_actual = ColumnVectorD::<$Scalar>::zeros_from_length(4);
                let mut mat_c_expect = ColumnVectorD::<$Scalar>::zeros_from_length(4);

                let dist = StandardNormal;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                mat_a.fill_from_rand_standard_normal(&mut rng);
                mat_b.fill_from_rand_standard_normal(&mut rng);
                mat_c_actual.fill_from_rand_standard_normal(&mut rng);

                for index in 0..mat_c_actual.layout().number_of_elements() {
                    *mat_c_expect.get1d_mut(index).unwrap() =
                        mat_c_actual.get1d_value(index).unwrap();
                }

                let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
                let beta = <$Scalar>::random_scalar(&mut rng, &dist);

                matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
                <$Scalar>::matmul(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

                for index in 0..mat_c_expect.layout().number_of_elements() {
                    let val1 = mat_c_actual.get1d_value(index).unwrap();
                    let val2 = mat_c_expect.get1d_value(index).unwrap();
                    assert_ulps_eq!(&val1, &val2, max_ulps = 100);
                }
            }
        };
    }

    macro_rules! row_matvec_test {
        ($Scalar:ty, $fname:ident) => {
            #[test]
            fn $fname() {
                let mut mat_a = RowVectorD::<$Scalar>::zeros_from_length(4);
                let mut mat_b = MatrixD::<$Scalar>::zeros_from_dim(4, 6);
                let mut mat_c_actual = RowVectorD::<$Scalar>::zeros_from_length(6);
                let mut mat_c_expect = RowVectorD::<$Scalar>::zeros_from_length(6);

                let dist = StandardNormal;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                mat_a.fill_from_rand_standard_normal(&mut rng);
                mat_b.fill_from_rand_standard_normal(&mut rng);
                mat_c_actual.fill_from_rand_standard_normal(&mut rng);

                for index in 0..mat_c_actual.layout().number_of_elements() {
                    *mat_c_expect.get1d_mut(index).unwrap() =
                        mat_c_actual.get1d_value(index).unwrap();
                }

                let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
                let beta = <$Scalar>::random_scalar(&mut rng, &dist);

                matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
                <$Scalar>::matmul(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

                for index in 0..mat_c_expect.layout().number_of_elements() {
                    let val1 = mat_c_actual.get1d_value(index).unwrap();
                    let val2 = mat_c_expect.get1d_value(index).unwrap();
                    assert_ulps_eq!(&val1, &val2, max_ulps = 100);
                }
            }
        };
    }

    matmul_test!(f32, test_matmul_f32);
    matmul_test!(f64, test_matmul_f64);
    matmul_test!(c32, test_matmul_c32);
    matmul_test!(c64, test_matmul_c64);

    row_matvec_test!(f32, test_row_matvec_f32);
    row_matvec_test!(f64, test_row_matvec_f64);
    row_matvec_test!(c32, test_row_matvec_c32);
    row_matvec_test!(c64, test_row_matvec_c64);

    col_matvec_test!(f64, test_col_matvec_f64);
    col_matvec_test!(f32, test_col_matvec_f32);
    col_matvec_test!(c32, test_col_matvec_c32);
    col_matvec_test!(c64, test_col_matvec_c64);
}
