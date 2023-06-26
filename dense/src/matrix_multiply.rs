//! Implementation of matrix multiplication.
//!
//! This module implements the matrix multiplication. The current implementation
//! uses the [rlst-blis] crate. Two traits are
//! defined. A low-level trait [MultiplyAdd] provides the method
//! [multiply_add](MultiplyAdd::multiply_add), which
//! performs `mat_c = alpha * mat_a * mat_b + beta * mat_c`, where `*` is to be
//! understood as matrix multipolitcation. A higher level [Dot] trait implements
//! the operation `mat_c = mat_a.dot(&mat_b)`. The latter allocates new memory,
//! while the former relies on suitable memory being allocated.
//!
//! The [MultiplyAdd] trait is currently only implemented for dynamic matrices.
//! Implementations for fixed size matrices will be added in the future.

use crate::data_container::{DataContainer, DataContainerMut, VectorContainer};
use crate::matrix::GenericBaseMatrix;
use crate::traits::*;
use crate::types::*;
use num;
use rlst_blis;
use rlst_blis::interface::{gemm::Gemm, types::TransMode};

/// This trait provides a high-level interface for the multiplication of a matrix
/// with another matrix. The result is a new matrix, hence memory allocation takes place.
pub trait Dot<Rhs> {
    type Output;

    /// Return the matrix product with a right-hand side.
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

/// This trait is an interface for the `dgemm` operation `mat_c = alpha * mat_a * mat_b + beta * mat_c`.
pub trait MultiplyAdd<
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
    fn multiply_add(
        alpha: Item,
        mat_a: &GenericBaseMatrix<Item, Data1, RS1, CS1>,
        mat_b: &GenericBaseMatrix<Item, Data2, RS2, CS2>,
        beta: Item,
        mat_c: &mut GenericBaseMatrix<Item, Data3, RS3, CS3>,
    );
}

// Matrix x Matrix = Matrix
impl<T: Scalar, Data1: DataContainer<Item = T>, Data2: DataContainer<Item = T>>
    Dot<GenericBaseMatrix<T, Data2, Dynamic, Dynamic>>
    for GenericBaseMatrix<T, Data1, Dynamic, Dynamic>
where
    T: MultiplyAdd<
        T,
        Data1,
        Data2,
        VectorContainer<T>,
        Dynamic,
        Dynamic,
        Dynamic,
        Dynamic,
        Dynamic,
        Dynamic,
    >,
{
    type Output = GenericBaseMatrix<T, VectorContainer<T>, Dynamic, Dynamic>;

    fn dot(&self, rhs: &GenericBaseMatrix<T, Data2, Dynamic, Dynamic>) -> Self::Output {
        let mut res = crate::rlst_mat!(T, (self.layout().dim().0, rhs.layout().dim().1));
        T::multiply_add(
            num::cast::<f64, T>(1.0).unwrap(),
            self,
            rhs,
            num::cast::<f64, T>(0.0).unwrap(),
            &mut res,
        );
        res
    }
}

impl<
        T: Scalar,
        Data1: DataContainer<Item = T>,
        Data2: DataContainer<Item = T>,
        Data3: DataContainerMut<Item = T>,
    > MultiplyAdd<T, Data1, Data2, Data3, Dynamic, Dynamic, Dynamic, Dynamic, Dynamic, Dynamic>
    for T
where
    T: Gemm,
{
    fn multiply_add(
        alpha: T,
        mat_a: &GenericBaseMatrix<T, Data1, Dynamic, Dynamic>,
        mat_b: &GenericBaseMatrix<T, Data2, Dynamic, Dynamic>,
        beta: T,
        mat_c: &mut GenericBaseMatrix<T, Data3, Dynamic, Dynamic>,
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

        let stride_c = mat_c.stride();

        <T as Gemm>::gemm(
            TransMode::NoTrans,
            TransMode::NoTrans,
            dim3.0,
            dim3.1,
            dim1.1,
            alpha,
            mat_a.data(),
            mat_a.stride().0,
            mat_a.stride().1,
            mat_b.data(),
            mat_b.stride().0,
            mat_b.stride().1,
            beta,
            mat_c.data_mut(),
            stride_c.0,
            stride_c.1,
        );
    }
}

#[cfg(test)]
mod test {

    use super::*;
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
                let mut mat_a = crate::rlst_mat!($Scalar, (4, 6));
                let mut mat_b = crate::rlst_mat!($Scalar, (6, 5));
                let mut mat_c_actual = crate::rlst_mat!($Scalar, (4, 5));
                let mut mat_c_expect = crate::rlst_mat!($Scalar, (4, 5));

                let dist = StandardNormal;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                mat_a.fill_from_equally_distributed(&mut rng);
                mat_b.fill_from_equally_distributed(&mut rng);
                mat_c_actual.fill_from_equally_distributed(&mut rng);

                for index in 0..mat_c_actual.layout().number_of_elements() {
                    *mat_c_expect.get1d_mut(index).unwrap() =
                        mat_c_actual.get1d_value(index).unwrap();
                }

                let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
                let beta = <$Scalar>::random_scalar(&mut rng, &dist);

                matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
                <$Scalar>::multiply_add(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

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
                let mut mat_a = crate::rlst_mat![$Scalar, (4, 6)];
                let mut mat_b = crate::rlst_col_vec![$Scalar, 6];
                let mut mat_c_actual = crate::rlst_col_vec![$Scalar, 4];
                let mut mat_c_expect = crate::rlst_col_vec![$Scalar, 4];

                let dist = StandardNormal;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                mat_a.fill_from_equally_distributed(&mut rng);
                mat_b.fill_from_equally_distributed(&mut rng);
                mat_c_actual.fill_from_equally_distributed(&mut rng);

                for index in 0..mat_c_actual.layout().number_of_elements() {
                    *mat_c_expect.get1d_mut(index).unwrap() =
                        mat_c_actual.get1d_value(index).unwrap();
                }

                let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
                let beta = <$Scalar>::random_scalar(&mut rng, &dist);

                matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
                <$Scalar>::multiply_add(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

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
                let mut mat_a = crate::rlst_row_vec![$Scalar, 4];
                let mut mat_b = crate::rlst_mat![$Scalar, (4, 6)];
                let mut mat_c_actual = crate::rlst_row_vec![$Scalar, 6];
                let mut mat_c_expect = crate::rlst_row_vec![$Scalar, 6];

                let dist = StandardNormal;

                let mut rng = rand::rngs::StdRng::seed_from_u64(0);

                mat_a.fill_from_equally_distributed(&mut rng);
                mat_b.fill_from_equally_distributed(&mut rng);
                mat_c_actual.fill_from_equally_distributed(&mut rng);

                for index in 0..mat_c_actual.layout().number_of_elements() {
                    *mat_c_expect.get1d_mut(index).unwrap() =
                        mat_c_actual.get1d_value(index).unwrap();
                }

                let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
                let beta = <$Scalar>::random_scalar(&mut rng, &dist);

                matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
                <$Scalar>::multiply_add(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

                for index in 0..mat_c_expect.layout().number_of_elements() {
                    let val1 = mat_c_actual.get1d_value(index).unwrap();
                    let val2 = mat_c_expect.get1d_value(index).unwrap();
                    assert_ulps_eq!(&val1, &val2, max_ulps = 10);
                }
            }
        };
    }

    // matmul_test!(f32, test_matmul_f32);
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
