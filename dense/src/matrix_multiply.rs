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

// use crate::data_container::{DataContainer, DataContainerMut, VectorContainer};
// use crate::matrix::{GenericBaseMatrix, Matrix};
// use crate::types::*;
// use crate::{traits::*, MatrixD};
// use num;
// use rlst_blis;
// use rlst_blis::interface::{gemm::Gemm, types::TransMode};
// use rlst_common::traits::FillFrom;

// /// This trait provides a high-level interface for the multiplication of a matrix
// /// with another matrix. The result is a new matrix, hence memory allocation takes place.
// pub trait Dot<Rhs> {
//     type Output;

//     /// Return the matrix product with a right-hand side.
//     fn dot(&self, rhs: &Rhs) -> Self::Output;
// }

// /// This trait is an interface for the `dgemm` operation `mat_c = alpha * mat_a * mat_b + beta * mat_c`.
// pub trait MultiplyAdd<
//     Item: Scalar,
//     Data1: DataContainer<Item = Item>,
//     Data2: DataContainer<Item = Item>,
//     Data3: DataContainerMut<Item = Item>,
//     S1: SizeIdentifier,
//     S2: SizeIdentifier,
//     S3: SizeIdentifier,
// >
// {
//     /// Perform the operation `mat_c = alpha * mat_a * mat_b + beta * mat_c`.
//     fn multiply_add(
//         alpha: Item,
//         mat_a: &GenericBaseMatrix<Item, Data1, S1>,
//         mat_b: &GenericBaseMatrix<Item, Data2, S2>,
//         beta: Item,
//         mat_c: &mut GenericBaseMatrix<Item, Data3, S3>,
//     );
// }

// // Matrix x Matrix = Matrix
// impl<
//         T: Scalar,
//         MatImpl1: MatrixImplTrait<T, S1>,
//         MatImpl2: MatrixImplTrait<T, S2>,
//         S1: SizeIdentifier,
//         S2: SizeIdentifier,
//     > Dot<Matrix<T, MatImpl2, S2>> for Matrix<T, MatImpl1, S1>
// where
//     T: MultiplyAdd<
//         T,
//         VectorContainer<T>,
//         VectorContainer<T>,
//         VectorContainer<T>,
//         Dynamic,
//         Dynamic,
//         Dynamic,
//     >,
// {
//     type Output = MatrixD<T>;

//     fn dot(&self, rhs: &Matrix<T, MatImpl2, S2>) -> Self::Output {
//         // We evaluate self and the other matrix and then perform the multiplication.
//         let mut left = crate::rlst_dynamic_mat![T, self.shape()];
//         let mut right = crate::rlst_dynamic_mat![T, rhs.shape()];

//         left.fill_from(self);
//         right.fill_from(rhs);

//         let mut res = crate::rlst_dynamic_mat!(T, (self.shape().0, rhs.shape().1));

//         T::multiply_add(
//             num::cast::<f64, T>(1.0).unwrap(),
//             &left,
//             &right,
//             num::cast::<f64, T>(0.0).unwrap(),
//             &mut res,
//         );
//         res
//     }
// }

// impl<
//         T: Scalar,
//         Data1: DataContainer<Item = T>,
//         Data2: DataContainer<Item = T>,
//         Data3: DataContainerMut<Item = T>,
//     > MultiplyAdd<T, Data1, Data2, Data3, Dynamic, Dynamic, Dynamic> for T
// where
//     T: Gemm,
// {
//     fn multiply_add(
//         alpha: T,
//         mat_a: &GenericBaseMatrix<T, Data1, Dynamic>,
//         mat_b: &GenericBaseMatrix<T, Data2, Dynamic>,
//         beta: T,
//         mat_c: &mut GenericBaseMatrix<T, Data3, Dynamic>,
//     ) {
//         let dim1 = mat_a.layout().dim();
//         let dim2 = mat_b.layout().dim();
//         let dim3 = mat_c.layout().dim();

//         assert!(
//                     (dim1.1 == dim2.0) & (dim3.0 == dim1.0) & (dim3.1 == dim2.1),
//                     "Matrix multiply incompatible dimensions for C = A * B: A = {:#?}, B = {:#?}, C = {:#?}",
//                     dim1,
//                     dim2,
//                     dim3
//                 );

//         let stride_c = mat_c.stride();

//         <T as Gemm>::gemm(
//             TransMode::NoTrans,
//             TransMode::NoTrans,
//             dim3.0,
//             dim3.1,
//             dim1.1,
//             alpha,
//             mat_a.data(),
//             mat_a.stride().0,
//             mat_a.stride().1,
//             mat_b.data(),
//             mat_b.stride().0,
//             mat_b.stride().1,
//             beta,
//             mat_c.data_mut(),
//             stride_c.0,
//             stride_c.1,
//         );
//     }
// }

// #[cfg(test)]
// mod test {

//     use super::*;
//     use approx::assert_ulps_eq;
//     use rand_distr::StandardNormal;
//     use rlst_common::assert_matrix_relative_eq;
//     use rlst_common::tools::RandScalar;
//     use rlst_common::traits::Eval;

//     use rand::prelude::*;

//     fn matmul_expect<
//         Item: Scalar,
//         Data1: DataContainer<Item = Item>,
//         Data2: DataContainer<Item = Item>,
//         Data3: DataContainerMut<Item = Item>,
//         S1: SizeIdentifier,
//         S2: SizeIdentifier,
//         S3: SizeIdentifier,
//     >(
//         alpha: Item,
//         mat_a: &GenericBaseMatrix<Item, Data1, S1>,
//         mat_b: &GenericBaseMatrix<Item, Data2, S2>,
//         beta: Item,
//         mat_c: &mut GenericBaseMatrix<Item, Data3, S3>,
//     ) {
//         let m = mat_a.layout().dim().0;
//         let k = mat_a.layout().dim().1;
//         let n = mat_b.layout().dim().1;

//         for m_index in 0..m {
//             for n_index in 0..n {
//                 *mat_c.get_mut(m_index, n_index).unwrap() *= beta;
//                 for k_index in 0..k {
//                     *mat_c.get_mut(m_index, n_index).unwrap() += alpha
//                         * mat_a.get_value(m_index, k_index).unwrap()
//                         * mat_b.get_value(k_index, n_index).unwrap();
//                 }
//             }
//         }
//     }

//     macro_rules! matmul_test {
//         ($Scalar:ty, $fname:ident) => {
//             #[test]
//             fn $fname() {
//                 let mut mat_a = crate::rlst_dynamic_mat!($Scalar, (4, 6));
//                 let mut mat_b = crate::rlst_dynamic_mat!($Scalar, (6, 5));
//                 let mut mat_c_actual = crate::rlst_dynamic_mat!($Scalar, (4, 5));
//                 let mut mat_c_expect = crate::rlst_dynamic_mat!($Scalar, (4, 5));

//                 let dist = StandardNormal;

//                 let mut rng = rand::rngs::StdRng::seed_from_u64(0);

//                 mat_a.fill_from_equally_distributed(&mut rng);
//                 mat_b.fill_from_equally_distributed(&mut rng);
//                 mat_c_actual.fill_from_equally_distributed(&mut rng);

//                 for index in 0..mat_c_actual.layout().number_of_elements() {
//                     *mat_c_expect.get1d_mut(index).unwrap() =
//                         mat_c_actual.get1d_value(index).unwrap();
//                 }

//                 let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
//                 let beta = <$Scalar>::random_scalar(&mut rng, &dist);

//                 matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
//                 <$Scalar>::multiply_add(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

//                 for index in 0..mat_c_expect.layout().number_of_elements() {
//                     let val1 = mat_c_actual.get1d_value(index).unwrap();
//                     let val2 = mat_c_expect.get1d_value(index).unwrap();
//                     assert_ulps_eq!(&val1, &val2, max_ulps = 100);
//                 }
//             }
//         };
//     }

//     macro_rules! col_matvec_test {
//         ($Scalar:ty, $fname:ident) => {
//             #[test]
//             fn $fname() {
//                 let mut mat_a = crate::rlst_dynamic_mat![$Scalar, (4, 6)];
//                 let mut mat_b = crate::rlst_col_vec![$Scalar, 6];
//                 let mut mat_c_actual = crate::rlst_col_vec![$Scalar, 4];
//                 let mut mat_c_expect = crate::rlst_col_vec![$Scalar, 4];

//                 let dist = StandardNormal;

//                 let mut rng = rand::rngs::StdRng::seed_from_u64(0);

//                 mat_a.fill_from_equally_distributed(&mut rng);
//                 mat_b.fill_from_equally_distributed(&mut rng);
//                 mat_c_actual.fill_from_equally_distributed(&mut rng);

//                 for index in 0..mat_c_actual.layout().number_of_elements() {
//                     *mat_c_expect.get1d_mut(index).unwrap() =
//                         mat_c_actual.get1d_value(index).unwrap();
//                 }

//                 let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
//                 let beta = <$Scalar>::random_scalar(&mut rng, &dist);

//                 matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
//                 <$Scalar>::multiply_add(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

//                 for index in 0..mat_c_expect.layout().number_of_elements() {
//                     let val1 = mat_c_actual.get1d_value(index).unwrap();
//                     let val2 = mat_c_expect.get1d_value(index).unwrap();
//                     assert_ulps_eq!(&val1, &val2, max_ulps = 100);
//                 }
//             }
//         };
//     }

//     macro_rules! row_matvec_test {
//         ($Scalar:ty, $fname:ident) => {
//             #[test]
//             fn $fname() {
//                 let mut mat_a = crate::rlst_row_vec![$Scalar, 4];
//                 let mut mat_b = crate::rlst_dynamic_mat![$Scalar, (4, 6)];
//                 let mut mat_c_actual = crate::rlst_row_vec![$Scalar, 6];
//                 let mut mat_c_expect = crate::rlst_row_vec![$Scalar, 6];

//                 let dist = StandardNormal;

//                 let mut rng = rand::rngs::StdRng::seed_from_u64(0);

//                 mat_a.fill_from_equally_distributed(&mut rng);
//                 mat_b.fill_from_equally_distributed(&mut rng);
//                 mat_c_actual.fill_from_equally_distributed(&mut rng);

//                 for index in 0..mat_c_actual.layout().number_of_elements() {
//                     *mat_c_expect.get1d_mut(index).unwrap() =
//                         mat_c_actual.get1d_value(index).unwrap();
//                 }

//                 let alpha = <$Scalar>::random_scalar(&mut rng, &dist);
//                 let beta = <$Scalar>::random_scalar(&mut rng, &dist);

//                 matmul_expect(alpha, &mat_a, &mat_b, beta, &mut mat_c_expect);
//                 <$Scalar>::multiply_add(alpha, &mat_a, &mat_b, beta, &mut mat_c_actual);

//                 for index in 0..mat_c_expect.layout().number_of_elements() {
//                     let val1 = mat_c_actual.get1d_value(index).unwrap();
//                     let val2 = mat_c_expect.get1d_value(index).unwrap();
//                     assert_ulps_eq!(&val1, &val2, max_ulps = 10);
//                 }
//             }
//         };
//     }

//     // matmul_test!(f32, test_matmul_f32);
//     matmul_test!(f64, test_matmul_f64);
//     matmul_test!(c32, test_matmul_c32);
//     matmul_test!(c64, test_matmul_c64);

//     row_matvec_test!(f32, test_row_matvec_f32);
//     row_matvec_test!(f64, test_row_matvec_f64);
//     row_matvec_test!(c32, test_row_matvec_c32);
//     row_matvec_test!(c64, test_row_matvec_c64);

//     col_matvec_test!(f64, test_col_matvec_f64);
//     col_matvec_test!(f32, test_col_matvec_f32);
//     col_matvec_test!(c32, test_col_matvec_c32);
//     col_matvec_test!(c64, test_col_matvec_c64);

//     #[test]
//     fn test_dot_matvec() {
//         let mut mat1 = crate::rlst_dynamic_mat![f64, (2, 2)];
//         let mut mat2 = crate::rlst_dynamic_mat![f64, (2, 2)];

//         mat1.fill_from_seed_equally_distributed(0);
//         mat2.fill_from_seed_equally_distributed(1);

//         let mut mat3 = crate::rlst_dynamic_mat![f64, (2, 3)];

//         mat3.fill_from_seed_equally_distributed(2);

//         let actual = (mat1.view() + mat2.view()).dot(&mat3);
//         let mut expect = crate::rlst_dynamic_mat![f64, (2, 3)];

//         let mat_sum = (mat1.view() + mat2.view()).eval();

//         for row in 0..2 {
//             for col in 0..3 {
//                 for k in 0..2 {
//                     expect[[row, col]] += mat_sum[[row, k]] * mat3[[k, col]];
//                 }
//             }
//         }

//         assert_matrix_relative_eq!(expect, actual, 1E-14);
//     }
// }

use rlst_blis::interface::gemm::Gemm;
use rlst_blis::interface::types::TransMode;
use rlst_common::traits::*;
use rlst_common::{traits::RawAccess, types::Scalar};

pub fn matrix_multiply<
    Item: Scalar + Gemm,
    MatA: RawAccess<Item = Item> + Shape<2> + Stride<2>,
    MatB: RawAccess<Item = Item> + Shape<2> + Stride<2>,
    MatC: RawAccessMut<Item = Item> + Shape<2> + Stride<2>,
>(
    transa: TransMode,
    transb: TransMode,
    alpha: Item,
    mat_a: &MatA,
    mat_b: &MatB,
    beta: Item,
    mat_c: &mut MatC,
) {
    let m = mat_c.shape()[0];
    let n = mat_c.shape()[1];

    let a_shape = match transa {
        TransMode::NoTrans => mat_a.shape(),
        TransMode::ConjNoTrans => mat_a.shape(),
        TransMode::Trans => [mat_a.shape()[1], mat_a.shape()[0]],
        TransMode::ConjTrans => [mat_a.shape()[1], mat_a.shape()[0]],
    };

    let b_shape = match transb {
        TransMode::NoTrans => mat_b.shape(),
        TransMode::ConjNoTrans => mat_b.shape(),
        TransMode::Trans => [mat_b.shape()[1], mat_b.shape()[0]],
        TransMode::ConjTrans => [mat_b.shape()[1], mat_b.shape()[0]],
    };

    assert_eq!(m, a_shape[0], "Wrong dimension. {} != {}", m, a_shape[0]);
    assert_eq!(n, b_shape[1], "Wrong dimension. {} != {}", n, b_shape[1]);
    assert_eq!(
        a_shape[1], b_shape[0],
        "Wrong dimension. {} != {}",
        a_shape[1], b_shape[0]
    );

    let [rsa, csa] = mat_a.stride();
    let [rsb, csb] = mat_b.stride();
    let [rsc, csc] = mat_c.stride();

    <Item as Gemm>::gemm(
        transa,
        transb,
        m,
        n,
        a_shape[1],
        alpha,
        mat_a.data(),
        rsa,
        csa,
        mat_b.data(),
        rsb,
        csb,
        beta,
        mat_c.data_mut(),
        rsc,
        csc,
    )
}

#[cfg(test)]
mod test {

    use rlst_common::assert_array_relative_eq;
    use rlst_common::types::{c32, c64};

    use super::*;
    use crate::array::Array;
    use crate::base_array::BaseArray;
    use crate::data_container::VectorContainer;
    use crate::rlst_dynamic_array2;
    use paste::paste;

    macro_rules! mat_mul_test_impl {
        ($ScalarType:ty, $eps:expr) => {
            paste! {
                fn [<test_mat_mul_impl_$ScalarType>](transa: TransMode, transb: TransMode, shape_a: [usize; 2], shape_b: [usize; 2], shape_c: [usize; 2]) {

                    let mut mat_a = rlst_dynamic_array2!($ScalarType, shape_a);
                    let mut mat_b = rlst_dynamic_array2!($ScalarType, shape_b);
                    let mut mat_c = rlst_dynamic_array2!($ScalarType, shape_c);
                    let mut expected = rlst_dynamic_array2!($ScalarType, shape_c);

                    mat_a.fill_from_seed_equally_distributed(0);
                    mat_b.fill_from_seed_equally_distributed(1);
                    //mat_c.fill_from_seed_equally_distributed(2);

                    expected.fill_from(mat_c.view_mut());

                    matrix_multiply(
                        transa,
                        transb,
                        <$ScalarType as Scalar>::from_real(1.),
                        &mat_a,
                        &mat_b,
                        <$ScalarType as Scalar>::from_real(1.),
                        &mut mat_c,
                    );
                    matrix_multiply_compare(
                        transa,
                        transb,
                        <$ScalarType as Scalar>::from_real(1.),
                        &mat_a,
                        &mat_b,
                        <$ScalarType as Scalar>::from_real(1.),
                        &mut expected,
                    );

                    assert_array_relative_eq!(mat_c, expected, $eps);
                }

                #[test]
                fn [<test_mat_mul_$ScalarType>]() {

                    [<test_mat_mul_impl_$ScalarType>](TransMode::NoTrans, TransMode::NoTrans, [3, 5], [5, 7], [3, 7]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::ConjNoTrans, TransMode::ConjNoTrans, [3, 5], [5, 7], [3, 7]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::ConjNoTrans, TransMode::NoTrans, [3, 5], [5, 7], [3, 7]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::NoTrans, TransMode::ConjNoTrans, [3, 5], [5, 7], [3, 7]);

                    [<test_mat_mul_impl_$ScalarType>](TransMode::NoTrans, TransMode::Trans, [3, 5], [7, 5], [3, 7]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::Trans, TransMode::NoTrans, [2, 1], [2, 1], [1, 1]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::Trans, TransMode::Trans, [5, 3], [7, 5], [3, 7]);

                    [<test_mat_mul_impl_$ScalarType>](TransMode::NoTrans, TransMode::ConjTrans, [3, 5], [7, 5], [3, 7]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::ConjTrans, TransMode::NoTrans, [5, 3], [5, 7], [3, 7]);
                    [<test_mat_mul_impl_$ScalarType>](TransMode::ConjTrans, TransMode::ConjTrans, [5, 3], [7, 5], [3, 7]);


                }

            }
        };
    }

    fn matrix_multiply_compare<Item: Scalar>(
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        mat_a: &Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
        mat_b: &Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
        beta: Item,
        mat_c: &mut Array<Item, BaseArray<Item, VectorContainer<Item>, 2>, 2>,
    ) {
        let a_shape = match transa {
            TransMode::NoTrans => mat_a.shape(),
            TransMode::ConjNoTrans => mat_a.shape(),
            TransMode::Trans => [mat_a.shape()[1], mat_a.shape()[0]],
            TransMode::ConjTrans => [mat_a.shape()[1], mat_a.shape()[0]],
        };

        let b_shape = match transb {
            TransMode::NoTrans => mat_b.shape(),
            TransMode::ConjNoTrans => mat_b.shape(),
            TransMode::Trans => [mat_b.shape()[1], mat_b.shape()[0]],
            TransMode::ConjTrans => [mat_b.shape()[1], mat_b.shape()[0]],
        };

        let mut a_actual = rlst_dynamic_array2!(Item, a_shape);
        let mut b_actual = rlst_dynamic_array2!(Item, b_shape);

        match transa {
            TransMode::NoTrans => a_actual.fill_from(mat_a.view()),
            TransMode::ConjNoTrans => a_actual.fill_from(mat_a.view().conj()),
            TransMode::Trans => a_actual.fill_from(mat_a.view().transpose()),
            TransMode::ConjTrans => a_actual.fill_from(mat_a.view().conj().transpose()),
        }

        match transb {
            TransMode::NoTrans => b_actual.fill_from(mat_b.view()),
            TransMode::ConjNoTrans => b_actual.fill_from(mat_b.view().conj()),
            TransMode::Trans => b_actual.fill_from(mat_b.view().transpose()),
            TransMode::ConjTrans => b_actual.fill_from(mat_b.view().conj().transpose()),
        }

        let m = mat_c.shape()[0];
        let n = mat_c.shape()[1];
        let k = a_actual.shape()[1];

        for row in 0..m {
            for col in 0..n {
                mat_c[[row, col]] *= beta;
                for index in 0..k {
                    mat_c[[row, col]] += alpha * a_actual[[row, index]] * b_actual[[index, col]];
                }
            }
        }
    }

    mat_mul_test_impl!(f64, 1E-14);
    mat_mul_test_impl!(f32, 1E-5);
    mat_mul_test_impl!(c32, 1E-5);
    mat_mul_test_impl!(c64, 1E-14);
}
