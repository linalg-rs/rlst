//! Implementation of matrix multiplication.
//!
//! This module implements the matrix multiplication. The current implementation
//! uses the [rlst-blis] crate.

use crate::traits::*;
use rlst_blis::interface::gemm::Gemm;
use rlst_blis::interface::types::TransMode;
use rlst_common::types::RlstScalar;

pub fn matrix_multiply<
    Item: RlstScalar + Gemm,
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

    use crate::assert_array_relative_eq;
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
                        <$ScalarType as RlstScalar>::from_real(1.),
                        &mat_a,
                        &mat_b,
                        <$ScalarType as RlstScalar>::from_real(1.),
                        &mut mat_c,
                    );
                    matrix_multiply_compare(
                        transa,
                        transb,
                        <$ScalarType as RlstScalar>::from_real(1.),
                        &mat_a,
                        &mat_b,
                        <$ScalarType as RlstScalar>::from_real(1.),
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

    fn matrix_multiply_compare<Item: RlstScalar>(
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
