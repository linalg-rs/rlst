//! Tests of array algebray operations

extern crate blas_src;
extern crate lapack_src;

use approx::assert_relative_eq;
use paste::paste;
use rlst::dense;
use rlst::dense::assert_array_relative_eq;
use rlst::dense::layout::convert_1d_nd_from_shape;
use rlst::prelude::rlst_dynamic_array3;
use rlst::prelude::*;

#[test]
fn test_addition() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);
    let mut res_chunked = rlst_dynamic_array3!(f64, shape);
    let mut expected = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);

    let arr3 = arr1.view() + arr2.view();

    for (multi_index, elem) in expected.iter_mut().enumerate().multi_index(shape) {
        *elem = arr1[multi_index] + arr2[multi_index];
    }

    res_chunked.fill_from_chunked::<_, 31>(arr3.view());
    res.fill_from(arr3.view());

    assert_array_relative_eq!(res_chunked, expected, 1E-14);
    assert_array_relative_eq!(res, expected, 1E-14);
}

#[test]
fn test_subtraction() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);
    let mut res_chunked = rlst_dynamic_array3!(f64, shape);
    let mut expected = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);

    let arr3 = arr1.view() - arr2.view();

    for (multi_index, elem) in expected.iter_mut().enumerate().multi_index(shape) {
        *elem = arr1[multi_index] - arr2[multi_index];
    }

    res_chunked.fill_from_chunked::<_, 31>(arr3.view());
    res.fill_from(arr3.view());

    assert_array_relative_eq!(res_chunked, expected, 1E-14);
    assert_array_relative_eq!(res, expected, 1E-14);
}

#[test]
fn test_multiple_operations() {
    let shape = [3, 4, 8];
    let nelements = shape.iter().product();

    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res_chunked = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);
    let mut compare = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);
    res.fill_from_seed_equally_distributed(2);
    res_chunked.fill_from_seed_equally_distributed(2);
    compare.fill_from(res.view());

    let arr3 = 3.6 * arr1.view() - arr2.view();
    res_chunked.sum_into_chunked::<_, 64>(arr3.view());
    res.sum_into(arr3.view());

    for index in 0..nelements {
        let indices = convert_1d_nd_from_shape(index, res.shape());

        approx::assert_relative_eq!(
            res[indices],
            compare[indices] + 3.6 * arr1[indices] - arr2[indices],
            epsilon = 1E-14
        );
        approx::assert_relative_eq!(
            res_chunked[indices],
            compare[indices] + 3.6 * arr1[indices] - arr2[indices],
            epsilon = 1E-14
        );
    }
}

#[test]
fn test_cmp_wise_multiplication() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res_chunked = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);
    let mut expected = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);

    let arr3 = arr1.view() * arr2.view();

    res_chunked.fill_from_chunked::<_, 31>(arr3.view());
    res.fill_from(arr3.view());

    for (multi_index, elem) in expected.iter_mut().enumerate().multi_index(shape) {
        *elem = arr1[multi_index] * arr2[multi_index];
    }

    assert_array_relative_eq!(res_chunked, expected, 1E-14);
    assert_array_relative_eq!(res, expected, 1E-14);
}

#[test]
fn test_conj() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(c64, shape);
    let mut res_chunked = rlst_dynamic_array3!(c64, shape);
    let mut res = rlst_dynamic_array3!(c64, shape);
    let mut expected = rlst_dynamic_array3!(c64, shape);

    arr1.fill_from_seed_equally_distributed(0);

    let arr3 = arr1.view().conj();

    res_chunked.fill_from_chunked::<_, 31>(arr3.view());
    res.fill_from(arr3.view());

    for (multi_index, elem) in expected.iter_mut().enumerate().multi_index(shape) {
        *elem = arr1[multi_index].conj();
    }

    assert_array_relative_eq!(res_chunked, expected, 1E-14);
    assert_array_relative_eq!(res, expected, 1E-14);
}

#[test]
fn test_transpose() {
    let shape = [3, 4, 8];
    let permutation = [1, 2, 0];
    let new_shape = [4, 8, 3];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    arr1.fill_from_seed_equally_distributed(0);
    let mut res = rlst_dynamic_array3!(f64, new_shape);
    let mut res_chunked = rlst_dynamic_array3!(f64, new_shape);
    let mut expected = rlst_dynamic_array3!(f64, new_shape);

    let arr3 = arr1.view().permute_axes(permutation);
    res.fill_from(arr3.view());
    res_chunked.fill_from_chunked::<_, 31>(arr3.view());

    assert_eq!(arr3.shape(), [shape[1], shape[2], shape[0]]);

    for (multi_index, elem) in arr3.iter().enumerate().multi_index(res.shape()) {
        let original_index = [multi_index[2], multi_index[0], multi_index[1]];

        assert_eq!(elem, arr1[original_index]);
        expected[multi_index] = elem;
    }

    assert_array_relative_eq!(res, expected, 1E-14);
    assert_array_relative_eq!(res_chunked, expected, 1E-14);
}

#[test]
fn test_cmp_wise_division() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res_chunked = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);
    let mut expected = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);

    let arr3 = arr1.view() / arr2.view();

    res_chunked.fill_from_chunked::<_, 31>(arr3.view());
    res.fill_from(arr3.view());

    for (multi_index, elem) in expected.iter_mut().enumerate().multi_index(shape) {
        *elem = arr1[multi_index] / arr2[multi_index];
    }

    assert_array_relative_eq!(res_chunked, expected, 1E-14);
    assert_array_relative_eq!(res, expected, 1E-14);
}

#[test]
fn test_to_complex() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut res_chunked = rlst_dynamic_array3!(c64, shape);
    let mut res = rlst_dynamic_array3!(c64, shape);
    let mut expected = rlst_dynamic_array3!(c64, shape);

    arr1.fill_from_seed_equally_distributed(0);

    let arr3 = arr1.view().to_complex();

    res_chunked.fill_from_chunked::<_, 31>(arr3.view());
    res.fill_from(arr3.view());

    for (multi_index, elem) in expected.iter_mut().enumerate().multi_index(shape) {
        *elem = c64::from_real(arr1[multi_index]);
    }

    assert_array_relative_eq!(res_chunked, expected, 1E-14);
    assert_array_relative_eq!(res, expected, 1E-14);
}

#[test]
fn test_iter() {
    let mut arr = crate::rlst_dynamic_array3![f64, [1, 3, 2]];

    for (index, item) in arr.iter_mut().enumerate() {
        *item = index as f64;
    }

    assert_eq!(arr[[0, 0, 0]], 0.0);
    assert_eq!(arr[[0, 1, 0]], 1.0);
    assert_eq!(arr[[0, 2, 0]], 2.0);
    assert_eq!(arr[[0, 0, 1]], 3.0);
    assert_eq!(arr[[0, 1, 1]], 4.0);
    assert_eq!(arr[[0, 2, 1]], 5.0);
}

#[test]
fn test_row_iter() {
    let shape = [2, 3];
    let mut arr = crate::rlst_dynamic_array2![f64, shape];

    arr.fill_from_seed_equally_distributed(0);

    let mut row_count = 0;
    for (row_index, row) in arr.row_iter().enumerate() {
        for col_index in 0..shape[1] {
            assert_eq!(row[[col_index]], arr[[row_index, col_index]]);
        }
        row_count += 1;
    }
    assert_eq!(row_count, shape[0]);
}

#[test]
fn test_row_iter_mut() {
    let shape = [2, 3];
    let mut arr = crate::rlst_dynamic_array2![f64, shape];
    let mut arr2 = crate::rlst_dynamic_array2![f64, shape];

    arr.fill_from_seed_equally_distributed(0);
    arr2.fill_from(arr.view());

    let mut row_count = 0;
    for (row_index, mut row) in arr.row_iter_mut().enumerate() {
        for col_index in 0..shape[1] {
            row[[col_index]] *= 2.0;
            assert_relative_eq!(
                row[[col_index]],
                2.0 * arr2[[row_index, col_index]],
                epsilon = 1E-14
            );
        }
        row_count += 1;
    }
    assert_eq!(row_count, shape[0]);
}

#[test]
fn test_col_iter() {
    let shape = [2, 3];
    let mut arr = crate::rlst_dynamic_array2![f64, shape];

    arr.fill_from_seed_equally_distributed(0);

    let mut col_count = 0;
    for (col_index, col) in arr.col_iter().enumerate() {
        for row_index in 0..shape[0] {
            assert_eq!(col[[row_index]], arr[[row_index, col_index]]);
        }
        col_count += 1;
    }

    assert_eq!(col_count, shape[1]);
}

#[test]
fn test_col_iter_mut() {
    let shape = [2, 3];
    let mut arr = crate::rlst_dynamic_array2![f64, shape];
    let mut arr2 = crate::rlst_dynamic_array2![f64, shape];

    arr.fill_from_seed_equally_distributed(0);
    arr2.fill_from(arr.view());

    let mut col_count = 0;
    for (col_index, mut col) in arr.col_iter_mut().enumerate() {
        for row_index in 0..shape[0] {
            col[[row_index]] *= 2.0;
            assert_relative_eq!(
                col[[row_index]],
                2.0 * arr2[[row_index, col_index]],
                epsilon = 1E-14
            );
        }
        col_count += 1;
    }
    assert_eq!(col_count, shape[1]);
}

#[test]
fn test_convert_1d_nd() {
    let multi_index: [usize; 3] = [2, 3, 7];
    let shape: [usize; 3] = [3, 4, 8];
    let stride = dense::layout::stride_from_shape(shape);

    let index_1d = dense::layout::convert_nd_raw(multi_index, stride);
    let actual_nd = dense::layout::convert_1d_nd_from_shape(index_1d, shape);

    println!("{}, {:#?}", index_1d, actual_nd);

    for (&expected, actual) in multi_index.iter().zip(actual_nd) {
        assert_eq!(expected, actual)
    }
}

#[test]
fn test_insert_axis_back() {
    let shape = [3, 7, 6];
    let mut arr = rlst_dynamic_array3!(f64, shape);
    arr.fill_from_seed_equally_distributed(0);

    let new_arr = arr.view().insert_empty_axis(AxisPosition::Back);

    assert_eq!(new_arr.shape(), [3, 7, 6, 1]);

    assert_eq!(new_arr[[1, 2, 5, 0]], arr[[1, 2, 5]]);

    assert_eq!(new_arr.stride(), [1, 3, 21, 126])
}

#[test]
fn test_insert_axis_front() {
    let shape = [3, 7, 6];
    let mut arr = rlst_dynamic_array3!(f64, shape);
    arr.fill_from_seed_equally_distributed(0);

    let new_arr = arr.view().insert_empty_axis(AxisPosition::Front);

    assert_eq!(new_arr.shape(), [1, 3, 7, 6]);

    assert_eq!(new_arr[[0, 1, 2, 5]], arr[[1, 2, 5]]);

    assert_eq!(new_arr.stride(), [1, 1, 3, 21])
}

#[test]
fn test_create_slice() {
    let shape = [3, 7, 6];
    let mut arr = rlst_dynamic_array3!(f64, shape);

    arr.fill_from_seed_equally_distributed(0);

    let slice = arr.view().slice(1, 2);

    assert_eq!(slice[[1, 5]], arr[[1, 2, 5]]);

    assert_eq!(slice.shape(), [3, 6]);

    let stride_expected = [arr.stride()[0], arr.stride()[2]];
    let stride_actual = slice.stride();

    assert_eq!(stride_expected, stride_actual);

    let orig_data = arr.data();
    let slice_data = slice.data();

    let orig_raw_index = dense::layout::convert_nd_raw([1, 2, 5], arr.stride());
    let slice_raw_index = dense::layout::convert_nd_raw([1, 5], slice.stride());

    assert_eq!(orig_data[orig_raw_index], slice_data[slice_raw_index]);
    assert_eq!(slice_data[slice_raw_index], slice[[1, 5]]);

    let last_raw_index = dense::layout::convert_nd_raw([2, 5], slice.stride());
    assert_eq!(slice_data[last_raw_index], slice[[2, 5]]);
}

#[test]
fn test_multiple_slices() {
    let shape = [3, 7, 6];
    let mut arr = rlst_dynamic_array3!(f64, shape);
    arr.fill_from_seed_equally_distributed(0);

    let mut slice = arr.view_mut().slice(1, 3).slice(1, 1);

    slice[[2]] = 5.0;

    assert_eq!(slice.shape(), [3]);
    assert_eq!(arr[[2, 3, 1]], 5.0);
}

#[test]
fn test_slice_of_subview() {
    let shape = [3, 7, 6];
    let mut arr = rlst_dynamic_array3!(f64, shape);
    arr.fill_from_seed_equally_distributed(0);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    arr2.fill_from(arr.view());

    let slice = arr.into_subview([1, 2, 4], [2, 3, 2]).slice(1, 2);

    assert_eq!(slice.shape(), [2, 2]);

    assert_eq!(slice[[1, 0]], arr2[[2, 4, 4]]);

    let slice_data = slice.data();
    let arr_data = arr2.data();

    let slice_index = dense::layout::convert_nd_raw([1, 0], slice.stride());
    let array_index = dense::layout::convert_nd_raw([2, 4, 4], arr2.stride());

    assert_eq!(slice_data[slice_index], arr_data[array_index]);
}

macro_rules! mat_mul_test_impl {
        ($ScalarType:ty, $eps:expr) => {
            paste! {
                fn [<test_mat_mul_impl_$ScalarType>](transa: TransMode, transb: TransMode, shape_a: [usize; 2], shape_b: [usize; 2], shape_c: [usize; 2]) {

                    use rlst::dense::matrix_multiply::matrix_multiply;

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
