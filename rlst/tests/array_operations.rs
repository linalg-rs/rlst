//! Tests of array algebray operations

use approx::assert_relative_eq;
use rlst::rlst_dynamic_array3;
use rlst_common::types::*;
use rlst_common::{assert_array_relative_eq, traits::*};
use rlst_dense::{array::iterators::AsMultiIndex, layout::convert_1d_nd_from_shape};

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
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    arr1.fill_from_seed_equally_distributed(0);

    let res = arr1.permute_axes([1, 2, 0]);

    assert_eq!(res.shape(), [4, 8, 3]);
}
