//! Tests of array algebray operations

use rlst::rlst_dynamic_array3;
use rlst_common::traits::*;
use rlst_dense::layout::{convert_1d_nd, stride_from_shape};

#[test]
fn test_addition() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);

    let arr3 = arr1.view() + arr2.view();

    res.fill_from_chunked::<_, 31>(arr3.view());

    for (res_item, (arr1_item, arr2_item)) in res.iter().zip(arr1.iter().zip(arr2.iter())) {
        approx::assert_relative_eq!(res_item, arr1_item + arr2_item, epsilon = 1E-14);
    }
}

#[test]
fn test_subtraction() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);

    let arr3 = arr1.view() - arr2.view();

    res.fill_from_chunked::<_, 31>(arr3.view());

    for (res_item, (arr1_item, arr2_item)) in res.iter().zip(arr1.iter().zip(arr2.iter())) {
        assert_eq!(res_item, arr1_item - arr2_item)
    }
}

#[test]
fn test_multiple_operations() {
    let shape = [3, 4, 8];
    let nelements = shape.iter().product();

    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);
    let mut compare = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(1);
    res.fill_from_seed_equally_distributed(2);
    compare.fill_from(res.view());

    let arr3 = 3.6 * arr1.view() - arr2.view();
    res.sum_into_chunked::<_, 64>(arr3.view());

    for index in 0..nelements {
        let indices = convert_1d_nd(index, stride_from_shape(shape));

        approx::assert_relative_eq!(
            res[indices],
            compare[indices] + 3.6 * arr1[indices] - arr2[indices],
            epsilon = 1E-14
        );
    }
}
