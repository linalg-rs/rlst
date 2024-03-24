//! Tests of array algebray operations

use rlst::dense::traits::DefaultIterator;
use rlst::prelude::rlst_dynamic_array3;

pub fn main() {
    let shape = [3, 4, 8];
    let mut arr1 = rlst_dynamic_array3!(f64, shape);
    let mut arr2 = rlst_dynamic_array3!(f64, shape);
    let mut res = rlst_dynamic_array3!(f64, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(0);

    let arr3 = arr1.view() + arr2.view();

    res.fill_from(arr3.view());

    for (res_item, (arr1_item, arr2_item)) in res.iter().zip(arr1.iter().zip(arr2.iter())) {
        assert_eq!(res_item, arr1_item + arr2_item)
    }
}
