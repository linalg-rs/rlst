//! Operators on arrays

pub mod addition;
pub mod cast;
pub mod cmp_wise_division;
pub mod cmp_wise_product;
pub mod conjugate;
pub mod other;
pub mod scalar_mult;
pub mod subtraction;
pub mod to_complex;
pub mod transpose;

/// Test SIMD
pub fn test_simd() {
    use crate::rlst_dynamic_array2;

    let shape = [200, 300];
    let mut arr1 = rlst_dynamic_array2!(f32, shape);
    let mut arr2 = rlst_dynamic_array2!(f32, shape);
    let mut res = rlst_dynamic_array2!(f32, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(0);

    // let arr3 = arr1.view() + arr2.view();

    let arr3 = 3.0 * arr1 + arr2;

    res.fill_from_chunked::<_, 512>(arr3.view());
    //res.fill_from(arr3.view());

    println!("{}", res[[0, 0]]);
}
