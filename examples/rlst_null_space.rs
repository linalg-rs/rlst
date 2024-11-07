//! Demo the inverse of a matrix
use rlst::dense::linalg::null_space::NullSpaceType;
pub use rlst::prelude::*;

//Here we compute the null space (B) of the rowspace of a short matrix (A).
pub fn main() {
    let mut arr = rlst_dynamic_array2!(f64, [3, 4]);
    arr.fill_from_seed_equally_distributed(0);

    let null_res = arr.view_mut().into_null_alloc(NullSpaceType::Row).unwrap();
    let res = empty_array().simple_mult_into_resize(arr.view_mut(), null_res.null_space_arr.view());

    println!("Value of |A*B|_2, where B=null(A): {}", res.view_flat().norm_2());

}
