//! Demo the null space of a matrix.
use rlst::dense::linalg::null_space::Method;
pub use rlst::prelude::*;

//Here we compute the null space (B) of the rowspace of a short matrix (A).
pub fn main() {
    let mut arr = rlst_dynamic_array2!(f64, [3, 4]);
    arr.fill_from_seed_equally_distributed(0);
    let tol = 1e-15;
    let null_res = arr.r_mut().into_null_alloc(tol, Method::Svd).unwrap();
    let res: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> =
        empty_array().simple_mult_into_resize(arr.r_mut(), null_res.null_space_arr.r());
    //Result
    println!(
        "Nullification via SVD. Value of |A*B|_2, where B=null(A): {}",
        res.view_flat().norm_2()
    );

    let mut arr = rlst_dynamic_array2!(f64, [3, 4]);
    arr.fill_from_seed_equally_distributed(0);
    let shape = arr.shape();
    let mut arr_qr = rlst_dynamic_array2!(f64, [shape[1], shape[0]]);
    arr_qr.fill_from(arr.r().conj().transpose());
    let null_res = arr_qr.r_mut().into_null_alloc(tol, Method::Qr).unwrap();
    let res: Array<f64, BaseArray<f64, VectorContainer<f64>, 2>, 2> =
        empty_array().simple_mult_into_resize(arr.r_mut(), null_res.null_space_arr.r());
    //Result
    println!(
        "Nullification via QR. Value of |A*B|_2, where B=null(A): {}",
        res.view_flat().norm_2()
    );
}
