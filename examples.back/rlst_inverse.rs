//! Demo the inverse of a matrix

pub use rlst::prelude::*;

pub fn main() {
    let mut arr = rlst_dynamic_array2!(f64, [2, 2]);
    let mut inverse = rlst_dynamic_array2!(f64, [2, 2]);
    arr.fill_from_seed_equally_distributed(0);
    inverse.fill_from(arr.r());

    inverse.r_mut().into_inverse_alloc().unwrap();

    println!("The original matrix is:");
    arr.pretty_print();

    println!("The inverse matrix is:");
    inverse.pretty_print();

    println!("The product is:");
    empty_array()
        .simple_mult_into_resize(arr.r(), inverse.r())
        .pretty_print();
}
