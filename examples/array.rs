//! Test some array functions.

use rlst::{rlst_dynamic_array, AijIteratorByValue};

fn main() {
    // This is a placeholder for the main function.
    // You can add tests or examples of how to use the traits defined above.
    let mut arr = rlst_dynamic_array!(f64, [4, 5]);
    arr.fill_from_seed_equally_distributed(0);

    for ([i, j], v) in arr.iter_aij_value() {
        println!("{i} {j} {v}");
    }
}
