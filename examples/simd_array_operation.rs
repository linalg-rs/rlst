//! Testing SIMD Array Operations

use rlst::prelude::*;

fn main() {
    let mut a = rlst_dynamic_array1!(f32, [20]);
    let mut b = rlst_dynamic_array1!(f32, [20]);

    a.fill_from_seed_normally_distributed(1);

    b.fill_from(3.0 * a);

    println!("Finished: {:?}", b[[0]]);
}
