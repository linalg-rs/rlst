//! Fixed size matrix example
//!
extern crate rlst_dense_proc_macro;

use rlst_dense::ExperimentalSizeIdentifier;
use rlst_dense_proc_macro::rlst_fixed_size;

#[rlst_fixed_size(33, 4)]
pub struct MySizeType;

pub fn main() {
    // match MySizeType::SIZE {
    //     rlst_dense::SizeValue::Fixed(m, n) => println!("{:#?}", (m, n)),
    //     rlst_dense::SizeValue::Dynamic => println!("Dynamic"),
    // }
}
