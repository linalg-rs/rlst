//! Fixed size matrix example
//!

use rlst_dense::SizeIdentifier;
use rlst_proc_macro::rlst_static_size;

#[rlst_static_size(33, 4)]
pub struct MySizeType;

pub fn main() {
    match MySizeType::SIZE {
        rlst_dense::SizeIdentifierValue::Static(m, n) => println!("{:#?}", (m, n)),
        rlst_dense::SizeIdentifierValue::Dynamic => println!("Dynamic"),
    }
}
