//! Fixed size matrix example
//!

use rlst_common::types::c64;
use rlst_proc_macro::rlst_static_array;

pub fn main() {
    let _arr = rlst_static_array!(c64, 2, 2);
}
