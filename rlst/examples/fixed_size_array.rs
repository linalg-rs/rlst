//! Fixed size matrix example
//!

use rlst_dense::types::c64;
use rlst_proc_macro::rlst_static_array;
use rlst_proc_macro::rlst_static_type;

pub fn main() {
    let _arr = rlst_static_array!(c64, 2, 2, 5);

    let _tmp: rlst_static_type!(f64, 3, 5);
}
