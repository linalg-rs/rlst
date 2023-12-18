//! Modify the threading behaviour in BLIS

use rlst_blis::interface::threading;

fn main() {
    println!("Num threads: {}", threading::get_num_threads());
    println!("Setting threads to maximum cpu threads.");
    threading::enable_threading();
    println!("Num threads: {}", threading::get_num_threads());
}
