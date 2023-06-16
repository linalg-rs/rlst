//! Set global threading for BLIS

use crate::raw::{self, bli_thread_set_num_threads};
use num_cpus;

/// Get the current number of threads used by BLIS.
pub fn get_num_threads() -> usize {
    let threads = unsafe { raw::bli_thread_get_num_threads() };

    threads as usize
}

/// Set threads to a given number of threads.
pub fn set_num_threads(nthreads: usize) {
    unsafe { bli_thread_set_num_threads(nthreads as i64) };
}

/// Set threads to the number of logical cpus.
pub fn enable_threading() {
    let num_cpus = num_cpus::get();

    unsafe { bli_thread_set_num_threads(num_cpus as i64) };
}

/// Set number of threads to 1.
pub fn disable_threading() {
    unsafe { bli_thread_set_num_threads(1) };
}
