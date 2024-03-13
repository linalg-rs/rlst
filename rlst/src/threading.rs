//! Control BLAS threading if possible

use num_cpus;

#[cfg(feature = "blis")]
extern "C" {
    fn bli_thread_set_num_threads(n_threads: i64);
    fn bli_thread_get_num_threads() -> i64;
}

#[cfg(feature = "blis")]
/// Get the current number of threads used by Blis.
pub fn get_num_threads() -> usize {
    let threads = unsafe { bli_thread_get_num_threads() };

    threads as usize
}

#[cfg(feature = "blis")]
/// Set threads to a given number of threads.
pub fn set_num_threads(nthreads: usize) {
    unsafe { bli_thread_set_num_threads(nthreads as i64) };
}

#[cfg(feature = "blis")]
/// Set threads to the number of logical cpus.
pub fn enable_threading() {
    let num_cpus = num_cpus::get();

    unsafe { bli_thread_set_num_threads(num_cpus as i64) };
}

#[cfg(feature = "blis")]
/// Set number of threads to 1.
pub fn disable_threading() {
    unsafe { bli_thread_set_num_threads(1) };
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_threading() {
        println!("Threads: {}", get_num_threads());
    }
}
