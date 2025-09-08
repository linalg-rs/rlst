//! BLAS threading control. This module provides interfaces to threading control functions in BLAS libraries that support it.

use std::ffi::c_int;

/// Return the number of physical CPU cores.
///
/// Note that physical CPU core information is only
/// support on Linux, Mac OS or Windows. On other systems
/// the logical number of cores is returned, which may be higher.
pub fn get_num_cores() -> usize {
    num_cpus::get()
}

#[cfg(feature = "blis_threading")]
extern "C" {
    fn bli_thread_set_num_threads(nthreads: c_int);
}

#[cfg(feature = "mkl_threading")]
extern "C" {
    fn mkl_set_num_threads(nthreads: c_int);
}

#[cfg(feature = "mkl_threading")]
extern "C" {
    fn openblas_set_num_threads(nthreads: c_int);
}

/// Set the number of threads for BLAS threading.
pub fn set_blas_threads(nthreads: usize) {
    #[cfg(feature = "blis_threading")]
    bli_thread_set_num_threads(nthreads as c_int);

    #[cfg(feature = "mkl_threading")]
    mkl_set_num_threads(nthreads as c_int);

    #[cfg(feature = "openblas_threading")]
    openblas_set_num_threads(nthreads as c_int);
}
