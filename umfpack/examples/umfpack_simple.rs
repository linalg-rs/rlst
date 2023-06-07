//! Translation of umfpack_simple.c from the UMFPACK User Guide
use std::ffi::c_void;

use rlst_umfpack as umfpack;

pub fn main() {
    let n = 5;

    let ai = vec![0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4];
    let ap = vec![0, 2, 5, 9, 10, 12];
    let ax = vec![2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0];
    let b = vec![8.0, 45.0, -3.0, 3.0, 19.0];
    let mut symbolic = std::ptr::null_mut::<c_void>();
    let mut numeric = std::ptr::null_mut::<c_void>();
    let mut x = vec![0.0; 5];

    let _ = unsafe {
        umfpack::umfpack_di_symbolic(
            n,
            n,
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            &mut symbolic,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };

    let _ = unsafe {
        umfpack::umfpack_di_numeric(
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            symbolic,
            &mut numeric,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };

    let _ = unsafe { umfpack::umfpack_di_free_symbolic(&mut symbolic) };

    let _ = unsafe {
        umfpack::umfpack_di_solve(
            umfpack::UMFPACK_A as i32,
            ap.as_ptr(),
            ai.as_ptr(),
            ax.as_ptr(),
            x.as_mut_ptr(),
            b.as_ptr(),
            numeric,
            std::ptr::null(),
            std::ptr::null_mut(),
        )
    };

    let _ = unsafe { umfpack::umfpack_di_free_numeric(&mut numeric) };

    println!("Solution: {:#?}", x);
}
