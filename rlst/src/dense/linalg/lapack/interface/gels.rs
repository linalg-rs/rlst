//! Implementation of ?gels - Solve the least squares problem A * X = B or A^H * X = B.

use std::ffi::c_char;

use lapack::{cgels, sgels, zgels};

use crate::base_types::LapackResult;
use crate::base_types::{LapackError, c32, c64};

use crate::dense::linalg::lapack::interface::lapack_return;

use num::{Zero, complex::ComplexFloat};

/// Transposition modes for `?gels`.
pub enum GelsTransMode {
    /// No transpose
    NoTranspose,
    /// Conjugate transpose
    ConjugateTranspose,
}

/// ?gels - Solve the least squares problem A * X = B or A^H * X = B.
pub trait Gels: Sized {
    /// Solve the least squares problem A * X = B or A^H * X = B.
    ///
    /// **Arguments:**
    /// - `trans`: Transpose mode, either 'N' for no transpose or 'C' for conjugate transpose.
    /// - `m`: Number of rows in matrix A.
    /// - `n`: Number of columns in matrix A.
    /// - `nrhs`: Number of right-hand sides (columns in B).
    /// - `a`: The matrix A.
    /// - `lda`: Leading dimension of A.
    /// - `b`: The right-hand side matrix B.
    /// - `ldb`: Leading dimension of B.
    ///
    /// **Returns:**
    /// A `LapackResult<()>` indicating success or failure.
    #[allow(clippy::too_many_arguments)]
    fn gels(
        trans: GelsTransMode,
        m: usize,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> LapackResult<()>;
}

trait TransposeToChar {
    fn to_char(trans: GelsTransMode) -> u8;
}

macro_rules! implement_transpose_to_char {
    ($scalar:ty, $char:expr) => {
        impl TransposeToChar for $scalar {
            fn to_char(trans: GelsTransMode) -> u8 {
                match trans {
                    GelsTransMode::NoTranspose => b'N',
                    GelsTransMode::ConjugateTranspose => $char,
                }
            }
        }
    };
}

implement_transpose_to_char!(f32, b'T');
implement_transpose_to_char!(f64, b'T');
implement_transpose_to_char!(c32, b'C');
implement_transpose_to_char!(c64, b'C');

macro_rules! implement_gels {
    ($scalar:ty, $gels:expr) => {
        impl Gels for $scalar {
            fn gels(
                trans: GelsTransMode,
                m: usize,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> LapackResult<()> {
                let mut info = 0;

                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n
                );

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, m)` {}.",
                    lda,
                    std::cmp::max(1, m)
                );

                assert_eq!(
                    b.len(),
                    ldb * nrhs,
                    "Require `b.len()` {} == `ldb * nrhs` {}.",
                    b.len(),
                    ldb * nrhs
                );

                assert!(
                    ldb >= std::cmp::max(1, std::cmp::max(m, n)),
                    "Require `ldb` {} >= `max(1, max(m, n))` {}.",
                    ldb,
                    std::cmp::max(1, std::cmp::max(m, n))
                );

                let mut work = vec![<$scalar>::zero(); 1];

                let trans = <$scalar as TransposeToChar>::to_char(trans);

                unsafe {
                    $gels(
                        trans,
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        &mut work,
                        -1,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar>::zero(); lwork as usize];

                unsafe {
                    $gels(
                        trans,
                        m as i32,
                        n as i32,
                        nrhs as i32,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

unsafe extern "C" {

    /// FFI bindings for the LAPACK `dgels` function.
    ///
    /// This is necessary because of a bug in the `lapack-sys` library that adds a non-existent
    /// parameter to the `dgels` signature.
    pub fn dgels_(
        trans: *const std::ffi::c_char,
        m: *const std::ffi::c_int,
        n: *const std::ffi::c_int,
        nrhs: *const std::ffi::c_int,
        A: *mut f64,
        lda: *const std::ffi::c_int,
        B: *mut f64,
        ldb: *const std::ffi::c_int,
        work: *mut f64,
        lwork: *const std::ffi::c_int,
        info: *mut std::ffi::c_int,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn dgels(
    trans: u8,
    m: i32,
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    b: &mut [f64],
    ldb: i32,
    work: &mut [f64],
    lwork: i32,
    info: &mut i32,
) {
    unsafe {
        dgels_(
            &(trans as c_char),
            &m,
            &n,
            &nrhs,
            a.as_mut_ptr(),
            &lda,
            b.as_mut_ptr(),
            &ldb,
            work.as_mut_ptr(),
            &lwork,
            info,
        )
    }
}

implement_gels!(f32, sgels);
implement_gels!(f64, dgels);
implement_gels!(c32, cgels);
implement_gels!(c64, zgels);
