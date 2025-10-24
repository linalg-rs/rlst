//! Non-Symmetric Arnoldi Update
use crate::RlstScalar;
use arpack_ng_sys::{__BindgenComplex, cnaupd_c, dnaupd_c, snaupd_c, znaupd_c};
use num::Complex;

/// Implementation of the Non-Symmetric Arnoldi Update
pub trait NonSymmetricArnoldiUpdate: RlstScalar {
    /// Non-Symmetric Arnoldi Update ARPACK function
    fn naupd(
        ido: &mut i32,
        bmat: &str,
        dim: i32,
        which: &str,
        nev: i32,
        tol: <Self as RlstScalar>::Real,
        resid: &mut [Self],
        ncv: i32,
        v: &mut [Self],
        ldv: i32,
        iparam: &mut [i32; 11],
        ipntr: &mut [i32; 14],
        workd: &mut [Self],
        workl: &mut [Self],
        lworkl: i32,
        rwork: &mut [<Self as RlstScalar>::Real],
        info: &mut i32,
    );
}

macro_rules! impl_naupd_complex_c {
    ($scalar:ty, $naupd_c:expr) => {
        impl NonSymmetricArnoldiUpdate for Complex<$scalar> {
            fn naupd(
                ido: &mut i32,
                bmat: &str,
                dim: i32,
                which: &str,
                nev: i32,
                tol: $scalar,
                resid: &mut [Complex<$scalar>],
                ncv: i32,
                v: &mut [Complex<$scalar>],
                ldv: i32,
                iparam: &mut [i32; 11],
                ipntr: &mut [i32; 14],
                workd: &mut [Complex<$scalar>],
                workl: &mut [Complex<$scalar>],
                lworkl: i32,
                rwork: &mut [$scalar],
                info: &mut i32,
            ) {
                unsafe {
                    $naupd_c(
                        ido,
                        bmat.as_ptr() as *const i8,
                        dim,
                        which.as_ptr() as *const i8,
                        nev,
                        tol,
                        resid.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        ncv,
                        v.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        ldv,
                        iparam.as_mut_ptr(),
                        ipntr.as_mut_ptr(),
                        workd.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        workl.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        lworkl,
                        rwork.as_mut_ptr(),
                        info,
                    );
                }
            }
        }
    };
}

macro_rules! impl_naupd_real_c {
    ($scalar:ty, $naupd_c:expr) => {
        impl NonSymmetricArnoldiUpdate for $scalar {
            fn naupd(
                ido: &mut i32,
                bmat: &str,
                dim: i32,
                which: &str,
                nev: i32,
                tol: $scalar,
                resid: &mut [$scalar],
                ncv: i32,
                v: &mut [$scalar],
                ldv: i32,
                iparam: &mut [i32; 11],
                ipntr: &mut [i32; 14],
                workd: &mut [$scalar],
                workl: &mut [$scalar],
                lworkl: i32,
                _rwork: &mut [$scalar],
                info: &mut i32,
            ) {
                unsafe {
                    $naupd_c(
                        ido,
                        bmat.as_ptr() as *const i8,
                        dim,
                        which.as_ptr() as *const i8,
                        nev,
                        tol,
                        resid.as_mut_ptr(),
                        ncv,
                        v.as_mut_ptr(),
                        ldv,
                        iparam.as_mut_ptr(),
                        ipntr.as_mut_ptr(),
                        workd.as_mut_ptr(),
                        workl.as_mut_ptr(),
                        lworkl,
                        info,
                    );
                }
            }
        }
    };
}

// Usage for complex types:
impl_naupd_complex_c!(f32, cnaupd_c);
impl_naupd_complex_c!(f64, znaupd_c);

// Usage for real types:
impl_naupd_real_c!(f32, snaupd_c);
impl_naupd_real_c!(f64, dnaupd_c);
