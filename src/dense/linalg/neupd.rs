//! Non-Symmetric Arnoldi Eigenvector Extraction
use crate::RlstScalar;
use arpack_ng_sys::{__BindgenComplex, cneupd_c, dneupd_c, sneupd_c, zneupd_c};
use num::Complex;

/// Implementation of the Non-Symmetric Arnoldi Eigenvector Extraction
pub trait NonSymmetricArnoldiExtract: RlstScalar {
    /// Non-Symmetric Arnoldi Eigenvector Extraction ARPACK function
    fn neupd(
        rvec: i32,
        howmny: &str,
        select: &mut [i32],
        d: &mut [<Self as RlstScalar>::Complex],
        z: &mut [Self],
        ldz: i32,
        sigmar: <Self as RlstScalar>::Real,
        sigmai: <Self as RlstScalar>::Real,
        workev: &mut [Self],
        bmat: &str,
        n: i32,
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

macro_rules! impl_neupd_complex_c {
    ($scalar:ty, $neupd_c:expr) => {
        impl NonSymmetricArnoldiExtract for Complex<$scalar> {
            fn neupd(
                rvec: i32,
                howmny: &str,
                select: &mut [i32],
                d: &mut [Complex<$scalar>],
                z: &mut [Complex<$scalar>],
                ldz: i32,
                sigmar: $scalar,
                sigmai: $scalar,
                workev: &mut [Complex<$scalar>],
                bmat: &str,
                n: i32,
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
                let sigmar_c = __BindgenComplex {
                    re: sigmar,
                    im: sigmai,
                };
                unsafe {
                    $neupd_c(
                        rvec,
                        howmny.as_ptr() as *const i8,
                        select.as_mut_ptr(),
                        d.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        z.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        ldz,
                        sigmar_c as __BindgenComplex<$scalar>,
                        workev.as_mut_ptr() as *mut __BindgenComplex<$scalar>,
                        bmat.as_ptr() as *const i8,
                        n,
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

impl_neupd_complex_c!(f32, cneupd_c);
impl_neupd_complex_c!(f64, zneupd_c);

macro_rules! impl_neupd_real_c {
    ($scalar:ty, $neupd_c:expr) => {
        impl NonSymmetricArnoldiExtract for $scalar {
            fn neupd(
                rvec: i32,
                howmny: &str,
                select: &mut [i32],
                d: &mut [Complex<Self>],
                z: &mut [Self],
                ldz: i32,
                sigmar: Self,
                sigmai: Self,
                workev: &mut [Self],
                bmat: &str,
                n: i32,
                which: &str,
                nev: i32,
                tol: $scalar,
                resid: &mut [Self],
                ncv: i32,
                v: &mut [Self],
                ldv: i32,
                iparam: &mut [i32; 11],
                ipntr: &mut [i32; 14],
                workd: &mut [Self],
                workl: &mut [Self],
                lworkl: i32,
                _rwork: &mut [Self], // unused in real case
                info: &mut i32,
            ) {
                let mut di: Vec<Self> = (0..nev).map(|_| num::Zero::zero()).collect();
                let mut dr: Vec<Self> = (0..nev).map(|_| num::Zero::zero()).collect();
                unsafe {
                    $neupd_c(
                        rvec,
                        howmny.as_ptr() as *const i8,
                        select.as_mut_ptr(),
                        dr.as_mut_ptr(),
                        di.as_mut_ptr(),
                        z.as_mut_ptr(),
                        ldz,
                        sigmar,
                        sigmai,
                        workev.as_mut_ptr(),
                        bmat.as_ptr() as *const i8,
                        n,
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

                for ((d_elem, &re), &im) in d.iter_mut().zip(dr.iter()).zip(di.iter()) {
                    *d_elem = Complex::new(re, im);
                }
            }
        }
    };
}

impl_neupd_real_c!(f32, sneupd_c);
impl_neupd_real_c!(f64, dneupd_c);
