//! Givens rotations
use crate::RlstScalar;
use num::Complex;

#[inline]
unsafe fn dlartgp(f: &[f64], g: &[f64], cs: &mut [f64], sn: &mut [f64], r: &mut [f64]) {
    assert!(!f.is_empty() && !g.is_empty());
    assert!(!cs.is_empty() && !sn.is_empty() && !r.is_empty());

    let a = f[0];
    let b = g[0];

    if b == 0.0 {
        cs[0] = 1.0;
        sn[0] = 0.0;
        r[0] = a;
    } else {
        let rho = (a * a + b * b).sqrt();
        cs[0] = a / rho;
        sn[0] = b / rho;
        r[0] = rho;
    }
}

#[inline]
unsafe fn slartgp(f: &[f32], g: &[f32], cs: &mut [f32], sn: &mut [f32], r: &mut [f32]) {
    assert!(!f.is_empty() && !g.is_empty());
    assert!(!cs.is_empty() && !sn.is_empty() && !r.is_empty());

    let a = f[0];
    let b = g[0];

    if b == 0.0 {
        cs[0] = 1.0;
        sn[0] = 0.0;
        r[0] = a;
    } else {
        let rho = (a * a + b * b).sqrt();
        cs[0] = a / rho;
        sn[0] = b / rho;
        r[0] = rho;
    }
}

#[inline]
unsafe fn zlartg(
    f: &[Complex<f64>],
    g: &[Complex<f64>],
    cs: &mut [f64],
    sn: &mut [Complex<f64>],
    r: &mut [Complex<f64>],
) {
    use lapack_sys::__BindgenComplex;

    assert!(!f.is_empty() && !g.is_empty());
    assert!(!cs.is_empty() && !sn.is_empty() && !r.is_empty());

    let f_ptr = f.as_ptr() as *const __BindgenComplex<f64>;
    let g_ptr = g.as_ptr() as *const __BindgenComplex<f64>;
    let cs_ptr = cs.as_mut_ptr();
    let sn_ptr = sn.as_mut_ptr() as *mut __BindgenComplex<f64>;
    let r_ptr = r.as_mut_ptr() as *mut __BindgenComplex<f64>;

    lapack_sys::zlartg_(f_ptr, g_ptr, cs_ptr, sn_ptr, r_ptr)
}

#[inline]
unsafe fn clartg(
    f: &[Complex<f32>],
    g: &[Complex<f32>],
    cs: &mut [f32],
    sn: &mut [Complex<f32>],
    r: &mut [Complex<f32>],
) {
    use lapack_sys::__BindgenComplex;

    assert!(!f.is_empty() && !g.is_empty());
    assert!(!cs.is_empty() && !sn.is_empty() && !r.is_empty());

    let f_ptr = f.as_ptr() as *const __BindgenComplex<f32>;
    let g_ptr = g.as_ptr() as *const __BindgenComplex<f32>;
    let cs_ptr = cs.as_mut_ptr();
    let sn_ptr = sn.as_mut_ptr() as *mut __BindgenComplex<f32>;
    let r_ptr = r.as_mut_ptr() as *mut __BindgenComplex<f32>;

    lapack_sys::clartg_(f_ptr, g_ptr, cs_ptr, sn_ptr, r_ptr)
}

/// This structure stores a set of Givens rotations extracted from matrix
pub struct GivensRotationsData<Item: RlstScalar> {
    /// Cosine
    pub c: Vec<<Item as RlstScalar>::Real>,
    /// Sin
    pub s: Vec<Item>,
    /// Magnitude
    pub r: Vec<Item>,
}

/// Implementation of Givens Rotations
pub trait GivensRotations<Item: RlstScalar> {
    /// Create the Givens Rotations structure
    fn new() -> Self;
    /// Creation of Givens Rotation
    fn add(&mut self, f: Item, g: Item);

    /// Application of Givens Rotation for a vector vec
    fn apply_rotation(&self, vec: &mut [Item; 2], ind: usize);

    /// Return rotated [f, g] vector
    fn get_rotated_vector(&self, ind: usize) -> [Item; 2];

    /// Return the last set of givens rotations
    fn get_last(&self) -> (<Item as RlstScalar>::Real, Item, Item);
}

macro_rules! impl_givens_rot {
    ($scalar:ty, $lartg:expr) => {
        impl GivensRotations<$scalar> for GivensRotationsData<$scalar> {
            fn new() -> Self {
                let c = Vec::new();
                let s = Vec::new();
                let r = Vec::new();

                Self { c, s, r }
            }

            fn add(&mut self, f: $scalar, g: $scalar) {
                match f.is_finite() && g.is_finite() {
                    true => {
                        let mut c = [num::Zero::zero()];
                        let mut s = [num::Zero::zero()];
                        let mut r = [num::Zero::zero()];
                        let f_arr = [f];
                        let g_arr = [g];
                        unsafe {
                            $lartg(&f_arr, &g_arr, &mut c, &mut s, &mut r);
                        }

                        self.c.push(c[0]);
                        self.s.push(s[0]);
                        self.r.push(r[0]);
                    }
                    _ => panic!("At least one value in the Givens rotation is not finite."),
                }
            }

            fn apply_rotation(&self, vec: &mut [$scalar; 2], ind: usize) {
                let v0 = vec[0];
                let v1 = vec[1];

                vec[0] = self.c[ind] * v0 + self.s[ind] * v1;
                vec[1] = -self.s[ind].conj() * v0 + self.c[ind] * v1;
            }

            fn get_rotated_vector(&self, ind: usize) -> [$scalar; 2] {
                [self.r[ind], num::Zero::zero()]
            }

            fn get_last(&self) -> (<$scalar as RlstScalar>::Real, $scalar, $scalar) {
                if let (Some(c), Some(s), Some(r)) = (self.c.last(), self.s.last(), self.r.last()) {
                    (*c, *s, *r)
                } else {
                    panic!("No Givens rotations computed yet")
                }
            }
        }
    };
}

// Implementations of the trait for different types
impl_givens_rot!(f32, slartgp);
impl_givens_rot!(f64, dlartgp);
impl_givens_rot!(Complex<f32>, clartg);
impl_givens_rot!(Complex<f64>, zlartg);
