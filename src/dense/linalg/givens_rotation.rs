//! Givens rotations
use crate::{rlst_dynamic_array2, DynamicArray, RlstScalar};
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

/// Data associated to a Givens Rotation
pub struct GivensRotationData<Item: RlstScalar> {
    /// Cosine
    pub c: <Item as RlstScalar>::Real,
    /// Sin
    pub s: Item,
    /// Magnitude
    pub r: Item,
}

/// Implementation of a Givens Rotation
pub trait GivensRotation<Item: RlstScalar> {
    /// Creation of Givens Rotation
    fn new(f: Item, g: Item) -> Self;
    /// Extract the givens matrix
    fn get_givens_matrix(&self) -> DynamicArray<Item, 2>;
}

macro_rules! givens_rot {
    ($scalar:ty, $lartg:expr) => {
        impl GivensRotation<$scalar> for GivensRotationData<$scalar> {
            fn new(f: $scalar, g: $scalar) -> Self {
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

                        Self {
                            c: c[0],
                            s: s[0],
                            r: r[0],
                        }
                    }
                    _ => panic!("At least one value in the Givens rotation is not finite."),
                }
            }

            fn get_givens_matrix(&self) -> DynamicArray<$scalar, 2> {
                let mut mat = rlst_dynamic_array2!($scalar, [2, 2]);
                mat.r_mut()[[0, 0]] = <$scalar as RlstScalar>::from_real(self.c);
                mat.r_mut()[[0, 1]] = self.s;
                mat.r_mut()[[1, 0]] = -self.s.conj();
                mat.r_mut()[[1, 1]] = <$scalar as RlstScalar>::from_real(self.c);

                mat
            }
        }
    };
}

givens_rot!(f32, slartgp);
givens_rot!(f64, dlartgp);
givens_rot!(Complex<f32>, clartg);
givens_rot!(Complex<f64>, zlartg);

/// A set of consecutive Givens rotations
pub type GivensRotations<T> = Vec<GivensRotationData<T>>;

/// Define a trait for GivensRotations operations
pub trait GivensRotationsOps<Item: RlstScalar> {
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

// Implement the trait for Vec<GivensRotationData<Item>>
impl<Item: RlstScalar> GivensRotationsOps<Item> for Vec<GivensRotationData<Item>>
where
    GivensRotationData<Item>: GivensRotation<Item>,
{
    fn new() -> Self {
        Vec::new()
    }
    fn add(&mut self, f: Item, g: Item) {
        self.push(GivensRotation::new(f, g));
    }

    fn apply_rotation(&self, vec: &mut [Item; 2], ind: usize) {
        let gr = &self[ind];
        let c = Item::from_real(gr.c);
        let s = gr.s;
        let v0 = vec[0];
        let v1 = vec[1];

        vec[0] = c * v0 + s * v1;
        vec[1] = -s.conj() * v0 + c * v1;
    }

    fn get_rotated_vector(&self, ind: usize) -> [Item; 2] {
        let gr = &self[ind];
        [gr.r, num::Zero::zero()]
    }

    fn get_last(&self) -> (<Item as RlstScalar>::Real, Item, Item) {
        if let Some(gr) = self.last() {
            (gr.c, gr.s, gr.r)
        } else {
            panic!("No Givens rotations computed yet")
        }
    }
}
