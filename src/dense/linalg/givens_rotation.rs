//! Givens rotations
use crate::RlstScalar;
use num::Complex;

// We expose the relevant LAPACK givens rotations
extern "C" {
    /// LAPACK Givens rotation for single precision
    pub fn slartg(f: *const f32, g: *const f32, c: *mut f32, s: *mut f32, r: *mut f32);
    /// LAPACK Givens rotation for double precision
    pub fn dlartg(f: *const f64, g: *const f64, c: *mut f64, s: *mut f64, r: *mut f64);
    /// LAPACK Givens rotation for complex single precision
    pub fn clartg(
        f: *const Complex<f32>,
        g: *const Complex<f32>,
        c: *mut f32,
        s: *mut Complex<f32>,
        r: *mut Complex<f32>,
    );
    /// LAPACK Givens rotation for complex double precision
    pub fn zlartg(
        f: *const Complex<f64>,
        g: *const Complex<f64>,
        c: *mut f64,
        s: *mut Complex<f64>,
        r: *mut Complex<f64>,
    );
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
                        let (mut c, mut s, mut r) =
                            (num::Zero::zero(), num::Zero::zero(), num::Zero::zero());
                        unsafe {
                            $lartg(&f, &g, &mut c, &mut s, &mut r);
                        }

                        self.c.push(c);
                        self.s.push(s);
                        self.r.push(r);
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
impl_givens_rot!(f32, slartg);
impl_givens_rot!(f64, dlartg);
impl_givens_rot!(Complex<f32>, clartg);
impl_givens_rot!(Complex<f64>, zlartg);
