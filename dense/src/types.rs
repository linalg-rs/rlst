//! HScalar types used by the library

/// The [HScalar] trait describes scalar floating types. It is implemented
/// for [f32], [f64], [c32], [c64].
pub use cauchy::Scalar;
pub trait HScalar: cauchy::Scalar {
    type Real;
    fn atan2(self, x: Self) -> Self;
}

/// Single precision complex type.
pub use cauchy::c32;

/// Double precision complex type.
pub use cauchy::c64;

/// The index type used throughout this crate. By default
/// it is set to `usize`.
pub type IndexType = usize;

impl HScalar for f32 {
    type Real = f32;

    fn atan2(self, x: f32) -> f32 {
        self.atan2(x)
    }
}

impl HScalar for f64 {
    type Real = f64;

    fn atan2(self, x: f64) -> f64 {
        self.atan2(x)
    }
}

impl HScalar for c32 {
    type Real = f32;

    fn atan2(self, _x: c32) -> c32 {
        unimplemented!("atan2 is not defined for complex numbers");
    }
}

impl HScalar for c64 {
    type Real = f64;
    fn atan2(self, _x: c64) -> c64 {
        unimplemented!("atan2 is not defined for complex numbers");
    }
}
