//! Type forwards

/// The [Scalar] trait describes scalar floating types. It is implemented
/// for [f32], [f64], [c32], [c64].
pub use rlst_common::types::Scalar;
// pub trait Scalar: cauchy::Scalar {
//     type Real;
//     fn atan2(self, x: Self) -> Self;
// }

/// Single precision complex type.
pub use rlst_common::types::c32;

/// Double precision complex type.
pub use rlst_common::types::c64;
