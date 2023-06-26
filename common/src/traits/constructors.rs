//! Traits for the creation of new entities.

/// Create a new object from `self` initialized to zero.
pub trait NewLikeSelf {
    type Out;
    fn new_like_self(&self) -> Self::Out;
}

/// Create a new object from `self` with dimensions transposed.
pub trait NewLikeTranspose {
    type Out;
    fn new_like_transpose(&self) -> Self::Out;
}

/// Evaluate a matrix/vector expression into a new matrix/vector.
pub trait Eval {
    type Out;
    fn eval(&self) -> Self::Out;
}

/// Copy a matrix/vector expression into a new matrix/vector.
pub trait Copy {
    type Out;
    fn copy(&self) -> Self::Out;
}

pub trait Identity {
    type Out;

    /// Return an identity matrix with given shape.
    fn identity(shape: (usize, usize)) -> Self::Out;
}
