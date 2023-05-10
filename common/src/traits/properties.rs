//! Traits describing properties of objects.

/// Return the shape of the object.
pub trait Shape {
    fn shape(&self) -> (usize, usize);
}

/// Return the stride of the object.
pub trait Stride {
    fn stride(&self) -> (usize, usize);
}

/// Return the number of elements.
pub trait NumberOfElements {
    fn number_of_elements(&self) -> usize;
}
