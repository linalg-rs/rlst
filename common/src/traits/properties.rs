//! Traits describing properties of objects.

/// Return the shape of the object.
pub trait Shape {
    fn shape(&self) -> (usize, usize);

    /// Return true if a dimension is 0.
    fn is_empty(&self) -> bool {
        let shape = self.shape();

        shape.0 == 0 || shape.1 == 0
    }
}

/// Return the stride of the object.
pub trait Stride {
    fn stride(&self) -> (usize, usize);
}

/// Return the number of elements.
pub trait NumberOfElements {
    fn number_of_elements(&self) -> usize;
}
