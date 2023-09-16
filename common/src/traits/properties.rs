//! Traits describing properties of objects.

/// Return the shape of the object.
pub trait Shape<const NDIM: usize> {
    fn shape(&self) -> [usize; NDIM];

    /// Return true if a dimension is 0.
    fn is_empty(&self) -> bool {
        let shape = self.shape();
        for elem in shape {
            if elem == 0 {
                return true;
            }
        }
        false
    }
}

/// Return the stride of the object.
pub trait Stride<const NDIM: usize> {
    fn stride(&self) -> [usize; NDIM];
}

/// Return the number of elements.
pub trait NumberOfElements {
    fn number_of_elements(&self) -> usize;
}
