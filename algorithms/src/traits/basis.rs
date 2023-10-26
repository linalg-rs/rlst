//! Trait describing a vector space basis.

pub trait Basis {
    type Element;

    /// Dimension of the basis.
    fn dim() -> usize;

    /// Create a basis element.
    fn basis_element(index: usize) -> Option<Self::Element>;
}

pub trait GrowableBasis: Basis {
    fn add_element(elem: Self::Element);
}
