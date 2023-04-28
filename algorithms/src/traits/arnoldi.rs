use rlst_common::basic_traits::Apply;
/// Arnoldi implemented for operators
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::MatrixD;

pub trait Arnoldi {
    // The Scalar Type
    type T: Scalar;

    // Element type of the underlying vector space
    type Element;

    // Initialize a new Arnoldi iteration.
    fn initialize(start: Self::Element, max_steps: usize) -> Self;

    // Perform a single Arnoldi step.
    fn arnoldi_step<Op: Apply<Self::Element, T = Self::T, Range = Self::Element>>(
        &self,
        operator: &Op,
        step_count: usize,
    ) -> RlstResult<()>;

    // Return a given basis element. Returns ```None``` if index out of bounds.
    fn basis_element(&self, index: usize) -> Option<&Self::Element>;

    // Return the projected Hessenberg matrix.
    fn hessenberg_matrix(&self) -> &MatrixD<Self::T>;

    // Consume the Arnoldi structure and return the basis and Hessenberg matrix.
    fn to_inner(self) -> (Vec<Self::Element>, MatrixD<Self::T>);
}
