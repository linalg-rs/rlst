//! Definition of a general linear operator.

use crate::{
    operator::{
        abstract_operator::{Operator, RlstOperatorReference},
        element::Element,
    },
    LinearSpace,
};

/// A base operator trait.
pub trait OperatorBase {
    /// Domain space type
    type Domain: LinearSpace;
    /// Range space type
    type Range: LinearSpace;

    /// Get the domain
    fn domain(&self) -> &Self::Domain;
    /// Get the range
    fn range(&self) -> &Self::Range;

    /// Convert to RLST reference
    fn r(&self) -> Operator<RlstOperatorReference<'_, Self>>
    where
        Self: Sized,
    {
        Operator::new(RlstOperatorReference::new(self))
    }

    /// Apply an operator as y -> alpha * Ax + beta y
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut Element<Self::Range>,
    );
}
