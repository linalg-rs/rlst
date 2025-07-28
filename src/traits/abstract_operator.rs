//! Definition of a general linear operator.

use num::{One, Zero};

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

    /// Apply an operator as y -> alpha * Ax + beta y.
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut Element<Self::Range>,
    );

    /// Apply the operator to `x`.
    #[inline(always)]
    fn dot(&self, x: &Element<Self::Domain>) -> Element<Self::Range>
    where
        <Self::Range as crate::LinearSpace>::F: One + Zero,
    {
        let mut res = self.range().zero();
        self.apply(One::one(), x, Zero::zero(), &mut res);
        res
    }
}
