//! Definition of a general linear operator.

use crate::{FieldType, LinearSpace};
use num::{One, Zero};
use rlst_dense::types::RlstResult;
use std::fmt::Debug;

/// A base operator trait.
pub trait OperatorBase: Debug {
    /// Domain space type
    type Domain: LinearSpace;
    /// Range space type
    type Range: LinearSpace;

    /// Get the domain
    fn domain(&self) -> &Self::Domain;
    /// Get the range
    fn range(&self) -> &Self::Range;
    /// Convert to RLST reference
    fn as_ref_obj(&self) -> RlstOperatorReference<'_, Self>
    where
        Self: Sized,
    {
        RlstOperatorReference(self)
    }
    /// Form a new operator alpha * self.
    fn scale(self, alpha: <Self::Range as LinearSpace>::F) -> ScalarTimesOperator<Self>
    where
        Self: Sized,
    {
        ScalarTimesOperator(self, alpha)
    }
    /// Form a new operator self + other.
    fn sum<Op: OperatorBase<Domain = Self::Domain, Range = Self::Range> + Sized>(
        self,
        other: Op,
    ) -> OperatorSum<Self::Domain, Self::Range, Self, Op>
    where
        Self: Sized,
    {
        OperatorSum(self, other)
    }

    /// Form a new operator self * other.
    fn product<Op: OperatorBase<Range = Self::Domain>>(
        self,
        other: Op,
    ) -> OperatorProduct<Self::Domain, Op, Self>
    where
        Self: Sized,
    {
        OperatorProduct {
            op1: other,
            op2: self,
        }
    }
}

/// Apply an operator as y -> alpha * Ax + beta y
pub trait AsApply: OperatorBase {
    /// Apply an operator as y -> alpha * Ax + beta y
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()>;

    /// Apply
    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        let mut out = self.range().zero();

        self.apply_extended(
            <FieldType<Self::Range> as One>::one(),
            x,
            <FieldType<Self::Range> as Zero>::zero(),
            &mut out,
        )
        .unwrap();
        out
    }
}

/// Operator reference
pub struct RlstOperatorReference<'a, Op: OperatorBase>(&'a Op);

impl<Op: OperatorBase> std::fmt::Debug for RlstOperatorReference<'_, Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorReference").field(&self.0).finish()
    }
}

impl<Op: OperatorBase> OperatorBase for RlstOperatorReference<'_, Op> {
    type Domain = Op::Domain;

    type Range = Op::Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }
}

impl<Op: AsApply> AsApply for RlstOperatorReference<'_, Op> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.0.apply_extended(alpha, x, beta, y)
    }
}

/// Operator sum
pub struct OperatorSum<
    Domain: LinearSpace,
    Range: LinearSpace,
    Op1: OperatorBase<Domain = Domain, Range = Range>,
    Op2: OperatorBase<Domain = Domain, Range = Range>,
>(Op1, Op2);

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        Op1: OperatorBase<Domain = Domain, Range = Range>,
        Op2: OperatorBase<Domain = Domain, Range = Range>,
    > std::fmt::Debug for OperatorSum<Domain, Range, Op1, Op2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorSum")
            .field(&&self.0)
            .field(&&self.1)
            .finish()
    }
}

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        Op1: OperatorBase<Domain = Domain, Range = Range>,
        Op2: OperatorBase<Domain = Domain, Range = Range>,
    > OperatorBase for OperatorSum<Domain, Range, Op1, Op2>
{
    type Domain = Domain;

    type Range = Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }
}

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        Op1: AsApply<Domain = Domain, Range = Range>,
        Op2: AsApply<Domain = Domain, Range = Range>,
    > AsApply for OperatorSum<Domain, Range, Op1, Op2>
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.0.apply_extended(alpha, x, beta, y)?;
        self.1
            .apply_extended(alpha, x, <<Self::Range as LinearSpace>::F as One>::one(), y)?;
        Ok(())
    }
}

/// Operator muiltiplied by a scalar
pub struct ScalarTimesOperator<Op: OperatorBase>(Op, <Op::Range as LinearSpace>::F);

impl<Op: OperatorBase> std::fmt::Debug for ScalarTimesOperator<Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorSum")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<Op: OperatorBase> OperatorBase for ScalarTimesOperator<Op> {
    type Domain = Op::Domain;

    type Range = Op::Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }
}

impl<Op: AsApply> AsApply for ScalarTimesOperator<Op> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.0.apply_extended(self.1 * alpha, x, beta, y)?;
        Ok(())
    }
}

/// The product op2 * op1.
pub struct OperatorProduct<
    Space: LinearSpace,
    Op1: OperatorBase<Range = Space>,
    Op2: OperatorBase<Domain = Space>,
> {
    op1: Op1,
    op2: Op2,
}

impl<Space: LinearSpace, Op1: OperatorBase<Range = Space>, Op2: OperatorBase<Domain = Space>> Debug
    for OperatorProduct<Space, Op1, Op2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProductOperator")
            .field("op1", &self.op1)
            .field("op2", &self.op2)
            .finish()
    }
}

impl<Space: LinearSpace, Op1: OperatorBase<Range = Space>, Op2: OperatorBase<Domain = Space>>
    OperatorBase for OperatorProduct<Space, Op1, Op2>
{
    type Domain = Op1::Domain;

    type Range = Op2::Range;

    fn domain(&self) -> &Self::Domain {
        self.op1.domain()
    }

    fn range(&self) -> &Self::Range {
        self.op2.range()
    }
}

impl<Space: LinearSpace, Op1: AsApply<Range = Space>, Op2: AsApply<Domain = Space>> AsApply
    for OperatorProduct<Space, Op1, Op2>
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.op2.apply_extended(alpha, &self.op1.apply(x), beta, y)
    }
}
