//! Definition of a general linear operator.

use crate::operator::element::Element;
use crate::operator::LinearSpace;
use crate::RlstScalar;
use num::One;
use std::fmt::Debug;
use std::rc::Rc;

/// A base operator trait.
pub trait OperatorBase: Debug {
    /// Domain space type
    type Domain: LinearSpace;
    /// Range space type
    type Range: LinearSpace;

    /// Get the domain
    fn domain(&self) -> Rc<Self::Domain>;
    /// Get the range
    fn range(&self) -> Rc<Self::Range>;

    /// Get a zero in the domain space.
    fn domain_zero(&self) -> <Self::Domain as LinearSpace>::E {
        <Self::Domain as LinearSpace>::zero(self.domain())
    }

    /// Get a zero in the range space.
    fn range_zero(&self) -> <Self::Range as LinearSpace>::E {
        <Self::Range as LinearSpace>::zero(self.range())
    }

    /// Convert to RLST reference
    fn r(&self) -> RlstOperatorReference<'_, Self>
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

    /// Take the difference self - other.
    fn sub<Op: OperatorBase<Domain = Self::Domain, Range = Self::Range> + Sized>(
        self,
        other: Op,
    ) -> OperatorSum<Self::Domain, Self::Range, Self, ScalarTimesOperator<Op>>
    where
        Self: Sized,
    {
        self.sum(other.neg())
    }

    /// Return the negative -self.
    fn neg(self) -> ScalarTimesOperator<Self>
    where
        Self: Sized,
    {
        ScalarTimesOperator(self, -<<Self::Range as LinearSpace>::F as One>::one())
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
    );

    /// Apply
    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E;
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

    fn domain(&self) -> Rc<Self::Domain> {
        self.0.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.0.range().clone()
    }
}

impl<Op: AsApply> AsApply for RlstOperatorReference<'_, Op> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) {
        self.0.apply_extended(alpha, x, beta, y);
    }

    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        self.0.apply(x)
    }
}

/// A concrete operator.
pub struct Operator<OpImpl: OperatorBase>(OpImpl);

impl<OpImpl: OperatorBase> std::fmt::Debug for Operator<OpImpl> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Operator").field(&self.0).finish()
    }
}

impl<OpImpl: OperatorBase> OperatorBase for Operator<OpImpl> {
    type Domain = OpImpl::Domain;

    type Range = OpImpl::Range;

    fn domain(&self) -> Rc<Self::Domain> {
        self.0.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.0.range().clone()
    }
}

impl<OpImpl: AsApply> AsApply for Operator<OpImpl> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) {
        self.0.apply_extended(alpha, x, beta, y);
    }

    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        self.0.apply(x)
    }
}

impl<OpImpl: OperatorBase> Operator<OpImpl> {
    /// Create a new Operator
    pub fn new(op: OpImpl) -> Self {
        Operator(op)
    }
}

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
        OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
    > std::ops::Add<Operator<OpImpl2>> for Operator<OpImpl1>
{
    type Output = Operator<OperatorSum<Domain, Range, OpImpl1, OpImpl2>>;

    fn add(self, rhs: Operator<OpImpl2>) -> Self::Output {
        Operator::new(self.0.sum(rhs.0))
    }
}

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
        OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
    > std::ops::Sub<Operator<OpImpl2>> for Operator<OpImpl1>
{
    type Output = Operator<OperatorSum<Domain, Range, OpImpl1, ScalarTimesOperator<OpImpl2>>>;

    fn sub(self, rhs: Operator<OpImpl2>) -> Self::Output {
        Operator::new(self.0.sub(rhs.0))
    }
}

/// Trait that is satisfied by each scalar type that can multiply an operator from the left.
pub trait OperatorLeftScalarMul<OpImpl: OperatorBase>: std::ops::Mul<Operator<OpImpl>> {}

impl<
        T: RlstScalar + std::ops::Mul<Operator<OpImpl>>,
        Domain: LinearSpace,
        Range: LinearSpace<F = T>,
        OpImpl: OperatorBase<Domain = Domain, Range = Range>,
    > OperatorLeftScalarMul<OpImpl> for T
{
}

macro_rules! impl_operator_mul {
    ($dtype:ty) => {
        impl<
                Domain: LinearSpace,
                Range: LinearSpace<F = $dtype>,
                OpImpl: OperatorBase<Domain = Domain, Range = Range>,
            > std::ops::Mul<Operator<OpImpl>> for $dtype
        {
            type Output = Operator<ScalarTimesOperator<OpImpl>>;

            fn mul(self, rhs: Operator<OpImpl>) -> Self::Output {
                Operator::new(rhs.0.scale(self))
            }
        }
    };
}

impl_operator_mul!(f32);
impl_operator_mul!(f64);
impl_operator_mul!(crate::dense::types::c32);
impl_operator_mul!(crate::dense::types::c64);

impl<OpImpl: OperatorBase> std::ops::Neg for Operator<OpImpl> {
    type Output = Operator<ScalarTimesOperator<OpImpl>>;

    fn neg(self) -> Self::Output {
        Operator::new(self.0.neg())
    }
}

impl<
        T: RlstScalar,
        Domain: LinearSpace,
        Range: LinearSpace<F = T>,
        OpImpl: OperatorBase<Domain = Domain, Range = Range>,
    > std::ops::Mul<T> for Operator<OpImpl>
{
    type Output = Operator<ScalarTimesOperator<OpImpl>>;

    fn mul(self, rhs: T) -> Self::Output {
        Operator::new(self.0.scale(rhs))
    }
}

impl<
        T: RlstScalar,
        Domain: LinearSpace,
        Range: LinearSpace<F = T>,
        OpImpl: OperatorBase<Domain = Domain, Range = Range>,
    > std::ops::Div<T> for Operator<OpImpl>
{
    type Output = Operator<ScalarTimesOperator<OpImpl>>;

    fn div(self, rhs: T) -> Self::Output {
        Operator::new(self.0.scale(T::one() / rhs))
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

    fn domain(&self) -> Rc<Self::Domain> {
        self.0.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.0.range().clone()
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
    ) {
        self.0.apply_extended(alpha, x, beta, y);
        self.1
            .apply_extended(alpha, x, <<Self::Range as LinearSpace>::F as One>::one(), y);
    }

    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        let mut sum_elem = self.0.apply(x);
        sum_elem.sum_inplace(&self.1.apply(x));
        sum_elem
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

    fn domain(&self) -> Rc<Self::Domain> {
        self.0.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.0.range().clone()
    }
}

impl<Op: AsApply> AsApply for ScalarTimesOperator<Op> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) {
        self.0.apply_extended(self.1 * alpha, x, beta, y);
    }

    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        let mut res = self.0.apply(x);
        res.scale_inplace(self.1);
        res
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

    fn domain(&self) -> Rc<Self::Domain> {
        self.op1.domain().clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.op2.range().clone()
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
    ) {
        self.op2.apply_extended(alpha, &self.op1.apply(x), beta, y);
    }

    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        self.op2.apply(&self.op1.apply(x))
    }
}
