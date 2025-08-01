//! Implementation of a general abstract operator.

use num::{One, Zero};

use crate::base_types::{c32, c64};
use crate::{abstract_operator::OperatorBase, LinearSpace};

/// A concrete operator.
pub struct Operator<OpImpl>(OpImpl);

impl<OpImpl> Operator<OpImpl>
where
    OpImpl: OperatorBase,
{
    /// Create a new operator.
    pub fn new(op_impl: OpImpl) -> Self {
        Self(op_impl)
    }
}

/// Operator reference
pub struct RlstOperatorReference<'a, Op>(&'a Op);

impl<'a, Op: OperatorBase> RlstOperatorReference<'a, Op> {
    /// Create a new operator reference.
    pub fn new(op: &'a Op) -> Self {
        Self(op)
    }

    /// Return an operator reference
    pub fn r(&self) -> Operator<RlstOperatorReference<'_, Self>> {
        Operator::new(RlstOperatorReference::new(self))
    }
}

impl<OpImpl: OperatorBase> OperatorBase for Operator<OpImpl> {
    type Domain = OpImpl::Domain;

    type Range = OpImpl::Range;

    #[inline(always)]
    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    #[inline(always)]
    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as crate::LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as crate::LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        self.0.apply(alpha, x, beta, y)
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

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as crate::LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as crate::LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        self.0.apply(alpha, x, beta, y)
    }
}

/// Sum of two operators.
pub struct OperatorSum<OpImpl1, OpImpl2>(Operator<OpImpl1>, Operator<OpImpl2>);

/// Scalar multiple of an operator.
pub struct OperatorScalarMul<Scalar, OpImpl>(Scalar, Operator<OpImpl>);

/// Difference of two operators.
pub struct OperatorSub<OpImpl1, OpImpl2>(Operator<OpImpl1>, Operator<OpImpl2>);

/// Product `Op1 * Op2` of two operators.
pub struct OperatorProduct<OpImpl1, OpImpl2>(Operator<OpImpl1>, Operator<OpImpl2>);

/// Negation of an operator.
pub struct OperatorNegation<OpImpl>(Operator<OpImpl>);

impl<Domain, Range, OpImpl1, OpImpl2> OperatorBase for OperatorSum<OpImpl1, OpImpl2>
where
    Domain: LinearSpace,
    Range: LinearSpace,
    Range::F: One + Copy,
    OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
    OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
{
    type Domain = Domain;

    type Range = Range;

    #[inline(always)]
    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    #[inline(always)]
    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as crate::LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as crate::LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        self.0.apply(alpha, x, beta, y);
        self.1.apply(alpha, x, One::one(), y);
    }
}

impl<Domain, Range, OpImpl1, OpImpl2> OperatorBase for OperatorSub<OpImpl1, OpImpl2>
where
    Domain: LinearSpace,
    Range: LinearSpace,
    Range::F: One + Copy + std::ops::Neg<Output = Range::F>,
    OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
    OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
{
    type Domain = Domain;

    type Range = Range;

    #[inline(always)]
    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    #[inline(always)]
    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as crate::LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as crate::LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        self.0.apply(alpha, x, beta, y);
        self.1.apply(std::ops::Neg::neg(alpha), x, One::one(), y);
    }
}

impl<Range, Scalar, OpImpl> OperatorBase for OperatorScalarMul<Scalar, OpImpl>
where
    Range: LinearSpace<F = Scalar>,
    Scalar: std::ops::Mul<Output = Scalar> + Copy,
    OpImpl: OperatorBase<Range = Range>,
{
    type Domain = OpImpl::Domain;

    type Range = OpImpl::Range;

    #[inline(always)]
    fn domain(&self) -> &Self::Domain {
        self.1.domain()
    }

    #[inline(always)]
    fn range(&self) -> &Self::Range {
        self.1.range()
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        self.1.apply(self.0 * alpha, x, beta, y)
    }
}

impl<Range, OpImpl1, OpImpl2> OperatorBase for OperatorProduct<Operator<OpImpl1>, Operator<OpImpl2>>
where
    Range: LinearSpace,
    Range::F: One + Zero,
    OpImpl1: OperatorBase<Domain = Range>,
    OpImpl2: OperatorBase<Range = Range>,
{
    type Domain = OpImpl2::Domain;

    type Range = OpImpl1::Range;

    #[inline(always)]
    fn domain(&self) -> &Self::Domain {
        self.1.domain()
    }

    #[inline(always)]
    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        let mut tmp = self.1.range().zero();

        self.1.apply(One::one(), x, Zero::zero(), &mut tmp);
        self.0.apply(alpha, &tmp, beta, y);
    }
}

impl<OpImpl> OperatorBase for OperatorNegation<OpImpl>
where
    OpImpl: OperatorBase,
    <OpImpl::Range as LinearSpace>::F: std::ops::Neg<Output = <OpImpl::Range as LinearSpace>::F>,
{
    type Domain = OpImpl::Domain;

    type Range = OpImpl::Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &super::element::Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut super::element::Element<Self::Range>,
    ) {
        self.0.apply(std::ops::Neg::neg(alpha), x, beta, y);
    }
}

impl<Domain, Range, OpImpl1, OpImpl2> std::ops::Add<Operator<OpImpl2>> for Operator<OpImpl1>
where
    Domain: LinearSpace,
    Range: LinearSpace,
    Range::F: One + Copy,
    OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
    OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
{
    type Output = Operator<OperatorSum<OpImpl1, OpImpl2>>;

    fn add(self, rhs: Operator<OpImpl2>) -> Self::Output {
        Operator::new(OperatorSum(self, rhs))
    }
}

impl<Domain, Range, OpImpl1, OpImpl2> std::ops::Sub<Operator<OpImpl2>> for Operator<OpImpl1>
where
    Domain: LinearSpace,
    Range: LinearSpace,
    Range::F: One + Copy + std::ops::Neg<Output = Range::F>,
    OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
    OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
{
    type Output = Operator<OperatorSub<OpImpl1, OpImpl2>>;

    fn sub(self, rhs: Operator<OpImpl2>) -> Self::Output {
        Operator::new(OperatorSub(self, rhs))
    }
}

impl<Scalar, OpImpl> std::ops::Mul<Scalar> for Operator<OpImpl>
where
    OpImpl: OperatorBase,
    OpImpl::Range: LinearSpace<F = Scalar>,
    Scalar: std::ops::Mul<Output = Scalar> + Copy,
{
    type Output = Operator<OperatorScalarMul<Scalar, OpImpl>>;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Operator::new(OperatorScalarMul(rhs, self))
    }
}

macro_rules! impl_scalar_mult {
    ($scalar:ty) => {
        impl<OpImpl> std::ops::Mul<Operator<OpImpl>> for $scalar
        where
            OpImpl: OperatorBase,
            OpImpl::Range: LinearSpace<F = $scalar>,
        {
            type Output = Operator<OperatorScalarMul<$scalar, OpImpl>>;

            fn mul(self, rhs: Operator<OpImpl>) -> Self::Output {
                Operator::new(OperatorScalarMul(self, rhs))
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);
