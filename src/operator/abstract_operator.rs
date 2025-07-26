//! Implementation of a general abstract operator.

use num::One;

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

// // impl<
//         Domain: LinearSpace,
//         Range: LinearSpace,
//         OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
//         OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
//     > std::ops::Add<Operator<OpImpl2>> for Operator<OpImpl1>
// {
//     type Output = Operator<OperatorSum<Domain, Range, OpImpl1, OpImpl2>>;

//     fn add(self, rhs: Operator<OpImpl2>) -> Self::Output {
//         Operator::new(self.0.sum(rhs.0))
//     }
// }

// impl<
//         Domain: LinearSpace,
//         Range: LinearSpace,
//         OpImpl1: OperatorBase<Domain = Domain, Range = Range>,
//         OpImpl2: OperatorBase<Domain = Domain, Range = Range>,
//     > std::ops::Sub<Operator<OpImpl2>> for Operator<OpImpl1>
// {
//     type Output = Operator<OperatorSum<Domain, Range, OpImpl1, ScalarTimesOperator<OpImpl2>>>;

//     fn sub(self, rhs: Operator<OpImpl2>) -> Self::Output {
//         Operator::new(self.0.sub(rhs.0))
//     }
// }

// /// Trait that is satisfied by each scalar type that can multiply an operator from the left.
// pub trait OperatorLeftScalarMul<OpImpl: OperatorBase>: std::ops::Mul<Operator<OpImpl>> {}

// impl<
//         T: RlstScalar + std::ops::Mul<Operator<OpImpl>>,
//         Domain: LinearSpace,
//         Range: LinearSpace<F = T>,
//         OpImpl: OperatorBase<Domain = Domain, Range = Range>,
//     > OperatorLeftScalarMul<OpImpl> for T
// {
// }

// macro_rules! impl_operator_mul {
//     ($dtype:ty) => {
//         impl<
//                 Domain: LinearSpace,
//                 Range: LinearSpace<F = $dtype>,
//                 OpImpl: OperatorBase<Domain = Domain, Range = Range>,
//             > std::ops::Mul<Operator<OpImpl>> for $dtype
//         {
//             type Output = Operator<ScalarTimesOperator<OpImpl>>;

//             fn mul(self, rhs: Operator<OpImpl>) -> Self::Output {
//                 Operator::new(rhs.0.scale(self))
//             }
//         }
//     };
// }

// impl_operator_mul!(f32);
// impl_operator_mul!(f64);
// impl_operator_mul!(crate::dense::types::c32);
// impl_operator_mul!(crate::dense::types::c64);

// impl<OpImpl: OperatorBase> std::ops::Neg for Operator<OpImpl> {
//     type Output = Operator<ScalarTimesOperator<OpImpl>>;

//     fn neg(self) -> Self::Output {
//         Operator::new(self.0.neg())
//     }
// }

// impl<
//         T: RlstScalar,
//         Domain: LinearSpace,
//         Range: LinearSpace<F = T>,
//         OpImpl: OperatorBase<Domain = Domain, Range = Range>,
//     > std::ops::Mul<T> for Operator<OpImpl>
// {
//     type Output = Operator<ScalarTimesOperator<OpImpl>>;

//     fn mul(self, rhs: T) -> Self::Output {
//         Operator::new(self.0.scale(rhs))
//     }
// }

// impl<
//         T: RlstScalar,
//         Domain: LinearSpace,
//         Range: LinearSpace<F = T>,
//         OpImpl: OperatorBase<Domain = Domain, Range = Range>,
//     > std::ops::Div<T> for Operator<OpImpl>
// {
//     type Output = Operator<ScalarTimesOperator<OpImpl>>;

//     fn div(self, rhs: T) -> Self::Output {
//         Operator::new(self.0.scale(T::one() / rhs))
//     }
// }

// /// Operator sum
// pub struct OperatorSum<
//     Domain: LinearSpace,
//     Range: LinearSpace,
//     Op1: OperatorBase<Domain = Domain, Range = Range>,
//     Op2: OperatorBase<Domain = Domain, Range = Range>,
// >(Op1, Op2);

// impl<
//         Domain: LinearSpace,
//         Range: LinearSpace,
//         Op1: OperatorBase<Domain = Domain, Range = Range>,
//         Op2: OperatorBase<Domain = Domain, Range = Range>,
//     > std::fmt::Debug for OperatorSum<Domain, Range, Op1, Op2>
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_tuple("OperatorSum")
//             .field(&&self.0)
//             .field(&&self.1)
//             .finish()
//     }
// }

// impl<
//         Domain: LinearSpace,
//         Range: LinearSpace,
//         Op1: OperatorBase<Domain = Domain, Range = Range>,
//         Op2: OperatorBase<Domain = Domain, Range = Range>,
//     > OperatorBase for OperatorSum<Domain, Range, Op1, Op2>
// {
//     type Domain = Domain;

//     type Range = Range;

//     fn domain(&self) -> Rc<Self::Domain> {
//         self.0.domain().clone()
//     }

//     fn range(&self) -> Rc<Self::Range> {
//         self.0.range().clone()
//     }
// }

// impl<
//         Domain: LinearSpace,
//         Range: LinearSpace,
//         Op1: AsApply<Domain = Domain, Range = Range>,
//         Op2: AsApply<Domain = Domain, Range = Range>,
//     > AsApply for OperatorSum<Domain, Range, Op1, Op2>
// {
//     fn apply_extended<
//         ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
//         ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
//     >(
//         &self,
//         alpha: <Self::Range as LinearSpace>::F,
//         x: Element<ContainerIn>,
//         beta: <Self::Range as LinearSpace>::F,
//         mut y: Element<ContainerOut>,
//     ) {
//         self.0.apply_extended(alpha, x.r(), beta, y.r_mut());
//         self.1
//             .apply_extended(alpha, x, <<Self::Range as LinearSpace>::F as One>::one(), y);
//     }
// }

// /// Operator muiltiplied by a scalar
// pub struct ScalarTimesOperator<Op: OperatorBase>(Op, <Op::Range as LinearSpace>::F);

// impl<Op: OperatorBase> std::fmt::Debug for ScalarTimesOperator<Op> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_tuple("OperatorSum")
//             .field(&self.0)
//             .field(&self.1)
//             .finish()
//     }
// }

// impl<Op: OperatorBase> OperatorBase for ScalarTimesOperator<Op> {
//     type Domain = Op::Domain;

//     type Range = Op::Range;

//     fn domain(&self) -> Rc<Self::Domain> {
//         self.0.domain().clone()
//     }

//     fn range(&self) -> Rc<Self::Range> {
//         self.0.range().clone()
//     }
// }

// impl<Op: AsApply> AsApply for ScalarTimesOperator<Op> {
//     fn apply_extended<
//         ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
//         ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
//     >(
//         &self,
//         alpha: <Self::Range as LinearSpace>::F,
//         x: Element<ContainerIn>,
//         beta: <Self::Range as LinearSpace>::F,
//         y: Element<ContainerOut>,
//     ) {
//         self.0.apply_extended(self.1 * alpha, x, beta, y);
//     }
// }

// /// The product op2 * op1.
// pub struct OperatorProduct<
//     Space: LinearSpace,
//     Op1: OperatorBase<Range = Space>,
//     Op2: OperatorBase<Domain = Space>,
// > {
//     op1: Op1,
//     op2: Op2,
// }

// impl<Space: LinearSpace, Op1: OperatorBase<Range = Space>, Op2: OperatorBase<Domain = Space>> Debug
//     for OperatorProduct<Space, Op1, Op2>
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("ProductOperator")
//             .field("op1", &self.op1)
//             .field("op2", &self.op2)
//             .finish()
//     }
// }

// impl<Space: LinearSpace, Op1: OperatorBase<Range = Space>, Op2: OperatorBase<Domain = Space>>
//     OperatorBase for OperatorProduct<Space, Op1, Op2>
// {
//     type Domain = Op1::Domain;

//     type Range = Op2::Range;

//     fn domain(&self) -> Rc<Self::Domain> {
//         self.op1.domain().clone()
//     }

//     fn range(&self) -> Rc<Self::Range> {
//         self.op2.range().clone()
//     }
// }

// impl<Space: LinearSpace, Op1: AsApply<Range = Space>, Op2: AsApply<Domain = Space>> AsApply
//     for OperatorProduct<Space, Op1, Op2>
// {
//     fn apply_extended<
//         ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
//         ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
//     >(
//         &self,
//         alpha: <Self::Range as LinearSpace>::F,
//         x: Element<ContainerIn>,
//         beta: <Self::Range as LinearSpace>::F,
//         y: Element<ContainerOut>,
//     ) {
//         self.op2.apply_extended(alpha, self.op1.apply(x), beta, y);
//     }
// }

// impl<
//         Space: LinearSpace,
//         OpImpl1: OperatorBase<Range = Space>,
//         OpImpl2: OperatorBase<Domain = Space>,
//     > std::ops::Mul<Operator<OpImpl1>> for Operator<OpImpl2>
// {
//     type Output = Operator<OperatorProduct<Space, OpImpl1, OpImpl2>>;

//     fn mul(self, rhs: Operator<OpImpl1>) -> Self::Output {
//         Operator::new(self.0.product(rhs.0))
//     }
// }
