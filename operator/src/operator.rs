//! Definition of a general linear operator.

use crate::LinearSpace;
use num::One;
use rlst_common::types::*;
use std::fmt::Debug;

pub type RlstOperator<'a, Domain, Range> =
    Box<dyn OperatorBase<Domain = Domain, Range = Range> + 'a>;

// A base operator trait.
pub trait OperatorBase: Debug {
    type Domain: LinearSpace;
    type Range: LinearSpace;

    /// Returns a reference to trait object that supports application of the operator.
    ///
    /// By default it returns an `Err`. But for concrete types
    /// that support matvecs it is specialised to return
    /// a dynamic reference.
    fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
        None
    }

    fn has_apply(&self) -> bool {
        self.as_apply().is_some()
    }

    fn domain(&self) -> &Self::Domain;

    fn range(&self) -> &Self::Range;
}

/// Apply an operator as y -> alpha * Ax + beta y
pub trait AsApply: OperatorBase {
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E<'_>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E<'_>,
    ) -> RlstResult<()>;
}

impl<In: LinearSpace, Out: LinearSpace> AsApply for dyn OperatorBase<Domain = In, Range = Out> {
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E<'_>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E<'_>,
    ) -> RlstResult<()> {
        if let Some(op) = self.as_apply() {
            op.apply(alpha, x, beta, y)
        } else {
            Err(RlstError::NotImplemented("Apply".to_string()))
        }
    }
}

pub struct RlstOperatorReference<'a, Domain: LinearSpace, Range: LinearSpace>(
    &'a RlstOperator<'a, Domain, Range>,
);

impl<'a, Domain: LinearSpace, Range: LinearSpace> std::fmt::Debug
    for RlstOperatorReference<'a, Domain, Range>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorReference").field(&self.0).finish()
    }
}

impl<'a, Domain: LinearSpace, Range: LinearSpace> OperatorBase
    for RlstOperatorReference<'a, Domain, Range>
{
    type Domain = Domain;

    type Range = Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
        Some(self)
    }
}

impl<'a, Domain: LinearSpace, Range: LinearSpace> AsApply
    for RlstOperatorReference<'a, Domain, Range>
{
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E<'_>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E<'_>,
    ) -> RlstResult<()> {
        if let Some(op) = self.0.as_apply() {
            op.apply(alpha, x, beta, y).unwrap();
            Ok(())
        } else {
            Err(RlstError::OperationFailed(
                "AsApply must be implemented for operator.".to_string(),
            ))
        }
    }
}

pub struct OperatorSum<'a, Domain: LinearSpace, Range: LinearSpace>(
    RlstOperator<'a, Domain, Range>,
    RlstOperator<'a, Domain, Range>,
);

impl<'a, Domain: LinearSpace, Range: LinearSpace> std::fmt::Debug
    for OperatorSum<'a, Domain, Range>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorSum")
            .field(&&self.0)
            .field(&&self.1)
            .finish()
    }
}

impl<'a, Domain: LinearSpace, Range: LinearSpace> OperatorBase for OperatorSum<'a, Domain, Range> {
    type Domain = Domain;

    type Range = Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
        Some(self)
    }
}

impl<'a, Domain: LinearSpace, Range: LinearSpace> AsApply for OperatorSum<'a, Domain, Range> {
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E<'_>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E<'_>,
    ) -> RlstResult<()> {
        if let (Some(op_a), Some(op_b)) = (self.0.as_apply(), self.1.as_apply()) {
            op_a.apply(alpha, x, beta, y).unwrap();
            op_b.apply(alpha, x, <<Self::Range as LinearSpace>::F as One>::one(), y)
                .unwrap();
            Ok(())
        } else {
            Err(RlstError::OperationFailed(
                "AsApply must be implemented for both operators in sum.".to_string(),
            ))
        }
    }
}

pub struct ScalarTimesOperator<'a, Domain: LinearSpace, Range: LinearSpace>(
    RlstOperator<'a, Domain, Range>,
    Range::F,
);

impl<'a, Domain: LinearSpace, Range: LinearSpace> std::fmt::Debug
    for ScalarTimesOperator<'a, Domain, Range>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorSum")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<'a, Domain: LinearSpace, Range: LinearSpace> OperatorBase
    for ScalarTimesOperator<'a, Domain, Range>
{
    type Domain = Domain;

    type Range = Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }

    fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
        Some(self)
    }
}

impl<'a, Domain: LinearSpace, Range: LinearSpace> AsApply
    for ScalarTimesOperator<'a, Domain, Range>
{
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E<'_>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E<'_>,
    ) -> RlstResult<()> {
        if let Some(op) = self.0.as_apply() {
            op.apply(self.1 * alpha, x, beta, y).unwrap();
            Ok(())
        } else {
            Err(RlstError::NotImplemented("Apply".to_string()))
        }
    }
}

macro_rules! impl_scalar_mult {
    ($scalar:ty) => {
        impl<'a, Domain: LinearSpace<F = $scalar> + 'a, Range: LinearSpace<F = $scalar> + 'a>
            std::ops::Mul<$scalar> for RlstOperator<'a, Domain, Range>
        {
            type Output = RlstOperator<'a, Domain, Range>;

            fn mul(self, rhs: Range::F) -> Self::Output {
                Box::new(ScalarTimesOperator(self, rhs))
            }
        }

        impl<'a, Domain: LinearSpace<F = $scalar> + 'a, Range: LinearSpace<F = $scalar> + 'a>
            std::ops::Mul<$scalar> for &'a RlstOperator<'a, Domain, Range>
        {
            type Output = RlstOperator<'a, Domain, Range>;

            fn mul(self, rhs: Range::F) -> Self::Output {
                Box::new(ScalarTimesOperator(
                    Box::new(RlstOperatorReference(self)),
                    rhs,
                ))
            }
        }

        impl<'a, Domain: LinearSpace<F = $scalar> + 'a, Range: LinearSpace<F = $scalar> + 'a>
            std::ops::Mul<RlstOperator<'a, Domain, Range>> for $scalar
        {
            type Output = RlstOperator<'a, Domain, Range>;

            fn mul(self, rhs: RlstOperator<'a, Domain, Range>) -> Self::Output {
                Box::new(ScalarTimesOperator(rhs, self))
            }
        }

        impl<'a, Domain: LinearSpace<F = $scalar> + 'a, Range: LinearSpace<F = $scalar> + 'a>
            std::ops::Mul<&'a RlstOperator<'a, Domain, Range>> for $scalar
        {
            type Output = RlstOperator<'a, Domain, Range>;

            fn mul(self, rhs: &'a RlstOperator<'a, Domain, Range>) -> Self::Output {
                Box::new(ScalarTimesOperator(
                    Box::new(RlstOperatorReference(rhs)),
                    self,
                ))
            }
        }
    };
}

impl_scalar_mult!(f32);
impl_scalar_mult!(f64);
impl_scalar_mult!(c32);
impl_scalar_mult!(c64);

impl<'a, Domain: LinearSpace + 'a, Range: LinearSpace + 'a>
    std::ops::Add<RlstOperator<'a, Domain, Range>> for RlstOperator<'a, Domain, Range>
{
    type Output = RlstOperator<'a, Domain, Range>;

    fn add(self, rhs: RlstOperator<'a, Domain, Range>) -> Self::Output {
        Box::new(OperatorSum(self, rhs))
    }
}

impl<'a, Domain: LinearSpace + 'a, Range: LinearSpace + 'a>
    std::ops::Add<&'a RlstOperator<'a, Domain, Range>> for RlstOperator<'a, Domain, Range>
{
    type Output = RlstOperator<'a, Domain, Range>;

    fn add(self, rhs: &'a RlstOperator<'a, Domain, Range>) -> Self::Output {
        Box::new(OperatorSum(self, Box::new(RlstOperatorReference(rhs))))
    }
}

impl<'a, Domain: LinearSpace + 'a, Range: LinearSpace + 'a>
    std::ops::Add<RlstOperator<'a, Domain, Range>> for &'a RlstOperator<'a, Domain, Range>
{
    type Output = RlstOperator<'a, Domain, Range>;

    fn add(self, rhs: RlstOperator<'a, Domain, Range>) -> Self::Output {
        Box::new(OperatorSum(Box::new(RlstOperatorReference(self)), rhs))
    }
}

impl<'a, Domain: LinearSpace + 'a, Range: LinearSpace + 'a>
    std::ops::Add<&'a RlstOperator<'a, Domain, Range>> for &'a RlstOperator<'a, Domain, Range>
{
    type Output = RlstOperator<'a, Domain, Range>;

    fn add(self, rhs: &'a RlstOperator<'a, Domain, Range>) -> Self::Output {
        Box::new(OperatorSum(
            Box::new(RlstOperatorReference(self)),
            Box::new(RlstOperatorReference(rhs)),
        ))
    }
}

#[cfg(test)]
mod test {

    use crate::{
        implementation::{
            array_vector_space::ArrayVectorSpace, dense_matrix_operator::DenseMatrixOperator,
        },
        Element, LinearSpace,
    };
    use rlst_dense::{assert_array_relative_eq, rlst_dynamic_array2, traits::*};

    #[test]
    fn test_operator_algebra() {
        let mut mat1 = rlst_dynamic_array2!(f64, [4, 3]);
        let mut mat2 = rlst_dynamic_array2!(f64, [4, 3]);

        let domain = ArrayVectorSpace::new(3);
        let range = ArrayVectorSpace::new(4);

        mat1.fill_from_seed_equally_distributed(0);
        mat2.fill_from_seed_equally_distributed(1);

        let op1 = DenseMatrixOperator::from(mat1, &domain, &range);
        let op2 = DenseMatrixOperator::from(mat2, &domain, &range);

        let mut x = domain.zero();
        let mut y = range.zero();
        let mut y_expected = range.zero();
        x.view_mut().fill_from_seed_equally_distributed(2);
        y.view_mut().fill_from_seed_equally_distributed(3);
        y_expected.view_mut().fill_from(y.view());

        op2.as_apply()
            .unwrap()
            .apply(2.0, &x, 3.5, &mut y_expected)
            .unwrap();
        op1.as_apply()
            .unwrap()
            .apply(10.0, &x, 1.0, &mut y_expected)
            .unwrap();

        let sum = 5.0 * &op1 + &op2;

        sum.as_apply().unwrap().apply(2.0, &x, 3.5, &mut y).unwrap();

        assert_array_relative_eq!(y.view(), y_expected.view(), 1E-12);
    }
}
