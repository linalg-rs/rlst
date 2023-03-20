//! General linear operator.

use crate::LinearSpace;
use rlst_common::types::*;
use std::fmt::Debug;

use crate::*;

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
}

/// Apply an operator.
pub trait AsApply: OperatorBase {
    fn apply(
        &self,
        x: ElementView<Self::Domain>,
        y: ElementViewMut<Self::Range>,
    ) -> SparseLinAlgResult<()>;
}

impl<In: LinearSpace, Out: LinearSpace> AsApply for dyn OperatorBase<Domain = In, Range = Out> {
    fn apply(
        &self,
        x: ElementView<Self::Domain>,
        y: ElementViewMut<Self::Range>,
    ) -> SparseLinAlgResult<()> {
        if let Some(op) = self.as_apply() {
            op.apply(x, y)
        } else {
            Err(SparseLinAlgError::NotImplemented("Apply".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {

    use std::marker::PhantomData;

    use super::*;

    #[derive(Debug)]
    struct SimpleSpace;
    impl LinearSpace for SimpleSpace {
        type F = f64;
        type E<'a> = SimpleVector;
    }

    #[derive(Debug)]
    struct SimpleVector {}

    #[derive(Debug)]
    struct View<'a> {
        marker: PhantomData<&'a ()>,
    }

    impl<'a> View<'a> {
        fn new() -> Self {
            Self {
                marker: PhantomData,
            }
        }
    }

    impl Element for SimpleVector {
        type Space = SimpleSpace;
        type View<'b> = View<'b> where Self: 'b;
        type ViewMut<'b> = View<'b> where Self: 'b;

        fn view<'b>(&'b self) -> Self::View<'b> {
            View::new()
        }

        fn view_mut<'b>(&'b mut self) -> Self::View<'b> {
            View::new()
        }
    }

    #[derive(Debug)]
    struct SparseMatrix;
    impl OperatorBase for SparseMatrix {
        type Domain = SimpleSpace;
        type Range = SimpleSpace;

        fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
            Some(self)
        }
        // fn as_matvec_h(
        //     &self,
        // ) -> Option<&dyn AsHermitianMatVec<Domain = Self::Domain, Range = Self::Range>> {
        //     Some(self)
        // }
    }
    impl AsApply for SparseMatrix {
        fn apply(
            &self,
            _x: ElementView<Self::Domain>,
            _y: ElementViewMut<Self::Range>,
        ) -> SparseLinAlgResult<()> {
            println!("{self:?} matvec");
            Ok(())
        }
    }

    // Finite difference matrices use the following formula where f is a
    // nonlinear function and x is a vector that we linearize around. It is not
    // tractable to apply the transpose or Hermitian adjoint without access to
    // the code that computes f.
    //
    // A y = (f(x + hy) - f(x)) / h
    #[derive(Debug)]
    struct FiniteDifference;
    impl OperatorBase for FiniteDifference {
        type Domain = SimpleSpace;
        type Range = SimpleSpace;
        fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
            Some(self)
        }
    }
    impl AsApply for FiniteDifference {
        fn apply(
            &self,
            _x: ElementView<Self::Domain>,
            _y: ElementViewMut<Self::Range>,
        ) -> SparseLinAlgResult<()> {
            println!("{self:?} matvec");
            Ok(())
        }
    }

    /// A fallible matrix
    #[derive(Debug)]
    struct SketchyMatrix;
    impl OperatorBase for SketchyMatrix {
        type Domain = SimpleSpace;
        type Range = SimpleSpace;
        fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
            Some(self)
        }
    }
    impl AsApply for SketchyMatrix {
        fn apply(
            &self,
            _x: ElementView<Self::Domain>,
            _y: ElementViewMut<Self::Range>,
        ) -> SparseLinAlgResult<()> {
            println!("{self:?} matvec");
            Err(SparseLinAlgError::OperationFailed("Apply".to_string()))
        }
    }
    #[test]
    fn test_mult_dyn() -> SparseLinAlgResult<()> {
        let x = SimpleVector {};
        let mut y = SimpleVector {};
        let ops: Vec<Box<dyn OperatorBase<Domain = SimpleSpace, Range = SimpleSpace>>> =
            vec![Box::new(SparseMatrix), Box::new(FiniteDifference)];
        for op in ops {
            op.apply(x.view(), y.view_mut())?;
        }
        Ok(())
    }

    #[test]
    fn test_mult() -> SparseLinAlgResult<()> {
        let x = SimpleVector {};
        let mut y = SimpleVector {};
        let a = SparseMatrix;
        // Static dispatch because we're using a struct that implements AsMatVec
        a.apply(x.view(), y.view_mut())?;
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_mult_sketchy() {
        let x = SimpleVector {};
        let mut y = SimpleVector {};
        let a = SketchyMatrix;
        // Static dispatch because we're using a struct that implements AsMatVec
        a.apply(x.view(), y.view_mut()).unwrap();
    }
}
