//! Arnoldi Iteration
use crate::dense::types::RlstScalar;
use crate::operator::{
    zero_element, AsApply, ElementImpl, ElementType, InnerProductSpace, LinearSpace, Operator,
};
use crate::{Element, ElementContainer, OperatorBase};
use num::One;

/// Iteration for CG
pub struct CgIteration<
    'a,
    Space: InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    Container: ElementContainer<E = Space::E>,
> where
    <Space::E as ElementImpl>::Space: InnerProductSpace,
{
    operator: Operator<OpImpl>,
    rhs: Element<Container>,
    x: ElementType<Space::E>,
    max_iter: usize,
    tol: <Space::F as RlstScalar>::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(&ElementType<Space::E>, &ElementType<Space::E>) + 'a>>,
    print_debug: bool,
}

impl<
        'a,
        Space: InnerProductSpace,
        OpImpl: AsApply<Domain = Space, Range = Space>,
        Container: ElementContainer<E = Space::E>,
    > CgIteration<'a, Space, OpImpl, Container>
where
    <Space::E as ElementImpl>::Space: InnerProductSpace,
{
    /// Create a new CG iteration
    pub fn new(op: Operator<OpImpl>, rhs: Element<Container>) -> Self {
        let domain = op.domain();
        Self {
            operator: op,
            rhs,
            x: zero_element(domain),
            max_iter: 1000,
            tol: num::cast::<f64, <Space::F as RlstScalar>::Real>(1E-6).unwrap(),
            callable: None,
            print_debug: false,
        }
    }

    /// Set x
    pub fn set_x<OtherElementContainer: ElementContainer<E = Space::E>>(
        mut self,
        x: Element<OtherElementContainer>,
    ) -> Self {
        self.x.fill_inplace(x);
        self
    }

    /// Set the tolerance
    pub fn set_tol(mut self, tol: <Space::F as RlstScalar>::Real) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum number of iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the cammable
    pub fn set_callable(
        mut self,
        callable: impl FnMut(&ElementType<Space::E>, &ElementType<Space::E>) + 'a,
    ) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Run CG
    pub fn run(mut self) -> (ElementType<Space::E>, <Space::F as RlstScalar>::Real) {
        fn print_success<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "CG converged in {} iterations with relative residual {:+E}.",
                it_count, rel_res
            );
        }

        fn print_fail<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "CG did not converge in {} iterations. Final relative residual is {:+E}.",
                it_count, rel_res
            );
        }

        let mut res = self.rhs.duplicate();
        res -= self.operator.apply(self.x.r(), crate::TransMode::NoTrans);

        let mut p = res.duplicate();

        // This syntax is only necessary because the type inference becomes confused for some reason.
        // If I write `let rhs_norm = self.rhs.norm()` the compiler thinks that `self.rhs` is a space and
        // not an element.
        let rhs_norm = self.rhs.norm();
        let mut res_inner = res.inner_product(res.r());
        let mut res_norm = res_inner.abs().sqrt();
        let mut rel_res = res_norm / rhs_norm;

        if rel_res < self.tol {
            if self.print_debug {
                print_success(0, rel_res);
            }
            return (self.x, rel_res);
        }

        for it_count in 0..self.max_iter {
            let p_conj_inner = self
                .operator
                .apply(p.r(), crate::TransMode::NoTrans)
                .inner_product(p.r());

            let alpha = res_inner / p_conj_inner;
            self.x.axpy_inplace(alpha, p.r());
            self.operator.apply_extended(
                -alpha,
                p.r(),
                <<Space as LinearSpace>::F as One>::one(),
                res.r_mut(),
            );
            if let Some(callable) = self.callable.as_mut() {
                callable(&self.x, &res);
            }
            let res_inner_previous = res_inner;
            res_inner = res.inner_product(res.r());
            res_norm = res_inner.abs().sqrt();
            rel_res = res_norm / rhs_norm;
            if rel_res < self.tol {
                if self.print_debug {
                    print_success(it_count, rel_res);
                }
                return (self.x, rel_res);
            }
            let beta = res_inner / res_inner_previous;
            p *= beta;
            p += res.r();
        }

        if self.print_debug {
            print_fail(self.max_iter, rel_res);
        }
        (self.x, rel_res)
    }
}
