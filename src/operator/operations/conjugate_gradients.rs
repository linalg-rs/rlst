//! Arnoldi Iteration
use crate::dense::types::RlstScalar;
use crate::operator::{AsApply, Element, InnerProductSpace, LinearSpace, NormedSpace};
use num::One;

/// Iteration for CG
pub struct CgIteration<'a, Space: InnerProductSpace, Op: AsApply<Domain = Space, Range = Space>> {
    operator: &'a Op,
    rhs: &'a Space::E,
    x: Space::E,
    max_iter: usize,
    tol: <Space::F as RlstScalar>::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(&<Space as LinearSpace>::E, &<Space as LinearSpace>::E) + 'a>>,
    print_debug: bool,
}

impl<'a, Space: InnerProductSpace, Op: AsApply<Domain = Space, Range = Space>>
    CgIteration<'a, Space, Op>
{
    /// Create a new CG iteration
    pub fn new(op: &'a Op, rhs: &'a Space::E) -> Self {
        Self {
            operator: op,
            rhs,
            x: <Space as LinearSpace>::zero(op.domain()),
            max_iter: 1000,
            tol: num::cast::<f64, <Space::F as RlstScalar>::Real>(1E-6).unwrap(),
            callable: None,
            print_debug: false,
        }
    }

    /// Set x
    pub fn set_x(mut self, x: &Space::E) -> Self {
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
    pub fn set_callable(mut self, callable: impl FnMut(&Space::E, &Space::E) + 'a) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Run CG
    pub fn run(mut self) -> (Space::E, <Space::F as RlstScalar>::Real) {
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

        let mut res: Space::E = <<Space as LinearSpace>::E as Clone>::clone(self.rhs);
        res.sum_inplace(&self.operator.apply(&self.x).neg());

        let mut p = res.clone();

        // This syntax is only necessary because the type inference becomes confused for some reason.
        // If I write `let rhs_norm = self.rhs.norm()` the compiler thinks that `self.rhs` is a space and
        // not an element.
        let rhs_norm = <Space as NormedSpace>::norm(&self.operator.range(), self.rhs);
        let mut res_inner =
            <Space as InnerProductSpace>::inner_product(&self.operator.range(), &res, &res);
        let mut res_norm = res_inner.abs().sqrt();
        let mut rel_res = res_norm / rhs_norm;

        if rel_res < self.tol {
            if self.print_debug {
                print_success(0, rel_res);
            }
            return (self.x, rel_res);
        }

        for it_count in 0..self.max_iter {
            let p_conj_inner = <Space as InnerProductSpace>::inner_product(
                &self.operator.range(),
                &self.operator.apply(&p),
                &p,
            );
            let alpha = res_inner / p_conj_inner;

            self.x.axpy_inplace(alpha, &p);
            self.operator.apply_extended(
                -alpha,
                &p,
                <<Space as LinearSpace>::F as One>::one(),
                &mut res,
            );
            if let Some(callable) = self.callable.as_mut() {
                callable(&self.x, &res);
            }
            let res_inner_previous = res_inner;
            res_inner =
                <Space as InnerProductSpace>::inner_product(&self.operator.range(), &res, &res);
            res_norm = res_inner.abs().sqrt();
            rel_res = res_norm / rhs_norm;
            if res_norm < self.tol {
                if self.print_debug {
                    print_success(it_count, rel_res);
                }
                return (self.x, rel_res);
            }
            let beta = res_inner / res_inner_previous;
            p.scale_inplace(beta);
            p.sum_inplace(&res);
        }

        if self.print_debug {
            print_fail(self.max_iter, rel_res);
        }
        (self.x, rel_res)
    }
}
