//! Arnoldi Iteration

use std::convert::AsRef;
use std::ops::{AddAssign, Mul};

use num::One;

use crate::{
    abstract_operator::OperatorBase,
    operator::{abstract_operator::Operator, element::Element},
    Inner, InnerProductSpace, Norm, NormedSpace, RlstScalar, Sqrt,
};

/// Iteration for CG
pub struct CgIteration<
    'a,
    Scalar: RlstScalar,
    Space: InnerProductSpace<F = Scalar>,
    OpImpl: OperatorBase<Domain = Space, Range = Space>,
> {
    operator: &'a Operator<OpImpl>,
    rhs: &'a Element<'a, Space>,
    x: &'a mut Element<'a, Space>,
    max_iter: usize,
    tol: Scalar::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(&Element<Space>, &Element<Space>) + 'a>>,
    print_debug: bool,
}

impl<
        'a,
        Scalar: RlstScalar,
        Space: InnerProductSpace<F = Scalar>,
        OpImpl: OperatorBase<Domain = Space, Range = Space>,
    > CgIteration<'a, Scalar, Space, OpImpl>
where
    Space: NormedSpace<Output = Scalar::Real>,
{
    /// Create a new CG iteration
    pub fn new(
        op: &'a Operator<OpImpl>,
        rhs: &'a Element<'a, Space>,
        x: &'a mut Element<'a, Space>,
    ) -> Self {
        Self {
            operator: op,
            rhs,
            x,
            max_iter: 1000,
            tol: num::cast(1E-6).unwrap(),
            callable: None,
            print_debug: false,
        }
    }

    /// Set tolerance
    pub fn set_tol(mut self, tol: Scalar::Real) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum number of iterations
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the callable
    pub fn set_callable(
        mut self,
        callable: impl FnMut(&Element<Space>, &Element<Space>) + 'a,
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
    pub fn run(mut self) -> Scalar::Real {
        fn print_success<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!("CG converged in {it_count} iterations with relative residual {rel_res:+E}.");
        }

        fn print_fail<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "CG did not converge in {it_count} iterations. Final relative residual is {rel_res:+E}.",
            );
        }

        let mut res = self.rhs.clone();
        self.operator
            .apply(-<Scalar as One>::one(), self.x, One::one(), &mut res);

        let mut p = res.clone();

        // This syntax is only necessary because the type inference becomes confused for some reason.
        // If I write `let rhs_norm = self.rhs.norm()` the compiler thinks that `self.rhs` is a space and
        // not an element.
        let mut res_inner = res.inner(&res);
        let rhs_norm = self.rhs.norm();
        let mut rel_res = Sqrt::sqrt(res_inner.abs()) / rhs_norm;

        if rel_res < self.tol {
            if self.print_debug {
                print_success(0, rel_res);
            }
            return rel_res;
        }

        for it_count in 0..self.max_iter {
            let a_p = self.operator.dot(&p);
            let p_conj_inner = a_p.inner(&p);

            let alpha = res_inner / p_conj_inner;
            *self.x += (&p).mul(alpha);
            res -= (&a_p).mul(alpha);

            if let Some(callable) = self.callable.as_mut() {
                callable(self.x, &res);
            }
            let res_inner_previous = res_inner;
            res_inner = res.inner(&res);
            rel_res = Sqrt::sqrt(res_inner.abs()) / rhs_norm;
            if rel_res < self.tol {
                if self.print_debug {
                    print_success(it_count, rel_res);
                }
                return rel_res;
            }
            let beta = res_inner / res_inner_previous;
            p *= beta;
            p += &res;
        }

        if self.print_debug {
            print_fail(self.max_iter, rel_res);
        }
        rel_res
    }
}
