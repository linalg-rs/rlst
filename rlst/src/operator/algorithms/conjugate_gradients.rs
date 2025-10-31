//! The Conjugate Gradient Method.
//!
//! The Conjugate Gradient Method (CG) solves operator equations `Ax=b`, where
//! `A` is a symmetric positive definite operator.

use std::ops::Mul;

use num::One;

use crate::{
    Inner, InnerProductSpace, Norm, NormedSpace, RlstScalar, Sqrt,
    abstract_operator::OperatorBase,
    operator::{abstract_operator::Operator, element::Element},
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
    /// Create a new CG iteration.
    ///
    /// This function takes an operator `op`, a right-hand side `rhs`, and an element `x` of the
    /// function space as starting approximation. Typically, `x` can be chosen to be the zero
    /// element of the space.
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

    /// Set the relative tolerance for convergence.
    ///
    /// The convergence criterion is `||b - Ax|| <= tol`. The default
    /// relative tolerance is 1E-6.
    pub fn set_tol(mut self, tol: Scalar::Real) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum number of iterations.
    ///
    /// The default maximum number of iterations is 1000.
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Define a callable.
    ///
    /// A callable is a function `FnMut(&Element, &Element)` that is
    /// executed in each step of the iteration. The first parameter is
    /// the current approximation `x` and the second parameter is the
    /// current residual `b - Ax`. This callable can be used to obtain
    /// information about each step of the iteration.
    pub fn set_callable(
        mut self,
        callable: impl FnMut(&Element<Space>, &Element<Space>) + 'a,
    ) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    ///
    /// Print debug information on `stdout`.
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Run the CG iteration.
    ///
    /// The output parameter is the relative residual at the end of the iteration.
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

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        Norm,
        abstract_operator::OperatorBase,
        dense::array::DynArray,
        operator::{
            abstract_operator::Operator, algorithms::conjugate_gradients::CgIteration,
            space::zero_element,
        },
    };

    #[test]
    fn test_cg() {
        let dim = 10;
        let tol = 1E-5;

        let mut residuals = Vec::<f64>::new();

        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut mat = DynArray::<f64, 2>::from_shape([dim, dim]);

        for index in 0..dim {
            mat[[index, index]] = rng.random_range(1.0..=2.0);
        }

        let op = Operator::from(&mat);

        let mut rhs = zero_element(op.range());
        rhs.imp_mut().fill_from_equally_distributed(&mut rng);
        let mut x = zero_element(op.domain());

        let cg = (CgIteration::new(&op, &rhs, &mut x))
            .set_callable(|_, res| {
                let res_norm = res.norm();
                residuals.push(res_norm);
            })
            .set_tol(tol)
            .print_debug();

        let res = cg.run();
        assert!(res < tol);
    }
}
