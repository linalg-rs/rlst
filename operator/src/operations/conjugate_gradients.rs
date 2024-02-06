//! Arnoldi Iteration
use crate::{AsApply, Element, InnerProductSpace, LinearSpace, NormedSpace};
use num::One;
use rlst_common::types::Scalar;

pub struct CgIteration<'a, Space: InnerProductSpace, Op: AsApply<Domain = Space, Range = Space>> {
    operator: &'a Op,
    space: &'a Space,
    rhs: &'a Space::E,
    x: Space::E,
    max_iter: usize,
    tol: <Space::F as Scalar>::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(&<Space as LinearSpace>::E, &<Space as LinearSpace>::E) + 'a>>,
    print_debug: bool,
}

impl<'a, Space: InnerProductSpace, Op: AsApply<Domain = Space, Range = Space>>
    CgIteration<'a, Space, Op>
{
    pub fn new(op: &'a Op, rhs: &'a Space::E) -> Self {
        Self {
            operator: op,
            space: op.domain(),
            rhs,
            x: op.domain().zero(),
            max_iter: 1000,
            tol: num::cast::<f64, <Space::F as Scalar>::Real>(1E-6).unwrap(),
            callable: None,
            print_debug: false,
        }
    }

    pub fn set_x(mut self, x: &Space::E) -> Self {
        self.x.fill_inplace(x);
        self
    }

    pub fn set_tol(mut self, tol: <Space::F as Scalar>::Real) -> Self {
        self.tol = tol;
        self
    }

    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn set_callable(mut self, callable: impl FnMut(&Space::E, &Space::E) + 'a) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    pub fn run(mut self) -> (Space::E, <Space::F as Scalar>::Real) {
        fn print_success<T: Scalar>(it_count: usize, rel_res: T) {
            println!(
                "CG converged in {} iterations with relative residual {:+E}.",
                it_count, rel_res
            );
        }

        fn print_fail<T: Scalar>(it_count: usize, rel_res: T) {
            println!(
                "CG did not converge in {} iterations. Final relative residual is {:+E}.",
                it_count, rel_res
            );
        }

        let mut res = self.space.new_from(self.rhs);
        res.sum_inplace(&self.operator.apply(&self.x).neg());

        let mut p = self.space.new_from(&res);

        let rhs_norm = self.space.norm(self.rhs);
        let mut res_inner = self.space.inner(&res, &res);
        let mut res_norm = res_inner.abs().sqrt();
        let mut rel_res = res_norm / rhs_norm;

        if rel_res < self.tol {
            if self.print_debug {
                print_success(0, rel_res);
            }
            return (self.x, rel_res);
        }

        for it_count in 0..self.max_iter {
            let p_conj_inner = self.space.inner(&self.operator.apply(&p), &p);
            let alpha = res_inner / p_conj_inner;

            self.x.axpy_inplace(alpha, &p);
            self.operator
                .apply_extended(
                    -alpha,
                    &p,
                    <<Space as LinearSpace>::F as One>::one(),
                    &mut res,
                )
                .unwrap();
            if let Some(callable) = self.callable.as_mut() {
                callable(&self.x, &res);
            }
            let res_inner_previous = res_inner;
            res_inner = self.space.inner(&res, &res);
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

#[cfg(test)]
mod test {

    use crate::interface::{
        array_vector_space::ArrayVectorSpace, dense_matrix_operator::DenseMatrixOperator,
    };

    use super::*;
    use rand::prelude::*;
    use rlst_dense::rlst_dynamic_array2;

    #[test]
    fn test_cg() {
        let dim = 10;
        let tol = 1E-5;

        let space = ArrayVectorSpace::<f64>::new(dim);
        let mut residuals = Vec::<f64>::new();

        let mut rng = rand::thread_rng();

        let mut mat = rlst_dynamic_array2!(f64, [dim, dim]);

        for index in 0..dim {
            mat[[index, index]] = rng.gen_range(1.0..=2.0);
        }

        let op = DenseMatrixOperator::new(mat.view(), &space, &space);

        let mut rhs = space.zero();
        rhs.view_mut().fill_from_equally_distributed(&mut rng);

        let cg = (CgIteration::new(&op, &rhs))
            .set_callable(|_, res| {
                let res_norm = space.norm(res);
                residuals.push(res_norm);
            })
            .set_tol(tol)
            .print_debug();

        let (_sol, res) = cg.run();
        assert!(res < tol);
    }
}
