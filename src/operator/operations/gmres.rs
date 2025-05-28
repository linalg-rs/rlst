//! Arnoldi Iteration
use crate::dense::types::RlstScalar;
use crate::operator::space::frame::Frame;
use crate::operator::{
    zero_element, AsApply, ElementImpl, ElementType, InnerProductSpace, Operator,
};
use crate::{
    rlst_dynamic_array2, DefaultIteratorMut, Element, ElementContainer, LinearSpace, OperatorBase,
    RawAccess, TriangularMatrix, TriangularOperations, TriangularType, VectorFrame,
};

fn apply_givens<Item: RlstScalar>(h: &mut [Item; 2], cs: Item, sn: Item) {
    let temp = cs * h[0] + sn * h[1];
    h[1] = -sn * h[0] + cs * h[1];
    h[0] = temp;
}

fn generate_givens<Item: RlstScalar>(a: Item, b: Item) -> (Item, Item) {
    if b == num::Zero::zero() {
        (num::One::one(), num::Zero::zero())
    } else {
        let r = (a * a + b * b).sqrt();
        let c = a / r;
        let s = b / r;
        (c, s)
    }
}

/// Iteration for GMRES
pub struct GmresIteration<
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
    restart: usize,
    dim: usize,
    tol: <Space::F as RlstScalar>::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<Box<dyn FnMut(&ElementType<Space::E>, <<Space as LinearSpace>::F as RlstScalar>::Real) + 'a>>,
    print_debug: bool,
}

type Field<T> = <TriangularMatrix<T> as TriangularOperations>::Item;

impl<
        'a,
        Space: InnerProductSpace,
        OpImpl: AsApply<Domain = Space, Range = Space>,
        Container: ElementContainer<E = Space::E>,
    > GmresIteration<'a, Space, OpImpl, Container>
where
    <Space::E as ElementImpl>::Space: InnerProductSpace,
    Space: LinearSpace,
    TriangularMatrix<<Space as LinearSpace>::F>:
        TriangularOperations<Item = <Space as LinearSpace>::F>,
{
    /// Create a new GMRES iteration
    pub fn new(op: Operator<OpImpl>, rhs: Element<Container>, dim: usize) -> Self {
        let domain = op.domain();
        Self {
            operator: op,
            rhs,
            x: zero_element(domain),
            max_iter: 1000,
            restart: 30,
            dim,
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

    /// Set restart
    pub fn set_restart(mut self, restart: usize) -> Self {
        self.restart = restart;
        self
    }

    /// Set the cammable
    pub fn set_callable(
        mut self,
        callable: impl FnMut(&ElementType<Space::E>, <<Space as LinearSpace>::F as RlstScalar>::Real) + 'a,
    ) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Run GMRES
    pub fn run(mut self) -> (ElementType<Space::E>, <Space::F as RlstScalar>::Real) {
        fn print_success<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "GMRES converged in {} iterations with relative residual {:+E}.",
                it_count, rel_res
            );
        }

        fn print_fail<T: RlstScalar>(it_count: usize, rel_res: T) {
            println!(
                "GMRES did not converge in {} iterations. Final relative residual is {:+E}.",
                it_count, rel_res
            );
        }

        if self.restart > self.dim {
            println!("Setting number of inner iterations (restart) to maximum allowed, which is the domain dimension");

            self.restart = self.dim;
        }

        let max_inner = self.restart;
        let max_outer = self.max_iter / max_inner;

        let mut res = self.rhs.duplicate();
        res -= self.operator.apply(self.x.r(), crate::TransMode::NoTrans);

        let res_inner = res.inner_product(res.r());
        let res_norm = res_inner.abs().sqrt();

        let mut rhs_norm = self.rhs.norm();
        let mut rel_res = res_norm / rhs_norm;

        if rhs_norm == num::Zero::zero() {
            rhs_norm = num::One::one();
        }

        if let Some(callable) = self.callable.as_mut() {
            callable(&self.x, rel_res);
        }

        if rel_res < self.tol {
            if self.print_debug {
                print_success(0, rel_res);
            }
            return (self.x, rel_res);
        }

        let res_0 = rhs_norm;

        let mut h_dim = 0;

        for outer in 0..max_outer {
            let mut v = VectorFrame::default();
            let mut h = rlst_dynamic_array2!(Field<Space::F>, [max_inner + 1, max_inner]);
            let mut g = vec![<Field<Space::F> as num::Zero>::zero(); max_inner + 1];
            g[0] = num::cast(res_norm).unwrap();

            let mut givens_cs = Vec::new();
            let mut givens_sn = Vec::new();

            let mut aux_rhs = self.rhs.duplicate();
            let alpha = <Space::F as num::One>::one() / Space::F::from_real(rhs_norm);
            aux_rhs.scale_inplace(alpha);
            v.push(aux_rhs);

            for it_count in 0..max_inner {
                let mut w = self
                    .operator
                    .apply(v.get(it_count).unwrap().r(), crate::TransMode::NoTrans);

                for k in 0..it_count + 1 {
                    let vk = v.get(k).unwrap();
                    let alpha = w.inner_product(vk.r());
                    h.r_mut()[[k, it_count]] = alpha;
                    w.axpy_inplace(-alpha, vk.r());
                }

                let w_norm = w.norm();
                h.r_mut()[[it_count + 1, it_count]] = num::cast(w_norm).unwrap();

                if w_norm != num::Zero::zero() {
                    let alpha = <Space::F as num::One>::one() / Space::F::from_real(w_norm);
                    w.scale_inplace(alpha);
                    v.push(w);
                }

                if it_count > 0 {
                    for i in 0..it_count {
                        let h_col = &mut [h.r()[[i, it_count]], h.r()[[i + 1, it_count]]];
                        apply_givens(h_col, givens_cs[i], givens_sn[i]);
                        h.r_mut()[[i, it_count]] = h_col[0];
                        h.r_mut()[[i + 1, it_count]] = h_col[1];
                    }
                }

                // Generate new Givens rotation to zero out H[j+1][j]
                let (cs, sn) =
                    generate_givens(h.r()[[it_count, it_count]], h.r()[[it_count + 1, it_count]]);
                givens_cs.push(cs);
                givens_sn.push(sn);
                // Apply to H
                let h_col = &mut [h.r()[[it_count, it_count]], h.r()[[it_count + 1, it_count]]];
                apply_givens(h_col, cs, sn);
                h.r_mut()[[it_count, it_count]] = h_col[0];
                h.r_mut()[[it_count + 1, it_count]] = h_col[1];

                // Apply to residual vector g
                let temp = cs * g[it_count] + sn * g[it_count + 1];

                g[it_count + 1] = -sn * g[it_count] + cs * g[it_count + 1];
                g[it_count] = temp;
                let residual = g[it_count + 1].abs();
                let rel_res = residual / res_0;
                h_dim = it_count + 1;

                if let Some(callable) = self.callable.as_mut() {
                    callable(&self.x, rel_res);
                }

                if it_count < max_inner - 1 {
                    if rel_res < self.tol {
                        break;
                    }
                }
            }

            let h_t = TriangularMatrix::new(
                &h.into_subview([0, 0], [h_dim, h_dim]),
                TriangularType::Upper,
            )
            .unwrap();
            let mut g_solve = rlst_dynamic_array2!(Field<Space::F>, [h_dim, 1]);

            for (item, other_item) in g_solve.iter_mut().zip(g[0..h_dim].iter()) {
                *item = *other_item;
            }
            h_t.solve(&mut g_solve, crate::Side::Left, crate::TransMode::NoTrans);
            for index in 0..h_dim {
                let elem = v.get(index).unwrap();
                let coeff = g_solve.r().data()[index];
                self.x.axpy_inplace(coeff, elem.r());
            }

            let mut res = self.rhs.duplicate();
            res -= self.operator.apply(self.x.r(), crate::TransMode::NoTrans);
            let res_inner = res.inner_product(res.r());
            let res_norm = res_inner.abs().sqrt();

            rel_res = res_norm / res_0;

            if let Some(callable) = self.callable.as_mut() {
                callable(&self.x, rel_res);
            }

            if rel_res < self.tol {
                print_success(outer * max_inner + h_dim, rel_res);
                return (self.x, rel_res);
            }
        }

        if self.print_debug {
            print_fail(self.max_iter, rel_res);
        }
        (self.x, rel_res)
    }
}
