//! Arnoldi Iteration
use crate::dense::types::RlstScalar;
use crate::operator;
use crate::operator::space::frame::Frame;
use crate::operator::{
    zero_element, AsApply, ElementImpl, ElementType, InnerProductSpace, Operator,
};
use crate::ElementContainerMut;
use crate::IndexableSpace;
use crate::{
    rlst_dynamic_array2, DefaultIteratorMut, Element, ElementContainer, GivensRotations,
    GivensRotationsOps, LinearSpace, OperatorBase, RawAccess, TransMode, TriangularMatrix,
    TriangularOperations, TriangularType, VectorFrame,
};
use core::f64;
use std::cmp::min;
use std::rc::Rc;

pub struct IdOperator<Space: IndexableSpace> {
    domain: Rc<Space>,
    range: Rc<Space>,
}

impl<Space: IndexableSpace> std::fmt::Debug for IdOperator<Space> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim_1 = self.domain().dimension();
        let dim_2 = self.range().dimension();
        write!(f, "Id Operator: [{}x{}]", dim_1, dim_2).unwrap();
        Ok(())
    }
}

impl<Space: IndexableSpace> OperatorBase for IdOperator<Space> {
    type Domain = Space;
    type Range = Space;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<Space: IndexableSpace> AsApply for IdOperator<Space> {
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        _trans_mode: TransMode,
    ) {
        y.fill_inplace(x);
    }

    fn apply<ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>>(
        &self,
        x: Element<ContainerIn>,
        trans_mode: TransMode,
    ) -> operator::ElementType<<Self::Range as LinearSpace>::E> {
        let mut y = zero_element(self.range());
        self.apply_extended(
            <<Self::Range as LinearSpace>::F as num::One>::one(),
            x,
            <<Self::Range as LinearSpace>::F as num::Zero>::zero(),
            y.r_mut(),
            trans_mode,
        );
        y
    }
}

impl<Space: IndexableSpace> IdOperator<Space> {
    pub fn new(domain: Rc<Space>, range: Rc<Space>) -> Self {
        IdOperator { domain, range }
    }
}

/// Iteration for GMRES
pub struct GmresIteration<
    'a,
    Space: InnerProductSpace,
    OpImpl: AsApply<Domain = Space, Range = Space>,
    Container: ElementContainer<E = Space::E>,
    PrecImpl: AsApply<Domain = Space, Range = Space> = IdOperator<Space>,
> where
    <Space::E as ElementImpl>::Space: InnerProductSpace,
{
    operator: Operator<OpImpl>,
    prec: Option<Operator<PrecImpl>>,
    rhs: Element<Container>,
    x: ElementType<Space::E>,
    max_iter: usize,
    restart: usize,
    dim: usize,
    tol: <Space::F as RlstScalar>::Real,
    #[allow(clippy::type_complexity)]
    callable: Option<
        Box<
            dyn FnMut(&ElementType<Space::E>, <<Space as LinearSpace>::F as RlstScalar>::Real) + 'a,
        >,
    >,
    print_debug: bool,
}

type Field<T> = <TriangularMatrix<T> as TriangularOperations>::Item;

impl<
        'a,
        Space: InnerProductSpace,
        OpImpl: AsApply<Domain = Space, Range = Space>,
        PrecImpl: AsApply<Domain = Space, Range = Space>,
        Container: ElementContainer<E = Space::E>,
    > GmresIteration<'a, Space, OpImpl, Container, PrecImpl>
where
    <Space::E as ElementImpl>::Space: InnerProductSpace,
    Space: LinearSpace,
    TriangularMatrix<<Space as LinearSpace>::F>:
        TriangularOperations<Item = <Space as LinearSpace>::F>,
    GivensRotations<<Space as LinearSpace>::F>: GivensRotationsOps<<Space as LinearSpace>::F>,
{
    /// Create a new GMRES iteration
    pub fn new(op: Operator<OpImpl>, rhs: Element<Container>, dim: usize) -> Self {
        let domain = op.domain();
        Self {
            operator: op,
            prec: None,
            rhs,
            x: zero_element(domain),
            max_iter: 10 * dim,
            restart: min(dim, 20),
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
        callable: impl FnMut(&ElementType<Space::E>, <<Space as LinearSpace>::F as RlstScalar>::Real)
            + 'a,
    ) -> Self {
        self.callable = Some(Box::new(callable));
        self
    }

    /// Enable debug printing
    pub fn print_debug(mut self) -> Self {
        self.print_debug = true;
        self
    }

    /// Set preconditioner
    pub fn set_preconditioner(mut self, prec: Operator<PrecImpl>) -> Self {
        self.prec = Some(prec);
        self
    }

    /// Run GMRES
    pub fn run(mut self) -> (ElementType<Space::E>, usize) {
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

        let rhs_norm = self.rhs.norm();
        let atol = self.tol * rhs_norm;

        let eps = num::cast::<f64, <Space::F as RlstScalar>::Real>(2.220446049250313e-16).unwrap(); //TODO: find a way to extract machine precision

        let p_norm = match self.prec.as_ref() {
            Some(prec) => prec.apply(self.rhs.r(), crate::TransMode::NoTrans).norm(),
            None => self.rhs.r().norm(),
        };

        let mut ptol_max_factor: <Space::F as RlstScalar>::Real = num::One::one();

        let mut ptol = p_norm * num::Float::min(ptol_max_factor, atol / rhs_norm);

        let mut presid = num::Zero::zero();

        let mut h = rlst_dynamic_array2!(Field<Space::F>, [self.restart, self.restart + 1]);

        let mut y = vec![<Field<Space::F> as num::Zero>::zero(); self.restart + 1];
        let mut res = self.rhs.duplicate();

        let mut res_norm = num::Zero::zero();
        let mut rel_res = num::Zero::zero();
        let mut breakdown = false;
        let mut inner_iter = 0;

        for iteration in 0..self.max_iter {
            let mut givens_rotations =
                <GivensRotations<Space::F> as GivensRotationsOps<Space::F>>::new();
            let mut v = VectorFrame::default();
            if iteration == 0 {
                res = res.r() - self.operator.apply(self.x.r(), crate::TransMode::NoTrans);
                let res_inner = res.inner_product(res.r());
                res_norm = res_inner.abs().sqrt();
                rel_res = res_norm / rhs_norm;
                if res_norm < atol {
                    if self.print_debug {
                        print_success(0, rel_res);
                    }
                    return (self.x, 0);
                }
            }

            res = match self.prec.as_ref() {
                Some(prec) => prec.apply(res, crate::TransMode::NoTrans),
                None => res,
            };

            let tmp = res.norm();
            let alpha = <Space::F as num::One>::one() / Space::F::from_real(tmp);
            res.scale_inplace(alpha);
            v.push(res.duplicate());

            y[0] = Space::F::from_real(tmp);

            let mut inner = 0;

            for it_count in 0..max_inner {
                let mut w = self
                    .operator
                    .apply(v.get(it_count).unwrap().r(), crate::TransMode::NoTrans);

                w = match self.prec.as_ref() {
                    Some(prec) => prec.apply(w.r(), crate::TransMode::NoTrans),
                    None => w,
                };

                let h0 = w.norm();

                breakdown = false;
                for k in 0..it_count + 1 {
                    let vk = v.get(k).unwrap();
                    let alpha = w.inner_product(vk.r());
                    h.r_mut()[[it_count, k]] = alpha;
                    w.axpy_inplace(-alpha, vk.r());
                }

                let h1 = w.norm();

                if h1 <= h0 * eps {
                    h.r_mut()[[it_count, it_count + 1]] = num::Zero::zero();
                    breakdown = true;
                } else {
                    h.r_mut()[[it_count, it_count + 1]] = num::cast(h1).unwrap();
                    let alpha = <Space::F as num::One>::one() / Space::F::from_real(h1);
                    w.scale_inplace(alpha);
                    v.push(w);
                }

                for i in 0..it_count {
                    let h_col = &mut [h.r()[[it_count, i]], h.r()[[it_count, i + 1]]];
                    givens_rotations.apply_rotation(h_col, i);
                    h.r_mut()[[it_count, i]] = h_col[0];
                    h.r_mut()[[it_count, i + 1]] = h_col[1];
                }

                givens_rotations.add(h.r()[[it_count, it_count]], h.r()[[it_count, it_count + 1]]);

                let (c, s, r) = givens_rotations.get_last();

                h.r_mut()[[it_count, it_count]] = r;
                h.r_mut()[[it_count, it_count + 1]] = num::Zero::zero();

                let c = <Space::F as RlstScalar>::from_real(c);
                let tmp = -s.conj() * y[it_count];
                y[it_count] = c * y[it_count];
                y[it_count + 1] = tmp;
                presid = tmp.abs();
                inner_iter += 1;

                inner = it_count;

                if let Some(callable) = self.callable.as_mut() {
                    rel_res = presid / p_norm;
                    callable(&self.x, rel_res);
                }

                if inner_iter == self.max_iter {
                    break;
                }

                if presid < ptol || breakdown {
                    break;
                }
            }

            let mut y_aux = vec![<Field<Space::F> as num::Zero>::zero(); inner + 1];
            if h.r()[[inner, inner]] == num::Zero::zero() {
                y[inner] = num::Zero::zero();
            }

            y_aux[..(inner + 1)].copy_from_slice(&y[..(inner + 1)]);

            let h_t = TriangularMatrix::new(
                &h.r().into_subview([0, 0], [inner + 1, inner + 1]),
                TriangularType::Lower,
            )
            .unwrap();

            let mut g_solve = rlst_dynamic_array2!(Field<Space::F>, [inner + 1, 1]);

            for (item, other_item) in g_solve.iter_mut().zip(y_aux[0..inner + 1].iter()) {
                *item = *other_item;
            }
            h_t.solve(&mut g_solve, crate::Side::Left, crate::TransMode::Trans);

            for index in 0..inner + 1 {
                let elem = v.get(index).unwrap();
                self.x.axpy_inplace(g_solve.r().data()[index], elem.r());
            }

            res = self.rhs.duplicate();
            res -= self.operator.apply(self.x.r(), crate::TransMode::NoTrans);

            let res_inner = res.inner_product(res.r());
            res_norm = res_inner.abs().sqrt();

            if inner_iter == self.max_iter {
                if res_norm <= atol {
                    return (self.x, 0);
                } else {
                    return (self.x, self.max_iter);
                }
            }

            if res_norm <= atol {
                print_success(inner_iter, rel_res);
                break;
            } else if breakdown {
                break;
            } else if presid <= ptol {
                ptol_max_factor = num::Float::max(
                    eps,
                    num::cast::<f64, <Space::F as RlstScalar>::Real>(0.25).unwrap()
                        * ptol_max_factor,
                );
            } else {
                ptol_max_factor = num::Float::min(
                    num::One::one(),
                    num::cast::<f64, <Space::F as RlstScalar>::Real>(1.5).unwrap()
                        * ptol_max_factor,
                );
            }

            ptol = presid * num::Float::min(ptol_max_factor, atol / res_norm);
        }

        let info = if res_norm <= atol {
            0
        } else {
            if self.print_debug {
                print_fail(self.max_iter, rel_res);
            }
            self.max_iter
        };

        (self.x, info)
    }
}
