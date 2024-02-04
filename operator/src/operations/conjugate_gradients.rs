//! Arnoldi Iteration
use crate::{frame::Frame, AsApply, Element, InnerProductSpace, LinearSpace, OperatorBase};
use num::{One, Zero};
use rlst_common::types::Scalar;
use rlst_dense::{
    array::DynamicArray,
    traits::{RandomAccessMut, Shape},
};

pub struct CgIteration<'a, Space: InnerProductSpace, Op: AsApply<Domain = Space, Range = Space>> {
    operator: &'a Op,
    rhs: &'a Space::E,
    res: Space::E,
    x: Space::E,
    p: Space::E,
    alpha: Space::F,
    beta: Space::F,
    iter_count: usize,
    max_steps: usize,
    rel_tol: <Space::F as Scalar>::Real,
}

impl<'a, Space: InnerProductSpace, Op: AsApply<Domain = Space, Range = Space>>
    CgIteration<'a, Space, Op>
{
    pub fn new(op: &'a Op, rhs: &'a Space::E) -> Self {
        Self {
            operator: op,
            rhs,
            res: op.domain().new_from(rhs),
            x: op.domain().zero(),
            p: op.domain().new_from(rhs),
            alpha: <Space::F as Zero>::zero(),
            beta: <Space::F as Zero>::zero(),
            iter_count: 0,
            max_steps: 1000,
            rel_tol: num::cast::<f64, <Space::F as Scalar>::Real>(1E-6).unwrap(),
        }
    }

    pub fn set_initial(self, x: &Space::E) -> Self {
        std::unimplemented!()
    }
}
