pub use rlst_common::types::*;
pub use rlst_operator::*;
use std::fmt::Debug;

pub struct PolynomialSpace;
impl LinearSpace for PolynomialSpace {
    type F = f64;
    type E<'a> = Polynomial;
}

#[derive(Debug)]
pub struct Polynomial {
    monomial_coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn from_monomial(monomial_coeffs: &[f64]) -> Self {
        Polynomial {
            monomial_coeffs: monomial_coeffs.to_owned(),
        }
    }
}

#[derive(Debug)]
pub struct PolynomialView<'a> {
    monomial_coeffs: &'a [f64],
}

impl<'a> PolynomialView<'a> {
    fn eval(&self, x: f64) -> f64 {
        self.monomial_coeffs.iter().rev().fold(0., |r, c| r * x + c)
    }
}
pub struct PolynomialViewMut<'a> {
    monomial_coeffs: &'a mut [f64],
}

impl Element for Polynomial {
    type Space = PolynomialSpace;
    type View<'b> = PolynomialView<'b> where Self: 'b ;
    type ViewMut<'b> = PolynomialViewMut<'b> where Self: 'b;

    fn view<'b>(&'b self) -> Self::View<'b> {
        PolynomialView {
            monomial_coeffs: &self.monomial_coeffs,
        }
    }
    fn view_mut<'b>(&'b mut self) -> PolynomialViewMut<'b> {
        PolynomialViewMut {
            monomial_coeffs: &mut self.monomial_coeffs,
        }
    }
}

pub struct PointwiseEvaluatorSpace;
impl LinearSpace for PointwiseEvaluatorSpace {
    type F = f64;
    type E<'a> = PointwiseEvaluate;
}
impl DualSpace for PointwiseEvaluatorSpace {
    type Space = PolynomialSpace;

    fn dual_pairing(
        &self,
        x: ElementView<Self>,
        p: ElementView<Self::Space>,
    ) -> RlstResult<Self::F> {
        Ok(x.scale * p.eval(x.x))
    }
}

pub struct PointwiseEvaluate {
    x: f64,
    scale: f64,
}
impl PointwiseEvaluate {
    pub fn new(x: f64) -> Self {
        PointwiseEvaluate { x, scale: 1. }
    }
}

impl Element for PointwiseEvaluate {
    type Space = PointwiseEvaluatorSpace;
    type View<'a> = &'a PointwiseEvaluate where Self: 'a;
    type ViewMut<'a> = &'a mut PointwiseEvaluate where Self: 'a;
    fn view<'a>(&'a self) -> Self::View<'a> {
        &self
    }
    fn view_mut<'a>(&'a mut self) -> Self::ViewMut<'a> {
        self
    }
}

#[derive(Debug)]
struct Derivative;
impl OperatorBase for Derivative {
    type Domain = PolynomialSpace;
    type Range = PolynomialSpace;
    fn as_apply(&self) -> Option<&dyn AsApply<Domain = Self::Domain, Range = Self::Range>> {
        Some(self)
    }
}
impl AsApply for Derivative {
    fn apply(&self, p: PolynomialView, dp: PolynomialViewMut) -> RlstResult<()> {
        for (i, c) in p.monomial_coeffs[1..].iter().enumerate() {
            dp.monomial_coeffs[i] = (1. + i as f64) * c;
        }
        dp.monomial_coeffs[p.monomial_coeffs.len() - 1] = 0.;
        Ok(())
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use crate::{
        AsApply, Derivative, DualSpace, Element, OperatorBase, PointwiseEvaluate,
        PointwiseEvaluatorSpace, Polynomial, PolynomialSpace,
    };
    use rlst_common::types::RlstResult;

    #[test]
    fn test_poly_eval() {
        let p = Polynomial::from_monomial(&[1., 2., 3.]);
        assert_eq!(p.view().eval(2.), 17.);
    }

    #[test]
    fn test_dual() -> RlstResult<()> {
        let ds = PointwiseEvaluatorSpace;
        let p = Polynomial::from_monomial(&[1., 2., 3.]);
        let n = PointwiseEvaluate::new(2.);
        let r = ds.dual_pairing(n.view(), p.view())?;
        assert_eq!(r, 17.);
        Ok(())
    }

    #[test]
    fn test_derivative() -> RlstResult<()> {
        let p = Polynomial::from_monomial(&[1., 2., 3.]);
        let mut dp = Polynomial::from_monomial(&[1., 1., 1.]);
        let d_ = Derivative;
        let d = &d_ as &dyn OperatorBase<Domain = PolynomialSpace, Range = PolynomialSpace>;
        d.apply(p.view(), dp.view_mut())?;
        assert_eq!(dp.monomial_coeffs, vec![2., 6., 0.]);
        Ok(())
    }
}
