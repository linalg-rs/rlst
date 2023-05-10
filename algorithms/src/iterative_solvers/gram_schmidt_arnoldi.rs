// use crate::traits::arnoldi::Arnoldi;
// use rlst_common::traits::*;
// use rlst_common::types::Scalar;
// use rlst_dense::MatrixD;

// #[allow(dead_code)]
// pub struct GramSchmidtArnoldi<
//     T: Scalar,
//     Element: Inner<T = T> + Scale<T = T> + MultSomeInto<T = T> + NewFromZero,
// > {
//     q: Vec<Element>,
//     h: MatrixD<T>,
//     max_steps: usize,
// }

// impl<
//         T: Scalar,
//         Element: Inner<T = T> + Scale<T = T> + MultSomeInto<T = T> + NewFromZero + Norm2<T = T>,
//     > Arnoldi for GramSchmidtArnoldi<T, Element>
// {
//     type T = T;
//     type Element = Element;

//     fn initialize(mut start: Self::Element, max_steps: usize) -> Self {
//         let mut q = Vec::<Self::Element>::new();
//         let h = rlst_dense::rlst_mat![T, (max_steps, max_steps)];
//         start.scale(T::one().div_real(start.norm2()));
//         q.push(start);
//         Self { q, h, max_steps }
//     }

//     #[allow(unused_variables)]
//     fn arnoldi_step<Op: Apply<Self::Element, T = Self::T, Range = Self::Element>>(
//         &self,
//         operator: &Op,
//         step_count: usize,
//     ) -> rlst_common::types::RlstResult<()> {
//         std::unimplemented!()
//     }

//     fn hessenberg_matrix(&self) -> &MatrixD<Self::T> {
//         &self.h
//     }

//     #[allow(unused_variables)]
//     fn basis_element(&self, index: usize) -> Option<&Self::Element> {
//         std::unimplemented!()
//     }

//     fn to_inner(self) -> (Vec<Self::Element>, MatrixD<Self::T>) {
//         std::unimplemented!()
//     }
// }
