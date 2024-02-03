//! Gram Schmidt orthogonalization
use crate::{
    frame::Frame, space::frame::DefaultFrame, Element, FieldType, InnerProductSpace, NormedSpace,
};
use num::One;
use rlst_common::types::Scalar;
use rlst_dense::{array::DynamicArray, rlst_dynamic_array2, traits::RandomAccessMut};
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    pub fn orthogonalize<'space: 'elem, 'elem, Space: InnerProductSpace>(
        frame: &'space DefaultFrame<'space, 'elem, Space>,
    ) -> (
        DefaultFrame<'space, 'elem, Space>,
        DynamicArray<FieldType<Space>, 2>,
    ) {
        let nelements = frame.len();
        let space = frame.space();

        let mut new_frame = DefaultFrame::new(frame.space());
        let mut r_mat = rlst_dynamic_array2!(FieldType<Space>, [nelements, nelements]);

        for (elem_index, elem) in frame.iter().enumerate() {
            let mut new_elem = space.clone(elem);
            for (other_index, other_elem) in new_frame.iter().take(elem_index).enumerate() {
                let inner = space.inner(&new_elem, other_elem);
                *r_mat.get_mut([other_index, elem_index]).unwrap() = inner;
                new_elem.sum_into(-inner, other_elem);
            }
            let norm = <Space::F as Scalar>::from_real(space.norm(&new_elem));
            *r_mat.get_mut([elem_index, elem_index]).unwrap() = norm;
            new_elem.scale_in_place(<Space::F as One>::one() / norm);
            new_frame.push(new_elem);
        }

        (new_frame, r_mat)
    }
}

// #[cfg(test)]
// mod test {

//     use approx::{assert_abs_diff_eq, assert_relative_eq};
//     use num::Zero;
//     use rlst_common::types::c64;
//     use rlst_dense::{rlst_dynamic_array1, traits::RawAccess};

//     use crate::{implementation::array_vector_space::ArrayVectorSpace, LinearSpace};

//     use super::*;

//     #[test]
//     pub fn test_gram_schmidt() {
//         let space = ArrayVectorSpace::<c64>::new(5);
//         let mut vec1 = space.zero();
//         let mut vec2 = space.zero();
//         let mut vec3 = space.zero();

//         vec1.view_mut().fill_from_seed_equally_distributed(0);
//         vec2.view_mut().fill_from_seed_equally_distributed(1);
//         vec3.view_mut().fill_from_seed_equally_distributed(2);

//         let mut frame = DefaultFrame::new(&space);

//         frame.push(vec1);
//         frame.push(vec2);
//         frame.push(vec3);

//         let (ortho_frame, r_mat) = ModifiedGramSchmidt::orthogonalize(&frame);

//         // Check orthogonality
//         for index1 in 0..3 {
//             for index2 in 0..3 {
//                 let inner = space.inner(
//                     ortho_frame.get(index1).unwrap(),
//                     ortho_frame.get(index2).unwrap(),
//                 );
//                 if index1 == index2 {
//                     assert_relative_eq!(inner, c64::one(), epsilon = 1E-12);
//                 } else {
//                     assert_abs_diff_eq!(inner, c64::zero(), epsilon = 1E-12);
//                 }
//             }
//         }

//         // Check that r is correct.
//         for (index, col) in r_mat.col_iter().enumerate() {
//             let mut actual = space.zero();
//             let expected = frame.get(index).unwrap();
//             let mut coeffs = rlst_dynamic_array1!(c64, [frame.len()]);
//             coeffs.fill_from(col.view());
//             ortho_frame.evaluate(coeffs.data(), &mut actual);
//             let rel_diff = (actual.view() - expected.view()).norm_2() / expected.view().norm_2();
//             assert_abs_diff_eq!(rel_diff, f64::zero(), epsilon = 1E-12);
//         }
//     }
// }
