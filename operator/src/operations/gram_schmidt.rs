//! Gram Schmidt orthogonalization
use crate::{space::frame::Frame, Element, FieldType, InnerProductSpace, LinearSpace, NormedSpace};
use rlst_common::types::Scalar;
use rlst_dense::{array::DynamicArray, rlst_dynamic_array2, traits::RandomAccessMut};
pub struct GramSchmidtOrthogonalization;

impl GramSchmidtOrthogonalization {
    pub fn evaluate<'space, Space: InnerProductSpace>(
        frame: &'space Frame<'space, Space>,
    ) -> (Frame<'space, Space>, DynamicArray<FieldType<Space>, 2>) {
        let nelements = frame.len();
        let space = frame.space();

        let mut new_frame = Frame::new(frame.space());
        let mut r_mat = rlst_dynamic_array2!(FieldType<Space>, [nelements, nelements]);

        for (elem_index, elem) in frame.iter().enumerate() {
            let mut new_elem = space.clone(elem);
            for (other_index, other_elem) in frame.iter().take(elem_index).enumerate() {
                let inner = space.inner(elem, other_elem);
                *r_mat.get_mut([other_index, elem_index]).unwrap() = inner;
                new_elem.sum_into(-inner, other_elem);
            }
            *r_mat.get_mut([elem_index, elem_index]).unwrap() =
                <Space::F as Scalar>::from_real(space.norm(&new_elem));

            new_frame.push(new_elem);
        }

        (new_frame, r_mat)
    }
}

#[cfg(test)]
mod test {

    use rlst_common::types::c64;
    use rlst_dense::rlst_dynamic_array1;

    use crate::implementation::array_vector_space::ArrayVectorSpace;
    use rlst_dense::{assert_array_relative_eq, rlst_dynamic_array2, traits::*};

    use super::*;

    #[test]
    pub fn test_gram_schmidt() {
        let space = ArrayVectorSpace::<f64>::new(5);
        let mut vec1 = space.zero();
        let mut vec2 = space.zero();
        let mut vec3 = space.zero();

        vec1.view_mut().fill_from_seed_equally_distributed(0);
        vec2.view_mut().fill_from_seed_equally_distributed(1);
        vec3.view_mut().fill_from_seed_equally_distributed(2);

        let mut frame = Frame::new(&space);

        frame.push(vec1);
        frame.push(vec2);
        frame.push(vec3);

        let (ortho_frame, r_mat) = GramSchmidtOrthogonalization::evaluate(&frame);

        // Check orthogonality
        for index1 in 0..3 {
            for index2 in 0..3 {
                let inner = space.inner(
                    ortho_frame.get(index1).unwrap(),
                    ortho_frame.get(index2).unwrap(),
                );
                println!("{} {}: {}", index1, index2, inner);
            }
        }
    }
}
