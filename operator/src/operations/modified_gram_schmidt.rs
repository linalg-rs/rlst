//! Gram Schmidt orthogonalization
use crate::{frame::Frame, Element, InnerProductSpace, NormedSpace};
use num::One;
use rlst_common::types::Scalar;
use rlst_dense::{
    array::DynamicArray,
    traits::{RandomAccessMut, Shape},
};
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    pub fn orthogonalize<
        Item: Scalar,
        Elem: Element<F = Item>,
        Space: InnerProductSpace<E = Elem, F = Item>,
        FrameType: Frame<E = Elem>,
    >(
        space: &Space,
        frame: &mut FrameType,
        r_mat: &mut DynamicArray<Item, 2>,
    ) {
        let nelements = frame.len();

        assert_eq!(r_mat.shape(), [nelements, nelements]);

        r_mat.set_zero();

        for elem_index in 0..nelements {
            let mut elem = space.new_from(frame.get(elem_index).unwrap());
            for (other_index, other_elem) in frame.iter().take(elem_index).enumerate() {
                let inner = space.inner(&elem, other_elem);
                *r_mat.get_mut([other_index, elem_index]).unwrap() = inner;
                elem.axpy_inplace(-inner, other_elem);
            }
            let norm = <Item as Scalar>::from_real(space.norm(&elem));
            *r_mat.get_mut([elem_index, elem_index]).unwrap() = norm;
            elem.scale_inplace(<Item as One>::one() / norm);
            frame.get_mut(elem_index).unwrap().fill_inplace(&elem);
        }
    }
}

#[cfg(test)]
mod test {

    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use num::Zero;
    use rlst_common::types::c64;
    use rlst_dense::rlst_dynamic_array2;
    use rlst_dense::{rlst_dynamic_array1, traits::RawAccess};

    use crate::space::frame::VectorFrame;
    use crate::{implementation::array_vector_space::ArrayVectorSpace, LinearSpace};

    use super::*;

    #[test]
    pub fn test_gram_schmidt() {
        let space = ArrayVectorSpace::<c64>::new(5);
        let mut vec1 = space.zero();
        let mut vec2 = space.zero();
        let mut vec3 = space.zero();

        vec1.view_mut().fill_from_seed_equally_distributed(0);
        vec2.view_mut().fill_from_seed_equally_distributed(1);
        vec3.view_mut().fill_from_seed_equally_distributed(2);

        let mut frame = VectorFrame::default();

        let mut original = VectorFrame::default();

        frame.push(vec1);
        frame.push(vec2);
        frame.push(vec3);

        for elem in frame.iter() {
            original.push(space.new_from(elem));
        }

        let mut r_mat = rlst_dynamic_array2!(c64, [3, 3]);

        ModifiedGramSchmidt::orthogonalize(&space, &mut frame, &mut r_mat);

        // Check orthogonality
        for index1 in 0..3 {
            for index2 in 0..3 {
                let inner = space.inner(frame.get(index1).unwrap(), frame.get(index2).unwrap());
                if index1 == index2 {
                    assert_relative_eq!(inner, c64::one(), epsilon = 1E-12);
                } else {
                    assert_abs_diff_eq!(inner, c64::zero(), epsilon = 1E-12);
                }
            }
        }

        // Check that r is correct.
        for (index, col) in r_mat.col_iter().enumerate() {
            let mut actual = space.zero();
            let expected = original.get(index).unwrap();
            let mut coeffs = rlst_dynamic_array1!(c64, [frame.len()]);
            coeffs.fill_from(col.view());
            frame.evaluate(coeffs.data(), &mut actual);
            let rel_diff = (actual.view() - expected.view()).norm_2() / expected.view().norm_2();
            assert_abs_diff_eq!(rel_diff, f64::zero(), epsilon = 1E-12);
        }
    }
}
