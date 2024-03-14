//! Gram Schmidt orthogonalization
use crate::{frame::Frame, Element, InnerProductSpace, NormedSpace};
use num::One;
use rlst_dense::types::RlstScalar;
use rlst_dense::{
    array::DynamicArray,
    traits::{RandomAccessMut, Shape},
};
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    pub fn orthogonalize<
        Item: RlstScalar,
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
            let norm = <Item as RlstScalar>::from_real(space.norm(&elem));
            *r_mat.get_mut([elem_index, elem_index]).unwrap() = norm;
            elem.scale_inplace(<Item as One>::one() / norm);
            frame.get_mut(elem_index).unwrap().fill_inplace(&elem);
        }
    }
}
