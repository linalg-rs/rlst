//! Gram Schmidt orthogonalization
use crate::dense::types::RlstScalar;
use crate::dense::{
    array::DynamicArray,
    traits::{RandomAccessMut, Shape},
};
use crate::operator::{frame::Frame, ElementImpl, InnerProductSpace};
/// Gram Schmidt orthogonalization
pub struct ModifiedGramSchmidt;

impl ModifiedGramSchmidt {
    /// Orthogonalize a matrix
    pub fn orthogonalize<
        Item: RlstScalar,
        ElemImpl: ElementImpl<F = Item>,
        FrameType: Frame<E = ElemImpl>,
    >(
        frame: &mut FrameType,
        r_mat: &mut DynamicArray<Item, 2>,
    ) where
        ElemImpl::Space: InnerProductSpace<E = ElemImpl, F = Item>,
    {
        let nelements = frame.len();

        assert_eq!(r_mat.shape(), [nelements, nelements]);

        r_mat.set_zero();

        for elem_index in 0..nelements {
            // The duplicate is necessary since we have to clone out the element from the frame. Otherwise
            // we would have the problem of accessing references to the frame while keeping the mutable
            // ref alive.
            let mut elem = frame.get(elem_index).unwrap().duplicate();
            for (other_index, other_elem) in frame.iter().take(elem_index).enumerate() {
                let inner = elem.inner_product(other_elem.r());
                *r_mat.get_mut([other_index, elem_index]).unwrap() = inner;
                elem.axpy_inplace(-inner, other_elem.r());
            }
            let norm = <Item as RlstScalar>::from_real(elem.norm());
            *r_mat.get_mut([elem_index, elem_index]).unwrap() = norm;
            elem /= norm;

            frame.get_mut(elem_index).unwrap().fill_inplace(elem);
        }
    }
}
