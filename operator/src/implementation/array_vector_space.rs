//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;

use crate::space::*;
use rlst_common::types::Scalar;
use rlst_dense::{
    array::{
        operators::scalar_mult::ArrayScalarMult,
        views::{ArrayView, ArrayViewMut},
        Array, DynamicArray,
    },
    base_array::BaseArray,
    data_container::VectorContainer,
    rlst_dynamic_array1,
};

pub struct ArrayVectorSpace<Item: Scalar> {
    dimension: usize,
    _marker: PhantomData<Item>,
}

pub struct ArrayVectorSpaceElement<'a, Item: Scalar> {
    elem: DynamicArray<Item, 1>,
    space: &'a ArrayVectorSpace<Item>,
}

impl<'a, Item: Scalar> ArrayVectorSpaceElement<'a, Item> {
    pub fn new(space: &'a ArrayVectorSpace<Item>) -> Self {
        Self {
            elem: rlst_dynamic_array1!(Item, [space.dimension()]),
            space,
        }
    }
}

impl<Item: Scalar> ArrayVectorSpace<Item> {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            _marker: PhantomData,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<Item: Scalar> LinearSpace for ArrayVectorSpace<Item> {
    type E<'b> = ArrayVectorSpaceElement<'b, Item>;

    type F = Item;

    fn zero(&self) -> Self::E<'_> {
        std::unimplemented!();
    }
}

impl<'a, Item: Scalar> Element for ArrayVectorSpaceElement<'a, Item> {
    type Space = ArrayVectorSpace<Item>;
    type View<'b> = Array<Item, ArrayView<'b, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1> where
        Self: 'b;

    type ViewMut<'b> = Array<Item, ArrayViewMut<'b, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1>
        where
            Self: 'b;

    fn scale_in_place(&mut self, alpha: <Self::Space as LinearSpace>::F) {
        self.view_mut().scale_in_place(alpha)
    }

    fn space(&self) -> &Self::Space {
        self.space
    }

    fn sum_into(&mut self, alpha: <Self::Space as LinearSpace>::F, other: &Self) {
        let result = alpha * other.view();
        self.view_mut().sum_into(alpha * other.view());
        // self.view_mut()
        //     .sum_into(Array::new(ArrayScalarMult::new(alpha, other.view())));
    }

    fn view(&self) -> Self::View<'_> {
        self.elem.view()
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        self.elem.view_mut()
    }
}

#[cfg(test)]
mod test {
    use rlst_dense::rlst_dynamic_array1;

    #[test]
    fn test_vec() {
        let mut mat = rlst_dynamic_array1!(f64, [3]);
        let mat2 = 3.0 * mat.view();
    }
}
