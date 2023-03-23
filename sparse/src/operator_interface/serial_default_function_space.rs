//! An indexable vector space has elements that can be indexed as n-dimensional vectors.

use std::marker::PhantomData;

use crate::index_layout::DefaultSerialIndexLayout;
use crate::traits::index_layout::IndexLayout;
use crate::traits::indexable_vector::{Inner, Norm2};
use crate::vector::DefaultSerialVector;
use rlst_common::types::{IndexType, Scalar};
use rlst_operator::{Element, IndexableSpace, InnerProductSpace, NormedSpace};

pub struct LocalIndexableVectorSpace<T: Scalar> {
    index_layout: DefaultSerialIndexLayout,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> LocalIndexableVectorSpace<T> {
    pub fn new(n: IndexType) -> Self {
        LocalIndexableVectorSpace {
            index_layout: DefaultSerialIndexLayout::new(n),
            _phantom: PhantomData,
        }
    }
}

pub struct LocalIndexableVectorSpaceElement<'a, T: Scalar> {
    space: &'a LocalIndexableVectorSpace<T>,
    data: crate::vector::DefaultSerialVector<T>,
}

impl<'a, T: Scalar> Element for LocalIndexableVectorSpaceElement<'a, T> {
    type Space = LocalIndexableVectorSpace<T>;
    type View<'b> = &'b crate::vector::DefaultSerialVector<T> where Self: 'b;
    type ViewMut<'b> = &'b mut crate::vector::DefaultSerialVector<T> where Self: 'b;

    fn space(&self) -> &Self::Space {
        self.space
    }

    fn view<'b>(&'b self) -> Self::View<'b> {
        &self.data
    }

    fn view_mut<'b>(&'b mut self) -> Self::ViewMut<'b> {
        &mut self.data
    }
}

impl<T: Scalar> rlst_operator::LinearSpace for LocalIndexableVectorSpace<T> {
    type F = T;
    type E<'a> = LocalIndexableVectorSpaceElement<'a, T> where Self: 'a;

    fn create_element<'a>(&'a self) -> Self::E<'a> {
        LocalIndexableVectorSpaceElement {
            space: &self,
            data: DefaultSerialVector::new(self.index_layout.number_of_global_indices()),
        }
    }
}

impl<T: Scalar> IndexableSpace for LocalIndexableVectorSpace<T> {
    fn dimension(&self) -> IndexType {
        self.index_layout.number_of_global_indices()
    }
}

impl<T: Scalar> InnerProductSpace for LocalIndexableVectorSpace<T> {
    fn inner<'a>(
        &self,
        x: &rlst_operator::ElementView<'a, Self>,
        other: &rlst_operator::ElementView<'a, Self>,
    ) -> rlst_common::types::SparseLinAlgResult<Self::F>
    where
        Self: 'a,
    {
        x.inner(other)
    }
}

impl<T: Scalar> NormedSpace for LocalIndexableVectorSpace<T> {
    fn norm<'a>(&'a self, x: &rlst_operator::ElementView<'a, Self>) -> <Self::F as Scalar>::Real {
        x.norm_2()
    }
}
