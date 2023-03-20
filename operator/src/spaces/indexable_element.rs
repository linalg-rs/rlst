//! An indexable element has an associated index set.
use crate::index_layout::IndexLayout;

pub trait IndexableElement: super::element::Element {
    type Ind: IndexLayout;
    fn index_layout(&self) -> &Self::Ind;
}
