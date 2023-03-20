use super::ElementView;
use super::LinearSpace;
use crate::types::SparseLinAlgResult;

pub trait InnerProductSpace: LinearSpace {
    fn inner<'a>(&self, x: &ElementView<'a, Self>, other: &ElementView<'a, Self>) -> SparseLinAlgResult<Self::F> where Self: 'a;
}
