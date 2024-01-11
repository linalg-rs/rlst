use super::ElementView;
use super::LinearSpace;
use rlst_common::types::RlstResult;

pub trait InnerProductSpace: LinearSpace {
    fn inner<'a>(
        &self,
        x: &ElementView<'a, Self>,
        other: &ElementView<'a, Self>,
    ) -> RlstResult<Self::F>
    where
        Self: 'a;
}
