use super::{ElementView, LinearSpace};
use crate::types::SparseLinAlgResult;

pub trait DualSpace: LinearSpace {
    type Space: LinearSpace<F = Self::F>;

    fn dual_pairing(
        &self,
        x: ElementView<Self>,
        other: ElementView<Self::Space>,
    ) -> SparseLinAlgResult<Self::F>;
}
