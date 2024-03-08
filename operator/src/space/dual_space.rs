use super::{ElementView, LinearSpace};
use rlst_dense::types::RlstResult;

pub trait DualSpace: LinearSpace {
    type Space: LinearSpace<F = Self::F>;

    fn dual_pairing(
        &self,
        x: ElementView<Self>,
        other: ElementView<Self::Space>,
    ) -> RlstResult<Self::F>;
}
