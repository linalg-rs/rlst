//! Dual space
use super::{ElementView, LinearSpace};
use crate::dense::types::RlstResult;

/// A dual space
pub trait DualSpace: LinearSpace {
    /// Space type
    type Space: LinearSpace<F = Self::F>;

    /// Dual pairing
    fn dual_pairing(
        &self,
        x: ElementView<Self>,
        other: ElementView<Self::Space>,
    ) -> RlstResult<Self::F>;
}
