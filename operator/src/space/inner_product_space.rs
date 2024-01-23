use crate::Element;

use super::LinearSpace;
use rlst_common::types::RlstResult;

pub trait InnerProductSpace: LinearSpace {
    fn inner<'a>(&'a self, x: &Self::E<'a>, other: &Self::E<'a>) -> Self::F
    where
        Self: 'a;
}
