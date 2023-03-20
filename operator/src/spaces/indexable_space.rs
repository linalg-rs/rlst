use super::LinearSpace;
use rlst_common::types::IndexType;

pub trait IndexableSpace: LinearSpace {
    fn dimension(&self) -> IndexType;
}
