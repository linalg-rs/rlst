use super::LinearSpace;
use crate::types::IndexType;
use crate::IndexLayout;

pub trait IndexableSpace: LinearSpace {
    type Ind: IndexLayout;
    fn dimension(&self) -> IndexType {
        self.index_layout().number_of_global_indices()
    }

    fn index_layout(&self) -> &Self::Ind;
}
