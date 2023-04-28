use super::LinearSpace;

pub trait IndexableSpace: LinearSpace {
    fn dimension(&self) -> usize;
}
