use super::LinearSpace;

pub trait InnerProductSpace: LinearSpace {
    fn inner(&self, x: &Self::E, other: &Self::E) -> Self::F;
}
