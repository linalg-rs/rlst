use crate::types::Scalar;

pub trait Scale {
    type T: Scalar;

    fn scale(&mut self, alpha: Self::T);
}

pub trait FillFrom {
    fn fill_from(&mut self, other: &Self);
}
