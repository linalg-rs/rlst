use crate::types::Scalar;

pub trait AijIterator {
    type T: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::T)>
    where
        Self: 'a;

    fn iter_aij<'a>(&'a self) -> Self::Iter<'a>;
}

pub trait ColumnMajorIterator {
    type T: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::T>
    where
        Self: 'a;

    fn iter_col_major<'a>(&'a self) -> Self::Iter<'a>;
}
