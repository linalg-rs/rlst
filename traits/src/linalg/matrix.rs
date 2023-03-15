//! Basic trait for matrices

use crate::types::{IndexType, Scalar};
use crate::IndexLayout;

pub trait Matrix {
    type T: Scalar;
    type Ind: IndexLayout;

    type View<'a>
    where
        Self: 'a;
    type ViewMut<'a>
    where
        Self: 'a;

    fn view<'a>(&'a self) -> Option<Self::View<'a>>;
    fn view_mut<'a>(&'a mut self) -> Option<Self::ViewMut<'a>>;

    fn column_layout(&self) -> &Self::Ind;
    fn row_layout(&self) -> &Self::Ind;

    fn shape(&self) -> (IndexType, IndexType) {
        (
            self.row_layout().number_of_global_indices(),
            self.column_layout().number_of_global_indices(),
        )
    }
}
